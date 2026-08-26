import numpy as np
import h5py
import os
import sys
from multiprocessing import Pool, shared_memory
import multiprocessing as mp

from . import worker_pool
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from .read_data import ReadData
from . import pair_kernel


class MeasureWBoxJackknife(MeasureIABase, ReadData):
	r"""Class that contains all methods for the measurements of $\xi_{gg}$ and $\xi_{g+}$ for $w_{gg}$ and $w_{g+}$
	including the jackknife realisations needed for the covariance estimation with Cartesian simulation data.

	Methods
	-------
	_measure_xi_rp_pi_box_jk_brute()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (rp, pi) grid binning including jackknife realisations in a periodic box
		using 1 CPU.
	_measure_xi_rp_pi_box_jk_tree()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (rp, pi) grid binning including jackknife realisations in a periodic box
		using 1 CPU and KDTree for extra speed.
	_measure_xi_rp_pi_box_jk_batch()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (rp, pi) grid binning including jackknife realisations in a periodic box
		using 1 CPU for a batch of indices.
		Support function of _measure_xi_rp_pi_box_jk_multiprocessing().
	_measure_xi_rp_pi_box_jk_multiprocessing()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (rp, pi) grid binning including jackknife realisations in a periodic
		box using >1 CPUs.

	Notes
	-----
	Inherits attributes from 'SimInfo', where 'boxsize', 'L_0p5' and 'snap_group' are used in this class.
	Inherits attributes from 'MeasureIABase', where 'data', 'output_file_name', 'periodicity', 'Num_position',
	'Num_shape', 'r_min', 'r_max', 'num_bins_r', 'num_bins_pi', 'r_bins', 'pi_bins', 'mu_r_bins' are used.

	"""

	def __init__(
			self,
			data,
			output_file_name,
			simulation=None,
			snapshot=None,
			separation_limits=[0.1, 20.0],
			num_bins_r=8,
			num_bins_pi=20,
			pi_max=None,
			boxsize=None,
			periodicity=True,
	):
		"""
		The __init__ method of the MeasureWSimulations class.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		return

	def _measure_xi_rp_pi_box_jk_brute(self, dataset_name, L_subboxes, masks=None, return_output=False,
									   jk_group_name="", ellipticity='distortion'):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (rp, pi) bins for an object created with MeasureIABox. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.

		Returns
		-------
		ndarrays
			$\xi_{gg}$ and $\xi_{g+}$, r_p bins, pi bins, S+D, DD, RR (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self, require_full_masks=True,
		)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks, L_subboxes)
		sample_set.jk_pos = jackknife_region_indices_pos
		sample_set.jk_shape = jackknife_region_indices_shape
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		weight_shape = sample_set.weight_shape
		e = sample_set.e
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		num_box = L_subboxes ** 3
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, backend="brute", jk=True, num_box=num_box)
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
		DD_jk = grids.DD_jk
		Splus_D_jk = grids.Splus_D_jk
		R_jk = pair_kernel.compute_R_jk(e, weight_shape, jackknife_region_indices_shape, num_box, getattr(self, "responsivity_correction", True))
		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / (num_box)
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_g_plus_denom = RR_g_plus.copy()  # guard against empty samples/bins in the divisions; raw RR grids are written to file
		RR_g_plus_denom[RR_g_plus_denom == 0] = 1
		RR_gg_denom = RR_gg.copy()
		RR_gg_denom[RR_gg_denom == 0] = 1
		correlation = Splus_D / RR_g_plus_denom  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus_denom  # (Scross_D - Scross_R) / RR_g_plus
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				corr = (Splus_D * (2 * R) - Splus_D_jk[i]) / (
						RR_jk_denom * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD", data=(Splus_D * (2 * R) - Splus_D_jk[i]) / (
						2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, pi_bins, Splus_D, DD, RR_g_plus


	def _measure_xi_rp_pi_box_jk_tree(self, dataset_name, L_subboxes, masks=None, return_output=False,
									  jk_group_name="", ellipticity='distortion'):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (rp, pi) bins for an object created with MeasureIABox. Uses 1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.

		Returns
		-------
		ndarrays
			$\xi_{gg}$ and $\xi_{g+}$, r_p bins, pi bins, S+D, DD, RR (if no output file is specified)

		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self, require_full_masks=True,
		)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks, L_subboxes)
		sample_set.jk_pos = jackknife_region_indices_pos
		sample_set.jk_shape = jackknife_region_indices_shape
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		weight_shape = sample_set.weight_shape
		e = sample_set.e
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		num_box = L_subboxes ** 3
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, backend="tree", jk=True, num_box=num_box)
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
		DD_jk = grids.DD_jk
		Splus_D_jk = grids.Splus_D_jk
		R_jk = pair_kernel.compute_R_jk(e, weight_shape, jackknife_region_indices_shape, num_box, getattr(self, "responsivity_correction", True))
		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / num_box
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_g_plus_denom = RR_g_plus.copy()  # guard against empty samples/bins in the divisions; raw RR grids are written to file
		RR_g_plus_denom[RR_g_plus_denom == 0] = 1
		RR_gg_denom = RR_gg.copy()
		RR_gg_denom[RR_gg_denom == 0] = 1
		correlation = Splus_D / RR_g_plus_denom  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus_denom  # (Scross_D - Scross_R) / RR_g_plus
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				corr = (Splus_D * (2 * R) - Splus_D_jk[i]) / (
						RR_jk_denom * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD", data=(Splus_D * (2 * R) - Splus_D_jk[i]) / (
						2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, pi_bins, Splus_D, DD, RR_g_plus


	def _measure_xi_rp_pi_box_jk_batch(self, i):
		r"""(rp, pi) jackknife shape-sample batch worker. Reads shared memory (incl. jk region
		indices) and delegates the counting + union-deletion jk accumulation to pair_kernel.accumulate
		(jk=True, reusing the parent's shared self.pos_tree). Support function for
		_measure_xi_rp_pi_box_jk_multiprocessing().
		"""
		if i + self.chunk_size > self.Num_shape_masked:
			i2 = self.Num_shape_masked
		else:
			i2 = i + self.chunk_size

		shms = []
		shared_data = {}
		for name, shape, dtype in self.shm_infos:
			shm = shared_memory.SharedMemory(name=name)
			shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
			shms.append(shm)

		sample_set = pair_kernel.SampleSet(
			pos=shared_data[f"positions_{self.ID_shm}"],
			pos_shape=shared_data[f"positions_shape_sample_{self.ID_shm}"][i:i2],
			weight=shared_data[f"weight_{self.ID_shm}"],
			weight_shape=shared_data[f"weight_shape_{self.ID_shm}"][i:i2],
			axis_direction=shared_data[f"axis_direction_{self.ID_shm}"][i:i2],
			e=shared_data[f"e_{self.ID_shm}"][i:i2],
			LOS_ind=self.LOS_ind,
			not_LOS=self.not_LOS,
			jk_pos=shared_data[f"jk_region_indices_pos_{self.ID_shm}"],
			jk_shape=shared_data[f"jk_region_indices_shape_{self.ID_shm}"][i:i2],
		)
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=self.R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, pos_tree=self.pos_tree,
									   jk=True, num_box=self.num_box)
		for shm in shms:
			shm.close()
		return grids.Splus_D, grids.Scross_D, grids.DD, grids.DD_jk, grids.Splus_D_jk

	def _measure_xi_rp_pi_box_jk_multiprocessing(self, dataset_name, L_subboxes, temp_file_path,
												 masks=None, return_output=False, jk_group_name="",
												 chunk_size=1000, num_nodes=1, ellipticity='distortion'
												 ):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (rp, pi) bins for an object created with MeasureIABox. Uses >1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
			Default is 1000.
		num_nodes : int, optional
			Number of CPUs used in the multiprocessing. Default is 1.
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.

		Returns
		-------
		ndarrays
			$\xi_{gg}$ and $\xi_{g+}$, r_p bins, pi bins, S+D, DD, RR (if no output file is specified)

		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self, require_full_masks=True,
		)
		positions = sample_set.pos
		positions_shape_sample = sample_set.pos_shape
		axis_direction = sample_set.axis_direction
		e = sample_set.e
		weight = sample_set.weight
		weight_shape = sample_set.weight_shape
		self.Num_position_masked = len(positions)
		self.Num_shape_masked = len(positions_shape_sample)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")
		self.LOS_ind = sample_set.LOS_ind
		self.not_LOS = sample_set.not_LOS
		self.R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		self.sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		self.sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		self.num_box = L_subboxes ** 3
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(
			masks,
			L_subboxes)

		# Build the shared position tree on whatever coordinates the binning
		# queries -- do not hardcode the projection here. BoxRpPi chooses between
		# the 3D positions and their 2D projection depending on the configuration
		# (benchmarks/FINDINGS.md F7), and the workers build their chunk trees with
		# binning.tree_coords, so a hardcoded convention here silently disagrees
		# with them -- scipy then raises "Trees passed to query_ball_tree have
		# different dimensionality".
		_binning = pair_kernel.BoxRpPi(self)
		self.pos_tree = KDTree(_binning.tree_coords(positions, self.not_LOS),
							   boxsize=self.boxsize)
		indices = np.arange(0, len(positions_shape_sample), chunk_size)
		self.chunk_size = chunk_size

		# create temp hdf5 from which data can be read. del self.data, but save it in this method to reduce RAM
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		file_temp = h5py.File(f"{temp_file_path}/w_{self.simname}_temp_data_{figname_dataset_name}.hdf5", "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
		write_dataset_hdf5(file_temp, "jackknife_region_indices_shape", jackknife_region_indices_shape)
		write_dataset_hdf5(file_temp, "jackknife_region_indices_pos", jackknife_region_indices_pos)
		file_temp.close()
		self.ID_shm = np.random.randint(100000)
		try:
			shared_data = {
				f"positions_{self.ID_shm}": positions,
				f"positions_shape_sample_{self.ID_shm}": positions_shape_sample,
				f"axis_direction_{self.ID_shm}": axis_direction,
				f"e_{self.ID_shm}": e,
				f"weight_{self.ID_shm}": weight,
				f"weight_shape_{self.ID_shm}": weight_shape,
				f"jk_region_indices_pos_{self.ID_shm}": jackknife_region_indices_pos,
				f"jk_region_indices_shape_{self.ID_shm}": jackknife_region_indices_shape,
			}
			for k in shared_data.keys():
				try:
					old = shared_memory.SharedMemory(name=k)
					old.unlink()
				except FileNotFoundError:
					pass
			shm_blocks, self.shm_infos = [], []
			for k in shared_data.keys():
				shm = shared_memory.SharedMemory(name=k, create=True, size=shared_data[k].nbytes)
				shared_arr = np.ndarray(shared_data[k].shape, dtype=shared_data[k].dtype, buffer=shm.buf)
				np.copyto(shared_arr, shared_data[k])
				shm_blocks.append(shm)
				self.shm_infos.append([k, shared_data[k].shape, shared_data[k].dtype])
			self.data = {}
			if masks is not None:
				masks = {}
			del shared_data, shared_arr
			del positions, positions_shape_sample, axis_direction, weight, weight_shape, jackknife_region_indices_pos, jackknife_region_indices_shape
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._measure_xi_rp_pi_box_jk_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			# restore self.data from the temp file even if a worker failed
			if os.path.exists(f"{temp_file_path}/w_{self.simname}_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"w_{self.simname}_temp_data_{figname_dataset_name}", None,
										   data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				self.data["LOS"] = self.LOS_ind
				jackknife_region_indices_pos = temp_data_obj_m.read_cat(f"jackknife_region_indices_pos")
				jackknife_region_indices_shape = temp_data_obj_m.read_cat(f"jackknife_region_indices_shape")
				os.remove(
					f"{temp_file_path}/w_{self.simname}_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		DD_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))
		Splus_D_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))

		for i in np.arange(len(result)):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]
			DD_jk += result[i][3]
			Splus_D_jk += result[i][4]

		if masks is None:
			weight_shape = self.data["weight_shape_sample"]
		else:
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		R_jk = pair_kernel.compute_R_jk(e, weight_shape, jackknife_region_indices_shape, self.num_box, getattr(self, "responsivity_correction", True))

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)

		RR_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (self.num_box - 1) / self.num_box
		for jk in np.arange(self.num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_g_plus_denom = RR_g_plus.copy()  # guard against empty samples/bins in the divisions; raw RR grids are written to file
		RR_g_plus_denom[RR_g_plus_denom == 0] = 1
		RR_gg_denom = RR_gg.copy()
		RR_gg_denom[RR_gg_denom == 0] = 1
		correlation = Splus_D / RR_g_plus_denom  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus_denom  # (Scross_D - Scross_R) / RR_g_plus
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, self.num_box):
				corr = (Splus_D * (2 * self.R) - Splus_D_jk[i]) / (
						RR_jk[i] * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD",
								   data=(Splus_D * (2 * self.R) - Splus_D_jk[i]) / (
										   2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, self.num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, pi_bins, Splus_D, DD, RR_g_plus


	def _count_pairs_xi_rp_pi_box_jk_brute(self, dataset_name, L_subboxes, masks=None, return_output=False,
										   jk_group_name=""):
		r"""Measures the projected clustering, $\xi_{gg}$, including jackknife realisations in (rp, pi) bins for an
		object created with MeasureIABox. DD-only twin of _measure_xi_rp_pi_box_jk_brute for corr_type='gg':
		skips all shape/ellipticity computation. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r_p bins, pi bins, DD, RR_gg (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self, require_full_masks=True,
		)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks, L_subboxes)
		sample_set.jk_pos = jackknife_region_indices_pos
		sample_set.jk_shape = jackknife_region_indices_shape
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		L3 = self.boxsize ** 3  # box volume
		num_box = L_subboxes ** 3
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, backend="brute", jk=True, num_box=num_box)
		DD = grids.DD
		DD_jk = grids.DD_jk
		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / (num_box)
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, pi_bins, DD, RR_gg


	def _count_pairs_xi_rp_pi_box_jk_tree(self, dataset_name, L_subboxes, masks=None, return_output=False,
										  jk_group_name=""):
		r"""Measures the projected clustering, $\xi_{gg}$, including jackknife realisations in (rp, pi) bins for an
		object created with MeasureIABox. DD-only twin of _measure_xi_rp_pi_box_jk_tree for corr_type='gg':
		skips all shape/ellipticity computation. Uses 1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r_p bins, pi bins, DD, RR_gg (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self, require_full_masks=True,
		)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks, L_subboxes)
		sample_set.jk_pos = jackknife_region_indices_pos
		sample_set.jk_shape = jackknife_region_indices_shape
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		L3 = self.boxsize ** 3  # box volume
		num_box = L_subboxes ** 3
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, backend="tree", jk=True, num_box=num_box)
		DD = grids.DD
		DD_jk = grids.DD_jk
		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / num_box
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, pi_bins, DD, RR_gg


	def _count_pairs_xi_rp_pi_box_jk_batch(self, i):
		r"""(rp, pi) DD-only jackknife shape-sample batch worker. Reads shared memory (incl. jk region
		indices) and delegates to pair_kernel.accumulate (shapes=False, jk=True). Support function for
		_count_pairs_xi_rp_pi_box_jk_multiprocessing().
		"""
		if i + self.chunk_size > self.Num_shape_masked:
			i2 = self.Num_shape_masked
		else:
			i2 = i + self.chunk_size

		shms = []
		shared_data = {}
		for name, shape, dtype in self.shm_infos:
			shm = shared_memory.SharedMemory(name=name)
			shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
			shms.append(shm)

		sample_set = pair_kernel.SampleSet(
			pos=shared_data[f"positions_{self.ID_shm}"],
			pos_shape=shared_data[f"positions_shape_sample_{self.ID_shm}"][i:i2],
			weight=shared_data[f"weight_{self.ID_shm}"],
			weight_shape=shared_data[f"weight_shape_{self.ID_shm}"][i:i2],
			LOS_ind=self.LOS_ind,
			not_LOS=self.not_LOS,
			jk_pos=shared_data[f"jk_region_indices_pos_{self.ID_shm}"],
			jk_shape=shared_data[f"jk_region_indices_shape_{self.ID_shm}"][i:i2],
		)
		binning = pair_kernel.BoxRpPi(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, pos_tree=self.pos_tree,
									   jk=True, num_box=self.num_box)
		for shm in shms:
			shm.close()
		return grids.DD, grids.DD_jk

	def _count_pairs_xi_rp_pi_box_jk_multiprocessing(self, dataset_name, L_subboxes, temp_file_path,
													 masks=None, return_output=False, jk_group_name="",
													 chunk_size=1000, num_nodes=1
													 ):
		r"""Measures the projected clustering, $\xi_{gg}$, including jackknife realisations in (rp, pi) bins for an
		object created with MeasureIABox. DD-only twin of _measure_xi_rp_pi_box_jk_multiprocessing for
		corr_type='gg': skips all shape/ellipticity computation. Uses >1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. Default is 1000.
		num_nodes : int, optional
			Number of CPUs used in the multiprocessing. Default is 1.

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r_p bins, pi bins, DD, RR_gg (if no output file is specified)

		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self, require_full_masks=True,
		)
		positions = sample_set.pos
		positions_shape_sample = sample_set.pos_shape
		weight = sample_set.weight
		weight_shape = sample_set.weight_shape
		self.Num_position_masked = len(positions)
		self.Num_shape_masked = len(positions_shape_sample)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")
		self.LOS_ind = sample_set.LOS_ind
		self.not_LOS = sample_set.not_LOS
		L3 = self.boxsize ** 3  # box volume
		self.sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		self.sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		self.num_box = L_subboxes ** 3
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(
			masks,
			L_subboxes)

		# Build the shared position tree on whatever coordinates the binning
		# queries -- do not hardcode the projection here. BoxRpPi chooses between
		# the 3D positions and their 2D projection depending on the configuration
		# (benchmarks/FINDINGS.md F7), and the workers build their chunk trees with
		# binning.tree_coords, so a hardcoded convention here silently disagrees
		# with them -- scipy then raises "Trees passed to query_ball_tree have
		# different dimensionality".
		_binning = pair_kernel.BoxRpPi(self)
		self.pos_tree = KDTree(_binning.tree_coords(positions, self.not_LOS),
							   boxsize=self.boxsize)
		indices = np.arange(0, len(positions_shape_sample), chunk_size)
		self.chunk_size = chunk_size

		# create temp hdf5 from which data can be read. del self.data, but save it in this method to reduce RAM
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		file_temp = h5py.File(f"{temp_file_path}/w_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5", "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
		write_dataset_hdf5(file_temp, "jackknife_region_indices_shape", jackknife_region_indices_shape)
		write_dataset_hdf5(file_temp, "jackknife_region_indices_pos", jackknife_region_indices_pos)
		file_temp.close()
		self.ID_shm = np.random.randint(100000)
		try:
			shared_data = {
				f"positions_{self.ID_shm}": positions,
				f"positions_shape_sample_{self.ID_shm}": positions_shape_sample,
				f"weight_{self.ID_shm}": weight,
				f"weight_shape_{self.ID_shm}": weight_shape,
				f"jk_region_indices_pos_{self.ID_shm}": jackknife_region_indices_pos,
				f"jk_region_indices_shape_{self.ID_shm}": jackknife_region_indices_shape,
			}
			for k in shared_data.keys():
				try:
					old = shared_memory.SharedMemory(name=k)
					old.unlink()
				except FileNotFoundError:
					pass
			shm_blocks, self.shm_infos = [], []
			for k in shared_data.keys():
				shm = shared_memory.SharedMemory(name=k, create=True, size=shared_data[k].nbytes)
				shared_arr = np.ndarray(shared_data[k].shape, dtype=shared_data[k].dtype, buffer=shm.buf)
				np.copyto(shared_arr, shared_data[k])
				shm_blocks.append(shm)
				self.shm_infos.append([k, shared_data[k].shape, shared_data[k].dtype])
			self.data = {}
			if masks is not None:
				masks = {}
			del shared_data, shared_arr
			del positions, positions_shape_sample, weight, weight_shape, jackknife_region_indices_pos, jackknife_region_indices_shape
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._count_pairs_xi_rp_pi_box_jk_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			# restore self.data from the temp file even if a worker failed
			if os.path.exists(f"{temp_file_path}/w_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"w_gg_{self.simname}_temp_data_{figname_dataset_name}", None,
										   data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				self.data["LOS"] = self.LOS_ind
				jackknife_region_indices_pos = temp_data_obj_m.read_cat(f"jackknife_region_indices_pos")
				jackknife_region_indices_shape = temp_data_obj_m.read_cat(f"jackknife_region_indices_shape")
				os.remove(
					f"{temp_file_path}/w_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		DD_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))

		for i in np.arange(len(result)):
			DD += result[i][0]
			DD_jk += result[i][1]

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)

		RR_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (self.num_box - 1) / self.num_box
		for jk in np.arange(self.num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs(
						self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], volume_jk, "cross",
						Num_position_jk, Num_shape_jk,
						self.num_overlap - self.overlap_jk_counts[jk])

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, self.num_box):
				RR_jk_denom = RR_jk[i].copy()  # guard against empty realisations/bins
				RR_jk_denom[RR_jk_denom == 0] = 1
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk_denom) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, pi_bins, DD, RR_gg


if __name__ == "__main__":
	pass
