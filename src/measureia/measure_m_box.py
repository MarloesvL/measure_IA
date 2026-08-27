import numpy as np
import h5py
import os
from multiprocessing import Pool, shared_memory
import multiprocessing as mp

from . import worker_pool
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from .read_data import ReadData
from . import pair_kernel


class MeasureMultipolesBox(MeasureIABase, ReadData):
	r"""Class that contains all methods for the measurements of $\xi_{gg}$ and $\xi_{g+}$ for $\tilde{\xi}_{gg,0}$ and
	$\tilde{\xi}_{g+,2}$ with Cartesian simulation data.

	Methods
	-------
	_measure_xi_r_mur_box_brute()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning in a periodic box using 1 CPU.
	_measure_xi_r_mur_box_tree()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning in a periodic box using 1 CPU and KDTree for extra speed.
	_measure_xi_r_mur_box_batch()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning in a periodic box using 1 CPU for a batch of indices.
		Support function of _measure_xi_r_mur_box_multiprocessing().
	_measure_xi_r_mur_box_multiprocessing()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning in a periodic box using >1 CPUs.

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
		The __init__ method of the MeasureMultipolesSimulations class.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		return

	def _measure_xi_r_mur_box_brute(self, dataset_name, masks=None, rp_cut=None, return_output=False, jk_group_name="",
									ellipticity='distortion'):
		r"""Measures the projected correlation functions, $\xi_{gg}$ and $\xi_{g+}$, in (r, mu_r) bins for an object
		created with MeasureIABox. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value is None.
		rp_cut : float, optional
			Limit for minimum r_p value for pairs to be included. Default value is None.
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
			$\xi_{gg}$ and $\xi_{g+}$, r bins, mu_r bins, S+D, DD, RR (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		weight_shape = sample_set.weight_shape
		e = sample_set.e
		if rp_cut == None:
			rp_cut = 0.0
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRMuR(self, rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, backend="brute")
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
		corrtype = "cross"  # auto-correlations are not supported; DD is always treated as a cross-count

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					Num_position, Num_shape, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

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
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus


	def _measure_xi_r_mur_box_tree(self, dataset_name, masks=None, rp_cut=None, return_output=False, jk_group_name="",
								   ellipticity='distortion'):
		r"""Measures the projected correlation functions, $\xi_{gg}$ and $\xi_{g+}$, in (r, mu_r) bins for an object
		created with MeasureIABox. Uses 1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value is None.
		rp_cut : float, optional
			Limit for minimum r_p value for pairs to be included. Default value is None.
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
			$\xi_{gg}$ and $\xi_{g+}$, r bins, mu_r bins, S+D, DD, RR (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		weight_shape = sample_set.weight_shape
		e = sample_set.e
		if rp_cut == None:
			rp_cut = 0.0
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRMuR(self, rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, backend="tree")
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
		corrtype = "cross"  # auto-correlations are not supported; DD is always treated as a cross-count

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					Num_position, Num_shape, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

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
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus


	def _measure_xi_r_mur_box_batch(self, i):
		r"""(r, mu_r) shape-sample batch worker. Reads shared memory and delegates the counting loop to
		pair_kernel.accumulate (BoxRMuR, shapes=True, reusing the parent's shared self.pos_tree). Support
		function for _measure_xi_r_mur_box_multiprocessing().
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
		)
		binning = pair_kernel.BoxRMuR(self, self.rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, R=self.R, shapes=True,
									   chunk_axis="shape", chunk_size_outer=100, pos_tree=self.pos_tree)
		for shm in shms:
			shm.close()
		return grids.Splus_D, grids.Scross_D, grids.DD

	def _measure_xi_r_mur_box_multiprocessing(self, dataset_name, temp_file_path, masks=None,
											  rp_cut=None, return_output=False, jk_group_name="",
											  chunk_size=100, num_nodes=1, ellipticity='distortion'):
		r"""Measures the projected correlation functions, $\xi_{gg}$ and $\xi_{g+}$, in (r, mu_r) bins for an object
		created with MeasureIABox. Uses >1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		num_nodes : int, optional
			Number of CPUs used in the multiprocessing. Default is 1.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		rp_cut : float, optional
			Limit for minimum r_p value for pairs to be included. Default value is None.
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
			$\xi_{gg}$ and $\xi_{g+}$, r bins, mu_r bins, S+D, DD, RR (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=True, ellipticity=ellipticity, base=self,
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
		if rp_cut == None:
			self.rp_cut = 0.0
		else:
			self.rp_cut = rp_cut
		self.LOS_ind = sample_set.LOS_ind
		self.not_LOS = sample_set.not_LOS
		self.R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
			if getattr(self, "responsivity_correction", True) and sum(weight_shape) > 0 else 0.5
		L3 = self.boxsize ** 3  # box volume
		# Build the shared position tree on whatever coordinates the binning
		# queries -- do not hardcode the projection here. BoxRpPi chooses between
		# the 3D positions and their 2D projection depending on the configuration
		# (benchmarks/FINDINGS.md F7), and the workers build their chunk trees with
		# binning.tree_coords, so a hardcoded convention here silently disagrees
		# with them -- scipy then raises "Trees passed to query_ball_tree have
		# different dimensionality".
		_binning = pair_kernel.BoxRMuR(self, rp_cut)
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
		file_temp = h5py.File(f"{temp_file_path}/m_{self.simname}_temp_data_{figname_dataset_name}.hdf5", "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
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
			del positions, positions_shape_sample, axis_direction, weight, weight_shape
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._measure_xi_r_mur_box_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			# restore self.data from the temp file even if a worker failed
			if os.path.exists(f"{temp_file_path}/m_{self.simname}_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"m_{self.simname}_temp_data_{figname_dataset_name}", None,
										   data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				self.data["LOS"] = self.LOS_ind
				os.remove(
					f"{temp_file_path}/m_{self.simname}_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		for i in np.arange(len(result)):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]

		corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)

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
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, xi_gg, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus


	def _count_pairs_xi_r_mur_box_brute(self, dataset_name, masks=None, rp_cut=None, return_output=False,
										jk_group_name=""):
		r"""Measures the clustering, $\xi_{gg}$, in (r, mu_r) bins for an object created with MeasureIABox.
		DD-only twin of _measure_xi_r_mur_box_brute for corr_type='gg': skips all shape/ellipticity computation.
		Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		rp_cut : float, optional
			Value of projected separation below which pairs are excluded. Default is None (no cut).
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r bins, mu_r bins, DD, RR_gg (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if rp_cut == None:
			rp_cut = 0.0
		L3 = self.boxsize ** 3  # box volume
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRMuR(self, rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, backend="brute")
		DD = grids.DD
		corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, mu_r_bins, DD, RR_gg


	def _count_pairs_xi_r_mur_box_tree(self, dataset_name, masks=None, rp_cut=None, return_output=False,
									   jk_group_name=""):
		r"""Measures the clustering, $\xi_{gg}$, in (r, mu_r) bins for an object created with MeasureIABox.
		DD-only twin of _measure_xi_r_mur_box_tree for corr_type='gg': skips all shape/ellipticity computation.
		Uses 1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		rp_cut : float, optional
			Value of projected separation below which pairs are excluded. Default is None (no cut).
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r bins, mu_r bins, DD, RR_gg (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if rp_cut == None:
			rp_cut = 0.0
		L3 = self.boxsize ** 3  # box volume
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.BoxRMuR(self, rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, backend="tree")
		DD = grids.DD
		corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape, self.num_overlap)

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, mu_r_bins, DD, RR_gg


	def _count_pairs_xi_r_mur_box_batch(self, i):
		r"""(r, mu_r) DD-only shape-sample batch worker. Reads shared memory and delegates the counting
		loop to pair_kernel.accumulate (BoxRMuR, shapes=False). Support function for
		_count_pairs_xi_r_mur_box_multiprocessing().
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
		)
		binning = pair_kernel.BoxRMuR(self, self.rp_cut)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="shape", chunk_size_outer=100, pos_tree=self.pos_tree)
		for shm in shms:
			shm.close()
		return grids.DD

	def _count_pairs_xi_r_mur_box_multiprocessing(self, dataset_name, temp_file_path, masks=None, rp_cut=None,
												  return_output=False, jk_group_name="", num_nodes=1,
												  chunk_size=1000):
		r"""Measures the clustering, $\xi_{gg}$, in (r, mu_r) bins for an object created with MeasureIABox.
		DD-only twin of _measure_xi_r_mur_box_multiprocessing for corr_type='gg': skips all shape/ellipticity
		computation. Uses >1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		rp_cut : float, optional
			Value of projected separation below which pairs are excluded. Default is None (no cut).
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Default value is "".
		num_nodes : int, optional
			Number of CPUs used in the multiprocessing. Default is 1.
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. Default is 1000.

		Returns
		-------
		ndarrays
			$\xi_{gg}$, r bins, mu_r bins, DD, RR_gg (if no output file is specified)
		"""
		sample_set = pair_kernel.prepare_box_samples(
			self.data, masks, self.Num_position, self.Num_shape,
			shapes=False, ellipticity='distortion', base=self,
		)
		positions = sample_set.pos
		positions_shape_sample = sample_set.pos_shape
		weight = sample_set.weight
		weight_shape = sample_set.weight_shape
		self.Num_position_masked = len(positions)
		self.Num_shape_masked = len(positions_shape_sample)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")
		if rp_cut == None:
			self.rp_cut = 0.0
		else:
			self.rp_cut = rp_cut
		self.LOS_ind = sample_set.LOS_ind
		self.not_LOS = sample_set.not_LOS
		L3 = self.boxsize ** 3  # box volume
		# Build the shared position tree on whatever coordinates the binning
		# queries -- do not hardcode the projection here. BoxRpPi chooses between
		# the 3D positions and their 2D projection depending on the configuration
		# (benchmarks/FINDINGS.md F7), and the workers build their chunk trees with
		# binning.tree_coords, so a hardcoded convention here silently disagrees
		# with them -- scipy then raises "Trees passed to query_ball_tree have
		# different dimensionality".
		_binning = pair_kernel.BoxRMuR(self, rp_cut)
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
		file_temp = h5py.File(f"{temp_file_path}/multipoles_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5",
							  "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
		file_temp.close()
		self.ID_shm = np.random.randint(100000)
		try:
			shared_data = {
				f"positions_{self.ID_shm}": positions,
				f"positions_shape_sample_{self.ID_shm}": positions_shape_sample,
				f"weight_{self.ID_shm}": weight,
				f"weight_shape_{self.ID_shm}": weight_shape,
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
			del positions, positions_shape_sample, weight, weight_shape
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._count_pairs_xi_r_mur_box_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			# restore self.data from the temp file even if a worker failed
			if os.path.exists(f"{temp_file_path}/multipoles_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"multipoles_gg_{self.simname}_temp_data_{figname_dataset_name}",
										   None, data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				self.data["LOS"] = self.LOS_ind
				os.remove(
					f"{temp_file_path}/multipoles_gg_{self.simname}_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		for i in np.arange(len(result)):
			DD += result[i]

		corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					self.Num_position_masked, self.Num_shape_masked, self.num_overlap)

		RR_gg_denom = RR_gg.copy()  # guard against empty samples/bins in the division; raw RR grid is written to file
		RR_gg_denom[RR_gg_denom == 0] = 1
		xi_gg = (DD / RR_gg_denom) - 1
		xi_gg[RR_gg == 0] = 0
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=xi_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return xi_gg, separation_bins, mu_r_bins, DD, RR_gg



if __name__ == "__main__":
	pass
