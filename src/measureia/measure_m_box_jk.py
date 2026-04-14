import numpy as np
import h5py
import pickle
import os
import sys
# from pathos.multiprocessing import ProcessingPool
from multiprocessing import Pool, shared_memory
import multiprocessing as mp
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from .read_data import ReadData

class MeasureMBoxJackknife(MeasureIABase, ReadData):
	r"""Class that contains all methods for the measurements of $\xi_{gg}$ and $\xi_{g+}$ for $\tilde{\xi}_{gg,0}$ and
	$\tilde{\xi}_{g+,2}$ including the jackknife realisations needed for the covariance estimation with
	Cartesian simulation data.

	Methods
	-------
	_measure_xi_r_mur_box_jk_brute()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning including jackknife realisations in a periodic box
		using 1 CPU.
	_measure_xi_r_mur_box_jk_tree()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning including jackknife realisations in a periodic box
		using 1 CPU and KDTree for extra speed.
	_measure_xi_r_mur_box_jk_batch()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning including jackknife realisations in a periodic box
		using 1 CPU for a batch of indices.
		Support function of _measure_xi_r_mur_box_jk_multiprocessing().
	_measure_xi_r_mur_box_jk_multiprocessing()
		Measure $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) grid binning including jackknife realisations in a periodic
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

	def _measure_xi_r_mur_box_jk_brute(self, dataset_name, L_subboxes, masks=None, rp_cut=None, return_output=False,
									   jk_group_name="", ellipticity='distortion'):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (r, mu_r) bins for an object created with MeasureIABox. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
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

		if masks == None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(positions)
		Num_shape = len(positions_shape_sample)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		if ellipticity == 'distortion':
			e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		elif ellipticity == 'ellipticity':
			e = (1 - q) / (1 + q)
		else:
			raise ValueError("Invalid value for ellipticity. Choose 'distortion' or 'ellipticity'.")
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		if rp_cut == None:
			rp_cut = 0.0
		sub_box_len_logr = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks,
																										  L_subboxes)

		num_box = L_subboxes ** 3
		DD_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		Splus_D_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))

		for n in np.arange(0, len(positions)):
			separation = positions_shape_sample - positions[n]
			if self.periodicity:
				separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
				separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			with np.errstate(invalid='ignore'):
				separation_dir = (
						projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
			separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
			with np.errstate(invalid='ignore'):
				mu_r = LOS / separation_len
				del LOS, projected_sep, separation
				phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			e_plus, e_cross = self.get_ellipticity(e, phi)
			del phi, separation_dir
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0
			mu_r[np.isnan(e_plus)] = 0.0

			# get the indices for the binning
			mask = (
					(projected_separation_len > rp_cut)
					* (separation_len >= self.r_bins[0])
					* (separation_len < self.r_bins[-1])
			)
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(self.r_bins[0]) / sub_box_len_logr
			)
			del separation_len, projected_separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			ind_mu_r = np.array(ind_mu_r, dtype=int)
			if np.any(ind_mu_r == self.num_bins_pi):
				ind_mu_r[ind_mu_r >= self.num_bins_pi] -= 1
			if np.any(ind_r == self.num_bins_r):
				ind_r[ind_r >= self.num_bins_r] -= 1
			np.add.at(Splus_D, (ind_r, ind_mu_r), (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_mu_r), (weight[n] * weight_shape[mask] * e_cross[mask]) / (2 * R))
			np.add.at(variance, (ind_r, ind_mu_r), ((weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R)) ** 2)

			shape_mask = np.where(jackknife_region_indices_shape[mask] != jackknife_region_indices_pos[n])[0]
			np.add.at(Splus_D_jk, (jackknife_region_indices_pos[n], ind_r, ind_mu_r),
					  (weight[n] * weight_shape[mask] * e_plus[mask]))  # responsivity added later
			np.add.at(Splus_D_jk,
					  (jackknife_region_indices_shape[mask][shape_mask], ind_r[shape_mask], ind_mu_r[shape_mask]),
					  (weight[n] * weight_shape[mask][shape_mask] * e_plus[mask][
						  shape_mask]))  # responsivity added later

			del e_plus, e_cross
			np.add.at(DD, (ind_r, ind_mu_r), weight[n] * weight_shape[mask])
			np.add.at(DD_jk, (jackknife_region_indices_pos[n], ind_r, ind_mu_r),
					  (weight[n] * weight_shape[mask]))
			np.add.at(DD_jk,
					  (jackknife_region_indices_shape[mask][shape_mask], ind_r[shape_mask], ind_mu_r[shape_mask]),
					  (weight[n] * weight_shape[mask][shape_mask]))

		R_jk = np.zeros(num_box)
		for i in np.arange(num_box):
			jk_mask = np.where(jackknife_region_indices_shape != i)
			R_jk[i] = sum(weight_shape[jk_mask] * (1 - e[jk_mask] ** 2 / 2.0)) / sum(weight_shape[jk_mask])

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / (num_box)
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs_r_mur(
						self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], volume_jk,
						"cross",
						Num_position_jk, Num_shape_jk)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				corr = (Splus_D * (2 * R) - Splus_D_jk[i]) / (
						RR_jk[i] * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD", data=(Splus_D * (2 * R) - Splus_D_jk[i]) / (
						2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk[i]) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus

	def _measure_xi_r_mur_box_jk_tree(self, dataset_name, L_subboxes, masks=None, rp_cut=None, return_output=False,
									  jk_group_name="", ellipticity='distortion'):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (r, mu_r) bins for an object created with MeasureIABox. Uses 1 CPU. Uses KDTree for speedup.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.
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

		if masks == None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(positions)
		Num_shape = len(positions_shape_sample)
		print(
			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		if ellipticity == 'distortion':
			e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		elif ellipticity == 'ellipticity':
			e = (1 - q) / (1 + q)
		else:
			raise ValueError("Invalid value for ellipticity. Choose 'distortion' or 'ellipticity'.")
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		if rp_cut == None:
			rp_cut = 0.0
		sub_box_len_logr = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(masks,
																										  L_subboxes)
		num_box = L_subboxes ** 3
		DD_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		Splus_D_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")

		pos_tree = KDTree(positions, boxsize=self.boxsize)
		for i in np.arange(0, len(positions_shape_sample), 100):
			i2 = min(len(positions_shape_sample), i + 100)
			positions_shape_sample_i = positions_shape_sample[i:i2]
			axis_direction_i = axis_direction[i:i2]
			e_i = e[i:i2]
			weight_shape_i = weight_shape[i:i2]
			jackknife_region_indices_shape_i = jackknife_region_indices_shape[i:i2]
			shape_tree = KDTree(positions_shape_sample_i, boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(pos_tree, self.r_min)
			ind_max_i = shape_tree.query_ball_tree(pos_tree, self.r_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(positions_shape_sample_i)):
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, not_LOS]
					LOS = separation[:, LOS_ind]
					projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					with np.errstate(invalid='ignore'):
						separation_dir = (
								projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
					separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
					del separation, projected_sep
					with np.errstate(invalid='ignore'):
						mu_r = LOS / separation_len
						phi = np.arccos(
							separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
								n, 1])  # [0,pi]
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)
					del phi, LOS, separation_dir

					e_plus[np.isnan(e_plus)] = 0.0
					mu_r[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (
							(projected_separation_len > rp_cut)
							* (separation_len >= self.r_bins[0])
							* (separation_len < self.r_bins[-1])
					)
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(
							self.r_bins[0]) / sub_box_len_logr
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					if np.any(ind_mu_r == self.num_bins_pi):
						ind_mu_r[ind_mu_r >= self.num_bins_pi] -= 1
					if np.any(ind_r == self.num_bins_r):
						ind_r[ind_r >= self.num_bins_r] -= 1
					np.add.at(Splus_D, (ind_r, ind_mu_r),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_plus[mask]) / (2 * R))
					np.add.at(Scross_D, (ind_r, ind_mu_r),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_cross[mask]) / (2 * R))
					del separation_len
					np.add.at(DD, (ind_r, ind_mu_r), weight[ind_rbin_i[n]][mask] * weight_shape_i[n])

					pos_mask = \
						np.where(
							jackknife_region_indices_pos[ind_rbin_i[n]][mask] != jackknife_region_indices_shape_i[n])[
							0]
					np.add.at(Splus_D_jk, (jackknife_region_indices_shape_i[n], ind_r, ind_mu_r),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_plus[
								  mask]))  # responsivity added later
					np.add.at(Splus_D_jk,
							  (jackknife_region_indices_pos[ind_rbin_i[n]][mask][pos_mask], ind_r[pos_mask],
							   ind_mu_r[pos_mask]),
							  (weight[ind_rbin_i[n]][mask][pos_mask] * weight_shape_i[n] * e_plus[mask][
								  pos_mask]))  # responsivity added later

					del e_plus, e_cross
					np.add.at(DD_jk, (jackknife_region_indices_shape_i[n], ind_r, ind_mu_r),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n]))  # responsivity added later
					np.add.at(DD_jk,
							  (jackknife_region_indices_pos[ind_rbin_i[n]][mask][pos_mask], ind_r[pos_mask],
							   ind_mu_r[pos_mask]),
							  (weight[ind_rbin_i[n]][mask][pos_mask] * weight_shape_i[n]))  # responsivity added later

		R_jk = np.zeros(num_box)
		for i in np.arange(num_box):
			jk_mask = np.where(jackknife_region_indices_shape != i)
			R_jk[i] = sum(weight_shape[jk_mask] * (1 - e[jk_mask] ** 2 / 2.0)) / sum(weight_shape[jk_mask])

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					Num_position, Num_shape)

		RR_jk = np.zeros((num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (num_box - 1) / num_box
		for jk in np.arange(num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs_r_mur(
						self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], volume_jk,
						"cross",
						Num_position_jk, Num_shape_jk)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				corr = (Splus_D * (2 * R) - Splus_D_jk[i]) / (
						RR_jk[i] * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD", data=(Splus_D * (2 * R) - Splus_D_jk[i]) / (
						2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk[i]) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus

	def _measure_xi_r_mur_box_jk_batch(self, i):
		r"""Measures components of $\xi_{gg}$ and $\xi_{g+}$ in (r, mu_r) bins including jackknife realisations for a
		batch of indices from i to i+chunk_size. Support function for _measure_xi_r_mu_r_box_jk_multiprocessing().

		Parameters
		----------
		i: int
			Start index of the batch.

		Returns
		-------
		ndarrays
			S+D, SxD, DD, DD_jk, S+D_jk where the _jk versions store the necessary information of DD of S+D for
			each jackknife realisation.
		"""
		if i + self.chunk_size > self.Num_shape_masked:
			i2 = self.Num_shape_masked
		else:
			i2 = i + self.chunk_size

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		DD_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))
		Splus_D_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))

		shms = []
		shared_data = {}
		for name, shape, dtype in self.shm_infos:
			shm = shared_memory.SharedMemory(name=name)
			shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
			shms.append(shm)
		positions = shared_data[f"positions_{self.ID_shm}"]
		for j in np.arange(i, i2, 100):
			j2 = min(j + 100, i2)
			positions_shape_sample_i = shared_data[f"positions_shape_sample_{self.ID_shm}"][j:j2]
			axis_direction_i = shared_data[f"axis_direction_{self.ID_shm}"][j:j2]
			weight_shape_i = shared_data[f"weight_shape_{self.ID_shm}"][j:j2]
			e_i = shared_data[f"e_{self.ID_shm}"][j:j2]
			jackknife_region_indices_shape_i = shared_data[f"jk_region_indices_shape_{self.ID_shm}"][j:j2]
			jackknife_region_indices_pos = shared_data[f"jk_region_indices_pos_{self.ID_shm}"]

			shape_tree = KDTree(positions_shape_sample_i, boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(self.pos_tree, self.r_min)
			ind_max_i = shape_tree.query_ball_tree(self.pos_tree, self.r_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(positions_shape_sample_i)):
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, self.not_LOS]
					LOS = separation[:, self.LOS_ind]
					projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					with np.errstate(invalid='ignore'):
						separation_dir = (
								projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
					separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
					del separation, projected_sep
					with np.errstate(invalid='ignore'):
						mu_r = LOS / separation_len
						phi = np.arccos(
							separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
								n, 1])  # [0,pi]
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)
					del phi, LOS, separation_dir

					e_plus[np.isnan(e_plus)] = 0.0
					mu_r[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (
							(projected_separation_len > self.rp_cut)
							* (separation_len >= self.r_bins[0])
							* (separation_len < self.r_bins[-1])
					)
					ind_r = np.floor(
						np.log10(separation_len[mask]) / self.sub_box_len_logr - np.log10(
							self.r_bins[0]) / self.sub_box_len_logr
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / self.sub_box_len_mu_r - self.mu_r_bins[0] / self.sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					if np.any(ind_mu_r == self.num_bins_pi):
						ind_mu_r[ind_mu_r >= self.num_bins_pi] -= 1
					if np.any(ind_r == self.num_bins_r):
						ind_r[ind_r >= self.num_bins_r] -= 1
					weight_i_n = shared_data[f"weight_{self.ID_shm}"][ind_rbin_i[n]]
					np.add.at(Splus_D, (ind_r, ind_mu_r),
							  (weight_i_n[mask] * weight_shape_i[n] * e_plus[mask]) / (2 * self.R))
					np.add.at(Scross_D, (ind_r, ind_mu_r),
							  (weight_i_n[mask] * weight_shape_i[n] * e_cross[mask]) / (2 * self.R))
					np.add.at(DD, (ind_r, ind_mu_r), weight_i_n[mask] * weight_shape_i[n])
					del separation_len

					pos_mask = \
						np.where(
							jackknife_region_indices_pos[ind_rbin_i[n]][mask] != jackknife_region_indices_shape_i[n])[
							0]
					np.add.at(Splus_D_jk, (jackknife_region_indices_shape_i[n], ind_r, ind_mu_r),
							  (weight_i_n[mask] * weight_shape_i[n] * e_plus[
								  mask]))  # responsivity added later
					np.add.at(Splus_D_jk,
							  (jackknife_region_indices_pos[ind_rbin_i[n]][mask][pos_mask], ind_r[pos_mask],
							   ind_mu_r[pos_mask]),
							  (weight_i_n[mask][pos_mask] * weight_shape_i[n] * e_plus[mask][
								  pos_mask]))  # responsivity added later

					del e_plus, e_cross
					np.add.at(DD_jk, (jackknife_region_indices_shape_i[n], ind_r, ind_mu_r),
							  (weight_i_n[mask] * weight_shape_i[n]))  # responsivity added later
					np.add.at(DD_jk,
							  (jackknife_region_indices_pos[ind_rbin_i[n]][mask][pos_mask], ind_r[pos_mask],
							   ind_mu_r[pos_mask]),
							  (weight_i_n[mask][pos_mask] * weight_shape_i[n]))  # responsivity added later
		for shm in shms:
			shm.close()
		return Splus_D, Scross_D, DD, DD_jk, Splus_D_jk

	def _measure_xi_r_mur_box_jk_multiprocessing(self, dataset_name, L_subboxes, temp_file_path, masks=None,
												 rp_cut=None, return_output=False, jk_group_name="",
												 chunk_size=100, num_nodes=1, ellipticity='distortion'):
		r"""Measures the projected correlation functions including jackknife realisations, $\xi_{gg}$ and $\xi_{g+}$,
		in (r, mu_r) bins for an object created with MeasureIABox. Uses >1 CPU. Uses KDTree for speedup.

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
		rp_cut : float, optional
			Limit for minimum r_p value for pairs to be included. Default value is None.
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
			$\xi_{gg}$ and $\xi_{g+}$, r bins, mu_r bins, S+D, DD, RR (if no output file is specified)

		"""
		if masks is None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			del axis_direction_v, axis_direction_len
			q = self.data["q"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			del axis_direction_v, axis_direction_len
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		self.Num_position_masked = len(positions)
		self.Num_shape_masked = len(positions_shape_sample)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")

		self.LOS_ind = self.data["LOS"]  # eg 2 for z axis
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], self.LOS_ind, invert=True)]  # eg 0,1 for x&y
		if ellipticity == 'distortion':
			e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		elif ellipticity == 'ellipticity':
			e = (1 - q) / (1 + q)
		else:
			raise ValueError("Invalid value for ellipticity. Choose 'distortion' or 'ellipticity'.")
		del q
		self.R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		if rp_cut == None:
			self.rp_cut = 0.0
		else:
			self.rp_cut = rp_cut
		self.sub_box_len_logr = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		self.sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi

		jackknife_region_indices_pos, jackknife_region_indices_shape = self._get_jackknife_region_indices(
			masks,
			L_subboxes)
		self.num_box = L_subboxes ** 3

		self.pos_tree = KDTree(positions, boxsize=self.boxsize)
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
			mp.set_start_method("spawn", force=True)
			with Pool(num_nodes) as p:
				result = p.map(self._measure_xi_r_mur_box_jk_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()

		temp_data_obj_m = ReadData(self.simname, f"m_{self.simname}_temp_data_{figname_dataset_name}", None,
								   data_path=temp_file_path)
		for k in keys:
			self.data[k] = temp_data_obj_m.read_cat(k)
			if masks is not None:
				masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
		self.data["LOS"] = self.LOS_ind
		jackknife_region_indices_pos = temp_data_obj_m.read_cat(f"jackknife_region_indices_pos")
		jackknife_region_indices_shape = temp_data_obj_m.read_cat(f"jackknife_region_indices_shape")
		os.remove(
			f"{temp_file_path}/m_{self.simname}_temp_data_{figname_dataset_name}.hdf5")

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
		R_jk = np.zeros(self.num_box)
		for i in np.arange(self.num_box):
			jk_mask = np.where(jackknife_region_indices_shape != i)
			R_jk[i] = sum(weight_shape[jk_mask] * (1 - e[jk_mask] ** 2 / 2.0)) / sum(weight_shape[jk_mask])

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, "cross",
					self.Num_position_masked, self.Num_shape_masked)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], L3, corrtype,
					self.Num_position_masked, self.Num_shape_masked)

		RR_jk = np.zeros((self.num_box, self.num_bins_r, self.num_bins_pi))
		volume_jk = L3 * (self.num_box - 1) / self.num_box
		for jk in np.arange(self.num_box):
			Num_position_jk, Num_shape_jk = len(np.where(jackknife_region_indices_pos != jk)[0]), len(
				np.where(jackknife_region_indices_shape != jk)[0])
			for i in np.arange(0, self.num_bins_r):
				for p in np.arange(0, self.num_bins_pi):
					RR_jk[jk, i, p] = self.get_random_pairs_r_mur(
						self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p], volume_jk,
						"cross",
						Num_position_jk, Num_shape_jk)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, self.num_box):
				corr = (Splus_D * (2 * self.R) - Splus_D_jk[i]) / (
						RR_jk[i] * 2 * R_jk[i])  # Responsivity will be different for each realisation
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=corr)
				write_dataset_hdf5(group, dataset_name + f"_{i}_SplusD",
								   data=(Splus_D * (2 * self.R) - Splus_D_jk[i]) / (
										   2 * R_jk[i]))  # Splus_D_jk[i]/(2*R_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}")
			for i in np.arange(0, self.num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk[i]) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_DD", data=(DD - DD_jk[i]))
				write_dataset_hdf5(group, dataset_name + f"_{i}_RR", data=RR_jk[i])
				write_dataset_hdf5(group, dataset_name + f"_{i}_r", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins, Splus_D, DD, RR_g_plus


if __name__ == "__main__":
	pass
