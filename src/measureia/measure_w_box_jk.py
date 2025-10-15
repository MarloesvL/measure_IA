import numpy as np
import h5py
import pickle
from pathos.multiprocessing import ProcessingPool
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureWBoxJackknife(MeasureIABase):
	"""Class that contains all methods for the measurements of xi_gg and xi_g+ for w_gg and w_g+ with carthesian
	simulation data.

	Methods
	-------
	_measure_xi_rp_pi_sims_brute()
		Measure xi_gg or xi_g+ in (rp, pi) grid binning in a periodic box using 1 CPU.
	_measure_xi_rp_pi_sims_tree()
		Measure xi_gg or xi_g+ in (rp, pi) grid binning in a periodic box using 1 CPU and KDTree for extra speed.
	_measure_xi_rp_pi_sims_batch()
		Measure xi_gg or xi_g+ in (rp, pi) grid binning in a periodic box using 1 CPU for a batch of indices.
		Support function of _measure_xi_rp_pi_sims_multiprocessing().
	_measure_xi_rp_pi_sims_multiprocessing()
		Measure xi_gg or xi_g+ in (rp, pi) grid binning in a periodic box using >1 CPUs.

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

	def _get_jackknife_region_indices(self, masks, L_subboxes):
		if masks == None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
		L_sub = self.L_0p5 * 2.0 / L_subboxes
		jackknife_region_indices_pos = np.zeros(len(positions))
		jackknife_region_indices_shape = np.zeros(len(positions_shape_sample))
		num_box = 0
		for i in np.arange(0, L_subboxes):
			for j in np.arange(0, L_subboxes):
				for k in np.arange(0, L_subboxes):
					x_bounds = [i * L_sub, (i + 1) * L_sub]
					y_bounds = [j * L_sub, (j + 1) * L_sub]
					z_bounds = [k * L_sub, (k + 1) * L_sub]
					x_mask = (positions[:, 0] > x_bounds[0]) * (positions[:, 0] < x_bounds[1])
					y_mask = (positions[:, 1] > y_bounds[0]) * (positions[:, 1] < y_bounds[1])
					z_mask = (positions[:, 2] > z_bounds[0]) * (positions[:, 2] < z_bounds[1])
					mask_position = x_mask * y_mask * z_mask  # mask that is True for all positions in the subbox
					mask_shape = x_mask * y_mask * z_mask  # mask that is True for all positions not in the subbox
					jackknife_region_indices_pos[mask_position] = num_box
					jackknife_region_indices_shape[mask_shape] = num_box
					num_box += 1
		return np.array(jackknife_region_indices_pos, dtype=int), np.array(jackknife_region_indices_shape, dtype=int)

	def _combine_jackknife_information(self, dataset_name, jk_group_name, data, num_box, return_output=False):
		covs, stds = [], []
		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file[f"{self.snap_group}/{data[d]}/{jk_group_name}/"]
			# calculating mean of the datavectors
			mean_multipoles = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				mean_multipoles += group_multipoles[dataset_name + "_" + str(b)][:]
			mean_multipoles /= num_box

			# calculation the covariance matrix (multipoles) and the standard deviation (sqrt of diag of cov)
			cov = np.zeros((self.num_bins_r, self.num_bins_r))
			std = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				std += (group_multipoles[dataset_name + "_" + str(b)][:] - mean_multipoles) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group_multipoles[dataset_name + "_" + str(b)][:] - mean_multipoles) * (
							group_multipoles[dataset_name + "_" + str(b)][i] - mean_multipoles[i]
					)
			std *= (num_box - 1) / num_box  # see Singh 2023
			std = np.sqrt(std)  # size of errorbars
			cov *= (num_box - 1) / num_box  # cov not sqrt so to get std, sqrt of diag would need to be taken
			data_file.close()
			if return_output:
				covs.append(cov)
				stds.append(std)
			else:
				output_file = h5py.File(self.output_file_name, "a")
				group_multipoles = create_group_hdf5(output_file, f"{self.snap_group}/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_mean_" + str(num_box), data=mean_multipoles)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
		if return_output:
			return covs, stds
		else:
			return

	def _measure_xi_rp_pi_box_jk_brute(self, dataset_name, L_subboxes, masks=None, return_output=False,
									   print_num=True, jk_group_name=""):
		"""Measures the projected correlation functions, xi_g+ and xi_gg, in (rp, pi) bins for an object created with
		MeasureIABox. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		print_num : bool, optional
			If True, prints the number of objects in the shape and positon samples. Default value is True.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Is used when this method is called in
			MeasureJackknife. Default value is "".

		Returns
		-------
		ndarrays
			xi_g+, xi_gg, r_p bins, pi bins if no output file is specified

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
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
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
		# print(np.shape(jackknife_region_indices_pos),np.shape(jackknife_region_indices_shape)) #(766,) (766,)
		# print(Num_position,Num_shape) #766 766
		# print(np.shape(DD_jk),np.shape(Splus_D_jk)) #(8, 10, 8) (8, 10, 8)
		# print(np.shape(DD),np.shape(Splus_D)) #((10, 8) (10, 8)

		for n in np.arange(0, len(positions)):
			# for Splus_D (calculate ellipticities around position sample)
			separation = positions_shape_sample - positions[n]
			if self.periodicity:
				separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
				separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			del separation
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			with np.errstate(invalid='ignore'):
				separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
				del projected_sep
				phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			del separation_dir
			e_plus, e_cross = self.get_ellipticity(e, phi)
			del phi
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			# get the indices for the binning
			mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.r_bins[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			del LOS
			ind_pi = np.array(ind_pi, dtype=int)
			if np.any(ind_pi == self.num_bins_pi):
				ind_pi[ind_pi >= self.num_bins_pi] -= 1
			if np.any(ind_r == self.num_bins_r):
				ind_r[ind_r >= self.num_bins_r] -= 1
			np.add.at(Splus_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_cross[mask]) / (2 * R))
			np.add.at(variance, (ind_r, ind_pi), ((weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R)) ** 2)
			np.add.at(Splus_D_jk, (jackknife_region_indices_pos[n], ind_r, ind_pi),
					  (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))
			np.add.at(Splus_D_jk, (jackknife_region_indices_shape[mask], ind_r, ind_pi),
					  (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))

			del e_plus, e_cross
			np.add.at(DD, (ind_r, ind_pi), weight[n] * weight_shape[mask])
			np.add.at(Splus_D_jk, (jackknife_region_indices_pos[n], ind_r, ind_pi),
					  (weight[n] * weight_shape[mask]))
			np.add.at(Splus_D_jk, (jackknife_region_indices_shape[mask], ind_r, ind_pi),
					  (weight[n] * weight_shape[mask]))

		# if Num_position == Num_shape:
		# 	corrtype = "auto"
		# 	DD = DD / 2.0  # auto correlation, all pairs are double
		# else:

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		RR_jk = (num_box - 1) / num_box * RR_g_plus
		sigsq = variance / RR_g_plus ** 2
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
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=(Splus_D - Splus_D_jk[i]) / RR_jk)
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins, Splus_D, DD, RR_g_plus

	def _measure_xi_rp_pi_box_jk_tree(self, dataset_name, L_subboxes, masks=None, return_output=False,
									  print_num=True, jk_group_name="", file_tree_path=None, save_tree=False):
		"""Measures the projected correlation functions, xi_g+ and xi_gg, in (rp, pi) bins for an object created with
		MeasureIABox. Uses 1 CPU.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		masks : dict or NoneType, optional
			Dictionary with masks for the data to select only part of the data. Uses same keywords as data dictionary.
			Default value = None.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.
		print_num : bool, optional
			If True, prints the number of objects in the shape and positon samples. Default value is True.
		jk_group_name : str, optional
			Group in output file (hdf5) where jackknife realisations are stored. Is used when this method is called in
			MeasureJackknife. Default value is "".

		Returns
		-------
		ndarrays
			xi_g+, xi_gg, r_p bins, pi bins if no output file is specified

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
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
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

		pos_tree = KDTree(positions[:, not_LOS], boxsize=self.boxsize)
		for i in np.arange(0, len(positions_shape_sample), 100):
			i2 = min(len(positions_shape_sample), i + 100)
			positions_shape_sample_i = positions_shape_sample[i:i2]
			axis_direction_i = axis_direction[i:i2]
			e_i = e[i:i2]
			weight_shape_i = weight_shape[i:i2]
			shape_tree = KDTree(positions_shape_sample_i[:, not_LOS], boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(pos_tree, self.r_min)
			ind_max_i = shape_tree.query_ball_tree(pos_tree, self.r_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			if save_tree:
				with open(f"{file_tree_path}/w_{self.simname}_tree_{figname_dataset_name}.pickle", 'ab') as handle:
					pickle.dump(ind_rbin_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
			for n in np.arange(0, len(positions_shape_sample_i)):  # CHANGE2: loop now over shapes, not positions
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]
				if self.periodicity:
					separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
					separation[separation < -self.L_0p5] += self.boxsize
				projected_sep = separation[:, not_LOS]
				LOS = separation[:, LOS_ind]
				del separation
				separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
				with np.errstate(invalid='ignore'):
					separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
					del projected_sep
					phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
				del separation_dir
				e_plus, e_cross = self.get_ellipticity(e, phi)
				del phi
				e_plus[np.isnan(e_plus)] = 0.0
				e_cross[np.isnan(e_cross)] = 0.0

				# get the indices for the binning
				mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * (
						LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
				ind_r = np.floor(
					np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.r_bins[0]) / sub_box_len_logrp
				)
				del separation_len
				ind_r = np.array(ind_r, dtype=int)
				ind_pi = np.floor(
					LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
				)  # need length of LOS, so only positive values
				del LOS
				ind_pi = np.array(ind_pi, dtype=int)
				if np.any(ind_pi == self.num_bins_pi):
					ind_pi[ind_pi >= self.num_bins_pi] -= 1
				if np.any(ind_r == self.num_bins_r):
					ind_r[ind_r >= self.num_bins_r] -= 1
				np.add.at(Splus_D, (ind_r, ind_pi),
						  (weight[ind_rbin_i[n]] * weight_shape[mask] * e_plus[mask]) / (2 * R))
				np.add.at(Scross_D, (ind_r, ind_pi),
						  (weight[ind_rbin_i[n]] * weight_shape[mask] * e_cross[mask]) / (2 * R))
				np.add.at(variance, (ind_r, ind_pi),
						  ((weight[ind_rbin_i[n]] * weight_shape[mask] * e_plus[mask]) / (2 * R)) ** 2)
				np.add.at(Splus_D_jk, (jackknife_region_indices_pos[ind_rbin_i[n]], ind_r, ind_pi),
						  (weight[ind_rbin_i[n]] * weight_shape[mask] * e_plus[mask]) / (2 * R))
				np.add.at(Splus_D_jk, (jackknife_region_indices_shape[mask], ind_r, ind_pi),
						  (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))

				del e_plus, e_cross
				np.add.at(DD, (ind_r, ind_pi), weight[ind_rbin_i[n]] * weight_shape[mask])
				np.add.at(Splus_D_jk, (jackknife_region_indices_pos[ind_rbin_i[n]], ind_r, ind_pi),
						  (weight[ind_rbin_i[n]] * weight_shape[mask]))
				np.add.at(Splus_D_jk, (jackknife_region_indices_shape[mask], ind_r, ind_pi),
						  (weight[ind_rbin_i[n]] * weight_shape[mask]))

		# if Num_position == Num_shape:
		# 	corrtype = "auto"
		# 	DD = DD / 2.0  # auto correlation, all pairs are double
		# else:

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		RR_jk = (num_box - 1) / num_box * RR_g_plus
		sigsq = variance / RR_g_plus ** 2
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
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=(Splus_D - Splus_D_jk[i]) / RR_jk)
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			for i in np.arange(0, num_box):
				write_dataset_hdf5(group, dataset_name + f"_{i}", data=((DD - DD_jk[i]) / RR_jk) - 1)
				write_dataset_hdf5(group, dataset_name + f"_{i}_rp", data=separation_bins)
				write_dataset_hdf5(group, dataset_name + f"_{i}_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins, Splus_D, DD, RR_g_plus


# jk realisation = sum of all regions - any pairs with either shape or pos in sub region
# or
# jk realistation = sum of all included subregions where neither shape or position is in subregion
# save DD with either shape or pos in subregion and subtract for n1
#

if __name__ == "__main__":
	pass
