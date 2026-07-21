import random as _random

import numpy as np
import h5py
from kmeans_radec import kmeans_sample
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase


class MeasureJackknife(MeasureIABase):
	"""Class that contains all methods for jackknife covariance measurements for IA correlation functions.

	Methods
	-------
	_measure_jackknife_realisations_obs()
		Measures all jackknife realisations for MeasureIALightcone using 1 or more CPUs.
	_measure_jackknife_covariance_obs()
		Combines jackknife realisations for MeasureIALightcone into covariance.
	_measure_jackknife_realisations_obs_multiprocessing()
		Measures all jackknife realisations for MeasureIALightcone using >1 CPU.
	measure_covariance_multiple_datasets()
		Given the jackknife realisations of two datasets, creates the cross covariance.
	create_full_cov_matrix_projections()
		Creates larger covariance matrix of multiple datasets including cross terms.

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
		The __init__ method of the MeasureJackknife class.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		return

	def assign_jackknife_patches(self, data, randoms_data, num_jk, seed=None):
		"""Assigns jackknife patches to data and randoms given a number of patches.
		Based on https://github.com/esheldon/kmeans_radec

		Parameters
		----------
		data : dict
			Dictionary containing position and shape sample data. Keywords: "RA", "DEC", "RA_shape_sample",
			"DEC_shape_sample"
		randoms_data : dict
			Dictionary containing position and shape sample data of randoms. Keywords: "RA", "DEC", "RA_shape_sample",
			"DEC_shape_sample"
		num_jk : int
			Number of jackknife patches
		seed : int or NoneType, optional
			Seed for the k-means initialisation, making the patch assignment reproducible. If None (default),
			the patches differ between runs. The global numpy random state is restored afterwards.

		Returns
		-------
		dict
			Dictionary with patch numbers for each sample. Keywords: 'position', 'shape', 'randoms_position',
			'randoms_shape'

		"""

		jk_patches = {}

		# Read the randoms file from which the jackknife regions will be created
		RA = randoms_data['RA']
		DEC = randoms_data['DEC']

		# Define a number of jaccknife regions and find their centres using kmans
		X = np.column_stack((RA, DEC))
		if not (isinstance(num_jk, (int, np.integer)) and not isinstance(num_jk, bool) and num_jk >= 1):
			raise ValueError(f"num_jk must be an integer >= 1, got {num_jk!r}.")
		if num_jk > len(X):
			raise ValueError(
				f"num_jk ({num_jk}) cannot exceed the number of randoms ({len(X)}) used to build "
				f"the jackknife patches. Lower num_jk or provide more randoms.")
		if seed is None:
			km = kmeans_sample(X, num_jk, maxiter=100, tol=1.0e-5)
		else:
			# kmeans_radec draws its starting sample with the stdlib random module
			np_state, py_state = np.random.get_state(), _random.getstate()
			np.random.seed(seed)
			_random.seed(seed)
			try:
				km = kmeans_sample(X, num_jk, maxiter=100, tol=1.0e-5)
			finally:
				np.random.set_state(np_state)
				_random.setstate(py_state)
		jk_labels = km.labels

		jk_patches['randoms_position'] = jk_labels

		RA = randoms_data['RA_shape_sample']
		DEC = randoms_data['DEC_shape_sample']
		X2 = np.column_stack((RA, DEC))
		jk_labels = km.find_nearest(X2)

		jk_patches['randoms_shape'] = jk_labels

		RA_data = data['RA']
		DEC_data = data['DEC']
		X3 = np.column_stack((RA_data, DEC_data))
		jk_labels = km.find_nearest(X3)

		jk_patches['position'] = jk_labels

		RA_data = data['RA_shape_sample']
		DEC_data = data['DEC_shape_sample']
		X4 = np.column_stack((RA_data, DEC_data))
		jk_labels = km.find_nearest(X4)

		jk_patches['shape'] = jk_labels

		return jk_patches

	def _combine_jackknife_information(self, dataset_name, jk_group_name, corr_group, num_box, return_output=False):
		"""
		Combine jackknife realisations into a covariance matrix.

		Parameters
		----------
		dataset_name: str
			Name of the dataset in the output file.
		jk_group_name: str
			Name of the subgroup in the output file where the jackknife realisations are saved.
		corr_group: list of str
			Name of the subgroups in the output file denoting the correlation (e.g. w_g_plus, multipoles_gg etc).
		num_box: int
			Number of jackknife realisations.
		return_output: bool, optional
			When True, returns output, otherwise saves to output file.

		Returns
		-------
		list of ndarrays
			list of covariances for each entry in corr_group and list of standard deviations for each entry in corr_group

		"""
		covs, stds = [], []
		for d in np.arange(0, len(corr_group)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file[f"{self.snap_group}{corr_group[d]}/{jk_group_name}/"]
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
				group_multipoles = create_group_hdf5(output_file, f"{self.snap_group}" + corr_group[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_mean_" + str(num_box), data=mean_multipoles)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
		if return_output:
			return covs, stds
		else:
			return

	def _get_jackknife_region_indices(self, masks, L_subboxes):
		"""
		Split the box in L_subboxes^3 subboxes and return indices of which subbox objects are in for position and
		shape sample.

		Parameters
		----------
		masks: dict or NoneType
			Input in methods in MeasureIABox that masks the input data dictionary.
		L_subboxes: int
			Number of subboxes on one side of the box. L_subboxes^3 is the total number of jackknife realisations.

		Returns
		-------
		ndarrays
			indices of jackknife region of position sample and indices of jackknife region of shape sample

		"""
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
					x_mask_shape = (positions_shape_sample[:, 0] > x_bounds[0]) * (
							positions_shape_sample[:, 0] < x_bounds[1])
					y_mask_shape = (positions_shape_sample[:, 1] > y_bounds[0]) * (
							positions_shape_sample[:, 1] < y_bounds[1])
					z_mask_shape = (positions_shape_sample[:, 2] > z_bounds[0]) * (
							positions_shape_sample[:, 2] < z_bounds[1])
					mask_position = x_mask * y_mask * z_mask  # mask that is True for all positions in the subbox
					mask_shape = x_mask_shape * y_mask_shape * z_mask_shape  # mask that is True for all positions not in the subbox
					jackknife_region_indices_pos[mask_position] = num_box
					jackknife_region_indices_shape[mask_shape] = num_box
					num_box += 1
		return np.array(jackknife_region_indices_pos, dtype=int), np.array(jackknife_region_indices_shape, dtype=int)

	def measure_covariance_multiple_datasets(self, corr_types, dataset_names, num_box=27, return_output=False):
		"""Combines the jackknife measurements for different datasets into one covariance matrix.
		Author: Marta Garcia Escobar (starting from measure_jackknife methods); updated

		Parameters
		----------
		corr_types : list of str
			Which type of correlation is measured. Takes 'w_g_plus', 'w_gg', 'multipoles_g_plus' or 'multipoles_gg'.
		dataset_names : list of str
			List of the dataset names. If there is only one value, it calculates the covariance matrix with itself.
		num_box : int, optional
			Number of jackknife realisations. Default value is 27.
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.

		Returns
		-------
		ndarray, ndarray
			covariance, standard deviation

		"""
		# check if corr_type is valid
		valid_corr_types = ["w_g_plus", "multipoles_g_plus", "w_gg", "multipoles_gg"]
		for corr_type in corr_types:
			if corr_type not in valid_corr_types:
				raise ValueError("corr_type must be 'w_g_plus', 'w_gg', 'multipoles_g_plus' or 'multipoles_gg'.")

		data_file = h5py.File(self.output_file_name, "a")

		mean_list = []  # list of arrays

		for d, dataset_name in enumerate(dataset_names):
			group = data_file[f"{self.snap_group}{corr_types[d]}/{dataset_name}_jk{num_box}"]
			mean_multipoles = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				mean_multipoles += group[dataset_name + "_" + str(b)]
			mean_multipoles /= num_box
			mean_list.append(mean_multipoles)

		# calculation the covariance matrix and the standard deviation (sqrt of diag of cov)
		cov = np.zeros((self.num_bins_r, self.num_bins_r))
		std = np.zeros(self.num_bins_r)

		if len(dataset_names) == 1:  # covariance with itself
			dataset_name = dataset_names[0]
			group = data_file[f"{self.snap_group}{corr_types[0]}/{dataset_name}_jk{num_box}"]
			for b in np.arange(0, num_box):
				std += (group[dataset_name + "_" + str(b)] - mean_list[0]) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group[dataset_name + "_" + str(b)] - mean_list[0]) * (
							group[dataset_name + "_" + str(b)][i] - mean_list[0][i]
					)
		elif len(dataset_names) == 2:
			group0 = data_file[f"{self.snap_group}{corr_types[0]}/{dataset_names[0]}_jk{num_box}"]
			group1 = data_file[f"{self.snap_group}{corr_types[1]}/{dataset_names[1]}_jk{num_box}"]
			for b in np.arange(0, num_box):
				std += (group0[dataset_names[0] + "_" + str(b)] - mean_list[0]) * (
						group1[dataset_names[1] + "_" + str(b)] - mean_list[1])
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group0[dataset_names[0] + "_" + str(b)] - mean_list[0]) * (
							group1[dataset_names[1] + "_" + str(b)][i] - mean_list[1][i]
					)
		else:
			raise KeyError("Too many datasets given, choose either 1 or 2")

		std *= (num_box - 1) / num_box  # see Singh 2023
		std = np.sqrt(std)  # size of errorbars
		cov *= (num_box - 1) / num_box  # cov not sqrt so to get std, sqrt of diag would need to be taken

		data_file.close()
		if len(corr_types) == 1 or corr_types[0] == corr_types[1]:
			corr_group_name = corr_types[0]
		else:
			corr_group_name = f"{corr_types[0]}_{corr_types[1]}"

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}{corr_group_name}")
			if len(dataset_names) == 2:
				write_dataset_hdf5(group, dataset_names[0] + "_" + dataset_names[1] + "_jackknife_cov_" + str(
					num_box), data=cov)
				write_dataset_hdf5(group,
								   dataset_names[0] + "_" + dataset_names[1] + "_jackknife_" + str(num_box),
								   data=std)

			else:
				write_dataset_hdf5(group, dataset_names[0] + "_jackknife_cov_" + str(num_box), data=cov)
				write_dataset_hdf5(group, dataset_names[0] + "_jackknife_" + str(num_box), data=std)
			output_file.close()
			return
		else:
			return cov, std

	def create_full_cov_matrix_projections(self, corr_type, dataset_names=["LOS_x", "LOS_y", "LOS_z"], num_box=27,
										   return_output=False):
		"""Function that creates the full covariance matrix for all 3 projections and combined covariance for 2
		projections by combining previously obtained jackknife information. Generalised from Marta Garcia Escobar's code.

		Parameters
		----------
		corr_type : str
			Which type of correlation is measured. Takes 'w_g_plus', 'w_gg', 'multipoles_g_plus' or 'multipoles_gg'.
		num_box : int, optional
			Number of jackknife realisations. Default value is 27.
		dataset_names : list of str
			Dataset names of projections to be combined. Default value is ["LOS_x","LOS_y","LOS_z"].
		return_output : bool, optional
			If True, the output will be returned instead of written to a file. Default value is False.

		Returns
		-------
		ndarrays
			covariance for 3 projections, covariance for x and y, covariance for x and z, covariance for y and z

		"""
		# corr_type may be given as a single string (applies to all three
		# projections) or as a list/tuple of 3 strings (one per projection,
		# all of which must currently be identical).
		if isinstance(corr_type, (list, tuple)):
			if len(set(corr_type)) != 1:
				raise ValueError(
					"All entries of corr_type must currently be identical.")
			corr_type = corr_type[0]
		valid_corr_types = ["w_g_plus", "multipoles_g_plus", "w_gg", "multipoles_gg"]
		if corr_type not in valid_corr_types:
			raise ValueError("corr_type must be 'w_g_plus', 'w_gg', 'multipoles_g_plus' or 'multipoles_gg'.")

		self.measure_covariance_multiple_datasets(corr_types=[corr_type],
												  dataset_names=[dataset_names[0]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_types=[corr_type],
												  dataset_names=[dataset_names[1]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_types=[corr_type],
												  dataset_names=[dataset_names[2]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_types=[corr_type, corr_type],
												  dataset_names=[dataset_names[0], dataset_names[1]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_types=[corr_type, corr_type],
												  dataset_names=[dataset_names[0], dataset_names[2]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_types=[corr_type, corr_type],
												  dataset_names=[dataset_names[1], dataset_names[2]], num_box=num_box)

		# import needed datasets
		output_file = h5py.File(self.output_file_name, "a")
		group = output_file[f"{self.snap_group}{corr_type}"]

		# cov matrix between datasets
		cov_xx = group[f'{dataset_names[0]}_jackknife_cov_{num_box}'][:]
		cov_yy = group[f'{dataset_names[1]}_jackknife_cov_{num_box}'][:]
		cov_zz = group[f'{dataset_names[2]}_jackknife_cov_{num_box}'][:]
		cov_xy = group[f'{dataset_names[0]}_{dataset_names[1]}_jackknife_cov_{num_box}'][:]
		cov_xz = group[f'{dataset_names[0]}_{dataset_names[2]}_jackknife_cov_{num_box}'][:]
		cov_yz = group[f'{dataset_names[1]}_{dataset_names[2]}_jackknife_cov_{num_box}'][:]

		# 3 projections
		cov_top = np.concatenate((cov_xx, cov_xy, cov_xz), axis=1)
		cov_middle = np.concatenate((cov_xy.T, cov_yy, cov_yz), axis=1)  # cov_xy.T = cov_yx
		cov_bottom = np.concatenate((cov_xz.T, cov_yz.T, cov_zz), axis=1)
		cov3 = np.concatenate((cov_top, cov_middle, cov_bottom), axis=0)

		# all 2 projections
		cov_top = np.concatenate((cov_xx, cov_xy), axis=1)
		cov_middle = np.concatenate((cov_xy.T, cov_yy), axis=1)  # cov_xz.T = cov_zx
		cov2xy = np.concatenate((cov_top, cov_middle), axis=0)

		cov_top = np.concatenate((cov_xx, cov_xz), axis=1)
		cov_middle = np.concatenate((cov_xz.T, cov_zz), axis=1)  # cov_xz.T = cov_zx
		cov2xz = np.concatenate((cov_top, cov_middle), axis=0)

		cov_top = np.concatenate((cov_yy, cov_yz), axis=1)
		cov_middle = np.concatenate((cov_yz.T, cov_zz), axis=1)  # cov_xz.T = cov_zx
		cov2yz = np.concatenate((cov_top, cov_middle), axis=0)

		if return_output:
			return cov3, cov2xy, cov2xz, cov2yz
		else:
			write_dataset_hdf5(group,
							   f"{dataset_names[0]}_{dataset_names[1]}_{dataset_names[2]}_combined_jackknife_cov_{num_box}",
							   data=cov3)
			write_dataset_hdf5(group,
							   f'{dataset_names[0]}_{dataset_names[1]}_combined_jackknife_cov_{num_box}',
							   data=cov2xy)
			write_dataset_hdf5(group,
							   f'{dataset_names[0]}_{dataset_names[2]}_combined_jackknife_cov_{num_box}',
							   data=cov2xz)
			write_dataset_hdf5(group,
							   f'{dataset_names[1]}_{dataset_names[2]}_combined_jackknife_cov_{num_box}',
							   data=cov2yz)
			return


if __name__ == "__main__":
	pass
