import numpy as np
import h5py
from pathos.multiprocessing import ProcessingPool
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_w_lightcone import MeasureWLightcone
from .measure_m_lightcone import MeasureMultipolesLightcone


class MeasureJackknife(MeasureWLightcone,
					   MeasureMultipolesLightcone):
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

	def measure_covariance_multiple_datasets(self, corr_type, dataset_names, num_box=27, return_output=False):
		"""Combines the jackknife measurements for different datasets into one covariance matrix.
		Author: Marta Garcia Escobar (starting from measure_jackknife methods); updated

		Parameters
		----------
		corr_type : str
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
		if corr_type not in valid_corr_types:
			raise ValueError("corr_type must be 'w_g_plus', 'w_gg', 'multipoles_g_plus' or 'multipoles_gg'.")

		data_file = h5py.File(self.output_file_name, "a")

		mean_list = []  # list of arrays

		for dataset_name in dataset_names:
			group = data_file[f"{self.snap_group}/{corr_type}/{dataset_name}_jk{num_box}"]
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
			group = data_file[f"{self.snap_group}/{corr_type}/{dataset_name}_jk{num_box}"]
			for b in np.arange(0, num_box):
				std += (group[dataset_name + "_" + str(b)] - mean_list[0]) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group[dataset_name + "_" + str(b)] - mean_list[0]) * (
							group[dataset_name + "_" + str(b)][i] - mean_list[0][i]
					)
		elif len(dataset_names) == 2:
			group0 = data_file[f"{self.snap_group}/{corr_type}/{dataset_names[0]}_jk{num_box}"]
			group1 = data_file[f"{self.snap_group}/{corr_type}/{dataset_names[1]}_jk{num_box}"]
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

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/{corr_type}")
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
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[0], dataset_names[1]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[0], dataset_names[2]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[1], dataset_names[2]], num_box=num_box)

		# import needed datasets
		output_file = h5py.File(self.output_file_name, "a")
		group = output_file[f"{self.snap_group}/{corr_type}"]

		# cov matrix between datasets
		cov_xx = group[f'{dataset_names[0]}_jackknife_cov_{num_box}'][:]
		cov_yy = group[f'{dataset_names[1]}_jackknife_cov_{num_box}'][:]
		cov_zz = group[f'{dataset_names[2]}_jackknife_cov_{num_box}'][:]
		cov_xy = group[f'{dataset_names[0]}_{dataset_names[1]}_jackknife_cov_{num_box}'][:]
		cov_yz = group[f'{dataset_names[0]}_{dataset_names[2]}_jackknife_cov_{num_box}'][:]
		cov_xz = group[f'{dataset_names[1]}_{dataset_names[2]}_jackknife_cov_{num_box}'][:]

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
