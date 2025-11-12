import sympy
import numpy as np
from .measure_w_box_jk import MeasureWBoxJackknife
from .measure_m_box_jk import MeasureMBoxJackknife
from .measure_w_box import MeasureWBox
from .measure_m_box import MeasureMultipolesBox


class MeasureIABox(MeasureWBox, MeasureMultipolesBox, MeasureWBoxJackknife, MeasureMBoxJackknife):
	r"""Manages the IA correlation function measurement methods used in the MeasureIA package based on speed and input.
	This class is used to call the methods that measure $w_{gg}$, $w_{g+}$ and multipoles for simulations in cartesian
	coordinates. Depending on the input parameters, various correlations incl covariance estimates are measured for
	given data.

	Methods
	-------
	measure_xi_w()
		Compute projected correlations $w_{gg}$ and/or $w_{g+}$.
	measure_xi_multipoles()
		Compute multipoles of the correlation functions, $\tilde{\xi}_{gg,0}$ and/or $\tilde{\xi}_{g+,2}$.

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
			num_nodes=1,
	):
		"""
		The __init__ method of the MeasureIABox class.

		Parameters
		----------
		num_nodes : int, optional
			Number of cores to be used in multiprocessing. Default is 1.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		self.num_nodes = num_nodes
		self.randoms_data = None
		self.data_dir = None
		self.num_samples = None

		return

	def measure_xi_w(self, dataset_name, corr_type, num_jk=0, temp_file_path=None, masks=None,
					 ellipticity='distortion', chunk_size=1000):
		r"""Measures $\xi_{gg}$, $\xi_{g+}$ and $w_{gg}$, $w_{g+}$ including jackknife covariance if desired.
		Manages the various _measure_xi_rp_pi_box method options in MeasureWBox and MeasureWBoxJackknife.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		num_jk : int, optional
			Number of jackknife regions (needs to be x^3, with x an int) for the covariance measurement.
			Default is 0 (no covariance).
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Directory of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
			Default is 1000.
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.

		"""
		if num_jk > 0:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
			except AssertionError:
				raise ValueError(
					f"Use x^3 as input for num_jk, with x as an int. {float(int(num_jk ** (1. / 3)))},{num_jk ** (1. / 3)}")

		if temp_file_path == False:
			temp_storage = False
			temp_file_path = None
		else:
			temp_storage = True
		if temp_storage and temp_file_path == None:
			raise ValueError(
				"Input temp_file_path for faster computation. Do not want to save data temporarily? Input file_path_tree=False.")

		# replace by better checks of input data
		try:
			RA = self.data["RA"]
			print("Given data is lightcone, use measure_xi_w_lightcone method instead.")
			exit()
		except:
			pass

		if num_jk > 0:  # include covariance
			if self.num_nodes > 1 and temp_storage:
				self._measure_xi_rp_pi_box_jk_multiprocessing(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															  return_output=False,
															  num_nodes=self.num_nodes,
															  jk_group_name=f"{dataset_name}_jk{num_jk}",
															  chunk_size=chunk_size, ellipticity=ellipticity,
															  temp_file_path=temp_file_path)
			elif temp_storage:
				self._measure_xi_rp_pi_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
												   return_output=False, ellipticity=ellipticity,
												   jk_group_name=f"{dataset_name}_jk{num_jk}")
			else:
				self._measure_xi_rp_pi_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
													return_output=False, ellipticity=ellipticity,
													jk_group_name=f"{dataset_name}_jk{num_jk}")
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			for i in np.arange(num_jk):
				self._measure_w_g_i(corr_type=corr_type, dataset_name=f"{dataset_name}_{i}",
									jk_group_name=f"{dataset_name}_jk{num_jk}", return_output=False)
			if corr_type == "both":
				corr_group = ["w_g_plus", "w_gg"]
			elif corr_type == "g+":
				corr_group = ["w_g_plus"]
			elif corr_type == "gg":
				corr_group = ["w_gg"]
			else:
				raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:  # no covariance
			if self.num_nodes > 1 and temp_storage:
				self._measure_xi_rp_pi_box_multiprocessing(dataset_name=dataset_name, temp_file_path=temp_file_path,
														   masks=masks, return_output=False, num_nodes=self.num_nodes,
														   chunk_size=chunk_size, ellipticity=ellipticity)
			elif temp_storage:
				self._measure_xi_rp_pi_box_tree(masks=masks, dataset_name=dataset_name,
												return_output=False, ellipticity=ellipticity)
			else:
				self._measure_xi_rp_pi_box_brute(masks=masks, dataset_name=dataset_name,
												 return_output=False, ellipticity=ellipticity)
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		return

	def measure_xi_multipoles(self, dataset_name, corr_type, num_jk=0, temp_file_path=None, masks=None, rp_cut=None,
							  ellipticity='distortion', chunk_size=1000):
		r"""Measures $\xi_{gg}$, $\xi_{g+}$ and $\tilde{\xi}_{gg,0}$, $\tilde{\xi}_{g+,2}$ including jackknife covariance
		if desired. Manages the various _measure_xi_r_mur_box method options in MeasureMultipolesBox and
		MeasureMultipolesBoxJackknife.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		num_jk : int, optional
			Number of jackknife regions (needs to be x^3, with x an int) for the covariance measurement. Default is 0 (no covariance).
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Directory of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		rp_cut : float or NoneType, optional
			Applies a minimum r_p value condition for pairs to be included. Default is None.
		chunck_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.
		"""

		if num_jk > 0:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
			except AssertionError:
				raise ValueError(
					f"Use x^3 as input for num_jk, with x as an int. {float(int(num_jk ** (1. / 3)))},{num_jk ** (1. / 3)}")

		if temp_file_path == False:
			temp_storage = False
			temp_file_path = None
		else:
			temp_storage = True
		if temp_storage and temp_file_path == None:
			raise ValueError(
				"Input temp_file_path for faster computation. Do not want to save data temporarily? Input file_path_tree=False.")

		# replace by better checks of input data
		try:
			RA = self.data["RA"]
			print("Given data is lightcone, use measure_xi_w_lightcone method instead.")
			exit()
		except:
			pass

		if num_jk > 0:  # include covariance
			if self.num_nodes > 1 and temp_storage:
				self._measure_xi_r_mur_box_jk_multiprocessing(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															  return_output=False,
															  num_nodes=self.num_nodes,
															  jk_group_name=f"{dataset_name}_jk{num_jk}",
															  chunk_size=chunk_size, ellipticity=ellipticity,
															  temp_file_path=temp_file_path)
			elif temp_storage:
				self._measure_xi_r_mur_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
												   return_output=False, ellipticity=ellipticity,
												   jk_group_name=f"{dataset_name}_jk{num_jk}")
			else:
				self._measure_xi_r_mur_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
													return_output=False, ellipticity=ellipticity,
													jk_group_name=f"{dataset_name}_jk{num_jk}")
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			for i in np.arange(num_jk):
				self._measure_multipoles(corr_type=corr_type, dataset_name=f"{dataset_name}_{i}",
										 jk_group_name=f"{dataset_name}_jk{num_jk}", return_output=False)
			if corr_type == "both":
				corr_group = ["multipoles_g_plus", "multipoles_gg"]
			elif corr_type == "g+":
				corr_group = ["multipoles_g_plus"]
			elif corr_type == "gg":
				corr_group = ["multipoles_gg"]
			else:
				raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:  # no covariance
			if self.num_nodes > 1 and temp_storage:
				self._measure_xi_r_mur_box_multiprocessing(dataset_name=dataset_name, temp_file_path=temp_file_path,
														   masks=masks, return_output=False, num_nodes=self.num_nodes,
														   chunk_size=chunk_size, ellipticity=ellipticity)
			elif temp_storage:
				self._measure_xi_r_mur_box_tree(masks=masks, dataset_name=dataset_name,
												return_output=False,
												ellipticity=ellipticity)
			else:
				self._measure_xi_r_mur_box_brute(masks=masks, dataset_name=dataset_name,
												 return_output=False, ellipticity=ellipticity)
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		return


if __name__ == "__main__":
	pass
