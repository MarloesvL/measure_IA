import sympy
import numpy as np
from .measure_jackknife import MeasureJackknife
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


class MeasureIALightcone(MeasureJackknife):
	r"""Manages the IA correlation function measurement methods used in the MeasureIA package based on speed and input.
	This class is used to call the methods that measure w_gg, w_g+ and multipoles for simulations (and observations),
	with lightcone data.
	Depending on the input parameters, various correlations incl covariance estimates are measured for given data.

	Attributes
	----------
	data_dir : dict or NoneType
		Temporary storage space for added data directory to allow for flexibility in passing data or randoms to internal
		methods.
	num_samples : dict or NoneType
		Dictionary containing the numbers of objects for each sample for lightcone-type measurements. Filled internally,
		no input needed.

	Methods
	-------
	measure_xi_w()
		Compute projected correlations $w_{gg}$ and/or $w_{g+}$.
	measure_xi_multipoles()
		Compute multipoles of the correlation functions, $\tilde{\xi}_{gg,0}$ and/or $\tilde{\xi}_{g+,2}$.

	Notes
	-----
	Inherits attributes from 'SimInfo', where none are used in this class.
	Inherits attributes from 'MeasureIABase', where 'data', 'output_file_name', 'Num_position',
	'Num_shape', 'r_min', 'r_max', 'num_bins_r', 'num_bins_pi', 'r_bins', 'pi_bins', 'mu_r_bins' are used.

	"""

	def __init__(
			self,
			data,
			randoms_data,
			output_file_name,
			separation_limits=[0.1, 20.0],
			num_bins_r=8,
			num_bins_pi=20,
			pi_max=None,
			num_nodes=1,
	):
		"""
		The __init__ method of the MeasureIALightcone class.

		Parameters
		----------
		randoms_data : dict or NoneType
			Dictionary with data of the randoms needed for lightcone-type measurements.
			The keywords are:
			'Redshift' and 'Redshift_shape_sample': (N_p) and (N_s) ndarray with redshifts of position and shape samples.
			'RA' and 'RA_shape_sample': (N_p) and (N_s) ndarray with RA coordinate of position and shape samples.
			'DEC' and 'DEC_shape_sample': (N_p) and (N_s) ndarray with DEC coordinate of position and shape samples.
			If only 'Redshift', 'RA' and 'DEC' are added, the sample will be used for both position and shape sample randoms.
		num_nodes : int, optional
			Number of cores to be used in multiprocessing. Default is 1.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, False, None, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, None, False)
		self.num_nodes = num_nodes
		self.randoms_data = randoms_data
		self.data_dir = None
		self.num_samples = None

		return

	def measure_xi_w(self, IA_estimator, dataset_name, corr_type, jk_patches=None, num_jk=None,
					 measure_cov=True, masks=None, masks_randoms=None, cosmology=None, over_h=False):
		"""Measures xi_gg, xi_g+ and w_gg, w_g+ including jackknife covariance if desired for lightcone data.
		Manages the various _measure_xi_rp_pi_obs and _measure_jackknife_covariance options in MeasureWObservations
		and MeasureJackknife.

		Parameters
		----------
		IA_estimator : str
			Choose which type of xi estimator is used. Choose from "clusters" or "galaxies".
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		randoms_data : dict or NoneType
			Dictionary that includes the randoms data in the same form as the data dictionary.
		jk_patches : dict or NoneType, optional
			Dictionary with entries of the jackknife patch numbers (ndarray) for each sample, named "position", "shape"
			and "random". Default is None.
		num_jk : int, optional
			Number of jackknife patches to be generated internally. Default is None.
		measure_cov : bool, optional
			If True, jackknife errors are calculated. Default is True.
		masks : dict or NoneType, optional
			Dictionary of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		masks_randoms : dict or NoneType, optional
			Dictionary of mask information for the randoms data in the same form as the data dictionary,
			where the masks are placed over the data to apply selections. Default is None.
		cosmology : pyccl cosmology object or NoneType, optional
			Pyccl cosmology to use in the calculation. If None (default), the cosmology is used:
			ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
		over_h : bool, optional
			If True, the units are assumed to be in not-over-h and converted to over-h units. Default is False.

		"""
		if IA_estimator == "clusters":
			if self.randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				raise KeyError("This version does not work yet, add randoms.")
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if self.randoms_data == None:
				raise KeyError("No randoms given. Please provide input.")
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
				print("WARNING: this version of the code has not been fully validated. Proceed with caution.")
		else:
			raise KeyError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		# todo: Expand to include methods with trees and internal multiproc
		# todo: Checks to see if data directories include everything they need
		data = self.data  # temporary save so it can be restored at the end of the calculation

		try:  # Are there one or two random samples given?
			random_shape = self.randoms_data["RA_shape_sample"]
			one_random_sample = False
		except:
			one_random_sample = True
			self.randoms_data["RA_shape_sample"] = self.randoms_data["RA"]
			self.randoms_data["DEC_shape_sample"] = self.randoms_data["DEC"]
			self.randoms_data["Redshift_shape_sample"] = self.randoms_data["Redshift"]
		try:
			weight = self.randoms_data["weight"]
		except:
			self.randoms_data["weight"] = np.ones(len(self.randoms_data["RA"]))
		try:
			weight = self.randoms_data["weight_shape_sample"]
		except:
			if one_random_sample:
				self.randoms_data["weight_shape_sample"] = self.randoms_data["weight"]  # in case weights are given
			else:
				self.randoms_data["weight_shape_sample"] = np.ones(len(self.randoms_data["RA_shape_sample"]))

		if measure_cov:
			if jk_patches == None:
				if num_jk != None:
					jk_patches = self.assign_jackknife_patches(data, self.randoms_data, num_jk)
				else:
					raise ValueError("Set calc_errors to False, or provide either jk_patches or num_jk input.")
			else:
				if one_random_sample:
					jk_patches["randoms_position"] = jk_patches["randoms"]
					jk_patches["randoms_shape"] = jk_patches["randoms"]

		self.data_dir = data
		try:
			weight = self.data_dir["weight"]
		except:
			self.data_dir["weight"] = np.ones(len(self.data_dir["RA"]))
		try:
			weight = self.data_dir["weight_shape_sample"]
		except:
			self.data_dir["weight_shape_sample"] = np.ones(len(self.data_dir["RA_shape_sample"]))

		num_samples = {}  # Needed to correct for different number of randoms and galaxies/clusters in data
		if masks == None:
			num_samples["D"] = len(self.data_dir["RA"])
			num_samples["S"] = len(self.data_dir["RA_shape_sample"])
		else:
			num_samples["D"] = len(self.data_dir["RA"][masks["RA"]])
			num_samples["S"] = len(self.data_dir["RA_shape_sample"][masks["RA_shape_sample"]])
		if masks_randoms == None:
			num_samples["R_D"] = len(self.randoms_data["RA"])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"])
		else:
			num_samples["R_D"] = len(self.randoms_data["RA"][masks_randoms["RA"]])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"][masks_randoms["RA_shape_sample"]])
		# print(self.data_dir,self.randoms_data)

		# Shape-position combinations:
		# S+D (Cg+, Gg+)
		# S+R (Cg+, Gg+)
		if corr_type == "g+" or corr_type == "both":
			# S+D
			self.data = self.data_dir
			self._measure_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=dataset_name,
												   over_h=over_h,
												   cosmology=cosmology)
			# S+R
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"e1": self.data_dir["e1"],
				"e2": self.data_dir["e2"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			# print(self.data)
			self._measure_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=f"{dataset_name}_randoms",
												   over_h=over_h,
												   cosmology=cosmology)

		# Position-position combinations:
		# SD (Cgg, Ggg)
		# SR (Cg+, Cgg, Ggg)
		# RD (Cgg, Ggg)
		# RR (Cgg, Gg+, Ggg)

		if corr_type == "gg":  # already have it for 'both'
			# SD (Cgg, Ggg)
			self.data = {
				"Redshift": self.data_dir["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.data_dir["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.data_dir["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"weight": self.data_dir["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			self._count_pairs_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology,
													   data_suffix="_DD")

			# SR (Cg+, Cgg, Ggg) - watch name (Obs estimator) # if g+ or both, already have it
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			self._count_pairs_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology,
													   data_suffix="_SR")

		if corr_type == "gg" or corr_type == "both":
			# RD (Cgg, Ggg)
			self.data = {
				"Redshift": self.data_dir["Redshift"],
				"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
				"RA": self.data_dir["RA"],
				"RA_shape_sample": self.randoms_data["RA_shape_sample"],
				"DEC": self.data_dir["DEC"],
				"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
				"weight": self.data_dir["weight"],
				"weight_shape_sample": self.randoms_data["weight_shape_sample"]
			}
			self._count_pairs_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology,
													   data_suffix="_RD")

		if IA_estimator == "galaxies" or corr_type == "gg" or corr_type == "both":
			# RR (Cgg, Gg+, Ggg)
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.randoms_data["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.randoms_data["weight_shape_sample"]
			}
			self._count_pairs_xi_rp_pi_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology,
													   data_suffix="_RR")

		self._obs_estimator([corr_type, "w"], IA_estimator, dataset_name, f"{dataset_name}_randoms", num_samples)
		self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		if measure_cov:
			self.num_samples = {}
			min_patch, max_patch = int(min(jk_patches["shape"])), int(max(jk_patches["shape"]))
			for n in np.arange(min_patch, max_patch + 1):
				self.num_samples[f"{n}"] = {}

			# Shape-position combinations:
			# S+D (Cg+, Gg+)
			# S+R (Cg+, Gg+)
			if corr_type == "g+" or corr_type == "both":
				# S+D
				self.data = self.data_dir
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches["shape"],
																			   corr_type=[corr_type, "w"], masks=masks,
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   cosmology=cosmology, count_pairs=False,
																			   num_sample_names=["S", "D"])
				# S+R
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"e1": self.data_dir["e1"],
					"e2": self.data_dir["e2"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				# print(self.data)
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["shape"],
					corr_type=[corr_type, "w"], masks=masks,
					dataset_name=f"{dataset_name}_randoms",
					num_nodes=self.num_nodes, over_h=over_h,
					cosmology=cosmology, count_pairs=False,
					num_sample_names=["S", "R_D"])

			# Position-position combinations:
			# SD (Cgg, Ggg)
			# SR (Cg+, Cgg, Ggg)
			# RD (Cgg, Ggg)
			# RR (Cgg, Gg+, Ggg)

			if corr_type == "gg":  # already have it for 'both'
				# SD (Cgg, Ggg)
				self.data = {
					"Redshift": self.data_dir["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.data_dir["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.data_dir["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"weight": self.data_dir["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches["shape"],
																			   corr_type=["gg", "w"],
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   cosmology=cosmology, count_pairs=True,
																			   data_suffix="_DD",
																			   num_sample_names=["S", "D"])

				# SR (Cg+, Cgg, Ggg) - watch name (Obs estimator) # if g+ or both, already have it
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["shape"],
					corr_type=["gg", "w"],
					dataset_name=dataset_name,
					num_nodes=self.num_nodes, over_h=over_h,
					cosmology=cosmology, count_pairs=True,
					data_suffix="_SR",
					num_sample_names=["S", "R_D"])

			if corr_type == "gg" or corr_type == "both":
				# RD (Cgg, Ggg)
				self.data = {
					"Redshift": self.data_dir["Redshift"],
					"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
					"RA": self.data_dir["RA"],
					"RA_shape_sample": self.randoms_data["RA_shape_sample"],
					"DEC": self.data_dir["DEC"],
					"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
					"weight": self.data_dir["weight"],
					"weight_shape_sample": self.randoms_data["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches[
																				   "randoms_shape"],
																			   corr_type=["gg", "w"],
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   cosmology=cosmology, count_pairs=True,
																			   data_suffix="_RD",
																			   num_sample_names=["R_S", "D"])

			if IA_estimator == "galaxies" or corr_type == "gg" or corr_type == "both":
				# RR (Cgg, Gg+, Ggg)
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.randoms_data["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.randoms_data["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["randoms_shape"],
					corr_type=["gg", "w"],
					dataset_name=dataset_name,
					num_nodes=self.num_nodes, over_h=over_h,
					cosmology=cosmology, count_pairs=True,
					data_suffix="_RR",
					num_sample_names=["R_S", "R_D"])

			self._measure_jackknife_covariance_lightcone(IA_estimator=IA_estimator, max_patch=max(jk_patches['shape']),
														 min_patch=min(jk_patches["shape"]), corr_type=[corr_type, "w"],
														 dataset_name=dataset_name, randoms_suf="_randoms")
		self.data = data
		return

	def measure_xi_multipoles(self, IA_estimator, dataset_name, corr_type, jk_patches=None,
							  num_jk=None, calc_errors=True, masks=None, masks_randoms=None, cosmology=None,
							  over_h=False, rp_cut=None):
		"""Measures multipoles including jackknife covariance if desired for lightcone data.
		Manages the various _measure_xi_r_mu_r_obs and _measure_jackknife_covariance options in
		MeasureMultipolesObservations and MeasureJackknife.

		Parameters
		----------
		IA_estimator : str
			Choose which type of xi estimator is used. Choose from "clusters" or "galaxies".
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		randoms_data : dict or NoneType
			Dictionary that includes the randoms data in the same form as the data dictionary.
		jk_patches : dict or NoneType, optional
			Dictionary with entries of the jackknife patch numbers (ndarray) for each sample, named "position", "shape"
			and "random". Default is None.
		num_jk : int, optional
			Number of jackknife patches to be generated internally. Default is None.
		measure_cov : bool, optional
			If True, jackknife errors are calculated. Default is True.
		masks : dict or NoneType, optional
			Dictionary of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		masks_randoms : dict or NoneType, optional
			Dictionary of mask information for the randoms data in the same form as the data dictionary,
			where the masks are placed over the data to apply selections. Default is None.
		cosmology : pyccl cosmology object or NoneType, optional
			Pyccl cosmology to use in the calculation. If None (default), the cosmology is used:
			ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
		over_h : bool, optional
			If True, the units are assumed to be in not-over-h and converted to over-h units. Default is False.
		rp_cut : float or NoneType, optional
			Applies a minimum r_p value condition for pairs to be included. Default is None.

		Returns
		-------

		"""
		if IA_estimator == "clusters":
			if self.randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				raise KeyError("This version does not work yet, add randoms.")
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if self.randoms_data == None:
				raise KeyError("No randoms given. Please provide input.")
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
				print("WARNING: this version of the code has not been fully validated. Proceed with caution.")
		else:
			raise KeyError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		# todo: Expand to include methods with trees and internal multiproc
		# todo: Checks to see if data directories include everything they need
		data = self.data  # temporary save so it can be restored at the end of the calculation

		try:  # Are there one or two random samples given?
			random_shape = self.randoms_data["RA_shape_sample"]
			one_random_sample = False
		except:
			one_random_sample = True
			self.randoms_data["RA_shape_sample"] = self.randoms_data["RA"]
			self.randoms_data["DEC_shape_sample"] = self.randoms_data["DEC"]
			self.randoms_data["Redshift_shape_sample"] = self.randoms_data["Redshift"]
		try:
			weight = self.randoms_data["weight"]
		except:
			self.randoms_data["weight"] = np.ones(len(self.randoms_data["RA"]))
		try:
			weight = self.randoms_data["weight_shape_sample"]
		except:
			if one_random_sample:
				self.randoms_data["weight_shape_sample"] = self.randoms_data["weight"]  # in case weights are given
			else:
				self.randoms_data["weight_shape_sample"] = np.ones(len(self.randoms_data["RA_shape_sample"]))

		if calc_errors:
			if jk_patches == None:
				if num_jk != None:
					jk_patches = self.assign_jackknife_patches(data, self.randoms_data, num_jk)
				else:
					raise ValueError("Set calc_errors to False, or provide either jk_patches or num_jk input.")
			else:
				if one_random_sample:
					jk_patches["randoms_position"] = jk_patches["randoms"]
					jk_patches["randoms_shape"] = jk_patches["randoms"]

		self.data_dir = data
		try:
			weight = self.data_dir["weight"]
		except:
			self.data_dir["weight"] = np.ones(len(self.data_dir["RA"]))
		try:
			weight = self.data_dir["weight_shape_sample"]
		except:
			self.data_dir["weight_shape_sample"] = np.ones(len(self.data_dir["RA_shape_sample"]))

		num_samples = {}  # Needed to correct for different number of randoms and galaxies/clusters in data
		if masks == None:
			num_samples["D"] = len(self.data_dir["RA"])
			num_samples["S"] = len(self.data_dir["RA_shape_sample"])
		else:
			num_samples["D"] = len(self.data_dir["RA"][masks["RA"]])
			num_samples["S"] = len(self.data_dir["RA_shape_sample"][masks["RA_shape_sample"]])
		if masks_randoms == None:
			num_samples["R_D"] = len(self.randoms_data["RA"])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"])
		else:
			num_samples["R_D"] = len(self.randoms_data["RA"][masks_randoms["RA"]])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"][masks_randoms["RA_shape_sample"]])
		# print(self.data_dir,self.randoms_data)

		# Shape-position combinations:
		# S+D (Cg+, Gg+)
		# S+R (Cg+, Gg+)
		if corr_type == "g+" or corr_type == "both":
			# S+D
			self.data = self.data_dir
			self._measure_xi_r_mur_lightcone_brute(masks=masks, dataset_name=dataset_name,
												   over_h=over_h, rp_cut=rp_cut,
												   cosmology=cosmology)
			# S+R
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"e1": self.data_dir["e1"],
				"e2": self.data_dir["e2"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			# print(self.data)
			self._measure_xi_r_mur_lightcone_brute(masks=masks,
												   dataset_name=f"{dataset_name}_randoms",
												   over_h=over_h, rp_cut=rp_cut,
												   cosmology=cosmology)

		# Position-position combinations:
		# SD (Cgg, Ggg)
		# SR (Cg+, Cgg, Ggg)
		# RD (Cgg, Ggg)
		# RR (Cgg, Gg+, Ggg)

		if corr_type == "gg":  # already have it for 'both'
			# SD (Cgg, Ggg)
			self.data = {
				"Redshift": self.data_dir["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.data_dir["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.data_dir["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"weight": self.data_dir["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			self._count_pairs_xi_r_mur_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology,
													   data_suffix="_DD", rp_cut=rp_cut)

			# SR (Cg+, Cgg, Ggg) - watch name (Obs estimator) # if g+ or both, already have it
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.data_dir["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.data_dir["weight_shape_sample"]
			}
			self._count_pairs_xi_r_mur_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology, rp_cut=rp_cut,
													   data_suffix="_SR")

		if corr_type == "gg" or corr_type == "both":
			# RD (Cgg, Ggg)
			self.data = {
				"Redshift": self.data_dir["Redshift"],
				"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
				"RA": self.data_dir["RA"],
				"RA_shape_sample": self.randoms_data["RA_shape_sample"],
				"DEC": self.data_dir["DEC"],
				"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
				"weight": self.data_dir["weight"],
				"weight_shape_sample": self.randoms_data["weight_shape_sample"]
			}
			self._count_pairs_xi_r_mur_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology, rp_cut=rp_cut,
													   data_suffix="_RD")

		if IA_estimator == "galaxies" or corr_type == "gg" or corr_type == "both":
			# RR (Cgg, Gg+, Ggg)
			self.data = {
				"Redshift": self.randoms_data["Redshift"],
				"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
				"RA": self.randoms_data["RA"],
				"RA_shape_sample": self.randoms_data["RA_shape_sample"],
				"DEC": self.randoms_data["DEC"],
				"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
				"weight": self.randoms_data["weight"],
				"weight_shape_sample": self.randoms_data["weight_shape_sample"]
			}
			self._count_pairs_xi_r_mur_lightcone_brute(masks=masks, dataset_name=dataset_name, over_h=over_h,
													   cosmology=cosmology, rp_cut=rp_cut,
													   data_suffix="_RR")

		self._obs_estimator([corr_type, "multipoles"], IA_estimator, dataset_name, f"{dataset_name}_randoms",
							num_samples)
		self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		if calc_errors:
			self.num_samples = {}
			min_patch, max_patch = int(min(jk_patches["shape"])), int(max(jk_patches["shape"]))
			for n in np.arange(min_patch, max_patch + 1):
				self.num_samples[f"{n}"] = {}

			# Shape-position combinations:
			# S+D (Cg+, Gg+)
			# S+R (Cg+, Gg+)
			if corr_type == "g+" or corr_type == "both":
				# S+D
				self.data = self.data_dir
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches["shape"],
																			   corr_type=[corr_type, "multipoles"],
																			   masks=masks,
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   rp_cut=rp_cut,
																			   cosmology=cosmology, count_pairs=False,
																			   num_sample_names=["S", "D"])
				# S+R
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"e1": self.data_dir["e1"],
					"e2": self.data_dir["e2"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["shape"],
					corr_type=[corr_type, "multipoles"],
					masks=masks,
					dataset_name=f"{dataset_name}_randoms",
					num_nodes=self.num_nodes, over_h=over_h,
					rp_cut=rp_cut,
					cosmology=cosmology, count_pairs=False,
					num_sample_names=["S", "R_D"])

			# Position-position combinations:
			# SD (Cgg, Ggg)
			# SR (Cg+, Cgg, Ggg)
			# RD (Cgg, Ggg)
			# RR (Cgg, Gg+, Ggg)

			if corr_type == "gg":  # already have it for 'both'
				# SD (Cgg, Ggg)
				self.data = {
					"Redshift": self.data_dir["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.data_dir["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.data_dir["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"weight": self.data_dir["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches["shape"],
																			   corr_type=["gg", "multipoles"],
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   cosmology=cosmology, count_pairs=True,
																			   rp_cut=rp_cut,
																			   data_suffix="_DD",
																			   num_sample_names=["S", "D"])

				# SR (Cg+, Cgg, Ggg) - watch name (Obs estimator) # if g+ or both, already have it
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.data_dir["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.data_dir["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.data_dir["DEC_shape_sample"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.data_dir["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["shape"],
					corr_type=["gg", "multipoles"],
					dataset_name=dataset_name,
					num_nodes=self.num_nodes, over_h=over_h,
					cosmology=cosmology, count_pairs=True,
					rp_cut=rp_cut,
					data_suffix="_SR",
					num_sample_names=["S", "R_D"])

			if corr_type == "gg" or corr_type == "both":
				# RD (Cgg, Ggg)
				self.data = {
					"Redshift": self.data_dir["Redshift"],
					"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
					"RA": self.data_dir["RA"],
					"RA_shape_sample": self.randoms_data["RA_shape_sample"],
					"DEC": self.data_dir["DEC"],
					"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
					"weight": self.data_dir["weight"],
					"weight_shape_sample": self.randoms_data["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(patches_pos=jk_patches["position"],
																			   patches_shape=jk_patches[
																				   "randoms_shape"],
																			   corr_type=["gg", "multipoles"],
																			   dataset_name=dataset_name,
																			   num_nodes=self.num_nodes, over_h=over_h,
																			   cosmology=cosmology, count_pairs=True,
																			   rp_cut=rp_cut,
																			   data_suffix="_RD",
																			   num_sample_names=["R_S", "D"])

			if IA_estimator == "galaxies" or corr_type == "gg" or corr_type == "both":
				# RR (Cgg, Gg+, Ggg)
				self.data = {
					"Redshift": self.randoms_data["Redshift"],
					"Redshift_shape_sample": self.randoms_data["Redshift_shape_sample"],
					"RA": self.randoms_data["RA"],
					"RA_shape_sample": self.randoms_data["RA_shape_sample"],
					"DEC": self.randoms_data["DEC"],
					"DEC_shape_sample": self.randoms_data["DEC_shape_sample"],
					"weight": self.randoms_data["weight"],
					"weight_shape_sample": self.randoms_data["weight_shape_sample"]
				}
				self._measure_jackknife_realisations_lightcone_multiprocessing(
					patches_pos=jk_patches["randoms_position"],
					patches_shape=jk_patches["randoms_shape"],
					corr_type=["gg", "multipoles"],
					dataset_name=dataset_name,
					num_nodes=self.num_nodes, over_h=over_h,
					cosmology=cosmology, count_pairs=True,
					rp_cut=rp_cut,
					data_suffix="_RR",
					num_sample_names=["R_S", "R_D"])

			self._measure_jackknife_covariance_lightcone(IA_estimator=IA_estimator, max_patch=max(jk_patches['shape']),
														 min_patch=min(jk_patches["shape"]),
														 corr_type=[corr_type, "multipoles"],
														 dataset_name=dataset_name, randoms_suf="_randoms")
		self.data = data
		return


if __name__ == "__main__":
	pass
