import numpy as np
from .measure_w_lightcone_jk import MeasureWLightconeJackknife
from .measure_w_lightcone import MeasureWLightcone
from .measure_m_lightcone_jk import MeasureMultipolesLightconeJackknife
from .measure_m_lightcone import MeasureMultipolesLightcone
from .measure_jackknife import MeasureJackknife


class MeasureIALightcone(MeasureWLightcone, MeasureMultipolesLightcone, MeasureWLightconeJackknife,
						 MeasureMultipolesLightconeJackknife, MeasureJackknife):
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

	def _merged_masks(self, masks_position, masks_shape):
		"""Combines the masks for a pair-count pass where the position and shape slots of self.data may hold
		different samples (data or randoms). Each slot's mask is taken from the mask dictionary of the sample
		that occupies it. Must be called after self.data has been set for the pass.

		Parameters
		----------
		masks_position : dict or NoneType
			Mask dictionary of the sample occupying the position slots ('Redshift', 'RA', 'DEC', 'weight').
		masks_shape : dict or NoneType
			Mask dictionary of the sample occupying the shape slots ('*_shape_sample', 'e1', 'e2').

		Returns
		-------
		dict or NoneType
			Combined mask dictionary, or None if no mask applies to either slot. A missing dictionary means no
			selection (all True); missing keys default to the slot's coordinate mask so all fields of one
			sample stay aligned.

		"""
		if masks_position is None and masks_shape is None:
			return None
		if masks_position is None:
			masks_position = {}
		if masks_shape is None:
			masks_shape = {}
		pos_default = np.ones(len(self.data["RA"]), dtype=bool)
		shape_default = np.ones(len(self.data["RA_shape_sample"]), dtype=bool)
		pos_mask = masks_position.get("RA", pos_default)
		shape_mask = masks_shape.get("RA_shape_sample", shape_default)
		merged = {}
		for key in ("Redshift", "RA", "DEC", "weight"):
			merged[key] = masks_position.get(key, pos_mask)
		for key in ("Redshift_shape_sample", "RA_shape_sample", "DEC_shape_sample", "weight_shape_sample",
					"e1", "e2"):
			merged[key] = masks_shape.get(key, shape_mask)
		return merged

	def measure_xi_helper(self, method_count_pairs, method_shape_correlation, IA_estimator, dataset_name, corr_type,
						  masks=None, masks_randoms=None, cosmology=None, over_h=False, chunk_size=1000, num_nodes=1,
						  temp_file_path=None):
		# Shape-position combinations:
		# S+D (Cg+, Gg+)
		# S+R (Cg+, Gg+)
		if corr_type == "g+" or corr_type == "both":
			# S+D
			self.data = self.data_dir
			method_shape_correlation(masks=self._merged_masks(masks, masks), dataset_name=dataset_name,
									 over_h=over_h, data_suffix="_SplusD",
									 cosmology=cosmology, chunk_size=chunk_size, num_nodes=num_nodes,
									 temp_file_path=temp_file_path)
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
			method_shape_correlation(masks=self._merged_masks(masks_randoms, masks), dataset_name=f"{dataset_name}",
									 over_h=over_h, data_suffix="_SplusR",
									 cosmology=cosmology, chunk_size=chunk_size, num_nodes=num_nodes,
									 temp_file_path=temp_file_path)

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
			method_count_pairs(masks=self._merged_masks(masks, masks), dataset_name=dataset_name, over_h=over_h,
							   cosmology=cosmology,
							   data_suffix="_DD", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(masks=self._merged_masks(masks_randoms, masks), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   data_suffix="_SR", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(masks=self._merged_masks(masks, masks_randoms), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   data_suffix="_RD", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(masks=self._merged_masks(masks_randoms, masks_randoms), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   data_suffix="_RR", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)
		return

	def measure_xi_jk_helper(self, method_count_pairs, method_shape_correlation, IA_estimator, dataset_name,
							 corr_type, jk_patches=None, masks=None, masks_randoms=None, cosmology=None, over_h=False,
							 chunk_size=1000, num_nodes=1, temp_file_path=None):
		num_jk = max(jk_patches["shape"]) - min(jk_patches["shape"]) + 1
		# Shape-position combinations:
		# S+D (Cg+, Gg+)
		# S+R (Cg+, Gg+)
		if corr_type == "g+" or corr_type == "both":
			# S+D
			self.data = self.data_dir
			method_shape_correlation(jackknife_region_indices_pos=jk_patches["position"],
									 jackknife_region_indices_shape=jk_patches["shape"],
									 masks=self._merged_masks(masks, masks),
									 dataset_name=dataset_name,
									 jk_group_name=f"{dataset_name}_jk{num_jk}",
									 over_h=over_h, data_suffix="_SplusD",
									 cosmology=cosmology, chunk_size=chunk_size, num_nodes=num_nodes,
									 temp_file_path=temp_file_path)
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
			method_shape_correlation(jackknife_region_indices_pos=jk_patches["randoms_position"],
									 jackknife_region_indices_shape=jk_patches["shape"],
									 masks=self._merged_masks(masks_randoms, masks),
									 dataset_name=f"{dataset_name}", data_suffix="_SplusR",
									 over_h=over_h, jk_group_name=f"{dataset_name}_jk{num_jk}",
									 cosmology=cosmology, chunk_size=chunk_size, num_nodes=num_nodes,
									 temp_file_path=temp_file_path)

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
			method_count_pairs(jackknife_region_indices_pos=jk_patches["position"],
							   jackknife_region_indices_shape=jk_patches["shape"],
							   masks=self._merged_masks(masks, masks), dataset_name=dataset_name, over_h=over_h,
							   cosmology=cosmology,
							   jk_group_name=f"{dataset_name}_jk{num_jk}",
							   data_suffix="_DD", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(jackknife_region_indices_pos=jk_patches["randoms_position"],
							   jackknife_region_indices_shape=jk_patches["shape"],
							   masks=self._merged_masks(masks_randoms, masks), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   jk_group_name=f"{dataset_name}_jk{num_jk}",
							   data_suffix="_SR", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(jackknife_region_indices_pos=jk_patches["position"],
							   jackknife_region_indices_shape=jk_patches["randoms_shape"],
							   masks=self._merged_masks(masks, masks_randoms), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   jk_group_name=f"{dataset_name}_jk{num_jk}",
							   data_suffix="_RD", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

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
			method_count_pairs(jackknife_region_indices_pos=jk_patches["randoms_position"],
							   jackknife_region_indices_shape=jk_patches["randoms_shape"],
							   masks=self._merged_masks(masks_randoms, masks_randoms), dataset_name=dataset_name,
							   over_h=over_h, cosmology=cosmology,
							   jk_group_name=f"{dataset_name}_jk{num_jk}",
							   data_suffix="_RR", chunk_size=chunk_size, num_nodes=num_nodes,
							   temp_file_path=temp_file_path)

		return

	def measure_xi_w(self, IA_estimator, dataset_name, corr_type, jk_patches=None, num_jk=None,
					 measure_cov=True, masks=None, masks_randoms=None, cosmology=None, over_h=False, tree=True,
					 chunk_size=1000, temp_file_path=None, seed=None, responsivity=False):
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
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
			Default is 1000.
		seed : int or NoneType, optional
			Seed for the internal jackknife patch generation (used only when num_jk is given), making the patch
			assignment reproducible. Default is None.

		"""
		if IA_estimator == "clusters":
			if self.randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				raise KeyError("This version does not work yet, add randoms.")
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR + 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if self.randoms_data == None:
				raise KeyError("No randoms given. Please provide input.")
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR + 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		else:
			raise KeyError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		self.responsivity_correction = responsivity
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
			if jk_patches is None:
				if num_jk is not None:
					jk_patches = self.assign_jackknife_patches(data, self.randoms_data, num_jk, seed=seed)
				else:
					raise ValueError("Set measure_cov to False, or provide either jk_patches or num_jk input.")
			else:
				if one_random_sample:
					jk_patches["randoms_position"] = jk_patches["randoms"]
					jk_patches["randoms_shape"] = jk_patches["randoms"]
			min_patch = min(jk_patches["shape"])
			max_patch = max(jk_patches["shape"])
			num_jk = max_patch - min_patch + 1

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
			# Stack RA and DEC into coordinate pairs
			coords_D = np.column_stack((self.data_dir["RA"], self.data_dir["DEC"]))
			coords_S = np.column_stack((self.data_dir["RA_shape_sample"], self.data_dir["DEC_shape_sample"]))


		else:
			coords_D = np.column_stack((self.data_dir["RA"][masks["RA"]], self.data_dir["DEC"][masks["DEC"]]))
			coords_S = np.column_stack((self.data_dir["RA_shape_sample"][masks["RA_shape_sample"]],
										self.data_dir["DEC_shape_sample"][masks["DEC_shape_sample"]]))
		# Use a structured view so np.intersect1d compares full pairs
		D_view = coords_D.view([('', coords_D.dtype)] * 2)
		S_view = coords_S.view([('', coords_S.dtype)] * 2)

		overlap, ind_D, ind_S = np.intersect1d(D_view, S_view, return_indices=True)

		num_samples["D"] = len(coords_D)
		num_samples["S"] = len(coords_S)
		num_samples["D_S"] = len(overlap)
		if masks_randoms == None:
			num_samples["R_D"] = len(self.randoms_data["RA"])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"])
		else:
			num_samples["R_D"] = len(self.randoms_data["RA"][masks_randoms["RA"]])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"][masks_randoms["RA_shape_sample"]])

		# ToDo: deal with masks

		if measure_cov:
			if self.num_nodes == 1:
				if tree:
					self.measure_xi_jk_helper(self._count_pairs_xi_rp_pi_lightcone_jk_tree,
											  self._measure_xi_rp_pi_lightcone_jk_tree, IA_estimator, dataset_name,
											  corr_type, jk_patches=jk_patches, masks=masks,
											  masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
											  chunk_size=chunk_size, num_nodes=self.num_nodes,
											  temp_file_path=temp_file_path)
				else:
					self.measure_xi_jk_helper(self._count_pairs_xi_rp_pi_lightcone_jk_brute,
											  self._measure_xi_rp_pi_lightcone_jk_brute, IA_estimator, dataset_name,
											  corr_type, jk_patches=jk_patches, masks=masks,
											  masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
											  chunk_size=chunk_size, num_nodes=self.num_nodes,
											  temp_file_path=temp_file_path)
			else:
				self.measure_xi_jk_helper(self._count_pairs_xi_rp_pi_lightcone_jk_multiprocessing,
										  self._measure_xi_rp_pi_lightcone_jk_multiprocessing, IA_estimator,
										  dataset_name, corr_type, jk_patches=jk_patches, masks=masks,
										  masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
										  chunk_size=chunk_size, num_nodes=self.num_nodes,
										  temp_file_path=temp_file_path)
			self._obs_estimator([corr_type, "w"], IA_estimator, dataset_name, num_samples)
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			print(num_samples)
			for i in np.arange(num_jk):
				overlap_i = np.where(jk_patches["position"][ind_D] == (i + min_patch))
				num_samples_i = {
					"S": num_samples["S"] - sum(jk_patches["shape"] == (i + min_patch)),
					"D": num_samples["D"] - sum(jk_patches["position"] == (i + min_patch)),
					"R_S": num_samples["R_S"] - sum(jk_patches["randoms_shape"] == (i + min_patch)),
					"R_D": num_samples["R_D"] - sum(jk_patches["randoms_position"] == (i + min_patch)),
					"D_S": num_samples["D_S"] - len(overlap_i)
				}
				self._obs_estimator([corr_type, "w"], IA_estimator, f"{dataset_name}_{i}",
									num_samples_i, jk_group_name=f"{dataset_name}_jk{num_jk}")

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
			# ToDo: change _combine_jackknife_information to deal with min_patch=1
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:
			if tree:
				self.measure_xi_helper(self._count_pairs_xi_rp_pi_lightcone_tree,
									   self._measure_xi_rp_pi_lightcone_tree,
									   IA_estimator, dataset_name, corr_type, masks=masks,
									   masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
									   chunk_size=chunk_size, num_nodes=self.num_nodes,
									   temp_file_path=temp_file_path)
			else:
				self.measure_xi_helper(self._count_pairs_xi_rp_pi_lightcone_brute,
									   self._measure_xi_rp_pi_lightcone_brute,
									   IA_estimator, dataset_name, corr_type, masks=masks,
									   masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
									   chunk_size=chunk_size, num_nodes=self.num_nodes,
									   temp_file_path=temp_file_path)
			self._obs_estimator([corr_type, "w"], IA_estimator, dataset_name, num_samples)
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		self.data = data
		return

	def measure_xi_multipoles(self, IA_estimator, dataset_name, corr_type, jk_patches=None, num_jk=None,
							  measure_cov=True, masks=None, masks_randoms=None, cosmology=None, over_h=False,
							  tree=True, chunk_size=1000, temp_file_path=None, seed=None, responsivity=False):
		"""Measures xi_gg, xi_g+ and multipoles including jackknife covariance if desired for lightcone data.
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
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
			Default is 1000.
		seed : int or NoneType, optional
			Seed for the internal jackknife patch generation (used only when num_jk is given), making the patch
			assignment reproducible. Default is None.

		"""
		if IA_estimator == "clusters":
			if self.randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				raise KeyError("This version does not work yet, add randoms.")
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR + 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if self.randoms_data == None:
				raise KeyError("No randoms given. Please provide input.")
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR + 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		else:
			raise KeyError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		self.responsivity_correction = responsivity
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
			if jk_patches is None:
				if num_jk is not None:
					jk_patches = self.assign_jackknife_patches(data, self.randoms_data, num_jk, seed=seed)
				else:
					raise ValueError("Set measure_cov to False, or provide either jk_patches or num_jk input.")
			else:
				if one_random_sample:
					jk_patches["randoms_position"] = jk_patches["randoms"]
					jk_patches["randoms_shape"] = jk_patches["randoms"]
			min_patch = min(jk_patches["shape"])
			max_patch = max(jk_patches["shape"])
			num_jk = max_patch - min_patch + 1

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
			# Stack RA and DEC into coordinate pairs
			coords_D = np.column_stack((self.data_dir["RA"], self.data_dir["DEC"]))
			coords_S = np.column_stack((self.data_dir["RA_shape_sample"], self.data_dir["DEC_shape_sample"]))


		else:
			coords_D = np.column_stack((self.data_dir["RA"][masks["RA"]], self.data_dir["DEC"][masks["DEC"]]))
			coords_S = np.column_stack((self.data_dir["RA_shape_sample"][masks["RA_shape_sample"]],
										self.data_dir["DEC_shape_sample"][masks["DEC_shape_sample"]]))
		# Use a structured view so np.intersect1d compares full pairs
		D_view = coords_D.view([('', coords_D.dtype)] * 2)
		S_view = coords_S.view([('', coords_S.dtype)] * 2)

		overlap, ind_D, ind_S = np.intersect1d(D_view, S_view, return_indices=True)

		num_samples["D"] = len(coords_D)
		num_samples["S"] = len(coords_S)
		num_samples["D_S"] = len(overlap)
		if masks_randoms == None:
			num_samples["R_D"] = len(self.randoms_data["RA"])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"])
		else:
			num_samples["R_D"] = len(self.randoms_data["RA"][masks_randoms["RA"]])
			num_samples["R_S"] = len(self.randoms_data["RA_shape_sample"][masks_randoms["RA_shape_sample"]])

		# ToDo: deal with masks

		if measure_cov:
			if tree:
				self.measure_xi_jk_helper(self._count_pairs_xi_r_mur_lightcone_jk_tree,
										  self._measure_xi_r_mur_lightcone_jk_tree, IA_estimator, dataset_name,
										  corr_type, jk_patches=jk_patches, masks=masks,
										  masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
										  chunk_size=chunk_size, num_nodes=self.num_nodes,
										  temp_file_path=temp_file_path)
			else:
				self.measure_xi_jk_helper(self._count_pairs_xi_r_mur_lightcone_jk_brute,
										  self._measure_xi_r_mur_lightcone_jk_brute, IA_estimator, dataset_name,
										  corr_type, jk_patches=jk_patches, masks=masks,
										  masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h,
										  chunk_size=chunk_size, num_nodes=self.num_nodes,
										  temp_file_path=temp_file_path)
			self._obs_estimator([corr_type, "multipoles"], IA_estimator, dataset_name, num_samples)
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			for i in np.arange(num_jk):
				overlap_i = np.where(jk_patches["position"][ind_D] == (i + min_patch))
				num_samples_i = {
					"S": num_samples["S"] - sum(jk_patches["shape"] == (i + min_patch)),
					"D": num_samples["D"] - sum(jk_patches["position"] == (i + min_patch)),
					"R_S": num_samples["R_S"] - sum(jk_patches["randoms_shape"] == (i + min_patch)),
					"R_D": num_samples["R_D"] - sum(jk_patches["randoms_position"] == (i + min_patch)),
					"D_S": num_samples["D_S"] - len(overlap_i)
				}
				self._obs_estimator([corr_type, "multipoles"], IA_estimator, f"{dataset_name}_{i}",
									num_samples_i, jk_group_name=f"{dataset_name}_jk{num_jk}")

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
			# ToDo: change _combine_jackknife_information to deal with min_patch=1
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:
			if tree:
				self.measure_xi_helper(self._count_pairs_xi_r_mur_lightcone_tree,
									   self._measure_xi_r_mur_lightcone_tree,
									   IA_estimator, dataset_name, corr_type, masks=masks,
									   temp_file_path=temp_file_path,
									   chunk_size=chunk_size, num_nodes=self.num_nodes,
									   masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h)
			else:
				self.measure_xi_helper(self._count_pairs_xi_r_mur_lightcone_brute,
									   self._measure_xi_r_mur_lightcone_brute,
									   IA_estimator, dataset_name, corr_type, masks=masks,
									   temp_file_path=temp_file_path,
									   chunk_size=chunk_size, num_nodes=self.num_nodes,
									   masks_randoms=masks_randoms, cosmology=cosmology, over_h=over_h)
			self._obs_estimator([corr_type, "multipoles"], IA_estimator, dataset_name, num_samples)
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		self.data = data
		return


if __name__ == "__main__":
	pass
