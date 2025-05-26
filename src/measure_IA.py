import sympy
import numpy as np
from src.measure_jackknife import MeasureJackknife

KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureIA(MeasureJackknife):
	"""
	Manages the methods used in the MeasureIA class based on speed and input.
	:param data: Dictionary with data needed for calculations. See specifications for keywords.
	:param simulation: Indicator of simulation. Choose from [TNG100, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1_m8, FLAMINGO_L1_m9, FLAMINGO_L1_m10, FLAMINGO_L2p8_m9] for now.
	:param snapshot: Number of the snapshot
	:param separation_limits: Bounds of the (projected) separation vector length bins in cMpc/h (so, r or r_p)
	:param num_bins_r: Number of bins for (projected) separation vector.
	:param num_bins_pi: Number of bins for line of sight (LOS) vector, pi.
	:param PT: Number indicating particle type
	:param LOS_lim: Bound for line of sight bins. Bounds will be [-LOS_lim, LOS_lim]
	:param output_file_name: Name and filepath of the file where the output should be stored.
	:param boxsize: Specify the boxsize of the simulation if using a simulation that is not in SimInfo
	:param periodicity: Set to True (default) to include periodic boundary conditions. False to ignore. Note that the RR
	terms are calculated analytically for the simulations, so periodicity=False only works for the S+D and DD terms.
	:param num_nodes: Number of cores available for multiprocessing. (Influences which method is used)
	"""

	def __init__(
			self,
			data,
			simulation=None,
			snapshot=99,
			separation_limits=[0.1, 20.0],
			num_bins_r=8,
			num_bins_pi=20,
			PT=4,
			LOS_lim=None,
			output_file_name=None,
			boxsize=None,
			periodicity=True,
			num_nodes=1,
	):
		super().__init__(data, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi, PT,
						 LOS_lim, output_file_name, boxsize, periodicity)
		self.num_nodes = num_nodes

		return

	def measure_xi_w(self, dataset_name, corr_type, num_jk=0, calc_errors=True, file_tree_path=None, masks=None,
					 remove_tree_file=True):
		"""
		Manages the various measure_projected_correlation options in MeasureIABase.
		:param dataset_name: Name of the dataset in the output file.
		:param corr_type: Type of correlation to be measured. Choose from [g+, gg, both].
		:param num_jk: Number of jackknife regions (needs to be x^3, with x an int) for the error calculation.
		:param calc_errors: If True, jackknife errors are calculated.
		:param file_tree_path: Path to where the tree information is temporarily stored. If None (default), no trees
		are used in the calculation. Note that the use of trees speeds up the calculations significantly.
		:param masks: Directory of mask information in the same form as the data input, where the masks are placed over
		the data to apply selections.
		:param remove_tree_file: If True (default), the file that stores the tree information is removed after the
		calculations.
		:return:
		"""
		if calc_errors:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
			except AssertionError:
				print(
					f"Use x^3 as input for num_jk, with x as an int. {float(int(num_jk ** (1. / 3)))},{num_jk ** (1. / 3)}")
				exit()
			if self.num_nodes == 1:
				multiproc_bool = False
				save_tree = True
			elif num_jk > 0.5 * self.num_nodes:
				multiproc_bool = True
				save_tree = True
			else:
				multiproc_bool = True
				save_tree = False
		else:
			if self.num_nodes == 1:
				multiproc_bool = False
				save_tree = False
			else:
				multiproc_bool = True
				save_tree = True
		if save_tree and file_tree_path == None:
			raise ValueError(
				"Input file_tree_path for faster computation. Do not want to use trees? Input file_path_tree=False.")
		elif save_tree and file_tree_path == False:
			save_tree = False
			file_tree_path = None
		try:
			RA = self.data["RA"]
			sim_bool = False
		except:
			sim_bool = True
		if not sim_bool:
			print("Given data is observational, use measure_xi_w_obs method instead.")
		else:
			if multiproc_bool and save_tree:
				self.measure_projected_correlation_tree(tree_input=None, masks=masks, dataset_name=dataset_name,
														return_output=False, print_num=True, dataset_name_tree=None,
														save_tree=save_tree, file_tree_path=file_tree_path)
				self.measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors_multiprocessing(masks=masks, corr_type=[corr_type, "w"],
																  dataset_name=dataset_name, L_subboxes=L, rp_cut=None,
																  num_nodes=self.num_nodes, twoD=False, tree=True,
																  tree_saved=True, file_tree_path=file_tree_path,
																  remove_tree_file=remove_tree_file)
			elif not multiproc_bool and save_tree:
				self.measure_projected_correlation_tree(tree_input=None, masks=masks, dataset_name=dataset_name,
														return_output=False, print_num=True, dataset_name_tree=None,
														save_tree=save_tree, file_tree_path=file_tree_path)
				self.measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors(masks=masks, corr_type=[corr_type, "w"],
												  dataset_name=dataset_name, L_subboxes=L, rp_cut=None,
												  tree_saved=True, file_tree_path=file_tree_path,
												  remove_tree_file=remove_tree_file)
			else:
				self.measure_projected_correlation_multiprocessing(num_nodes=self.num_nodes, masks=masks,
																   dataset_name=dataset_name, return_output=False,
																   print_num=True)
				self.measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors_multiprocessing(masks=masks, corr_type=[corr_type, "w"],
																  dataset_name=dataset_name, L_subboxes=L, rp_cut=None,
																  num_nodes=self.num_nodes, twoD=False, tree=False,
																  tree_saved=False)

		return

	def measure_xi_multipoles(self, dataset_name, corr_type, num_jk, calc_errors=True, file_tree_path=None, masks=None,
							  remove_tree_file=True, rp_cut=None):
		"""
		Manages the various measure_projected_correlation options in MeasureIABase.
		:param dataset_name: Name of the dataset in the output file.
		:param corr_type: Type of correlation to be measured. Choose from [g+, gg, both].
		:param num_jk: Number of jackknife regions (needs to be x^3, with x an int) for the error calculation.
		:param calc_errors: If True, jackknife errors are calculated.
		:param file_tree_path: Path to where the tree information is temporarily stored. If None (default), no trees
		are used in the calculation. Note that the use of trees speeds up the calculations significantly.
		:param masks: Directory of mask information in the same form as the data input, where the masks are placed over
		the data to apply selections.
		:param remove_tree_file: If True (default), the file that stores the tree information is removed after the
		calculations.
		:param rp_cut: Applies a minimum r_p value condition for pairs to be included. Default is None.
		:return:
		"""
		if calc_errors:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
			except AssertionError:
				print("Use x^3 as input for num_jk, with x as an int.")
				exit()
			if self.num_nodes == 1:
				multiproc_bool = False
				save_tree = True
			elif num_jk > 0.5 * self.num_nodes:
				multiproc_bool = True
				save_tree = True
			else:
				multiproc_bool = True
				save_tree = False
		else:
			if self.num_nodes == 1:
				multiproc_bool = False
				save_tree = False
			else:
				multiproc_bool = True
				save_tree = True
		if save_tree and file_tree_path == None:
			raise ValueError(
				"Input file_tree_path for faster computation. Do not want to use trees? Input file_path_tree=False.")
		elif save_tree and file_tree_path == False:
			save_tree = False
			file_tree_path = None
		try:
			RA = self.data["RA"]
			sim_bool = False
		except:
			sim_bool = True
		if not sim_bool:
			print("Assuming observational data.")
			self.measure_xi_multipoles_obs()
		else:
			if multiproc_bool and save_tree:
				self.measure_projected_correlation_multipoles_tree(tree_input=None, masks=masks,
																   dataset_name=dataset_name,
																   return_output=False, print_num=True,
																   dataset_name_tree=None, rp_cut=rp_cut,
																   save_tree=save_tree, file_tree_path=file_tree_path)
				self.measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors_multiprocessing(masks=masks,
																  corr_type=[corr_type, "multipoles"],
																  dataset_name=dataset_name, L_subboxes=L,
																  rp_cut=rp_cut,
																  num_nodes=self.num_nodes, twoD=False, tree=True,
																  tree_saved=True, file_tree_path=file_tree_path,
																  remove_tree_file=remove_tree_file)
			elif not multiproc_bool and save_tree:
				self.measure_projected_correlation_multipoles_tree(tree_input=None, masks=masks,
																   dataset_name=dataset_name,
																   return_output=False, print_num=True,
																   dataset_name_tree=None, rp_cut=rp_cut,
																   save_tree=save_tree, file_tree_path=file_tree_path)
				self.measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors(masks=masks, corr_type=[corr_type, "multipoles"],
												  dataset_name=dataset_name, L_subboxes=L, rp_cut=rp_cut,
												  tree_saved=True, file_tree_path=file_tree_path,
												  remove_tree_file=remove_tree_file)
			else:
				self.measure_projected_correlation_multipoles_multiprocessing(num_nodes=self.num_nodes,
																			  masks=masks,
																			  dataset_name=dataset_name,
																			  return_output=False, rp_cut=rp_cut,
																			  print_num=True)
				self.measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
				if calc_errors:
					self.measure_jackknife_errors_multiprocessing(masks=masks,
																  corr_type=[corr_type, "multipoles"],
																  dataset_name=dataset_name, L_subboxes=L,
																  rp_cut=rp_cut,
																  num_nodes=self.num_nodes, twoD=False, tree=False,
																  tree_saved=False)

		return

	def measure_xi_w_obs(self, IA_estimator, dataset_name, corr_type, jk_patches=None, randoms_data=None,
						 calc_errors=True,
						 masks=None, masks_randoms=None, cosmology=None, over_h=False):
		"""
		Manages the measurement of observational wg+ in MeasureIABase.
		:param IA_estimator: Choose which type of xi estimator is used. Choose "clusters" or "galaxies".
		:param dataset_name: Name of the dataset in the output file.
		:param corr_type: ype of correlation to be measured. Choose from [g+, gg, both].
		:param jk_patches: Directory with entries of the jackknife patches for each sample, named "position", "shape"
		and "random".
		:param randoms_data: Data directory that includes the randoms information in the same form as the data input.
		:param calc_errors: If True, jackknife errors are calculated.
		:param masks: Directory of mask information in the same form as the data input, where the masks are placed over
		the data to apply selections.
		:param masks_randoms: Directory of mask information for the randoms data in the same form as the data input,
		where the masks are placed over the data to apply selections.
		:param cosmology: pyccl cosmology to use in the calculation. If None (default), a default cosmology is used.
		:param over_h: If True, the units are assumed to be in not-over-h and converted to over-h units. Default is False.
		:return:
		"""
		if IA_estimator == "clusters":
			if randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				print("This version does not work yet, add randoms.")
				exit()
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if randoms_data == None:
				print("No randoms given. Please provide input.")
				exit()
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		else:
			raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		if calc_errors and jk_patches == None:
			raise ValueError("No jackknife patches are given, but calc_errors is set to True.")

		# todo: Expand to include methods with trees and internal multiproc
		# todo: Checks to see if data directories include everything they need
		data = self.data  # temporary save so it can be restored at the end of the calculation

		self.randoms_data = randoms_data
		try:  # Are there one or two random samples given?
			random_shape = randoms_data["RA_shape_sample"]
			one_random_sample = False
		except:
			one_random_sample = True
			self.randoms_data["RA_shape_sample"] = randoms_data["RA"]
			self.randoms_data["DEC_shape_sample"] = randoms_data["DEC"]
			self.randoms_data["Redshift_shape_sample"] = randoms_data["Redshift"]
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

		dataset_names = [dataset_name, f"{dataset_name}_randoms"]

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
			self.measure_projected_correlation_obs_clusters(masks=masks, dataset_name=dataset_name,
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
			self.measure_projected_correlation_obs_clusters(masks=masks, dataset_name=f"{dataset_name}_randoms",
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
			self.count_pairs_xi_grid_w(masks=masks, dataset_name=dataset_name, over_h=over_h, cosmology=cosmology,
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
			self.count_pairs_xi_grid_w(masks=masks, dataset_name=dataset_name, over_h=over_h, cosmology=cosmology,
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
			self.count_pairs_xi_grid_w(masks=masks, dataset_name=dataset_name, over_h=over_h, cosmology=cosmology,
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
			self.count_pairs_xi_grid_w(masks=masks, dataset_name=dataset_name, over_h=over_h, cosmology=cosmology,
									   data_suffix="_RR")

		self.obs_estimator([corr_type, "w"], IA_estimator, dataset_name, f"{dataset_name}_randoms", num_samples)
		self.measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms_position"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
																		patches_shape=jk_patches["shape"],
																		corr_type=["gg", "w"],
																		dataset_name=dataset_name,
																		num_nodes=self.num_nodes, over_h=over_h,
																		cosmology=cosmology, count_pairs=True,
																		data_suffix="_DD", num_sample_names=["S", "D"])

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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms_position"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
																		patches_shape=jk_patches["randoms_shape"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms_position"],
																		patches_shape=jk_patches["randoms_shape"],
																		corr_type=["gg", "w"],
																		dataset_name=dataset_name,
																		num_nodes=self.num_nodes, over_h=over_h,
																		cosmology=cosmology, count_pairs=True,
																		data_suffix="_RR",
																		num_sample_names=["R_S", "R_D"])

			self.measure_jackknife_errors_obs(IA_estimator=IA_estimator, max_patch=max(jk_patches['shape']),
											  min_patch=min(jk_patches["shape"]), corr_type=[corr_type, "w"],
											  dataset_name=dataset_name, randoms_suf="_randoms")
		self.data = data
		return

	def measure_xi_multipoles_obs(self, IA_estimator, dataset_name, corr_type, jk_patches=None, randoms_data=None,
								  calc_errors=True, rp_cut=None,
								  masks=None, masks_randoms=None, cosmology=None, over_h=False):
		"""
		Manages the measurement of observational wg+ in MeasureIABase.
		:param IA_estimator: Choose which type of xi estimator is used. Choose "clusters" or "galaxies".
		:param dataset_name: Name of the dataset in the output file.
		:param corr_type: ype of correlation to be measured. Choose from [g+, gg, both].
		:param jk_patches: Directory with entries of the jackknife patches for each sample, named "position", "shape"
		and "random".
		:param randoms_data: Data directory that includes the randoms information in the same form as the data input.
		:param calc_errors: If True, jackknife errors are calculated.
		:param masks: Directory of mask information in the same form as the data input, where the masks are placed over
		the data to apply selections.
		:param masks_randoms: Directory of mask information for the randoms data in the same form as the data input,
		where the masks are placed over the data to apply selections.
		:param cosmology: pyccl cosmology to use in the calculation. If None (default), a default cosmology is used.
		:param over_h: If True, the units are assumed to be in not-over-h and converted to over-h units. Default is False.
		:return:
		"""
		if IA_estimator == "clusters":
			if randoms_data == None:
				print("No randoms given, correlation defined as S+D/DD")
				print("This version does not work yet, add randoms.")
				exit()
			else:
				print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		elif IA_estimator == "galaxies":
			if randoms_data == None:
				print("No randoms given. Please provide input.")
				exit()
			else:
				print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - RD - SR)/RR - 1")
				if masks != None and masks_randoms == None:
					print("Warning, masks given for data vector but not for randoms.")
		else:
			raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")

		if calc_errors and jk_patches == None:
			raise ValueError("No jackknife patches are given, but calc_errors is set to True.")

		# todo: Expand to include methods with trees and internal multiproc
		# todo: Checks to see if data directories include everything they need
		data = self.data  # temporary save so it can be restored at the end of the calculation

		self.randoms_data = randoms_data
		try:  # Are there one or two random samples given?
			random_shape = randoms_data["RA_shape_sample"]
			one_random_sample = False
		except:
			one_random_sample = True
			self.randoms_data["RA_shape_sample"] = randoms_data["RA"]
			self.randoms_data["DEC_shape_sample"] = randoms_data["DEC"]
			self.randoms_data["Redshift_shape_sample"] = randoms_data["Redshift"]
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

		self.data_dir = data
		try:
			weight = self.data_dir["weight"]
		except:
			self.data_dir["weight"] = np.ones(len(self.data_dir["RA"]))
		try:
			weight = self.data_dir["weight_shape_sample"]
		except:
			self.data_dir["weight_shape_sample"] = np.ones(len(self.data_dir["RA_shape_sample"]))

		dataset_names = [dataset_name, f"{dataset_name}_randoms"]

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
			self.measure_projected_correlation_multipoles_obs_clusters(masks=masks, dataset_name=dataset_name,
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
			self.measure_projected_correlation_multipoles_obs_clusters(masks=masks,
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
			self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
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
			self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
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
			self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
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
			self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
												cosmology=cosmology, rp_cut=rp_cut,
												data_suffix="_RR")

		self.obs_estimator([corr_type, "multipoles"], IA_estimator, dataset_name, f"{dataset_name}_randoms",
						   num_samples)
		self.measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
																		patches_shape=jk_patches["shape"],
																		corr_type=["gg", "multipoles"],
																		dataset_name=dataset_name,
																		num_nodes=self.num_nodes, over_h=over_h,
																		cosmology=cosmology, count_pairs=True,
																		rp_cut=rp_cut,
																		data_suffix="_DD", num_sample_names=["S", "D"])

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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
																		patches_shape=jk_patches["randoms"],
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
				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms"],
																		patches_shape=jk_patches["randoms"],
																		corr_type=["gg", "multipoles"],
																		dataset_name=dataset_name,
																		num_nodes=self.num_nodes, over_h=over_h,
																		cosmology=cosmology, count_pairs=True,
																		rp_cut=rp_cut,
																		data_suffix="_RR",
																		num_sample_names=["R_S", "R_D"])

			self.measure_jackknife_errors_obs(IA_estimator=IA_estimator, max_patch=max(jk_patches['shape']),
											  min_patch=min(jk_patches["shape"]), corr_type=[corr_type, "multipoles"],
											  dataset_name=dataset_name, randoms_suf="_randoms")
		self.data = data
		return

# def measure_xi_multipoles_obs(self, IA_estimator, dataset_name, corr_type, jk_patches=None, randoms_data=None,
# 							  calc_errors=True, rp_cut=None,
# 							  masks=None, masks_randoms=None, cosmology=None, over_h=False):
# 	"""
# 	Manages the measurement of observational multipoles in MeasureIABase.
# 	:param IA_estimator: Choose which type of xi estimator is used. Choose "clusters" or "galaxies".
# 	:param dataset_name: Name of the dataset in the output file.
# 	:param corr_type: ype of correlation to be measured. Choose from [g+, gg, both].
# 	:param jk_patches: Directory with entries of the jackknife patches for each sample, named "position", "shape"
# 	and "random".
# 	:param randoms_data: Data directory that includes the randoms information in the same form as the data input.
# 	:param calc_errors: If True, jackknife errors are calculated.
# 	:param rp_cut: Applies a minimum r_p value condition for pairs to be included. Default is None.
# 	:param masks: Directory of mask information in the same form as the data input, where the masks are placed over
# 	the data to apply selections.
# 	:param masks_randoms: Directory of mask information for the randoms data in the same form as the data input,
# 	where the masks are placed over the data to apply selections.
# 	:param cosmology: pyccl cosmology to use in the calculation. If None (default), a default cosmology is used.
# 	:param over_h: If True, the units are assumed to be in not-over-h and converted to over-h units. Default is False.
# 	:return:
# 	"""
# 	if IA_estimator == "clusters":
# 		if randoms_data == None:
# 			print("No randoms given, correlation defined as S+D/DD")
# 		else:
# 			print("xi_g+ defined as S+D/SD - S+R/SR, xi_gg as (DS - DR - SR)/RR - 1")
# 			if masks != None and masks_randoms == None:
# 				print("Warning, masks given for data vector but not for randoms.")
# 	elif IA_estimator == "galaxies":
# 		if randoms_data == None:
# 			print("No randoms given. Please provide input.")
# 			exit()
# 		else:
# 			print("xi_g+ defined as (S+D - S+R)/RR, xi_gg as (SD - DR - SR)/RR - 1")
# 			if masks != None and masks_randoms == None:
# 				print("Warning, masks given for data vector but not for randoms.")
# 	else:
# 		raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")
#
# 	if calc_errors and jk_patches == None:
# 		raise ValueError("No jackknife patches are given, but calc_errors is set to True.")
#
# 	# expand to include methods with trees and internal multiproc
# 	data = self.data  # temporary save so it can be restored at the end of the calculation
# 	self.randoms_data = randoms_data
# 	self.randoms_data["RA_shape_sample"] = data["RA_shape_sample"]
# 	self.randoms_data["DEC_shape_sample"] = data["DEC_shape_sample"]
# 	self.randoms_data["Redshift_shape_sample"] = data["Redshift_shape_sample"]
# 	self.randoms_data["e1"] = data["e1"]
# 	self.randoms_data["e2"] = data["e2"]
# 	try:
# 		weight = self.randoms_data["weight_shape_sample"]
# 	except:
# 		self.randoms_data["weight_shape_sample"] = np.ones(len(self.randoms_data["RA_shape_sample"]))
# 	self.data_dir = self.data
# 	dataset_names = [dataset_name, f"{dataset_name}_randoms"]
# 	jk_names = ["position", "randoms"]
#
# 	# more elaborate to include other types of estimators. Compute all elements, then overwrite the correlation with the correct combination
# 	for i, self.data in enumerate([self.data_dir, self.randoms_data]):
# 		try:
# 			weight = self.data["weight"]
# 		except:
# 			self.data["weight"] = np.ones(len(self.data["RA"]))
# 		try:
# 			weight = self.data["weight_shape_sample"]
# 		except:
# 			self.data["weight_shape_sample"] = np.ones(len(self.data["RA_shape_sample"]))
# 		self.measure_projected_correlation_multipoles_obs_clusters(masks=masks, dataset_name=dataset_names[i],
# 																   over_h=over_h, rp_cut=rp_cut,
# 																   cosmology=cosmology)
# 	if corr_type == "both" or corr_type == "gg":
# 		# 	get DR
# 		self.data = {
# 			"Redshift": self.data_dir["Redshift"],
# 			"Redshift_shape_sample": self.randoms_data["Redshift"],
# 			"RA": self.data_dir["RA"],
# 			"RA_shape_sample": self.randoms_data["RA"],
# 			"DEC": self.data_dir["DEC"],
# 			"DEC_shape_sample": self.randoms_data["DEC"],
# 		}
# 		self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
# 											cosmology=cosmology,
# 											data_suffix="_DR", rp_cut=rp_cut)
# 	if IA_estimator == "galaxies" or corr_type == "both" or corr_type == "gg":
# 		# 	get RR
# 		self.data = {
# 			"Redshift": self.randoms_data["Redshift"],
# 			"Redshift_shape_sample": self.randoms_data["Redshift"],
# 			"RA": self.randoms_data["RA"],
# 			"RA_shape_sample": self.randoms_data["RA"],
# 			"DEC": self.randoms_data["DEC"],
# 			"DEC_shape_sample": self.randoms_data["DEC"],
# 		}
# 		self.count_pairs_xi_grid_multipoles(masks=masks, dataset_name=dataset_name, over_h=over_h,
# 											cosmology=cosmology,
# 											data_suffix="_RR", rp_cut=rp_cut)
#
# 	self.obs_estimator([corr_type, "multipoles"], IA_estimator, dataset_name, f"{dataset_name}_randoms")
# 	self.measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
# 	if calc_errors:
# 		for i, self.data in enumerate([self.data_dir, self.randoms_data]):
# 			if self.num_nodes == 1:
# 				self.measure_jackknife_realisations_obs(patches_pos=jk_patches[jk_names[i]],
# 														patches_shape=jk_patches["shape"],
# 														corr_type=[corr_type, "multipoles"], rp_cut=rp_cut,
# 														dataset_name=dataset_names[i], over_h=over_h,
# 														cosmology=cosmology)
# 			else:
# 				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches[jk_names[i]],
# 																		patches_shape=jk_patches["shape"],
# 																		corr_type=[corr_type, "multipoles"],
# 																		dataset_name=dataset_names[i],
# 																		num_nodes=self.num_nodes, over_h=over_h,
# 																		rp_cut=rp_cut,
# 																		cosmology=cosmology)
#
# 		if corr_type == "both" or corr_type == "gg":
# 			# 	get DR
# 			self.data = {
# 				"Redshift": self.data_dir["Redshift"],
# 				"Redshift_shape_sample": self.randoms_data["Redshift"],
# 				"RA": self.data_dir["RA"],
# 				"RA_shape_sample": self.randoms_data["RA"],
# 				"DEC": self.data_dir["DEC"],
# 				"DEC_shape_sample": self.randoms_data["DEC"],
# 			}
# 			if self.num_nodes == 1:
# 				self.measure_jackknife_realisations_obs(patches_pos=jk_patches["position"],
# 														patches_shape=jk_patches["randoms"],
# 														corr_type=["gg", "multipoles"],
# 														dataset_name=dataset_name, over_h=over_h, rp_cut=rp_cut,
# 														cosmology=cosmology, count_pairs=True, data_suffix="_DR")
# 			else:
# 				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["position"],
# 																		patches_shape=jk_patches["randoms"],
# 																		corr_type=["gg", "multipoles"],
# 																		dataset_name=dataset_name,
# 																		num_nodes=self.num_nodes, over_h=over_h,
# 																		rp_cut=rp_cut,
# 																		cosmology=cosmology, count_pairs=True,
# 																		data_suffix="_DR")
# 		if IA_estimator == "galaxies" or corr_type == "both" or corr_type == "gg":
# 			# 	get RR
# 			self.data = {
# 				"Redshift": self.randoms_data["Redshift"],
# 				"Redshift_shape_sample": self.randoms_data["Redshift"],
# 				"RA": self.randoms_data["RA"],
# 				"RA_shape_sample": self.randoms_data["RA"],
# 				"DEC": self.randoms_data["DEC"],
# 				"DEC_shape_sample": self.randoms_data["DEC"],
# 			}
# 			if self.num_nodes == 1:
# 				self.measure_jackknife_realisations_obs(patches_pos=jk_patches["randoms"],
# 														patches_shape=jk_patches["randoms"],
# 														corr_type=["gg", "multipoles"],
# 														dataset_name=dataset_name, over_h=over_h, rp_cut=rp_cut,
# 														cosmology=cosmology, count_pairs=True, data_suffix="_RR")
# 			else:
# 				self.measure_jackknife_realisations_obs_multiprocessing(patches_pos=jk_patches["randoms"],
# 																		patches_shape=jk_patches["randoms"],
# 																		corr_type=["gg", "multipoles"],
# 																		dataset_name=dataset_name,
# 																		num_nodes=self.num_nodes, over_h=over_h,
# 																		rp_cut=rp_cut,
# 																		cosmology=cosmology, count_pairs=True,
# 																		data_suffix="_RR")
# 		# rewrite method to be adaptable to all types of estimators
# 		self.measure_jackknife_errors_obs(IA_estimator=IA_estimator, max_patch=max(jk_patches['shape']),
# 										  min_patch=min(jk_patches["shape"]),
# 										  corr_type=[corr_type, "multipoles"],
# 										  dataset_name=dataset_name, randoms_suf="_randoms")
# 	self.data = data
#
# 	return
