import math
import numpy as np
import h5py
from kmeans_radec import kmeans_sample
from scipy.special import lpmn
from .write_data import write_dataset_hdf5, create_group_hdf5
from .Sim_info import SimInfo


class MeasureIABase(SimInfo):
	"""Base class for MeasureIA package that includes some general methods used throughout the package.

	Attributes
	----------
	Num_position : int
		Number of objects in the position sample. This value is updated in jackknife realisations.
	Num_shape : int
		Number of objects in the shape sample. This value is updated in jackknife realisations.
	r_min : float
		Minimum bound of (projected) separation length; bin edge. Default is 0.1.
	r_max : float
		Maximum bound of (projected) separation length; bin edge. Default is 20.
	r_bins : ndarray
		Bin edges of the (projected) separation length (r_p or r).
	pi_bins : ndarray
		Bin edges of the line of sight (pi).
	mu_r_bins : ndarray
		Bin edges of the mu_r.

	Methods
	-------
	calculate_dot_product_arrays()
		Calculates dot product of elements of two arrays
	get_ellipticity()
		Given e and phi, e_+ and e_x components of ellipticity are returned.
	get_random_pairs()
		Analytical RR for a (rp,pi) bin.
	get_volume_spherical_cap()
		Volume of an (r,mu_r) bin.
	get_random_pairs_r_mur()
		Analytical RR for a (r,mu_r) bin.
	setdiff2D()
		Compares each row of a1 and a2 and returns the elements that do not overlap.
	setdiff_omit()
		For rows in nested list a1, whose index is included in incl_ind, returns elements that do not overlap between
		the row in a1 and a2.
	_measure_w_g_i()
		Measure wgg or wg+ from xi grid provided by MeasureWBox or MeasureWLightcone class methods.
	_measure_multipoles()
		Measure multipoles (gg or g+) from xi grid provided by MeasureMultipolesBox or MeasureMultipolesLightcone
		class methods.
	_obs_estimator()
		Combines elements (DD, RR, etc) of xi estimators into xi_gg or xi_g+ for MeasureIALightcone.
	assign_jackknife_patches()
		Given positions of multiple samples, defines jackknife patches and returns index of every object in the sample.

	Notes
	-----
	Inherits attributes from 'SimInfo', where 'boxsize', 'L_0p5' and 'snap_group' are used in this class.

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
		The __init__ method of the MeasureIABase class.

		Parameters
		----------
		data : dict or NoneType
			Dictionary with data needed for calculations.
			For cartesian coordinates, the keywords are:
			'Position' and 'Position_shape_sample': (N_p,3), (N_s,3) ndarrays with the x, y, z coordinates
			of the N_p, N_s objects in the position and shape samples, respectively.
			'Axis_Direction': (N_s,2) ndarray with the two elements of the unit vectors describing the
			axis direction of the projected axis of the object shape.
			'LOS': index referring back to the column number in the 'Position' samples that contains the
			line-of-sight coordinate. (e.g. if the shapes are projected over the z-axis, LOS=2)
			'q': (N_s) array containing the axis ratio q=b/a for each object in the shape sample.
			For lightcone coordinates, the keywords are:
			'Redshift' and 'Redshift_shape_sample': (N_p) and (N_s) ndarray with redshifts of position and shape samples.
			'RA' and 'RA_shape_sample': (N_p) and (N_s) ndarray with RA coordinate of position and shape samples.
			'DEC' and 'DEC_shape_sample': (N_p) and (N_s) ndarray with DEC coordinate of position and shape samples.
			'e1' and 'e2': (N_s) arrays with the two ellipticity components e1 and e2 of the shape sample objects.
		output_file_name : str
			Name and filepath of the file where the output should be stored. Needs to be hdf5-type.
		simulation : str or NoneType, optional
			Indicator of simulation, obtaining correct boxsize in cMpc/h automatically. 
			Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8].
			Default is None, in which case boxsize needs to be added manually; or in the case of observational data, 
			the pi_max.
		snapshot : int or str or NoneType, optional
			Number of the snapshot, which, if given, will ensure that the output file to contains a group
			'Snapshot_[snapshot]'. If None, the group is omitted from the output file structure. Default is None.
		separation_limits : iterable of 2 entries, optional
			Bounds of the (projected) separation vector length bins in cMpc/h (so, r or r_p). Default is [0.1,20].
		num_bins_r : int, optional
			Number of bins for (projected) separation vector. Default is 8.
		num_bins_pi : int, optional
			Number of bins for line of sight (LOS) vector, pi or mu_r when multipoles are measured. Default is 20.
		pi_max : int or float, optional
			Bound for line of sight bins. Bounds will be [-pi_max, pi_max]. Default is None, in which case half the
			boxsize will be used.
		boxsize : int or float or NoneType, optional
			If simulation is not included in SimInfo, a manual boxsize can be added here. Make sure simulation=None
			and the boxsize units are equal to those in the data dictionary. Default is None.
		periodicity : bool, optional
			If True, the periodic boundary conditions of the simulation box are taken into account. If False, they are
			ignored. Note that because this code used analytical randoms for the simulations, the correlations will not
			be correct in this case and only DD and S+D terms should be studied. Non-periodic randoms can be measured by
			providing random data to the code and considering the DD term that is measured. Correlations and covariance
			matrix will need to be reconstructed from parts. [Please add a request for teh integration of this method of
			this if you would like to use this option often.] Default is True.
		
		"""
		SimInfo.__init__(self, simulation, snapshot, boxsize)
		self.data = data
		self.output_file_name = output_file_name
		self.periodicity = periodicity
		if periodicity:
			periodic = "periodic "
		else:
			periodic = ""
		try:
			self.Num_position = len(data["Position"])  # number of halos in position sample
			self.Num_shape = len(data["Position_shape_sample"])  # number of halos in shape sample
		except:
			try:
				self.Num_position = len(data["RA"])
				self.Num_shape = len(data["RA_shape_sample"])
			except:
				self.Num_position = 0
				self.Num_shape = 0
				print("Warning: no Postion or Position_shape_sample given.")
		if self.Num_position > 0:
			try:
				weight = self.data["weight"]
			except:
				self.data["weight"] = np.ones(self.Num_position)
			try:
				weight = self.data["weight_shape_sample"]
			except:
				self.data["weight_shape_sample"] = np.ones(self.Num_shape)
		self.r_min = separation_limits[0]  # cMpc/h
		self.r_max = separation_limits[1]  # cMpc/h
		self.num_bins_r = num_bins_r
		self.num_bins_pi = num_bins_pi
		self.r_bins = np.logspace(np.log10(self.r_min), np.log10(self.r_max), self.num_bins_r + 1)
		if pi_max == None:
			if self.L_0p5 is None:
				raise ValueError(
					"Both pi_max and boxsize are None. Provide input on one of them to determine the integration limit pi_max.")
			else:
				pi_max = self.L_0p5
		self.pi_bins = np.linspace(-pi_max, pi_max, self.num_bins_pi + 1)
		self.mu_r_bins = np.linspace(-1, 1, self.num_bins_pi + 1)
		if simulation == False:
			print(f"MeasureIA object initialised with:\n \
					observational data.\n \
					There are {self.Num_shape} galaxies in the shape sample and {self.Num_position} galaxies in the position sample.\n\
					The separation bin edges are given by {self.r_bins} Mpc.\n \
					There are {num_bins_r} r or r_p bins and {num_bins_pi} pi bins.\n \
					The maximum pi used for binning is {pi_max}.\n \
					The data will be written to {self.output_file_name}")
		else:
			print(f"MeasureIA object initialised with:\n \
			simulation {simulation} that has a {periodic}boxsize of {self.boxsize} cMpc/h.\n \
			There are {self.Num_shape} galaxies in the shape sample and {self.Num_position} galaxies in the position sample.\n\
			The separation bin edges are given by {self.r_bins} cMpc/h.\n \
			There are {num_bins_r} r or r_p bins and {num_bins_pi} pi bins.\n \
			The maximum pi used for binning is {pi_max}.\n \
			The data will be written to {self.output_file_name}")
		return

	@staticmethod
	def calculate_dot_product_arrays(a1, a2):
		"""Calculates the dot product over 2 2D arrays across axis 1 so that dot_product[i] = np.dot(a1[i],a2[i])

		Parameters
		----------
		a1 : ndarray
			First array
		a2 : ndarray
			Second array

		Returns
		-------
		ndarray
			Dot product of columns of arrays

		"""
		dot_product = np.zeros(np.shape(a1)[0])
		for i in np.arange(0, np.shape(a1)[1]):
			dot_product += a1[:, i] * a2[:, i]
		return dot_product

	@staticmethod
	def get_ellipticity(e, phi):
		"""Calculates the radial and tangential components of the ellipticity, given the size of the ellipticty vector
		and the angle between the semimajor or semiminor axis and the separation vector.

		Parameters
		----------
		e : ndarray
			size of the ellipticity vector
		phi : ndarray
			angle between semimajor/semiminor axis and separation vector

		Returns
		-------
		ndarray
			e_+ and e_x

		"""
		e_plus, e_cross = e * np.cos(2 * phi), e * np.sin(2 * phi)
		return e_plus, e_cross

	@staticmethod
	def get_random_pairs(rp_max, rp_min, pi_max, pi_min, L3, corrtype, Num_position, Num_shape):
		"""Returns analytical value of the number of pairs expected in an r_p, pi bin for a random uniform distribution.
		(Singh et al. 2023)

		Parameters
		----------
		rp_max : float
			Upper bound of projected separation vector bin
		rp_min : float
			Lower bound of projected separation vector bin
		pi_max : float
			Upper bound of line of sight vector bin
		pi_min : float
			Lower bound of line of sight vector bin
		L3 : float or int
			Volume of the simulation box
		corrtype : str
			Correlation type, auto or cross. RR for auto is RR_cross/2.
		Num_position : int
			Number of objects in the position sample.
		Num_shape : int
			Number of objects in the shape sample.


		Returns
		-------
		float
			number of pairs in r_p, pi bin

		"""
		if corrtype == "auto":
			RR = (
					(Num_position - 1.0) * Num_shape / 2.0
					* np.pi
					* (rp_max ** 2 - rp_min ** 2)
					* abs(pi_max - pi_min)
					/ L3
			)  # volume is cylindrical pi*dr^2 * height
		elif corrtype == "cross":
			RR = Num_position * Num_shape * np.pi * (rp_max ** 2 - rp_min ** 2) * abs(pi_max - pi_min) / L3
		else:
			raise ValueError("Unknown input for corrtype, choose from auto or cross.")
		return RR

	def get_random_pairs_r_mur(self, r_max, r_min, mur_max, mur_min, L3, corrtype, Num_position, Num_shape):
		"""Returns analytical value of the number of pairs expected in an r, mu_r bin for a random uniform distribution.

		Parameters
		----------
		r_max : float
			Upper bound of separation vector bin
		r_min : float
			Lower bound of separation vector bin
		mur_max : float
			Upper bound of mu_r bin
		mur_min : float
			Lower bound of mu_r bin
		L3 : float
			Volume of the simulation box
		corrtype : str
			Correlation type, auto or cross. RR for auto is RR_cross/2.
		Num_position : int
			Number of objects in the position sample.
		Num_shape : int
			Number of objects in the shape sample.


		Returns
		-------
		float
			number of pairs in r, mu_r bin

		"""

		if corrtype == "auto":
			RR = (
					(Num_position - 1.0)
					/ 2.0
					* Num_shape
					* 2. * np.pi / 3. * (r_max ** 3 - r_min ** 3) * (mur_max - mur_min)
					/ L3
			)
		# volume is big cap - small cap for large - small radius
		elif corrtype == "cross":
			RR = (
					(Num_position - 1.0)
					* Num_shape
					* 2. * np.pi / 3. * (r_max ** 3 - r_min ** 3) * (mur_max - mur_min)
					/ L3
			)
		else:
			raise ValueError("Unknown input for corrtype, choose from auto or cross.")
		return abs(RR)

	@staticmethod
	def setdiff2D(a1, a2):
		"""Compares each row of a1 and a2 and returns the elements that do not overlap

		Parameters
		----------
		a1 : nested list
			List containing lists of elements to compare to a2
		a2 : nested list
			List containing lists of elements to compare to a1

		Returns
		-------
		nested list
			For each row, the not-overlapping elements between a1 and a2
		"""
		assert len(a1) == len(a2), "Lengths of lists where each row is to be compared, are not the same."
		diff = []
		for i in np.arange(0, len(a1)):
			setdiff = np.setdiff1d(a1[i], a2[i])
			diff.append(setdiff)
			del setdiff
		return diff

	@staticmethod
	def setdiff_omit(a1, a2, incl_ind):
		"""For rows in nested list a1, whose index is included in incl_ind, returns elements that do not overlap between
		the row in a1 and a2.

		Parameters
		----------
		a1 : nested list
			List of lists or arrays where indicated rows need to be compared to a2
		a2 : list or array
			Elements to be compared to the row in a1 [and not included in return values].
		incl_ind : list or array
			Indices of rows in a1 to be compared to a2.

		Returns
		-------
		nested list
			For each included row in a1, the not-overlapping elements between a1 and a2
		"""
		diff = []
		for i in np.arange(0, len(a1)):
			if np.isin(i, incl_ind):
				setdiff = np.setdiff1d(a1[i], a2)
				diff.append(setdiff)
				del setdiff
		return diff

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
			group_multipoles = data_file[f"{self.snap_group}/{corr_group[d]}/{jk_group_name}/"]
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
				group_multipoles = create_group_hdf5(output_file, f"{self.snap_group}/" + corr_group[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_mean_" + str(num_box), data=mean_multipoles)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
		if return_output:
			return covs, stds
		else:
			return

	def _measure_w_g_i(self, dataset_name, corr_type="both", return_output=False, jk_group_name=""):
		"""Measures w_gg or w_g+ for a given xi_gi dataset that has been calculated with the _measure_xi_rp_pi_sims
		methods. Integrates over pi bins via sum * dpi. Stores rp, and w_gg or w_g+.

		Parameters
		----------
		dataset_name : str
			Name of xi_gg or xi_g+ dataset and name given to w_gg or w_g+ dataset when stored.
		return_output : bool, optional
			Output is returned if True, saved to file if False. Default value = False
		corr_type : str, optional
			Type of correlation function. Choose from [g+,gg,both]. Default value = "both"
		jk_group_name : str, optional
			Name of subgroup in hdf5 file where jackknife realisations are stored. Default value = ""

		Returns
		-------
		ndarray
			[rp, wgg] or [rp, wg+] if return_output is True

		"""
		if corr_type == "both":
			xi_data = ["xi_g_plus", "xi_gg"]
			wg_data = ["w_g_plus", "w_gg"]
		elif corr_type == "g+":
			xi_data = ["xi_g_plus"]
			wg_data = ["w_g_plus"]
		elif corr_type == "gg":
			xi_data = ["xi_gg"]
			wg_data = ["w_gg"]
		else:
			raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
		for i in np.arange(0, len(xi_data)):
			correlation_data_file = h5py.File(self.output_file_name, "a")
			group = correlation_data_file[f"{self.snap_group}/w/{xi_data[i]}/{jk_group_name}"]
			correlation_data = group[dataset_name][:]
			pi = group[dataset_name + "_pi"]
			rp = group[dataset_name + "_rp"]
			dpi = (self.pi_bins[1:] - self.pi_bins[:-1])
			pi_bins = self.pi_bins[:-1] + abs(dpi) / 2.0  # middle of bins
			# variance = group[dataset_name + "_sigmasq"][:]
			if sum(np.isin(pi, pi_bins)) == len(pi):
				dpi = np.array([dpi] * len(correlation_data[:, 0]))
				correlation_data = correlation_data * abs(dpi)
			# sigsq_el = variance * dpi ** 2
			else:
				raise ValueError("Update pi bins in initialisation of object to match xi_g_plus dataset.")
			w_g_i = np.sum(correlation_data, axis=1)  # sum over pi values
			# sigsq = np.sum(sigsq_el, axis=1)
			if return_output:
				output_data = np.array([rp, w_g_i]).transpose()
				correlation_data_file.close()
				return output_data
			else:
				group_out = create_group_hdf5(correlation_data_file,
											  f"{self.snap_group}/{wg_data[i]}/{jk_group_name}")
				write_dataset_hdf5(group_out, dataset_name + "_rp", data=rp)
				write_dataset_hdf5(group_out, dataset_name, data=w_g_i)
				# write_dataset_hdf5(group_out, dataset_name + "_sigma", data=np.sqrt(sigsq))
				correlation_data_file.close()
		return

	def _measure_multipoles(self, dataset_name, corr_type="both", return_output=False, jk_group_name=""):
		"""Measures multipoles for a given xi_g+ or xi_gg measured by _measure_xi_r_pi_sims methods.
		The data assumes xi_g+ and xi_gg to be measured in bins of r and mu_r.

		Parameters
		----------
		dataset_name : str
			Name of xi_gg or xi_g+ dataset and name given to multipoles dataset when stored.
		corr_type : str, optional
			Type of correlation function. Choose from [g+,gg,both]. Default value = "both"
		return_output : bool, optional
			Output is returned if True, saved to file if False. Default value = False.
		jk_group_name : str, optional
			Name of subgroup in hdf5 file where jackknife realisations are stored. Default value = ""

		Returns
		-------
		ndarray
			[r, multipoles_gg] or [r, multipoles_g+] if return_output is True
		"""
		correlation_data_file = h5py.File(self.output_file_name, "a")
		if corr_type == "g+":  # todo: expand to include ++ option
			group = correlation_data_file[f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of r,mur
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [2]
			l_list = sab_list
			corr_type_list = ["g_plus"]
		elif corr_type == "gg":
			group = correlation_data_file[f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [0]
			l_list = sab_list
			corr_type_list = ["gg"]
		elif corr_type == "both":
			group = correlation_data_file[f"{self.snap_group}/multipoles/xi_g_plus/{jk_group_name}"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			group = correlation_data_file[f"{self.snap_group}/multipoles/xi_gg/{jk_group_name}"]
			correlation_data_list.append(group[dataset_name][:])  # xi_g+ in grid of rp,pi
			r_list.append(group[dataset_name + "_r"][:])
			mu_r_list.append(group[dataset_name + "_mu_r"][:])
			sab_list = [2, 0]
			l_list = sab_list
			corr_type_list = ["g_plus", "gg"]
		else:
			raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
		for i in np.arange(0, len(sab_list)):
			corr_type_i = corr_type_list[i]
			correlation_data = correlation_data_list[i]
			r = r_list[i]
			mu_r = mu_r_list[i]
			sab = sab_list[i]
			l = l_list[i]
			L = np.zeros((len(r), len(mu_r)))
			mu_r = np.array(list(mu_r) * len(r)).reshape((len(r), len(mu_r)))  # make pi into grid for mu

			r = np.array(list(r) * len(mu_r)).reshape((len(r), len(mu_r)))
			r = r.transpose()
			for n in np.arange(0, len(mu_r[:, 0])):
				for m in np.arange(0, len(mu_r[0])):
					L_m, dL = lpmn(l, sab, mu_r[n, m])  # make associated Legendre polynomial grid
					L[n, m] = L_m[-1, -1]  # grid ranges from 0 to sab and 0 to l, so last element is what we seek
			dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1])
			dmu_r_array = np.array(list(dmur) * len(r)).reshape((len(r), len(dmur)))
			multipoles = (
					(2 * l + 1)
					/ 2.0
					* math.factorial(l - sab)
					/ math.factorial(l + sab)
					* L
					* correlation_data
					* dmu_r_array
			)
			multipoles = np.sum(multipoles, axis=1)
			dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
			separation = self.r_bins[:-1] + abs(dsep)  # middle of bins
			if return_output:
				correlation_data_file.close()
				np.array([separation, multipoles]).transpose()
			else:
				group_out = create_group_hdf5(
					correlation_data_file, f"{self.snap_group}/multipoles_{corr_type_i}/{jk_group_name}"
				)
				write_dataset_hdf5(group_out, dataset_name + "_r", data=separation)
				write_dataset_hdf5(group_out, dataset_name, data=multipoles)
		correlation_data_file.close()
		return

	def _obs_estimator(self, corr_type, IA_estimator, dataset_name, dataset_name_randoms, num_samples,
					   jk_group_name="", jk_group_name_randoms=""):
		"""Reads various components of xi and combines into correct estimator for cluster or galaxy
		lightcone alignment correlations. It then writes the xi_gg or xi_g+ in the correct place in the output file.

		Parameters
		----------
		corr_type : list of 2 str elements
			First element: ['gg', 'g+', 'both'], second: 'w' or 'multipoles'
		IA_estimator : str
			Chooser from 'clusters' or 'galaxies' for different estimator definition.
		dataset_name : str
			Name of the dataset
		dataset_name_randoms : str
			Name of the dataset for data with randoms as positions
		num_samples : dict
			Dictionary of samples sizes for position, shape and random samples. Keywords: D, S, R_D, R_S
		jk_group_name : str
			Name of subgroup in hdf5 file where jackknife realisations are stored. Default value = ""

		Returns
		-------

		"""
		output_file = h5py.File(self.output_file_name, "a")
		if corr_type[0] == "g+" or corr_type[0] == "both":
			group_gp = output_file[
				f"{self.snap_group}/{corr_type[1]}/xi_g_plus/{jk_group_name}"]
			group_gp_r = output_file[
				f"{self.snap_group}/{corr_type[1]}/xi_g_plus/{jk_group_name_randoms}"]
			SpD = group_gp[f"{dataset_name}_SplusD"][:]
			SpD /= (num_samples["S"] * num_samples["D"])
			SpR = group_gp_r[f"{dataset_name_randoms}_SplusD"][:]
			SpR /= (num_samples["S"] * num_samples["R_D"])
		group_gg = output_file[f"{self.snap_group}/{corr_type[1]}/xi_gg/{jk_group_name}"]
		group_gg_r = output_file[f"{self.snap_group}/{corr_type[1]}/xi_gg/{jk_group_name_randoms}"]
		DD = group_gg[f"{dataset_name}_DD"][:]
		DD /= (num_samples["D"] * num_samples["S"])

		if IA_estimator == "clusters":
			if corr_type[0] == "gg":
				SR = group_gg[f"{dataset_name}_SR"][:]
				SR /= (num_samples["S"] * num_samples["R_D"])
			else:
				SR = group_gg_r[f"{dataset_name_randoms}_DD"][:]
				SR /= (num_samples["S"] * num_samples["R_D"])
			if corr_type[0] == "g+" or corr_type[0] == "both":
				correlation_gp = SpD / DD - SpR / SR
				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
			if corr_type[0] == "gg" or corr_type[0] == "both":
				RD = group_gg[f"{dataset_name}_RD"][:]
				RD /= (num_samples["D"] * num_samples["R_S"])
				RR = group_gg[f"{dataset_name}_RR"][:]
				RR /= (num_samples["R_D"] * num_samples["R_S"])
				correlation_gg = (DD - RD - SR) / RR - 1
				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		elif IA_estimator == "galaxies":
			RR = group_gg[f"{dataset_name}_RR"][:]
			RR /= (num_samples["R_D"] * num_samples["R_S"])
			if corr_type[0] == "g+" or corr_type[0] == "both":
				correlation_gp = (SpD - SpR) / RR
				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
			if corr_type[0] == "gg" or corr_type[0] == "both":
				RD = group_gg[f"{dataset_name}_RD"][:]
				RD /= (num_samples["D"] * num_samples["R_S"])
				if corr_type[0] == "gg":
					SR = group_gg[f"{dataset_name}_SR"][:]
					SR /= (num_samples["S"] * num_samples["R_D"])
				else:
					SR = group_gg_r[f"{dataset_name_randoms}_DD"][:]
					SR /= (num_samples["S"] * num_samples["R_D"])
				correlation_gg = (DD - RD - SR) / RR + 1
				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		else:
			raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")
		# output_file = h5py.File(self.output_file_name, "a")
		# 		group_gg = output_file[f"{self.snap_group}/{corr_type[1]}/xi_gg/{jk_group_name}"]
		# 		if corr_type[0] == "g+" or corr_type[0] == "both":
		# 			group_gp = output_file[
		# 				f"{self.snap_group}/{corr_type[1]}/xi_g_plus/{jk_group_name}"]
		# 			SpD = group_gp[f"{dataset_name}_SplusD"][:]
		# 			SpD /= (num_samples["S"] * num_samples["D"] - num_samples["D_S"])
		# 			SpR = group_gp[f"{dataset_name}_SplusR"][:]
		# 			SpR /= (num_samples["S"] * num_samples["R_D"])
		# 		if corr_type[0] == "gg" or corr_type[0] == "both":
		# 			SR = group_gg[f"{dataset_name}_SR"][:]
		# 			SR /= (num_samples["S"] * num_samples["R_D"])
		# 			RD = group_gg[f"{dataset_name}_RD"][:]
		# 			RD /= (num_samples["D"] * num_samples["R_S"])
		# 		if IA_estimator == 'clusters' or corr_type[0] == "gg" or corr_type[0] == "both":
		# 			DD = group_gg[f"{dataset_name}_DD"][:]
		# 			DD[DD == 0] = 1.
		# 			DD /= (num_samples["D"] * num_samples["S"] - num_samples["D_S"])
		# 		if IA_estimator == "galaxies" or corr_type[0] == "gg" or corr_type[0] == "both":
		# 			RR = group_gg[f"{dataset_name}_RR"][:]
		# 			RR /= (num_samples["R_D"] * num_samples["R_S"])
		#
		# 		if IA_estimator == "clusters":
		# 			if corr_type[0] == "g+" or corr_type[0] == "both":
		# 				correlation_gp = SpD / DD - SpR / SR
		# 				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
		# 			if corr_type[0] == "gg" or corr_type[0] == "both":
		# 				correlation_gg = (DD - RD - SR) / RR + 1
		# 				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		# 		elif IA_estimator == "galaxies":
		# 			if corr_type[0] == "g+" or corr_type[0] == "both":
		# 				correlation_gp = (SpD - SpR) / RR
		# 				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
		# 			if corr_type[0] == "gg" or corr_type[0] == "both":
		# 				correlation_gg = (DD - RD - SR) / RR + 1
		# 				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		# 		else:
		# 			raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")
		output_file.close()
		return

	def assign_jackknife_patches(self, data, randoms_data, num_jk):
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
		km = kmeans_sample(X, num_jk, maxiter=100, tol=1.0e-5)
		jk_labels = km.labels

		jk_patches['randoms_position'] = jk_labels

		RA = randoms_data['RA_shape_sample']
		DEC = randoms_data['DEC_shape_sample']
		X2 = np.column_stack((RA, DEC))
		jk_labels = km.find_nearest(X2)

		jk_patches['randoms_shape'] = jk_labels

		RA = data['RA']
		DEC = data['DEC']
		X2 = np.column_stack((RA, DEC))
		jk_labels = km.find_nearest(X2)

		jk_patches['position'] = jk_labels

		RA = data['RA_shape_sample']
		DEC = data['DEC_shape_sample']
		X2 = np.column_stack((RA, DEC))
		jk_labels = km.find_nearest(X2)

		jk_patches['shape'] = jk_labels

		return jk_patches


if __name__ == "__main__":
	pass
