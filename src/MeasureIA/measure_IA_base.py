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
		"""Calculates the dot product over 2 2D arrays across axis 1 so that
		dot_product[i] = np.dot(a1[i],a2[i])

		Parameters
		----------
		a1 :
			First array
		a2 :
			Second array

		Returns
		-------
		type
			Dot product of columns of arrays

		"""
		dot_product = np.zeros(np.shape(a1)[0])
		for i in np.arange(0, np.shape(a1)[1]):
			dot_product += a1[:, i] * a2[:, i]
		return dot_product

	def measure_3D_orientation_separation_correlation(self, masks=None, dataset_name="All_galaxies"):
		"""NEEDS MORE EXTENSIVE TESTS
		Measures the 3D orientation-separation correlation function for given positions and minor axis directions.

		Parameters
		----------
		masks :
			Directory of masks for the data that makes a selection in the data. (Default value = None)
		dataset_name :
			Name of the dataset in the hdf5 file specified in output file name. (Default value = "All_galaxies")

		Returns
		-------
		type
			correlation, separation bin means (log) if output file name not specified.

		"""
		print("WARNING: this method has not been tested and is likely not correct.")
		exit()
		if masks == None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
		n_pairs = [0] * self.num_bins_r
		inner_product = [0] * self.num_bins_r
		for n in np.arange(0, len(positions)):
			separation = positions_shape_sample - positions[n]
			separation[separation > self.L_0p5] -= self.boxsize
			separation[separation < -self.L_0p5] += self.boxsize
			separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
			separation_dir = (separation.transpose() / separation_len).transpose()
			inner_product_n = self.calculate_dot_product_arrays(separation_dir, axis_direction) ** 2
			for i in np.arange(0, self.num_bins_r):
				lower_limit_mask = separation_len > self.r_bins[i]
				upper_limit_mask = separation_len < self.r_bins[i + 1]
				mask = lower_limit_mask * upper_limit_mask
				n_pairs[i] += sum(mask)
				inner_product[i] += sum(inner_product_n[mask])
		correlation = np.array(inner_product) / np.array(n_pairs) - 1.0 / 3
		dsep = (self.r_bins[:-1] - self.r_bins[1:]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/3D_correlations")
			write_dataset_hdf5(group, dataset_name, data=np.array([separation_bins, correlation]).transpose())
			output_file.close()
			return
		else:
			return correlation, separation_bins

	@staticmethod
	def get_ellipticity(e, phi):
		"""Calculates the radial and tangential components of the ellipticity, given the size of the ellipticty vector
		and the angle between the semimajor or semiminor axis and the separation vector.

		Parameters
		----------
		e :
			size of the ellipticity vector
		phi :
			angle between semimajor/semiminor axis and separation vector

		Returns
		-------
		type
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
		rp_max :
			upper bound of projected separation vector bin
		rp_min :
			lower bound of projected separation vector bin
		pi_max :
			upper bound of line of sight vector bin
		pi_min :
			lower bound of line of sight vector bin
		L3 :
			volume of the simulation box
		corrtype :
			Correlation type, auto or cross. RR for auto is RR_cross/2.
		Num_position :

		Num_shape :


		Returns
		-------
		type
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

	@staticmethod
	def get_volume_spherical_cap(mur, r):
		"""Calculate the volume of a spherical cap.

		Parameters
		----------
		mur :
			cos(theta), where theta is the polar angle between the apex and disk of the cap.
		r :
			radius

		Returns
		-------
		type
			Volume of the spherical cap.

		"""
		return np.pi / 3.0 * r ** 3 * (2 + mur) * (1 - mur) ** 2

	def get_random_pairs_r_mur(self, r_max, r_min, mur_max, mur_min, L3, corrtype, Num_position, Num_shape):
		"""Retruns analytical value of the number of pairs expected in an r_p, pi bin for a random uniform distribution.
		(Singh et al. 2023)

		Parameters
		----------
		r_max :
			upper bound of projected separation vector bin
		r_min :
			lower bound of projected separation vector bin
		mur_max :
			upper bound of mu_r bin
		mur_min :
			lower bound of mu_r bin
		L3 :
			volume of the simulation box
		corrtype :
			Correlation type, auto or cross. RR for auto is RR_cross/2.
		Num_position :

		Num_shape :


		Returns
		-------
		type
			number of pairs in r, mu_r bin

		"""

		if corrtype == "auto":
			RR = (
					(Num_position - 1.0)
					/ 2.0
					* Num_shape
					* (
							self.get_volume_spherical_cap(mur_min, r_max)
							- self.get_volume_spherical_cap(mur_max, r_max)
							- (self.get_volume_spherical_cap(mur_min, r_min) - self.get_volume_spherical_cap(mur_max,
																											 r_min))
					)
					/ L3
			)
		# volume is big cap - small cap for large - small radius
		elif corrtype == "cross":
			RR = (
					(Num_position - 1.0)
					* Num_shape
					* (
							self.get_volume_spherical_cap(mur_min, r_max)
							- self.get_volume_spherical_cap(mur_max, r_max)
							- (self.get_volume_spherical_cap(mur_min, r_min) - self.get_volume_spherical_cap(mur_max,
																											 r_min))
					)
					/ L3
			)
		else:
			raise ValueError("Unknown input for corrtype, choose from auto or cross.")
		return abs(RR)

	@staticmethod
	def setdiff2D(a1, a2):
		"""

		Parameters
		----------
		a1 :

		a2 :


		Returns
		-------

		"""
		diff = []
		for i in np.arange(0, len(a1)):
			setdiff = np.setdiff1d(a1[i], a2[i])
			diff.append(setdiff)
			del setdiff
		return diff

	@staticmethod
	def setdiff_omit(a1, a2, incl_ind):
		"""

		Parameters
		----------
		a1 :

		a2 :

		incl_ind :


		Returns
		-------

		"""
		diff = []
		for i in np.arange(0, len(a1)):
			if np.isin(i, incl_ind):
				setdiff = np.setdiff1d(a1[i], a2)
				diff.append(setdiff)
				del setdiff
		return diff



	def measure_projected_correlation_save_pairs(self, output_file_pairs="", masks=None, dataset_name="All_galaxies",
												 print_num=True):
		"""Measures the projected correlation function (xi_g_plus, xi_gg) for given coordinates of the position and shape sample
		(Position, Position_shape_sample), the projected axis direction (Axis_Direction), the ratio between projected
		axes, q=b/a (q) and the index of the direction of the line of sight (LOS=2 for z axis).
		Positions are assumed to be given in cMpc/h.

		Parameters
		----------
		masks :
			the masks for the data to select only part of the data (Default value = None)
		dataset_name :
			the dataset name given in the hdf5 file. (Default value = "All_galaxies")
		return_output :
			Output is returned if True, saved to file if False.
		output_file_pairs :
			 (Default value = "")
		print_num :
			 (Default value = True)

		Returns
		-------
		type
			xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified

		"""

		if masks == None:
			positions = self.data["Position"]
			positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
		Num_position = len(positions)
		Num_shape = len(positions_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		output_file_pairs = h5py.File(output_file_pairs, "a")
		group = create_group_hdf5(output_file_pairs, "w_g_plus")

		indices_shape = np.arange(0, len(positions_shape_sample))
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
			separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
			del projected_sep
			phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			del separation_dir
			e_plus, e_cross = self.get_ellipticity(e, phi)
			del phi
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			write_data = (np.array(
				[[n] * len(indices_shape), indices_shape, separation_len, LOS, e_plus / (2 * R)]).transpose())
			# np.array(
			# [[n] * len(ind_r), indices_shape[mask], ind_r, ind_pi, e_plus[mask] / (2 * R)]).transpose())
			if n == 0:
				group.create_dataset(dataset_name, data=write_data, maxshape=(None, 5), chunks=True)
			else:
				group[dataset_name].resize((group[dataset_name].shape[0] + write_data.shape[0]), axis=0)
				group[dataset_name][-write_data.shape[0]:] = write_data

		output_file_pairs.close()
		return

	def _measure_w_g_i(self, corr_type="both", dataset_name="All_galaxies", return_output=False, jk_group_name=""):
		"""Measures w_gi for a given xi_gi dataset that has been calculated with the measure projected correlation
		method. Sums over pi values. Stores [rp, w_gi]. i can be + or g

		Parameters
		----------
		dataset_name :
			Name of xi_gi dataset and name given to w_gi dataset when stored. (Default value = "All_galaxies")
		return_output :
			Output is returned if True, saved to file if False. (Default value = False)
		corr_type :
			Type of correlation function. Choose from [g+,gg,both]. (Default value = "both")
		jk_group_name :
			 (Default value = "")

		Returns
		-------

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

	def _measure_multipoles(self, corr_type="both", dataset_name="All_galaxies", return_output=False, jk_group_name=""):
		"""Measures multipoles for a given xi_g+ calculated by measure projected correlation.
		The data assumes xi_g+ to be measured in bins of rp and pi. It measures mu_r and r
		and saves the multipoles in the (r,mu_r) space. Should be binned into r bins.

		Parameters
		----------
		corr_type :
			Default value of g+, ensuring correct dataset and sab and l to be 2.
		dataset_name :
			Name of the dataset of xi_g+ and multipoles. (Default value = "All_galaxies")
		return_output :
			Output is returned if True, saved to file if False. (Default value = False)
		jk_group_name :
			 (Default value = "")

		Returns
		-------

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
					   jk_group_name=""):
		"""Reads various components of xi and combines into correct estimator for cluster or galaxy observational alignments

		Parameters
		----------
		corr_type :
			w or multipoles
		IA_estimator :
			clusters or galaxies
		dataset_name :
			Name of the dataset
		dataset_name_randoms :
			Name of the dataset for data with randoms as positions
		num_samples :

		jk_group_name :
			 (Default value = "")

		Returns
		-------

		"""
		output_file = h5py.File(self.output_file_name, "a")
		if corr_type[0] == "g+" or corr_type[0] == "both":
			group_gp = output_file[
				f"{self.snap_group}/{corr_type[1]}/xi_g_plus/{jk_group_name}"]  # /w/xi_g_plus/
			SpD = group_gp[f"{dataset_name}_SplusD"][:]
			SpR = group_gp[f"{dataset_name_randoms}_SplusD"][:]
		group_gg = output_file[f"{self.snap_group}/{corr_type[1]}/xi_gg/{jk_group_name}"]
		DD = group_gg[f"{dataset_name}_DD"][:]

		if IA_estimator == "clusters":
			if corr_type[0] == "gg":
				SR = group_gg[f"{dataset_name}_SR"][:]
			else:
				SR = group_gg[f"{dataset_name_randoms}_DD"][:]
			SR *= num_samples["D"] / num_samples["R_D"]
			if corr_type[0] == "g+" or corr_type[0] == "both":
				SpR *= num_samples["D"] / num_samples["R_D"]
				correlation_gp = SpD / DD - SpR / SR
				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
			if corr_type[0] == "gg" or corr_type[0] == "both":
				RD = group_gg[f"{dataset_name}_RD"][:]
				RR = group_gg[f"{dataset_name}_RR"][:]
				RD *= num_samples["S"] / num_samples["R_S"]
				RR *= (num_samples["S"] / num_samples["R_S"]) * (num_samples["D"] / num_samples["R_D"])
				correlation_gg = (DD - RD - SR) / RR - 1
				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		elif IA_estimator == "galaxies":
			RR = group_gg[f"{dataset_name}_RR"][:]
			RR *= (num_samples["S"] / num_samples["R_S"]) * (num_samples["D"] / num_samples["R_D"])
			if corr_type[0] == "g+" or corr_type[0] == "both":
				SpR *= num_samples["D"] / num_samples["R_D"]
				correlation_gp = (SpD - SpR) / RR
				write_dataset_hdf5(group_gp, dataset_name, correlation_gp)
			if corr_type[0] == "gg" or corr_type[0] == "both":
				RD = group_gg[f"{dataset_name}_RD"][:]
				if corr_type[0] == "gg":
					SR = group_gg[f"{dataset_name}_SR"][:]
				else:
					SR = group_gg[f"{dataset_name_randoms}_DD"][:]
				RD *= num_samples["S"] / num_samples["R_S"]
				SR *= num_samples["D"] / num_samples["R_D"]
				correlation_gg = (DD - RD - SR) / RR - 1
				write_dataset_hdf5(group_gg, dataset_name, correlation_gg)
		else:
			raise ValueError("Unknown input for IA_estimator, choose from [clusters, galaxies].")
		output_file.close()
		return

	def assign_jackknife_patches(self, data, randoms_data, num_jk):
		"""Assigns jackknife patches to data and randoms given a number of patches.
		Based on https://github.com/esheldon/kmeans_radec

		Parameters
		----------
		data :
			directory containing position and shape sample data
		randoms_data :
			directory containing position and shape sample data of randoms
		num_jk :
			number of jackknife patches

		Returns
		-------
		type
			directory with patch numbers for each sample

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

	def measure_misalignment_angle(self, vector1_name, vector2_name, normalise=False):
		"""NOT TESTED
		Calculate the misalignment angle between two given vectors. Assumes the vectors to be normalised unless
		otherwise specified.

		Parameters
		----------
		vector1_name :
			Name in data of the first vector.
		vector2_name :
			Name in data of the second vector
		normalise :
			If True, the vectors are divided by their length. Default is False.

		Returns
		-------
		type
			the misalignment angle, unless an output file name is given.

		"""

		eigen_vector1 = self.data[vector1_name]
		eigen_vector2 = self.data[vector2_name]
		if normalise:
			eigen_vector1 = (eigen_vector1.transpose() / np.sqrt(np.sum(eigen_vector1 ** 2, axis=1))).transpose()
			eigen_vector2 = (eigen_vector2.transpose() / np.sqrt(np.sum(eigen_vector2 ** 2, axis=1))).transpose()
		misalignment_angle = np.arccos(self.calculate_dot_product_arrays(eigen_vector1, eigen_vector2))

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/Misalignment_angels")
			write_dataset_hdf5(group, vector1_name + "_" + vector2_name, data=misalignment_angle)
			output_file.close()
		else:
			return misalignment_angle
		return


if __name__ == "__main__":
	pass
