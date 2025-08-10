import math
import numpy as np
import h5py
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
from scipy.special import lpmn
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.Sim_info import SimInfo
from astropy.cosmology import LambdaCDM, z_at_value

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureIABase(SimInfo):
	"""
	Measures intrinsic alignment correlation functions including errors. Different samples for shapes and positions
	can be used. Currently allows for w_g+, w_gg and multipoles to be calculated.
	:param data: Dictionary with data needed for calculations. See specifications for keywords.
	:param simulation: Indicator of simulation. Choose from [TNG100, TNG300] for now.
	:param snapshot: Number of the snapshot
	:param separation_limits: Bounds of the (projected) separation vector length bins in cMpc/h (so, r or r_p)
	:param num_bins_r: Number of bins for (projected) separation vector.
	:param num_bins_pi: Number of bins for line of sight (LOS) vector, pi.
	:param PT: Number indicating particle type
	:param LOS_lim: Bound for line of sight bins. Bounds will be [-LOS_lim, LOS_lim]
	:param output_file_name: Name and filepath of the file where the output should be stored.
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
	):
		if type(simulation) == str:  # simulation is a tag that is hardcoded into SimInfo
			SimInfo.__init__(self, simulation, snapshot, PT)
			self.boxsize /= 1000.0  # ckpc -> cMpc
			self.L_0p5 /= 1000.0
		elif boxsize is not None:  # boxsize is given manually
			self.boxsize = boxsize
			self.L_0p5 = boxsize / 2.0
			self.PT = PT
			self.snapshot = snapshot
		elif simulation == False:
			print("Assuming observational data. Only use observational methods.")
			self.L_0p5 = None
			self.PT = None
			self.snapshot = None
		else:
			SimInfo.__init__(self, simulation,
							 snapshot, PT)  # simulation is a SimInfo object created in the file that calls this class
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
		self.separation_min = separation_limits[0]  # cMpc/h
		self.separation_max = separation_limits[1]  # cMpc/h
		self.num_bins_r = num_bins_r
		self.num_bins_pi = num_bins_pi
		self.bin_edges = np.logspace(np.log10(self.separation_min), np.log10(self.separation_max), self.num_bins_r + 1)
		if LOS_lim == None:
			pi_max = self.L_0p5
		else:
			pi_max = LOS_lim
		self.pi_bins = np.linspace(-pi_max, pi_max, self.num_bins_pi + 1)
		self.bins_mu_r = np.linspace(-1, 1, self.num_bins_pi + 1)
		if simulation == False:
			print(f"MeasureIA object initialised with:\n \
					observational data.\n \
					There are {self.Num_shape} galaxies in the shape sample and {self.Num_position} galaxies in the position sample.\n\
					The separation bin edges are given by {self.bin_edges} Mpc.\n \
					There are {num_bins_r} r or r_p bins and {num_bins_pi} pi bins.\n \
					The maximum pi used for binning is {pi_max}.\n \
					The data will be written to {self.output_file_name}")
		else:
			print(f"MeasureIA object initialised with:\n \
			simulation {simulation} that has a {periodic}boxsize of {self.boxsize} cMpc/h.\n \
			There are {self.Num_shape} galaxies in the shape sample and {self.Num_position} galaxies in the position sample.\n\
			The separation bin edges are given by {self.bin_edges} cMpc/h.\n \
			There are {num_bins_r} r or r_p bins and {num_bins_pi} pi bins.\n \
			The maximum pi used for binning is {pi_max}.\n \
			The data will be written to {self.output_file_name}")
		return

	@staticmethod
	def calculate_dot_product_arrays(a1, a2):
		"""
		Calculates the dot product over 2 2D arrays across axis 1 so that
		dot_product[i] = np.dot(a1[i],a2[i])
		:param a1: First array
		:param a2: Second array
		:return: Dot product of columns of arrays
		"""
		dot_product = np.zeros(np.shape(a1)[0])
		for i in np.arange(0, np.shape(a1)[1]):
			dot_product += a1[:, i] * a2[:, i]
		return dot_product

	def measure_3D_orientation_separation_correlation(self, masks=None, dataset_name="All_galaxies"):
		"""
		NEEDS MORE EXTENSIVE TESTS
		Measures the 3D orientation-separation correlation function for given positions and minor axis directions.
		:param masks: Directory of masks for the data that makes a selection in the data.
		:param dataset_name: Name of the dataset in the hdf5 file specified in output file name.
		:return: correlation, separation bin means (log) if output file name not specified.
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
				lower_limit_mask = separation_len > self.bin_edges[i]
				upper_limit_mask = separation_len < self.bin_edges[i + 1]
				mask = lower_limit_mask * upper_limit_mask
				n_pairs[i] += sum(mask)
				inner_product[i] += sum(inner_product_n[mask])
		correlation = np.array(inner_product) / np.array(n_pairs) - 1.0 / 3
		dsep = (self.bin_edges[:-1] - self.bin_edges[1:]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/3D_correlations")
			write_dataset_hdf5(group, dataset_name, data=np.array([separation_bins, correlation]).transpose())
			output_file.close()
			return
		else:
			return correlation, separation_bins

	@staticmethod
	def get_ellipticity(e, phi):
		"""
		Calculates the radial and tangential components of the ellipticity, given the size of the ellipticty vector
		and the angle between the semimajor or semiminor axis and the separation vector.
		:param e: size of the ellipticity vector
		:param phi: angle between semimajor/semiminor axis and separation vector
		:return: e_+ and e_x
		"""
		e_plus, e_cross = e * np.cos(2 * phi), e * np.sin(2 * phi)
		return e_plus, e_cross

	@staticmethod
	def get_random_pairs(rp_max, rp_min, pi_max, pi_min, L3, corrtype, Num_position, Num_shape):
		"""
		Returns analytical value of the number of pairs expected in an r_p, pi bin for a random uniform distribution.
		(Singh et al. 2023)
		:param rp_max: upper bound of projected separation vector bin
		:param rp_min: lower bound of projected separation vector bin
		:param pi_max: upper bound of line of sight vector bin
		:param pi_min: lower bound of line of sight vector bin
		:param L3: volume of the simulation box
		:param corrtype: Correlation type, auto or cross. RR for auto is RR_cross/2.
		:return: number of pairs in r_p, pi bin
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
		"""
		Calculate the volume of a spherical cap.
		:param mur: cos(theta), where theta is the polar angle between the apex and disk of the cap.
		:param r: radius
		:return: Volume of the spherical cap.
		"""
		return np.pi / 3.0 * r ** 3 * (2 + mur) * (1 - mur) ** 2

	def get_random_pairs_r_mur(self, r_max, r_min, mur_max, mur_min, L3, corrtype, Num_position, Num_shape):
		"""
		Retruns analytical value of the number of pairs expected in an r_p, pi bin for a random uniform distribution.
		(Singh et al. 2023)
		:param r_max: upper bound of projected separation vector bin
		:param r_min: lower bound of projected separation vector bin
		:param mur_max: upper bound of mu_r bin
		:param mur_min: lower bound of mu_r bin
		:param L3: volume of the simulation box
		:param corrtype:  Correlation type, auto or cross. RR for auto is RR_cross/2.
		:return: number of pairs in r, mu_r bin
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
		diff = []
		for i in np.arange(0, len(a1)):
			setdiff = np.setdiff1d(a1[i], a2[i])
			diff.append(setdiff)
			del setdiff
		return diff

	@staticmethod
	def setdiff_omit(a1, a2, incl_ind):
		diff = []
		for i in np.arange(0, len(a1)):
			if np.isin(i, incl_ind):
				setdiff = np.setdiff1d(a1[i], a2)
				diff.append(setdiff)
				del setdiff
		return diff

	# def get_cosmo_points(self, data, cosmology=cosmo):
	# 	'''convert from astropy table of RA, DEC, and redshift to 3D cartesian coordinates in Mpc/h
	# 	Assumes col0=RA, col1=DEC, col2=Z'''
	# 	comoving_dist = cosmo.comoving_distance(data[:, 2]).to(u.Mpc)
	# 	points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data[:, 1]) * u.deg,
	# 												np.asarray(data[:, 0]) * u.deg)  # in Mpc
	# 	return np.asarray(points).transpose() * cosmology.h  # in Mpc/h
	#
	# def get_pair_coords(self, obs_pos1, obs_pos2, use_center_origin=True, cosmology=cosmo):
	# 	'''
	# 	Takes in observed positions of galaxy pairs and returns comoving coordinates, in Mpc/h, with the orgin at the center of the pair.
	# 	The first coordinate (x-axis) is along the LOS
	# 	The second coordinate (y-axis) is along 'RA'
	# 	The third coordinate (z-axis) along 'DEC', i.e. aligned with North in origional coordinates.
	#
	# 	INPUT
	# 	-------
	# 	obs_pos1, obs_pos2: table with columns: 'RA', 'DEC', z_column
	# 	use_center_origin: True for coordinate orgin at center of pair, othersise centers on first position
	# 	cosmology: astropy.cosmology
	#
	# 	RETURNS
	# 	-------
	# 	numpy array of cartesian coordinates, in Mpc/h. Shape (2,3)
	#
	# 	'''
	# 	cartesian_coords = self.get_cosmo_points(vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
	# 	# find center position of coordinates
	# 	origin = cartesian_coords[0]
	# 	if use_center_origin == True:
	# 		origin = np.mean(cartesian_coords, axis=0)
	# 	cartesian_coords -= origin
	# 	return cartesian_coords  # in Mpc/h
	#
	# def measure_projected_correlation_obs(self, masks=None, dataset_name="All_galaxies",
	# 									  return_output=False, print_num=True):
	# 	"""
	# 	Measures the projected correlation function (xi_g_plus, xi_gg) for given coordinates of the position and shape sample
	# 	(Position, Position_shape_sample), the projected axis direction (Axis_Direction), the ratio between projected
	# 	axes, q=b/a (q) and the index of the direction of the line of sight (LOS=2 for z axis).
	# 	Positions are assumed to be given in cMpc/h.
	# 	:param masks: the masks for the data to select only part of the data
	# 	:param dataset_name: the dataset name given in the hdf5 file.
	# 	:param return_output: Output is returned if True, saved to file if False.
	# 	:return: xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified
	# 	"""
	#
	# 	if masks == None:
	# 		positions = self.data["Position"]
	# 		positions_shape_sample = self.data["Position_shape_sample"]
	# 		axis_direction_v = self.data["Axis_Direction"]
	# 		axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
	# 		axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
	# 		try:
	# 			q = self.data["q"]
	# 		except:
	# 			e1, e2 = self.data["e1"], self.data["e2"]
	# 		weight = self.data["weight"]
	# 		weight_shape = self.data["weight_shape_sample"]
	# 	else:
	# 		positions = self.data["Position"][masks["Position"]]
	# 		positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
	# 		axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
	# 		axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
	# 		axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
	# 		try:
	# 			q = self.data["q"][masks["q"]]
	# 		except:
	# 			e1, e2 = self.data["e1"][masks["e1"]], self.data["e2"][masks["e2"]]
	# 		weight = self.data["weight"][masks["weight"]]
	# 		weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
	# 	Num_position = len(positions)
	# 	Num_shape = len(positions_shape_sample)
	# 	if print_num:
	# 		print(
	# 			f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
	#
	# 	LOS_ind = 2  # redshift column #self.data["LOS"]  # eg 2 for z axis
	# 	not_LOS = [0, 1]  # np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
	# 	try:
	# 		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
	# 	except:
	# 		e_comp = e1 + 1j * e2
	# 		e = np.sqrt(e_comp.real ** 2 + e_comp.imag ** 2)
	# 	R = 1 - np.mean(e ** 2) / 2.0  # responsivity factor
	# 	L3 = self.boxsize ** 3  # box volume
	# 	sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
	# 	sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
	# 	DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	# 	Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	# 	Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	# 	RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	# 	RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	# 	variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
	#
	# 	pos_tree = KDTree(positions, boxsize=self.boxsize)
	# 	shape_tree = KDTree(positions_shape_sample, boxsize=self.boxsize)
	# 	ind_min = shape_tree.query_ball_tree(pos_tree, self.separation_min)
	# 	ind_max = shape_tree.query_ball_tree(pos_tree, self.separation_max)
	# 	ind_rbin = self.setdiff2D(ind_max, ind_min)
	# 	del ind_min, ind_max
	#
	# 	for n in np.arange(0, len(positions_shape_sample)):
	# 		# for Splus_D (calculate ellipticities around position sample)
	# 		separation = self.get_pair_coords(positions_shape_sample[n], positions[ind_rbin[n]],
	# 										  use_center_origin=False)
	# 		# separation = positions_shape_sample[n] - positions[ind_rbin[n]]
	# 		# separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
	# 		# separation[separation < -self.L_0p5] += self.boxsize
	# 		projected_sep = separation[:, not_LOS]
	# 		LOS = separation[:, LOS_ind]
	# 		separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
	# 		separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
	# 		phi = np.arccos(separation_dir[:, 0] * axis_direction[n, 0] + separation_dir[:, 1] * axis_direction[
	# 			n, 1])  # [0,pi]
	# 		e_plus, e_cross = self.get_ellipticity(e[n], phi)
	# 		e_plus[np.isnan(e_plus)] = 0.0
	# 		e_cross[np.isnan(e_cross)] = 0.0
	#
	# 		# get the indices for the binning
	# 		mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
	# 				LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
	# 		ind_r = np.floor(
	# 			np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.bin_edges[0]) / sub_box_len_logrp
	# 		)
	# 		ind_r = np.array(ind_r, dtype=int)
	# 		ind_pi = np.floor(
	# 			LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
	# 		)  # need length of LOS, so only positive values
	# 		ind_pi = np.array(ind_pi, dtype=int)
	# 		np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask] / (2 * R))
	# 		np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask] / (2 * R))
	# 		np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
	# 		np.add.at(DD, (ind_r, ind_pi), 1.0)
	#
	# 	if Num_position == Num_shape:
	# 		DD = DD / 2.0  # auto correlation, all pairs are double
	#
	# 	for i in np.arange(0, self.num_bins_r):
	# 		for p in np.arange(0, self.num_bins_pi):
	# 			RR_g_plus[i, p] = self.get_random_pairs(
	# 				self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
	# 				Num_position, Num_shape)
	# 			RR_gg[i, p] = self.get_random_pairs(
	# 				self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "auto",
	# 				Num_position, Num_shape)
	# 	correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
	# 	xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
	# 	sigsq = variance / RR_g_plus ** 2
	# 	dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
	# 	separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
	# 	dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
	# 	pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins
	#
	# 	if (self.output_file_name != None) & return_output == False:
	# 		output_file = h5py.File(self.output_file_name, "a")
	# 		group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
	# 		write_dataset_hdf5(group, dataset_name, data=correlation)
	# 		write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
	# 		write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
	# 		write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
	# 		write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
	# 		write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
	# 		group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
	# 		write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
	# 		write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
	# 		write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
	# 		write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
	# 		write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
	# 		write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
	# 		group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
	# 		write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
	# 		write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
	# 		write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
	# 		write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
	# 		write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
	# 		write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
	# 		output_file.close()
	# 		return
	# 	else:
	# 		return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins

	def measure_projected_correlation_save_pairs(self, output_file_pairs="", masks=None, dataset_name="All_galaxies",
												 print_num=True):
		"""
		Measures the projected correlation function (xi_g_plus, xi_gg) for given coordinates of the position and shape sample
		(Position, Position_shape_sample), the projected axis direction (Axis_Direction), the ratio between projected
		axes, q=b/a (q) and the index of the direction of the line of sight (LOS=2 for z axis).
		Positions are assumed to be given in cMpc/h.
		:param masks: the masks for the data to select only part of the data
		:param dataset_name: the dataset name given in the hdf5 file.
		:param return_output: Output is returned if True, saved to file if False.
		:return: xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified
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

	def measure_w_g_i(self, corr_type="both", dataset_name="All_galaxies", return_output=False):
		"""
		Measures w_gi for a given xi_gi dataset that has been calculated with the measure projected correlation
		method. Sums over pi values. Stores [rp, w_gi]. i can be + or g
		:param dataset_name: Name of xi_gi dataset and name given to w_gi dataset when stored.
		:param return_output: Output is returned if True, saved to file if False.
		:param corr_type: Type of correlation function. Choose from [g+,gg,both].
		:return:
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
			group = correlation_data_file[f"Snapshot_{self.snapshot}/w/" + xi_data[i]]
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
				group_out = create_group_hdf5(correlation_data_file, f"Snapshot_{self.snapshot}/" + wg_data[i])
				write_dataset_hdf5(group_out, dataset_name + "_rp", data=rp)
				write_dataset_hdf5(group_out, dataset_name, data=w_g_i)
				# write_dataset_hdf5(group_out, dataset_name + "_sigma", data=np.sqrt(sigsq))
				correlation_data_file.close()
		return

	def measure_multipoles(self, corr_type="both", dataset_name="All_galaxies", return_output=False):
		"""
		Measures multipoles for a given xi_g+ calculated by measure projected correlation.
		The data assumes xi_g+ to be measured in bins of rp and pi. It measures mu_r and r
		and saves the multipoles in the (r,mu_r) space. Should be binned into r bins.
		:param corr_type: Default value of g+, ensuring correct dataset and sab and l to be 2.
		:param dataset_name: Name of the dataset of xi_g+ and multipoles.
		:param return_output: Output is returned if True, saved to file if False.
		:return:
		"""
		correlation_data_file = h5py.File(self.output_file_name, "a")
		if corr_type == "g+":  # todo: expand to include ++ option
			group = correlation_data_file[f"Snapshot_{self.snapshot}/multipoles/xi_g_plus"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of r,mur
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [2]
			l_list = sab_list
			corr_type_list = ["g_plus"]
		elif corr_type == "gg":
			group = correlation_data_file[f"Snapshot_{self.snapshot}/multipoles/xi_gg"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [0]
			l_list = sab_list
			corr_type_list = ["gg"]
		elif corr_type == "both":
			group = correlation_data_file[f"Snapshot_{self.snapshot}/multipoles/xi_g_plus"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			group = correlation_data_file[f"Snapshot_{self.snapshot}/multipoles/xi_gg"]
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
			dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1])
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
			dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
			separation = self.bin_edges[:-1] + abs(dsep)  # middle of bins
			if return_output:
				correlation_data_file.close()
				np.array([separation, multipoles]).transpose()
			else:
				group_out = create_group_hdf5(
					correlation_data_file, f"Snapshot_{self.snapshot}/multipoles_" + corr_type_i
				)
				write_dataset_hdf5(group_out, dataset_name + "_r", data=separation)
				write_dataset_hdf5(group_out, dataset_name, data=multipoles)
		correlation_data_file.close()
		return

	def obs_estimator(self, corr_type, IA_estimator, dataset_name, dataset_name_randoms, num_samples):
		'''
		Reads various components of xi and combines into correct estimator for cluster or galaxy observational alignments
		:param corr_type: w or multipoles
		:param IA_estimator: clusters or galaxies
		:param dataset_name: Name of the dataset
		:param dataset_name_randoms: Name of the dataset for data with randoms as positions
		:return:
		'''
		output_file = h5py.File(self.output_file_name, "a")
		if corr_type[0] == "g+" or corr_type[0] == "both":
			group_gp = output_file[f"Snapshot_{self.snapshot}/{corr_type[1]}/xi_g_plus"]  # /w/xi_g_plus/
			SpD = group_gp[f"{dataset_name}_SplusD"][:]
			SpR = group_gp[f"{dataset_name_randoms}_SplusD"][:]
		group_gg = output_file[f"Snapshot_{self.snapshot}/{corr_type[1]}/xi_gg"]
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
		'''
		Assigns jackknife patches to data and randoms given a number of patches.
		Based on https://github.com/esheldon/kmeans_radec
		:param data: directory containing position and shape sample data
		:param randoms_data: directory containing position and shape sample data of randoms
		:param num_jk: number of jackknife patches
		:return: directory with patch numbers for each sample
		'''

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
		"""
		NOT TESTED
		Calculate the misalignment angle between two given vectors. Assumes the vectors to be normalised unless
		otherwise specified.
		:param vector1_name: Name in data of the first vector.
		:param vector2_name: Name in data of the second vector
		:param normalise: If True, the vectors are divided by their length. Default is False.
		:return: the misalignment angle, unless an output file name is given.
		"""

		eigen_vector1 = self.data[vector1_name]
		eigen_vector2 = self.data[vector2_name]
		if normalise:
			eigen_vector1 = (eigen_vector1.transpose() / np.sqrt(np.sum(eigen_vector1 ** 2, axis=1))).transpose()
			eigen_vector2 = (eigen_vector2.transpose() / np.sqrt(np.sum(eigen_vector2 ** 2, axis=1))).transpose()
		misalignment_angle = np.arccos(self.calculate_dot_product_arrays(eigen_vector1, eigen_vector2))

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/Misalignment_angels")
			write_dataset_hdf5(group, vector1_name + "_" + vector2_name, data=misalignment_angle)
			output_file.close()
		else:
			return misalignment_angle
		return


if __name__ == "__main__":
	pass
