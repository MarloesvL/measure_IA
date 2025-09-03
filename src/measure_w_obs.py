import numpy as np
import h5py
import pyccl as ccl
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.measure_IA_base import MeasureIABase
from astropy.table import vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureWObservations(MeasureIABase):
	"""Measures intrinsic alignment correlation functions including errors. Different samples for shapes and positions
		can be used. Currently allows for w_g+, w_gg and multipoles to be calculated.

	Parameters
	----------
	data :
		Dictionary with data needed for calculations. See specifications for keywords.
	simulation :
		Indicator of simulation. Choose from [TNG100, TNG300] for now.
	snapshot :
		Number of the snapshot
	separation_limits :
		Bounds of the (projected) separation vector length bins in cMpc/h (so, r or r_p)
	num_bins_r :
		Number of bins for (projected) separation vector.
	num_bins_pi :
		Number of bins for line of sight (LOS) vector, pi.
	PT :
		Number indicating particle type
	LOS_lim :
		Bound for line of sight bins. Bounds will be [-LOS_lim, LOS_lim]
	output_file_name :
		Name and filepath of the file where the output should be stored.

	Returns
	-------

	"""

	def __init__(
			self,
			data,
			simulation=None,
			snapshot=None,
			separation_limits=[0.1, 20.0],
			num_bins_r=8,
			num_bins_pi=20,
			LOS_lim=None,
			output_file_name=None,
			boxsize=None,
			periodicity=True,
	):
		super().__init__(data, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 LOS_lim, output_file_name, boxsize, periodicity)
		return

	def _measure_xi_rp_pi_obs_brute(self, masks=None, dataset_name="All_galaxies", return_output=False,
									print_num=True, over_h=False, cosmology=None, jk_group_name=""
									):
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
			Output is returned if True, saved to file if False. (Default value = False)
		print_num :
			 (Default value = True)
		over_h :
			 (Default value = False)
		cosmology :
			 (Default value = None)
		jk_group_name :
			 (Default value = "")

		Returns
		-------
		type
			xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified

		"""

		if masks == None:
			redshift = self.data["Redshift"]
			redshift_shape_sample = self.data["Redshift_shape_sample"]
			RA = self.data["RA"]
			RA_shape_sample = self.data["RA_shape_sample"]
			DEC = self.data["DEC"]
			DEC_shape_sample = self.data["DEC_shape_sample"]
			e1 = self.data["e1"]
			e2 = self.data["e2"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			redshift = self.data["Redshift"][masks["Redshift"]]
			redshift_shape_sample = self.data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
			RA = self.data["RA"][masks["RA"]]
			RA_shape_sample = self.data["RA_shape_sample"][masks["RA_shape_sample"]]
			DEC = self.data["DEC"][masks["DEC"]]
			DEC_shape_sample = self.data["DEC_shape_sample"][masks["DEC_shape_sample"]]
			e1 = self.data["e1"][masks["e1"]]
			e2 = self.data["e2"][masks["e2"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if cosmology == None:
			cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
			if print_num:
				print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
		h = cosmology["h"]

		LOS_all = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift))
		LOS_all_shape_sample = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift_shape_sample))
		if over_h:
			LOS_all *= h
			LOS_all_shape_sample *= h

		theta = 1. / 2 * np.arctan2(e2, e1)  # e1 = |e| cos(2theta), e2 = |e| sin(2theta)
		Semimajor_Axis_Direction = np.array([np.cos(theta), np.sin(theta)])
		axis_direction_len = np.sqrt(np.sum(Semimajor_Axis_Direction ** 2, axis=0))
		axis_direction = Semimajor_Axis_Direction / axis_direction_len
		e = np.sqrt(e1 ** 2 + e2 ** 2)
		phi_axis_dir = np.arctan2(axis_direction[1], axis_direction[0])

		for n in np.arange(0, len(RA)):
			# for Splus_D (calculate ellipticities around position sample)
			LOS = LOS_all_shape_sample - LOS_all[n]
			dra = (RA_shape_sample - RA[n]) / 180 * np.pi
			ddec = (DEC_shape_sample - DEC[n]) / 180 * np.pi
			dx = dra * LOS_all[n] * np.cos(DEC[n] / 180 * np.pi)
			dy = ddec * LOS_all[n]
			projected_sep = np.array([dx, dy])
			if over_h:
				projected_sep *= h
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=0))
			with np.errstate(invalid='ignore'):
				separation_dir = (projected_sep / separation_len)  # normalisation of rp
				del projected_sep
				phi_sep_dir = np.arctan2(separation_dir[1], separation_dir[0])
				phi = phi_axis_dir - phi_sep_dir
			del separation_dir
			e_plus, e_cross = self.get_ellipticity(-e, phi)
			del phi
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			# get the indices for the binning
			mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.bin_edges[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			del LOS
			ind_pi = np.array(ind_pi, dtype=int)
			if np.any(ind_r == np.shape(Splus_D)[0]):
				ind_r[np.where(ind_r == np.shape(Splus_D)[0])] = np.shape(Splus_D)[0] - 1
			if np.any(ind_pi == np.shape(Splus_D)[1]):
				ind_pi[np.where(ind_pi == np.shape(Splus_D)[1])] = np.shape(Splus_D)[1] - 1
			try:
				np.add.at(Splus_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_plus[mask]))
			except:
				print(ind_r, np.shape(Splus_D)[0], ind_r == 10, np.sum(ind_r == int(np.shape(Splus_D)[0])) > 0,
					  ind_r == int(np.shape(Splus_D)[0]))
			np.add.at(Scross_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_cross[mask]))
			del e_plus, e_cross
			np.add.at(DD, (ind_r, ind_pi), weight[n] * weight_shape[mask])

		# if Num_position == Num_shape:
		# 	DD = DD / 2.0  # auto correlation, all pairs are double

		DD[np.where(DD == 0)] = 1

		correlation = Splus_D / DD
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return Splus_D, DD, separation_bins, pi_bins

	def _count_pairs_xi_rp_pi_obs_brute(self, masks=None, dataset_name="All_galaxies", return_output=False,
										print_num=True, over_h=False, cosmology=None, data_suffix="_DD", jk_group_name=""
										):
		"""Measures the projected clustering (xi_gg) for given coordinates of the position and shape sample
		(Position, Position_shape_sample) and the index of the direction of the line of sight (LOS=2 for z axis).
		Positions are assumed to be given in cMpc/h.

		Parameters
		----------
		masks :
			the masks for the data to select only part of the data (Default value = None)
		dataset_name :
			the dataset name given in the hdf5 file. (Default value = "All_galaxies")
		return_output :
			Output is returned if True, saved to file if False. (Default value = False)
		print_num :
			 (Default value = True)
		over_h :
			 (Default value = False)
		cosmology :
			 (Default value = None)
		data_suffix :
			 (Default value = "_DD")
		jk_group_name :
			 (Default value = "")

		Returns
		-------
		type
			xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified

		"""

		if masks == None:
			redshift = self.data["Redshift"]
			redshift_shape_sample = self.data["Redshift_shape_sample"]
			RA = self.data["RA"]
			RA_shape_sample = self.data["RA_shape_sample"]
			DEC = self.data["DEC"]
			DEC_shape_sample = self.data["DEC_shape_sample"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			redshift = self.data["Redshift"][masks["Redshift"]]
			redshift_shape_sample = self.data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
			RA = self.data["RA"][masks["RA"]]
			RA_shape_sample = self.data["RA_shape_sample"][masks["RA_shape_sample"]]
			DEC = self.data["DEC"][masks["DEC"]]
			DEC_shape_sample = self.data["DEC_shape_sample"][masks["DEC_shape_sample"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if cosmology == None:
			cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
			if print_num:
				print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
		h = cosmology["h"]

		LOS_all = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift))
		LOS_all_shape_sample = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift_shape_sample))
		if over_h:
			LOS_all *= h
			LOS_all_shape_sample *= h

		for n in np.arange(0, len(RA)):
			# for Splus_D (calculate ellipticities around position sample)
			LOS = LOS_all_shape_sample - LOS_all[n]
			dra = (RA_shape_sample - RA[n]) / 180 * np.pi
			ddec = (DEC_shape_sample - DEC[n]) / 180 * np.pi
			dx = dra * LOS_all[n] * np.cos(DEC[n] / 180 * np.pi)
			dy = ddec * LOS_all[n]
			projected_sep = np.array([dx, dy])
			if over_h:
				projected_sep *= h
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=0))

			# get the indices for the binning
			mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.bin_edges[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			del LOS
			ind_pi = np.array(ind_pi, dtype=int)
			np.add.at(DD, (ind_r, ind_pi), weight[n] * weight_shape[mask])

		# if Num_position == Num_shape:
		# 	DD = DD / 2.0  # auto correlation, all pairs are double

		DD[np.where(DD == 0)] = 1

		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=DD)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return DD, separation_bins, pi_bins

	def get_cosmo_points(self, data, cosmology=cosmo):
		"""convert from astropy table of RA, DEC, and redshift to 3D cartesian coordinates in Mpc/h
		Assumes col0=RA, col1=DEC, col2=Z

		Parameters
		----------
		data :

		cosmology :
			 (Default value = cosmo)

		Returns
		-------

		"""
		comoving_dist = cosmo.comoving_distance(data[:, 2]).to(u.Mpc)
		print(min(data[:, 1]), max(data[:, 1]))
		points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data[:, 1]) * u.deg,
													np.asarray(data[:, 0]) * u.deg)  # in Mpc
		return np.asarray(points).transpose() * cosmology.h  # in Mpc/h

	def get_pair_coords(self, obs_pos1, obs_pos2, use_center_origin=True, cosmology=cosmo):
		"""Takes in observed positions of galaxy pairs and returns comoving coordinates, in Mpc/h, with the orgin at the center of the pair.
		The first coordinate (x-axis) is along the LOS
		The second coordinate (y-axis) is along 'RA'
		The third coordinate (z-axis) along 'DEC', i.e. aligned with North in origional coordinates.
		
		INPUT
		-------
		obs_pos1, obs_pos2: table with columns: 'RA', 'DEC', z_column
		use_center_origin: True for coordinate orgin at center of pair, othersise centers on first position
		cosmology: astropy.cosmology

		Parameters
		----------
		obs_pos1 :

		obs_pos2 :

		use_center_origin :
			 (Default value = True)
		cosmology :
			 (Default value = cosmo)

		Returns
		-------

		
		"""
		cartesian_coords = self.get_cosmo_points(np.vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
		# find center position of coordinates
		origin = cartesian_coords[0]
		if use_center_origin == True:
			origin = np.mean(cartesian_coords, axis=0)
		cartesian_coords -= origin
		return cartesian_coords  # in Mpc/h

	def count_pairs_xi_grid_w_update(self, masks=None, dataset_name="All_galaxies", return_output=False,
									 print_num=True, over_h=False, cosmology=None, data_suffix="_DD", jk_group_name=""
									 ):
		"""Measures the projected clustering (xi_gg) for given coordinates of the position and shape sample
		(Position, Position_shape_sample) and the index of the direction of the line of sight (LOS=2 for z axis).
		Positions are assumed to be given in cMpc/h.

		Parameters
		----------
		masks :
			the masks for the data to select only part of the data (Default value = None)
		dataset_name :
			the dataset name given in the hdf5 file. (Default value = "All_galaxies")
		return_output :
			Output is returned if True, saved to file if False. (Default value = False)
		print_num :
			 (Default value = True)
		over_h :
			 (Default value = False)
		cosmology :
			 (Default value = None)
		data_suffix :
			 (Default value = "_DD")
		jk_group_name :
			 (Default value = "")

		Returns
		-------
		type
			xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified

		"""

		if masks == None:
			redshift = self.data["Redshift"]
			redshift_shape_sample = self.data["Redshift_shape_sample"]
			RA = self.data["RA"]
			RA_shape_sample = self.data["RA_shape_sample"]
			DEC = self.data["DEC"]
			DEC_shape_sample = self.data["DEC_shape_sample"]
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			redshift = self.data["Redshift"][masks["Redshift"]]
			redshift_shape_sample = self.data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
			RA = self.data["RA"][masks["RA"]]
			RA_shape_sample = self.data["RA_shape_sample"][masks["RA_shape_sample"]]
			DEC = self.data["DEC"][masks["DEC"]]
			DEC_shape_sample = self.data["DEC_shape_sample"][masks["DEC_shape_sample"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if cosmology == None:
			cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
			if print_num:
				print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
		h = cosmology["h"]

		LOS_all = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift))
		LOS_all_shape_sample = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift_shape_sample))
		if over_h:
			LOS_all *= h
			LOS_all_shape_sample *= h
		c = FlatLambdaCDM(H0=70.0, Om0=(0.225 + 0.045), Ob0=0.045)
		RA_rad = RA / 180 * np.pi
		RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
		DEC_rad = DEC / 180 * np.pi
		DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi

		for n in np.arange(0, len(RA)):
			# for Splus_D (calculate ellipticities around position sample)
			LOS = LOS_all_shape_sample - LOS_all[n]
			dra = (RA_shape_sample - RA[n]) / 180 * np.pi
			ddec = (DEC_shape_sample - DEC[n]) / 180 * np.pi
			dx = dra * LOS_all[n] * np.cos(DEC[n] / 180 * np.pi)
			dy = ddec * LOS_all[n]
			print(np.shape(np.array([[RA[n], DEC[n], redshift[n]]])), np.shape(np.array(
				[LOS_all_shape_sample, RA_shape_sample, DEC_shape_sample]).transpose()))
			points = self.get_pair_coords(np.array([[RA[n], DEC[n], redshift[n]]]),
										  np.array(
											  [RA_shape_sample, DEC_shape_sample,
											   redshift_shape_sample]).transpose(), use_center_origin=False,
										  cosmology=c)
			points = points[1:]
			LOS1, dx1, dy1 = points[:, 0], points[:, 1], points[:, 2]

			# DYI points
			# print(np.shape(LOS_all[n]))
			# LOS_all2 = cosmo.comoving_distance(redshift).to(u.Mpc)
			# print((LOS_all-np.asarray(LOS_all2))/LOS_all) # <1%

			# this is the same as 1 if cosmo.comoving_distance(redshift).to(u.Mpc) is used and not existing LOS
			# cart_coords = coordinates.spherical_to_cartesian(
			# 	np.abs(np.concatenate([[LOS_all[n]], LOS_all_shape_sample])),
			# 	np.concatenate([[DEC_rad[n]], DEC_shape_sample_rad]),
			# 	np.concatenate([[RA_rad[n]], RA_shape_sample_rad]))
			# cart_coords = np.asarray(cart_coords).transpose() * h
			# pos = cart_coords[0]
			# cart_coords = cart_coords[1:]
			# sep = cart_coords - pos
			# LOS2, dx2, dy2 = sep[:, 0], sep[:, 1], sep[:, 2]

			# try again
			# pos2 = coordinates.SkyCoord(ra=RA[n]*u.deg,dec=DEC[n]*u.deg,distance=LOS_all[n]/h *u.Mpc)
			# shape2 = coordinates.SkyCoord(ra=RA_shape_sample*u.deg,dec=DEC_shape_sample*u.deg,distance=LOS_all_shape_sample/h*u.Mpc)
			# LOS3, dx3, dy3 = shape2.x - pos2.x,

			# this is the same as 2
			# cart_coords = coordinates.spherical_to_cartesian(
			# 	np.abs(np.concatenate([[LOS_all[n]], LOS_all_shape_sample])),
			# 	np.concatenate([[DEC[n]], DEC_shape_sample])*u.deg,
			# 	np.concatenate([[RA[n]], RA_shape_sample])*u.deg)
			# cart_coords = np.asarray(cart_coords).transpose() * h
			# pos = cart_coords[0]
			# cart_coords = cart_coords[1:]
			# sep = cart_coords - pos
			# LOS3, dx3, dy3 = sep[:, 0], sep[:, 1], sep[:, 2]

			# again
			# x = d * cos(δ) * cos(α), y = d * cos(δ) * sin(α), and z = d * sin(δ)

			# print((LOS - LOS2) / LOS, (dx - dx2) / dx, (dy - dy2) / dy)  # <1% due to LOS cosmology
			# print(LOS2,dx2,dy2)
			# print(LOS3, dx3, dy3)

			print((LOS - LOS1) / LOS)  # not the same

			# cartesian_coords = self.get_cosmo_points(np.vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
			# find center position of coordinates
			# origin = cartesian_coords[0]
			# if use_center_origin == True:
			# 	origin = np.mean(cartesian_coords, axis=0)
			# cartesian_coords -= origin

			# comoving_dist = cosmo.comoving_distance(data[:, 2]).to(u.Mpc)
			# points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data[:, 1]) * u.deg,
			# 											np.asarray(data[:, 0]) * u.deg)  # in Mpc
			# return np.asarray(points).transpose() * cosmology.h  # in Mpc/h

			# print(points, dx, dy, LOS)
			# print(np.shape(points),len(LOS))
			exit()
			projected_sep = np.array([dx, dy])
			if over_h:
				projected_sep *= h
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=0))

			# get the indices for the binning
			mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.bin_edges[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			del LOS
			ind_pi = np.array(ind_pi, dtype=int)
			np.add.at(DD, (ind_r, ind_pi), weight[n] * weight_shape[mask])

		# if Num_position == Num_shape:
		# 	DD = DD / 2.0  # auto correlation, all pairs are double

		DD[np.where(DD == 0)] = 1

		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=DD)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return DD, separation_bins, pi_bins
