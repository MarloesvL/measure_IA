import numpy as np
import h5py
import pyccl as ccl
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase


class MeasureMultipolesLightcone(MeasureIABase):
	"""Class that contains all methods for the measurements of xi_gg and xi_g+ for w_gg and w_g+ with lightcone data.

	Notes
	-----
	Inherits attributes from 'SimInfo', where 'boxsize', 'L_0p5' and 'snap_group' are used in this class.
	Inherits attributes from 'MeasureIABase', where 'data', 'output_file_name', 'periodicity', 'Num_position',
	'Num_shape', 'r_min', 'r_max', 'num_bins_r', 'num_bins_pi', 'r_bins', 'mu_r_bins', 'mu_r_bins' are used.

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
		The __init__ method of the MeasureWObservations class.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		return

	def _measure_xi_r_mur_lightcone_brute(self, dataset_name, masks=None, return_output=False,
										  print_num=True, over_h=False, cosmology=None,
										  data_suffix="_SplusD", chunk_size=1000, num_nodes=1, temp_file_path=None
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
			xi_g_plus, xi_gg, separation_bins, mu_r_bins if no output file is specified

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
			if "weight" not in masks:
				masks["weight"] = masks["RA"]
			if "weight_shape_sample" not in masks:
				masks["weight_shape_sample"] = masks["RA_shape_sample"]
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		if data_suffix == "_SplusD":
			DD_suff = "_DD"
			Scross_suff = "_ScrossD"
		elif data_suffix == "_SplusR":
			DD_suff = "_SR"
			Scross_suff = "_ScrossR"
		else:
			raise ValueError("data_suffix must be _SplusD or _SplusR")
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi
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
		del redshift, redshift_shape_sample, cosmology

		if getattr(self, "responsivity_correction", False):
			R = sum(weight_shape * (1 - (e1 ** 2 + e2 ** 2) / 2.0)) / sum(weight_shape)
			e1, e2 = e1 / (2 * R), e2 / (2 * R)
		e = np.array([e1, e2]).transpose()
		RA_rad = RA / 180 * np.pi
		RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
		DEC_rad = DEC / 180 * np.pi
		DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi
		n_shape = np.array([np.cos(DEC_shape_sample_rad) * np.cos(RA_shape_sample_rad),
							np.cos(DEC_shape_sample_rad) * np.sin(RA_shape_sample_rad),
							np.sin(DEC_shape_sample_rad)]).transpose()
		del DEC_shape_sample, RA_shape_sample, DEC_shape_sample_rad, RA_shape_sample_rad
		s_shape = n_shape * np.array([LOS_all_shape_sample]).transpose()
		n_pos = np.array([np.cos(DEC_rad) * np.cos(RA_rad),
						  np.cos(DEC_rad) * np.sin(RA_rad),
						  np.sin(DEC_rad)]).transpose()
		east = np.array([-np.sin(RA_rad), np.cos(RA_rad), np.zeros(Num_position)]).transpose()
		north = np.array([
			-np.sin(DEC_rad) * np.cos(RA_rad),
			-np.sin(DEC_rad) * np.sin(RA_rad),
			np.cos(DEC_rad)
		]).transpose()
		del RA, DEC, DEC_rad, RA_rad
		s_pos = np.array([LOS_all]).transpose() * n_pos
		del LOS_all, LOS_all_shape_sample

		for n in np.arange(0, Num_position):
			L = s_pos[n] + s_shape
			n_LOS = L / np.sqrt(np.sum(L ** 2, axis=1))[:, None]
			# n_LOS = (n_pos[n] + n_shape) / np.array([np.sqrt(np.sum((n_pos[n] + n_shape) ** 2, axis=1))]).transpose()
			s = s_shape - s_pos[n]
			LOS = self.calculate_dot_product_arrays(s, n_LOS)
			separation_len = np.sqrt(np.sum(s ** 2, axis=1))
			mu_r = LOS / separation_len

			# Projected separation vector
			s_perp = s - np.sum(s * n_LOS, axis=1, keepdims=True) * n_LOS

			# Components of projected separation
			x = np.sum(s_perp * east[n], axis=1)
			y = np.sum(s_perp * north[n], axis=1)
			# phi = np.arctan2(x, y)  # angle from north toward east
			phi = np.arctan2(y, x)

			e_plus, e_cross = self.get_ellipticity(e, phi)
			# del phi_sep_dir
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			# get the indices for the binning
			mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.r_bins[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			del mu_r
			ind_mu_r = np.array(ind_mu_r, dtype=int)
			if np.any(ind_r == np.shape(Splus_D)[0]):
				ind_r[np.where(ind_r == np.shape(Splus_D)[0])] = np.shape(Splus_D)[0] - 1
			if np.any(ind_mu_r == np.shape(Splus_D)[1]):
				ind_mu_r[np.where(ind_mu_r == np.shape(Splus_D)[1])] = np.shape(Splus_D)[1] - 1
			np.add.at(Splus_D, (ind_r, ind_mu_r), (weight[n] * weight_shape[mask] * e_plus[mask]))
			np.add.at(Scross_D, (ind_r, ind_mu_r), (weight[n] * weight_shape[mask] * e_cross[mask]))
			np.add.at(DD, (ind_r, ind_mu_r), weight[n] * weight_shape[mask])
			del e_plus, e_cross


		# if Num_position == Num_shape:
		# 	DD = DD / 2.0  # auto correlation, all pairs are double

		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/")
			write_dataset_hdf5(group, dataset_name + Scross_suff, data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + DD_suff, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return Splus_D, DD, separation_bins, mu_r_bins

	def _measure_xi_r_mur_lightcone_tree(self, dataset_name, masks=None, return_output=False,
										 print_num=True, over_h=False, cosmology=None,
										 data_suffix="_SplusD", chunk_size=1000, num_nodes=1, temp_file_path=None
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
			xi_g_plus, xi_gg, separation_bins, mu_r_bins if no output file is specified

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
			if "weight" not in masks:
				masks["weight"] = masks["RA"]
			if "weight_shape_sample" not in masks:
				masks["weight_shape_sample"] = masks["RA_shape_sample"]
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		if data_suffix == "_SplusD":
			DD_suff = "_DD"
			Scross_suff = "_ScrossD"
		elif data_suffix == "_SplusR":
			DD_suff = "_SR"
			Scross_suff = "_ScrossR"
		else:
			raise ValueError("data_suffix must be _SplusD or _SplusR")
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi
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
		del redshift, redshift_shape_sample, cosmology

		if getattr(self, "responsivity_correction", False):
			R = sum(weight_shape * (1 - (e1 ** 2 + e2 ** 2) / 2.0)) / sum(weight_shape)
			e1, e2 = e1 / (2 * R), e2 / (2 * R)
		e = np.array([e1, e2]).transpose()
		RA_rad = RA / 180 * np.pi
		RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
		DEC_rad = DEC / 180 * np.pi
		DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi
		n_shape = np.array([np.cos(DEC_shape_sample_rad) * np.cos(RA_shape_sample_rad),
							np.cos(DEC_shape_sample_rad) * np.sin(RA_shape_sample_rad),
							np.sin(DEC_shape_sample_rad)]).transpose()
		del DEC_shape_sample, RA_shape_sample, DEC_shape_sample_rad, RA_shape_sample_rad
		s_shape = n_shape * np.array([LOS_all_shape_sample]).transpose()
		n_pos = np.array([np.cos(DEC_rad) * np.cos(RA_rad),
						  np.cos(DEC_rad) * np.sin(RA_rad),
						  np.sin(DEC_rad)]).transpose()
		east = np.array([-np.sin(RA_rad), np.cos(RA_rad), np.zeros(Num_position)]).transpose()
		north = np.array([
			-np.sin(DEC_rad) * np.cos(RA_rad),
			-np.sin(DEC_rad) * np.sin(RA_rad),
			np.cos(DEC_rad)
		]).transpose()
		del RA, DEC, DEC_rad, RA_rad
		s_pos = np.array([LOS_all]).transpose() * n_pos
		del LOS_all, LOS_all_shape_sample
		shape_tree = KDTree(s_shape)
		for i in np.arange(0, Num_position, 100):  # RAM optimisation
			i2 = min(Num_position, i + 100)
			s_pos_i = s_pos[i:i2]
			n_pos_i = n_pos[i:i2]
			weight_i = weight[i:i2]
			east_i = east[i:i2]
			north_i = north[i:i2]
			pos_tree = KDTree(s_pos_i)
			ind_min_i = pos_tree.query_ball_tree(shape_tree, self.r_min)
			ind_max_i = pos_tree.query_ball_tree(shape_tree, self.r_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(s_pos_i)):
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					L = s_pos_i[n] + s_shape[ind_rbin_i[n]]
					n_LOS = L / np.sqrt(np.sum(L ** 2, axis=1))[:, None]
					# n_LOS = (n_pos_i[n] + n_shape[ind_rbin_i[n]]) / np.array(
					# 	[np.sqrt(np.sum((n_pos_i[n] + n_shape[ind_rbin_i[n]]) ** 2, axis=1))]).transpose()
					s = s_shape[ind_rbin_i[n]] - s_pos_i[n]
					LOS = self.calculate_dot_product_arrays(s, n_LOS)
					separation_len = np.sqrt(np.sum(s ** 2, axis=1))
					mu_r = LOS / separation_len

					# Projected separation vector
					s_perp = s - np.sum(s * n_LOS, axis=1, keepdims=True) * n_LOS

					# Components of projected separation
					x = np.sum(s_perp * east_i[n], axis=1)
					y = np.sum(s_perp * north_i[n], axis=1)
					# phi = np.arctan2(x, y)  # angle from north toward east
					phi = np.arctan2(y, x)

					e_plus, e_cross = self.get_ellipticity(e[ind_rbin_i[n]], phi)
					# del phi_sep_dir
					e_plus[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(
							self.r_bins[0]) / sub_box_len_logrp
					)
					del separation_len
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					del mu_r
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					if np.any(ind_r == np.shape(Splus_D)[0]):
						ind_r[np.where(ind_r == np.shape(Splus_D)[0])] = np.shape(Splus_D)[0] - 1
					if np.any(ind_mu_r == np.shape(Splus_D)[1]):
						ind_mu_r[np.where(ind_mu_r == np.shape(Splus_D)[1])] = np.shape(Splus_D)[1] - 1
					np.add.at(Splus_D, (ind_r, ind_mu_r),
							  (weight_i[n] * weight_shape[ind_rbin_i[n]][mask] * e_plus[mask]))
					np.add.at(Scross_D, (ind_r, ind_mu_r),
							  (weight_i[n] * weight_shape[ind_rbin_i[n]][mask] * e_cross[mask]))
					np.add.at(DD, (ind_r, ind_mu_r), weight_i[n] * weight_shape[ind_rbin_i[n]][mask])
					del e_plus, e_cross

		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dmur = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/")
			write_dataset_hdf5(group, dataset_name + Scross_suff, data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + DD_suff, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return Splus_D, DD, separation_bins, mu_r_bins

	def _count_pairs_xi_r_mur_lightcone_brute(self, dataset_name, masks=None, return_output=False,
											  print_num=True, over_h=False, cosmology=None, data_suffix="_DD",
											  chunk_size=1000, num_nodes=1, temp_file_path=None
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
			if "weight" not in masks:
				masks["weight"] = masks["RA"]
			if "weight_shape_sample" not in masks:
				masks["weight_shape_sample"] = masks["RA_shape_sample"]
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi
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
		del redshift, redshift_shape_sample, cosmology

		RA_rad = RA / 180 * np.pi
		RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
		DEC_rad = DEC / 180 * np.pi
		DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi
		n_shape = np.array([np.cos(DEC_shape_sample_rad) * np.cos(RA_shape_sample_rad),
							np.cos(DEC_shape_sample_rad) * np.sin(RA_shape_sample_rad),
							np.sin(DEC_shape_sample_rad)]).transpose()
		del DEC_shape_sample, RA_shape_sample, DEC_shape_sample_rad, RA_shape_sample_rad
		s_shape = n_shape * np.array([LOS_all_shape_sample]).transpose()
		n_pos = np.array([np.cos(DEC_rad) * np.cos(RA_rad),
						  np.cos(DEC_rad) * np.sin(RA_rad),
						  np.sin(DEC_rad)]).transpose()
		del RA, DEC, RA_rad, DEC_rad
		s_pos = np.array([LOS_all]).transpose() * n_pos
		del LOS_all, LOS_all_shape_sample

		for n in np.arange(0, Num_position):
			n_LOS = (n_pos[n] + n_shape) / np.array([np.sqrt(np.sum((n_pos[n] + n_shape) ** 2, axis=1))]).transpose()
			s = s_shape - s_pos[n]
			LOS = self.calculate_dot_product_arrays(s, n_LOS)
			separation_len = np.sqrt(np.sum(s ** 2, axis=1))
			mu_r = LOS / separation_len

			# get the indices for the binning
			mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.r_bins[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			del mu_r
			ind_mu_r = np.array(ind_mu_r, dtype=int)
			np.add.at(DD, (ind_r, ind_mu_r), weight[n] * weight_shape[mask])

		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return DD, separation_bins, mu_r_bins

	def _count_pairs_xi_r_mur_lightcone_tree(self, dataset_name, masks=None, return_output=False,
											 print_num=True, over_h=False, cosmology=None,
											 data_suffix="_DD", chunk_size=1000, num_nodes=1, temp_file_path=None
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
			xi_g_plus, xi_gg, separation_bins, mu_r_bins if no output file is specified

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
			if "weight" not in masks:
				masks["weight"] = masks["RA"]
			if "weight_shape_sample" not in masks:
				masks["weight_shape_sample"] = masks["RA_shape_sample"]
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi
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
		RA_rad = RA / 180 * np.pi
		RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
		DEC_rad = DEC / 180 * np.pi
		DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi
		n_shape = np.array([np.cos(DEC_shape_sample_rad) * np.cos(RA_shape_sample_rad),
							np.cos(DEC_shape_sample_rad) * np.sin(RA_shape_sample_rad),
							np.sin(DEC_shape_sample_rad)]).transpose()
		del DEC_shape_sample, RA_shape_sample, DEC_shape_sample_rad, RA_shape_sample_rad
		s_shape = n_shape * np.array([LOS_all_shape_sample]).transpose()
		n_pos = np.array([np.cos(DEC_rad) * np.cos(RA_rad),
						  np.cos(DEC_rad) * np.sin(RA_rad),
						  np.sin(DEC_rad)]).transpose()
		del RA, DEC, DEC_rad, RA_rad
		s_pos = np.array([LOS_all]).transpose() * n_pos
		del LOS_all, LOS_all_shape_sample
		shape_tree = KDTree(s_shape)
		for i in np.arange(0, Num_position, 100):
			i2 = min(Num_position, i + 100)
			s_pos_i = s_pos[i:i2]
			n_pos_i = n_pos[i:i2]
			weight_i = weight[i:i2]
			pos_tree = KDTree(s_pos_i)
			ind_min_i = pos_tree.query_ball_tree(shape_tree, self.r_min)
			ind_max_i = pos_tree.query_ball_tree(shape_tree, self.r_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(s_pos_i)):  # CHANGE2: loop now over shapes, not positions
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					L = s_pos_i[n] + s_shape[ind_rbin_i[n]]
					n_LOS = L / np.sqrt(np.sum(L ** 2, axis=1))[:, None]
					# n_LOS = (n_pos_i[n] + n_shape[ind_rbin_i[n]]) / np.array(
					# 	[np.sqrt(np.sum((n_pos_i[n] + n_shape[ind_rbin_i[n]]) ** 2, axis=1))]).transpose()
					s = s_shape[ind_rbin_i[n]] - s_pos_i[n]
					LOS = self.calculate_dot_product_arrays(s, n_LOS)
					separation_len = np.sqrt(np.sum(s ** 2, axis=1))  # len of s-pi*nlos ->check
					mu_r = LOS / separation_len

					# get the indices for the binning
					mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(
							self.r_bins[0]) / sub_box_len_logrp
					)
					del separation_len
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / sub_box_len_mu_r - self.mu_r_bins[0] / sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					del mu_r
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					np.add.at(DD, (ind_r, ind_mu_r), weight_i[n] * weight_shape[ind_rbin_i[n]][mask])

		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		mu_r_bins = self.mu_r_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return DD, separation_bins, mu_r_bins


if __name__ == "__main__":
	pass
