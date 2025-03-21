import math
import numpy as np
import h5py
import time
import pickle
import os
import pyccl as ccl
from scipy.special import lpmn
from pathos.multiprocessing import ProcessingPool
from scipy.spatial import KDTree
# from scipy.spatial import cKDTree as KDTree
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.Sim_info import SimInfo
from astropy.table import Table, join, vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.cosmology import LambdaCDM, z_at_value

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureIA(SimInfo):
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

	def measure_projected_correlation(self, masks=None, dataset_name="All_galaxies", return_output=False, print_num=True
									  ):
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
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

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
			np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask] / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask] / (2 * R))
			np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
			del e_plus, e_cross, mask
			np.add.at(DD, (ind_r, ind_pi), 1.0)

		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins

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

	def measure_projected_correlation_tree(self, tree_input=None, masks=None, dataset_name="All_galaxies",
										   return_output=False, print_num=True, dataset_name_tree=None, save_tree=False,
										   file_tree_path=None):
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
		# masking changes the number of galaxies
		Num_position = len(positions)  # number of halos in position sample
		Num_shape = len(positions_shape_sample)  # number of halos in shape sample
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if tree_input != None:
			indices_not_position, indices_shape = tree_input[0], tree_input[1]
			Num_position -= len(indices_not_position)
			Num_shape = len(indices_shape)
			R = 1 - np.mean(e[indices_shape] ** 2) / 2.0
			tree_file = open(f"{file_tree_path}/{dataset_name_tree}.pickle", 'rb')
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		pos_tree = KDTree(positions[:, not_LOS], boxsize=self.boxsize)
		for i in np.arange(0, len(positions_shape_sample), 100):
			i2 = min(len(positions_shape_sample), i + 100)
			positions_shape_sample_i = positions_shape_sample[i:i2]
			axis_direction_i = axis_direction[i:i2]
			e_i = e[i:i2]
			if tree_input != None:
				ind_rbin = pickle.load(tree_file)
				indices_shape_i = indices_shape[(indices_shape >= i) * (indices_shape < i2)] - i
				ind_rbin_i = self.setdiff_omit(ind_rbin, indices_not_position, indices_shape_i)
				positions_shape_sample_i = positions_shape_sample_i[indices_shape_i]
				axis_direction_i = axis_direction_i[indices_shape_i]
				e_i = e_i[indices_shape_i]
			else:
				shape_tree = KDTree(positions_shape_sample_i[:, not_LOS], boxsize=self.boxsize)
				ind_min_i = shape_tree.query_ball_tree(pos_tree, self.separation_min)
				ind_max_i = shape_tree.query_ball_tree(pos_tree, self.separation_max)
				ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
				if save_tree:
					with open(f"{file_tree_path}/w_{self.simname}_tree_{figname_dataset_name}.pickle", 'ab') as handle:
						pickle.dump(ind_rbin_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
			for n in np.arange(0, len(positions_shape_sample_i)):  # CHANGE2: loop now over shapes, not positions
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]  # CHANGE1 & CHANGE2
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, not_LOS]
					LOS = separation[:, LOS_ind]
					separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
					del projected_sep, separation
					phi = np.arccos(
						separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
							n, 1])  # CHANGE2
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)  # CHANGE2
					del phi, separation_dir
					e_plus[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
							LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(
							self.bin_edges[0]) / sub_box_len_logrp
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_pi = np.floor(
						LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
					)  # need length of LOS, so only positive values
					ind_pi = np.array(ind_pi, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask] / (2 * R))
					np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask] / (2 * R))
					del e_plus, e_cross, separation_len, mask
					# np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
					np.add.at(DD, (ind_r, ind_pi), 1.0)
		if tree_input != None:
			tree_file.close()
		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins

	def measure_projected_correlation_tree_single(self, indices):
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		for j in np.arange(0, len(indices), 100):
			i = indices[j]
			i2 = min(indices[-1], i + 100)
			positions_shape_sample_i = self.positions_shape_sample[i:i2]
			axis_direction_i = self.axis_direction[i:i2]
			e_i = self.e[i:i2]

			shape_tree = KDTree(positions_shape_sample_i[:, self.not_LOS], boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(self.pos_tree, self.separation_min)
			ind_max_i = shape_tree.query_ball_tree(self.pos_tree, self.separation_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(positions_shape_sample_i)):  # CHANGE2: loop now over shapes, not positions
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - self.positions[ind_rbin_i[n]]  # CHANGE1 & CHANGE2
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, self.not_LOS]
					LOS = separation[:, self.LOS_ind]
					separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
					del projected_sep, separation
					phi = np.arccos(
						separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
							n, 1])  # CHANGE2
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)  # CHANGE2
					del phi, separation_dir
					e_plus[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
							LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / self.sub_box_len_logrp - np.log10(
							self.bin_edges[0]) / self.sub_box_len_logrp
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_pi = np.floor(
						LOS[mask] / self.sub_box_len_pi - self.pi_bins[0] / self.sub_box_len_pi
					)  # need length of LOS, so only positive values
					ind_pi = np.array(ind_pi, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask] / (2 * self.R))
					np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask] / (2 * self.R))
					del e_plus, e_cross, separation_len, mask
					# np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
					np.add.at(DD, (ind_r, ind_pi), 1.0)

		return Splus_D, Scross_D, DD, variance

	def measure_projected_correlation_tree_multiprocessing(self, num_nodes=9, masks=None,
														   dataset_name="All_galaxies", return_output=False,
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
			self.positions = self.data["Position"]
			self.positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
		else:
			self.positions = self.data["Position"][masks["Position"]]
			self.positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
		# masking changes the number of galaxies
		Num_position = len(self.positions)  # number of halos in position sample
		Num_shape = len(self.positions_shape_sample)  # number of halos in shape sample
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		self.LOS_ind = self.data["LOS"]  # eg 2 for z axis
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], self.LOS_ind, invert=True)]  # eg 0,1 for x&y
		self.e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		self.R = 1 - np.mean(self.e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		self.sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		self.sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		self.pos_tree = KDTree(self.positions[:, self.not_LOS], boxsize=self.boxsize)

		self.multiproc_chuncks = np.array_split(np.arange(len(self.positions_shape_sample)), num_nodes)
		result = ProcessingPool(nodes=num_nodes).map(
			self.measure_projected_correlation_tree_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(num_nodes):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]
			variance += result[i][3]

		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins

	def get_cosmo_points(self, data, cosmology=cosmo):
		'''convert from astropy table of RA, DEC, and redshift to 3D cartesian coordinates in Mpc/h
		Assumes col0=RA, col1=DEC, col2=Z'''
		comoving_dist = cosmo.comoving_distance(data[:, 2]).to(u.Mpc)
		points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data[:, 1]) * u.deg,
													np.asarray(data[:, 0]) * u.deg)  # in Mpc
		return np.asarray(points).transpose() * cosmology.h  # in Mpc/h

	def get_pair_coords(self, obs_pos1, obs_pos2, use_center_origin=True, cosmology=cosmo):
		'''
		Takes in observed positions of galaxy pairs and returns comoving coordinates, in Mpc/h, with the orgin at the center of the pair.
		The first coordinate (x-axis) is along the LOS
		The second coordinate (y-axis) is along 'RA'
		The third coordinate (z-axis) along 'DEC', i.e. aligned with North in origional coordinates.

		INPUT
		-------
		obs_pos1, obs_pos2: table with columns: 'RA', 'DEC', z_column
		use_center_origin: True for coordinate orgin at center of pair, othersise centers on first position
		cosmology: astropy.cosmology

		RETURNS
		-------
		numpy array of cartesian coordinates, in Mpc/h. Shape (2,3)

		'''
		cartesian_coords = self.get_cosmo_points(vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
		# find center position of coordinates
		origin = cartesian_coords[0]
		if use_center_origin == True:
			origin = np.mean(cartesian_coords, axis=0)
		cartesian_coords -= origin
		return cartesian_coords  # in Mpc/h

	def measure_projected_correlation_obs(self, masks=None, dataset_name="All_galaxies",
										  return_output=False, print_num=True):
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
			try:
				q = self.data["q"]
			except:
				e1, e2 = self.data["e1"], self.data["e2"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			try:
				q = self.data["q"][masks["q"]]
			except:
				e1, e2 = self.data["e1"][masks["e1"]], self.data["e2"][masks["e2"]]
		Num_position = len(positions)
		Num_shape = len(positions_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = 2  # redshift column #self.data["LOS"]  # eg 2 for z axis
		not_LOS = [0, 1]  # np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		try:
			e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		except:
			e_comp = e1 + 1j * e2
			e = np.sqrt(e_comp.real ** 2 + e_comp.imag ** 2)
		R = 1 - np.mean(e ** 2) / 2.0  # responsivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = (self.pi_bins[-1] - self.pi_bins[0]) / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		pos_tree = KDTree(positions, boxsize=self.boxsize)
		shape_tree = KDTree(positions_shape_sample, boxsize=self.boxsize)
		ind_min = shape_tree.query_ball_tree(pos_tree, self.separation_min)
		ind_max = shape_tree.query_ball_tree(pos_tree, self.separation_max)
		ind_rbin = self.setdiff2D(ind_max, ind_min)
		del ind_min, ind_max

		for n in np.arange(0, len(positions_shape_sample)):
			# for Splus_D (calculate ellipticities around position sample)
			separation = self.get_pair_coords(positions_shape_sample[n], positions[ind_rbin[n]],
											  use_center_origin=False)
			# separation = positions_shape_sample[n] - positions[ind_rbin[n]]
			# separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
			# separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
			phi = np.arccos(separation_dir[:, 0] * axis_direction[n, 0] + separation_dir[:, 1] * axis_direction[
				n, 1])  # [0,pi]
			e_plus, e_cross = self.get_ellipticity(e[n], phi)
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			# get the indices for the binning
			mask = (separation_len >= self.bin_edges[0]) * (separation_len < self.bin_edges[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.bin_edges[0]) / sub_box_len_logrp
			)
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			ind_pi = np.array(ind_pi, dtype=int)
			np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask] / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask] / (2 * R))
			np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
			np.add.at(DD, (ind_r, ind_pi), 1.0)

		if Num_position == Num_shape:
			DD = DD / 2.0  # auto correlation, all pairs are double

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "auto",
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins

	def measure_projected_correlation_obs_clusters(self, masks=None, dataset_name="All_galaxies", return_output=False,
												   print_num=True, over_h=True, cosmology=None
												   ):
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
			redshift = self.data["Redshift"]
			redshift_shape_sample = self.data["Redshift_shape_sample"]
			RA = self.data["RA"]
			RA_shape_sample = self.data["RA_shape_sample"]
			DEC = self.data["DEC"]
			DEC_shape_sample = self.data["DEC_shape_sample"]
			e1 = self.data["e1"]
			e2 = self.data["e2"]
		else:
			redshift = self.data["Redshift"][masks["Redshift"]]
			redshift_shape_sample = self.data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
			RA = self.data["RA"][masks["RA"]]
			RA_shape_sample = self.data["RA_shape_sample"][masks["RA_shape_sample"]]
			DEC = self.data["DEC"][masks["DEC"]]
			DEC_shape_sample = self.data["DEC_shape_sample"][masks["DEC_shape_sample"]]
			e1 = self.data["e1"][masks["e1"]]
			e2 = self.data["e2"][masks["e2"]]
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
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if cosmology == None:
			print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
			cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
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
			np.add.at(Splus_D, (ind_r, ind_pi), e_plus[mask])
			np.add.at(Scross_D, (ind_r, ind_pi), e_cross[mask])
			del e_plus, e_cross, mask
			np.add.at(DD, (ind_r, ind_pi), 1.0)

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
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/w/xi_gg")
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, DD, separation_bins, pi_bins

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

	def measure_projected_correlation_multipoles(
			self, masks=None, rp_cut=None, dataset_name="All_galaxies", return_output=False, print_num=True
	):
		"""
		Measures the projected correlation function (xi_g_plus) for given coordinates of the position and shape sample
		(Position, Position_shape_sample), the projected axis direction (Axis_Direction), the ratio between projected
		axes, q=b/a (q) and the index of the direction of the line of sight (LOS=2 for z axis).
		Positions are assumed to be given in Mpc.
		:param rp_cut: Limit for minimum r_p value for pairs to be included.
		:param masks: the masks for the data to select only part of the data
		:param dataset_name: the dataset name given in the hdf5 file.
		:param return_output: Output is returned if True, saved to file if False.
		:return: correlation, separation_bins, pi_bins if no output file is specified
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

		if rp_cut == None:
			rp_cut = 0.0
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logr = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		for n in np.arange(0, len(positions)):
			# for Splus_D (calculate ellipticities around position sample)
			separation = positions_shape_sample - positions[n]
			if self.periodicity:
				separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
				separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			separation_dir = (projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
			separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
			mu_r = LOS / separation_len
			del LOS, projected_sep, separation
			phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			e_plus, e_cross = self.get_ellipticity(e, phi)
			del phi, separation_dir
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0
			mu_r[np.isnan(e_plus)] = 0.0

			# get the indices for the binning
			mask = (
					(projected_separation_len > rp_cut)
					* (separation_len >= self.bin_edges[0])
					* (separation_len < self.bin_edges[-1])
			)
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(self.bin_edges[0]) / sub_box_len_logr
			)
			del separation_len, projected_separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.bins_mu_r[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			ind_mu_r = np.array(ind_mu_r, dtype=int)

			np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask] / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask] / (2 * R))
			del e_plus, e_cross, mask, mu_r
			np.add.at(DD, (ind_r, ind_mu_r), 1.0)

		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, corrtype,
					Num_position, Num_shape)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins

	def measure_projected_correlation_multipoles_tree(self, tree_input=None, masks=None, rp_cut=None,
													  dataset_name="All_galaxies", return_output=False, print_num=True,
													  dataset_name_tree=None, save_tree=False, file_tree_path=None):
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
		# masking changes the number of galaxies
		Num_position = len(positions)  # number of halos in position sample
		Num_shape = len(positions_shape_sample)  # number of halos in shape sample

		if rp_cut == None:
			rp_cut = 0.0
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		R = 1 - np.mean(e ** 2) / 2.0  # responsivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logr = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		if tree_input != None:
			indices_not_position, indices_shape = tree_input[0], tree_input[1]
			Num_position -= len(indices_not_position)
			Num_shape = len(indices_shape)
			R = 1 - np.mean(e[indices_shape] ** 2) / 2.0
			tree_file = open(f"{file_tree_path}/{dataset_name_tree}.pickle", 'rb')
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		pos_tree = KDTree(positions, boxsize=self.boxsize)
		for i in np.arange(0, len(positions_shape_sample), 100):
			i2 = min(len(positions_shape_sample), i + 100)
			positions_shape_sample_i = positions_shape_sample[i:i2]
			axis_direction_i = axis_direction[i:i2]
			e_i = e[i:i2]

			if tree_input != None:
				ind_rbin = pickle.load(tree_file)
				indices_shape_i = indices_shape[(indices_shape >= i) * (indices_shape < i2)] - i
				ind_rbin_i = self.setdiff_omit(ind_rbin, indices_not_position, indices_shape_i)
				positions_shape_sample_i = positions_shape_sample_i[indices_shape_i]
				axis_direction_i = axis_direction_i[indices_shape_i]
				e_i = e_i[indices_shape_i]
			else:
				shape_tree = KDTree(positions_shape_sample_i, boxsize=self.boxsize)
				ind_min_i = shape_tree.query_ball_tree(pos_tree, self.separation_min)
				ind_max_i = shape_tree.query_ball_tree(pos_tree, self.separation_max)
				ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
				if save_tree:
					with open(f"{file_tree_path}/m_{self.simname}_tree_{figname_dataset_name}.pickle", 'ab') as handle:
						pickle.dump(ind_rbin_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
			for n in np.arange(0, len(positions_shape_sample_i)):
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, not_LOS]
					LOS = separation[:, LOS_ind]
					projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					separation_dir = (
							projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
					separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
					del separation, projected_sep
					mu_r = LOS / separation_len
					phi = np.arccos(
						separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
							n, 1])  # [0,pi]
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)
					del phi, LOS, separation_dir

					e_plus[np.isnan(e_plus)] = 0.0
					mu_r[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (
							(projected_separation_len > rp_cut)
							* (separation_len >= self.bin_edges[0])
							* (separation_len < self.bin_edges[-1])
					)
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(
							self.bin_edges[0]) / sub_box_len_logr
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / sub_box_len_mu_r - self.bins_mu_r[0] / sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask] / (2 * R))
					np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask] / (2 * R))
					np.add.at(DD, (ind_r, ind_mu_r), 1.0)
					del e_plus, e_cross, mask, separation_len
		if tree_input != None:
			tree_file.close()
		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, corrtype,
					Num_position, Num_shape)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins

	def measure_projected_correlation_multipoles_tree_single(self, indices):
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		for j in np.arange(0, len(indices), 100):
			i = indices[j]
			i2 = min(indices[-1], i + 100)
			positions_shape_sample_i = self.positions_shape_sample[i:i2]
			axis_direction_i = self.axis_direction[i:i2]
			e_i = self.e[i:i2]

			shape_tree = KDTree(positions_shape_sample_i, boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(self.pos_tree, self.separation_min)
			ind_max_i = shape_tree.query_ball_tree(self.pos_tree, self.separation_max)
			ind_rbin_i = self.setdiff2D(ind_max_i, ind_min_i)
			for n in np.arange(0, len(positions_shape_sample_i)):
				if len(ind_rbin_i[n]) > 0:
					# for Splus_D (calculate ellipticities around position sample)
					separation = positions_shape_sample_i[n] - self.positions[ind_rbin_i[n]]
					if self.periodicity:
						separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
						separation[separation < -self.L_0p5] += self.boxsize
					projected_sep = separation[:, self.not_LOS]
					LOS = separation[:, self.LOS_ind]
					projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
					separation_dir = (
							projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
					separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
					del separation, projected_sep
					mu_r = LOS / separation_len
					phi = np.arccos(
						separation_dir[:, 0] * axis_direction_i[n, 0] + separation_dir[:, 1] * axis_direction_i[
							n, 1])  # [0,pi]
					e_plus, e_cross = self.get_ellipticity(e_i[n], phi)
					del phi, LOS, separation_dir

					e_plus[np.isnan(e_plus)] = 0.0
					mu_r[np.isnan(e_plus)] = 0.0
					e_cross[np.isnan(e_cross)] = 0.0

					# get the indices for the binning
					mask = (
							(projected_separation_len > self.rp_cut)
							* (separation_len >= self.bin_edges[0])
							* (separation_len < self.bin_edges[-1])
					)
					ind_r = np.floor(
						np.log10(separation_len[mask]) / self.sub_box_len_logr - np.log10(
							self.bin_edges[0]) / self.sub_box_len_logr
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_mu_r = np.floor(
						mu_r[mask] / self.sub_box_len_mu_r - self.bins_mu_r[0] / self.sub_box_len_mu_r
					)  # need length of LOS, so only positive values
					ind_mu_r = np.array(ind_mu_r, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask] / (2 * self.R))
					np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask] / (2 * self.R))
					np.add.at(DD, (ind_r, ind_mu_r), 1.0)
					del e_plus, e_cross, mask, separation_len
		return Splus_D, Scross_D, DD

	def measure_projected_correlation_multipoles_tree_multiprocessing(self, num_nodes=9, masks=None,
																	  rp_cut=None, dataset_name="All_galaxies",
																	  return_output=False, print_num=True):
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
			self.positions = self.data["Position"]
			self.positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
		else:
			self.positions = self.data["Position"][masks["Position"]]
			self.positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
		# masking changes the number of galaxies
		Num_position = len(self.positions)  # number of halos in position sample
		Num_shape = len(self.positions_shape_sample)  # number of halos in shape sample

		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		if rp_cut == None:
			self.rp_cut = 0.0
		else:
			self.rp_cut = rp_cut
		self.LOS_ind = self.data["LOS"]  # eg 2 for z axis
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], self.LOS_ind, invert=True)]  # eg 0,1 for x&y
		self.e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		self.R = 1 - np.mean(self.e ** 2) / 2.0  # responsivity factor
		L3 = self.boxsize ** 3  # box volume
		self.sub_box_len_logr = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		self.sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		self.pos_tree = KDTree(self.positions, boxsize=self.boxsize)

		self.multiproc_chuncks = np.array_split(np.arange(len(self.positions_shape_sample)), num_nodes)
		result = ProcessingPool(nodes=num_nodes).map(
			self.measure_projected_correlation_multipoles_tree_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(num_nodes):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]

		if Num_position == Num_shape:
			corrtype = "auto"
			DD = DD / 2.0  # auto correlation, all pairs are double
		else:
			corrtype = "cross"

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, corrtype,
					Num_position, Num_shape)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, mu_r_bins

	def measure_projected_correlation_multipoles_obs_clusters(self, masks=None, dataset_name="All_galaxies",
															  return_output=False,
															  print_num=True, over_h=True, cosmology=None, rp_cut=None
															  ):
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
			redshift = self.data["Redshift"]
			redshift_shape_sample = self.data["Redshift_shape_sample"]
			RA = self.data["RA"]
			RA_shape_sample = self.data["RA_shape_sample"]
			DEC = self.data["DEC"]
			DEC_shape_sample = self.data["DEC_shape_sample"]
			e1 = self.data["e1"]
			e2 = self.data["e2"]
		else:
			redshift = self.data["Redshift"][masks["Redshift"]]
			redshift_shape_sample = self.data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
			RA = self.data["RA"][masks["RA"]]
			RA_shape_sample = self.data["RA_shape_sample"][masks["RA_shape_sample"]]
			DEC = self.data["DEC"][masks["DEC"]]
			DEC_shape_sample = self.data["DEC_shape_sample"][masks["DEC_shape_sample"]]
			e1 = self.data["e1"][masks["e1"]]
			e2 = self.data["e2"][masks["e2"]]
		Num_position = len(RA)
		Num_shape = len(RA_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		if rp_cut == None:
			rp_cut = 0.0
		sub_box_len_logr = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		if cosmology == None:
			print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
			cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
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
			separation = np.array([dx, dy, LOS])
			if over_h:
				projected_sep *= h
			projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=0))
			separation_len = np.sqrt(np.sum(separation ** 2, axis=0))
			separation_dir = (projected_sep / projected_separation_len)  # normalisation of rp
			mu_r = LOS / separation_len
			del projected_sep
			phi_sep_dir = np.arctan2(separation_dir[1], separation_dir[0])
			phi = phi_axis_dir - phi_sep_dir
			# np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			del separation_dir
			e_plus, e_cross = self.get_ellipticity(-e, phi)
			del phi
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0
			mu_r[np.isnan(e_plus)] = 0.0

			# get the indices for the binning
			mask = (
					(projected_separation_len > rp_cut)
					* (separation_len >= self.bin_edges[0])
					* (separation_len < self.bin_edges[-1])
			)
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(self.bin_edges[0]) / sub_box_len_logr
			)
			del separation_len, projected_separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.bins_mu_r[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			ind_mu_r = np.array(ind_mu_r, dtype=int)
			del LOS
			np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask])
			np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask])
			del e_plus, e_cross, mask
			np.add.at(DD, (ind_r, ind_mu_r), 1.0)

		# if Num_position == Num_shape:
		# 	DD = DD / 2.0  # auto correlation, all pairs are double

		DD[np.where(DD == 0)] = 1

		correlation = Splus_D / DD
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, DD, separation_bins, mu_r_bins

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

	def measure_jackknife_errors(
			self, masks=None, corr_type=["both", "multipoles"], dataset_name="All_galaxies", L_subboxes=3, rp_cut=None,
			tree_saved=True, file_tree_path=None, remove_tree_file=True, num_nodes=None
	):
		"""
		Measures the errors in the projected correlation function using the jackknife method.
		The box is divided into L_subboxes^3 subboxes; the correlation function is calculated omitting one box at a time.
		Then the standard deviation is taken for wg+ and the covariance matrix is calculated for the multipoles.
		:param rp_cut: Limit for minimum r_p value for pairs to be included. (Needed for measure_projected_correlation_multipoles)
		:param corr_type: Array with two entries. For first choose from [gg, g+, both], for second from [w, multipoles]
		:param dataset_name: Name of the dataset
		:param L_subboxes: Integer by which the length of the side of the mox should be divided.
		:return:
		"""
		if corr_type[0] == "both":
			data = [corr_type[1] + "_g_plus", corr_type[1] + "_gg"]
		elif corr_type[0] == "g+":
			data = [corr_type[1] + "_g_plus"]
		elif corr_type[0] == "gg":
			data = [corr_type[1] + "_gg"]
		else:
			raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		L_sub = self.L_0p5 * 2.0 / L_subboxes
		num_box = 0
		for i in np.arange(0, L_subboxes):
			for j in np.arange(0, L_subboxes):
				for k in np.arange(0, L_subboxes):
					x_bounds = [i * L_sub, (i + 1) * L_sub]
					y_bounds = [j * L_sub, (j + 1) * L_sub]
					z_bounds = [k * L_sub, (k + 1) * L_sub]
					x_mask = (self.data["Position"][:, 0] > x_bounds[0]) * (self.data["Position"][:, 0] < x_bounds[1])
					y_mask = (self.data["Position"][:, 1] > y_bounds[0]) * (self.data["Position"][:, 1] < y_bounds[1])
					z_mask = (self.data["Position"][:, 2] > z_bounds[0]) * (self.data["Position"][:, 2] < z_bounds[1])
					mask_position = np.invert(
						x_mask * y_mask * z_mask
					)  # mask that is True for all positions not in the subbox
					if self.Num_position == self.Num_shape:
						mask_shape = mask_position
					else:
						x_mask = (self.data["Position_shape_sample"][:, 0] > x_bounds[0]) * (
								self.data["Position_shape_sample"][:, 0] < x_bounds[1]
						)
						y_mask = (self.data["Position_shape_sample"][:, 1] > y_bounds[0]) * (
								self.data["Position_shape_sample"][:, 1] < y_bounds[1]
						)
						z_mask = (self.data["Position_shape_sample"][:, 2] > z_bounds[0]) * (
								self.data["Position_shape_sample"][:, 2] < z_bounds[1]
						)
						mask_shape = np.invert(
							x_mask * y_mask * z_mask
						)  # mask that is True for all positions not in the subbox
					if tree_saved:
						if masks != None:
							indices_shape = np.where(mask_shape[masks["Position_shape_sample"]])[0]
							mask_not_position = np.invert(mask_position[masks["Position"]])
							indices_not_position = np.where(mask_not_position)[0]
							masks_total = {
								"Position": masks["Position"],
								"Position_shape_sample": masks["Position_shape_sample"],
								"Axis_Direction": masks["Position_shape_sample"],
								"q": masks["Position_shape_sample"],
							}
						else:
							indices_shape = np.where(mask_shape)[0]
							mask_not_position = np.invert(mask_position)
							indices_not_position = np.where(mask_not_position)[0]
							masks_total = None
						tree_input = [indices_not_position, indices_shape]

					else:
						if masks != None:
							mask_position = mask_position * masks["Position"]
							mask_shape = mask_shape * masks["Position_shape_sample"]
						tree_input = None
						masks_total = {
							"Position": mask_position,
							"Position_shape_sample": mask_shape,
							"Axis_Direction": mask_shape,
							"q": mask_shape,
						}
					if corr_type[1] == "multipoles":
						if num_nodes == None:
							self.measure_projected_correlation_multipoles_tree(
								tree_input=tree_input,
								masks=masks_total,
								rp_cut=rp_cut,
								dataset_name=dataset_name + "_" + str(num_box),
								print_num=False,
								save_tree=False,
								dataset_name_tree=f"m_{self.simname}_tree_{figname_dataset_name}",
								file_tree_path=file_tree_path,
							)
						else:
							self.measure_projected_correlation_multipoles_tree_multiprocessing(
								masks=masks_total,
								rp_cut=rp_cut,
								dataset_name=dataset_name + "_" + str(num_box),
								print_num=False,
								num_nodes=num_nodes
							)
						self.measure_multipoles(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(num_box))
					else:
						if num_nodes == None:
							self.measure_projected_correlation_tree(
								tree_input=tree_input,
								masks=masks_total,
								dataset_name=dataset_name + "_" + str(num_box),
								print_num=False,
								save_tree=False,
								dataset_name_tree=f"w_{self.simname}_tree_{figname_dataset_name}",
								file_tree_path=file_tree_path,
							)
						else:
							self.measure_projected_correlation_tree_multiprocessing(
								masks=masks_total,
								dataset_name=dataset_name + "_" + str(num_box),
								print_num=False,
								num_nodes=num_nodes
							)
						self.measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(num_box))

					num_box += 1
		if remove_tree_file and tree_saved:
			if corr_type[1] == "multipoles":
				os.remove(
					f"{file_tree_path}/m_{self.simname}_tree_{figname_dataset_name}.pickle")  # removes temp pickle file
			else:
				os.remove(
					f"{file_tree_path}/w_{self.simname}_tree_{figname_dataset_name}.pickle")  # removes temp pickle file
		covs, stds = [], []
		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file[f"Snapshot_{self.snapshot}/" + data[d]]
			# calculating mean of the datavectors
			mean_multipoles = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				mean_multipoles += group_multipoles[dataset_name + "_" + str(b)]
			mean_multipoles /= num_box

			# calculation the covariance matrix (multipoles) and the standard deviation (sqrt of diag of cov)
			cov = np.zeros((self.num_bins_r, self.num_bins_r))
			std = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				std += (group_multipoles[dataset_name + "_" + str(b)] - mean_multipoles) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group_multipoles[dataset_name + "_" + str(b)] - mean_multipoles) * (
							group_multipoles[dataset_name + "_" + str(b)][i] - mean_multipoles[i]
					)
			std *= (num_box - 1) / num_box  # see Singh 2023
			std = np.sqrt(std)  # size of errorbars
			cov *= (num_box - 1) / num_box  # cov not sqrt so to get std, sqrt of diag would need to be taken
			data_file.close()
			if self.output_file_name != None:
				output_file = h5py.File(self.output_file_name, "a")
				group_multipoles = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
			else:
				covs.append(cov)
				stds.append(std)
		if self.output_file_name != None:
			return
		else:
			return covs, stds

	def measure_jackknife_errors_multiprocessing(
			self, masks=None, corr_type=["both", "multipoles"], dataset_name="All_galaxies", L_subboxes=3, rp_cut=None,
			num_nodes=4, twoD=False, tree=True, tree_saved=True, file_tree_path=None, remove_tree_file=True
	):
		"""
		Measures the errors in the projected correlation function using the jackknife method, using multiple CPU cores.
		The box is divided into L_subboxes^3 subboxes; the correlation function is calculated omitting one box at a time.
		Then the standard deviation is taken for wg+ and the covariance matrix is calculated for the multipoles.
		:param twoD: Divide box into L_subboxes^2, no division in z-direction.
		:param num_nodes: Number of CPU nodes to use in multiprocessing.
		:param dataset_name: Name of the dataset
		:param L_subboxes: Integer by which the length of the side of the mox should be divided.
		:param rp_cut: Limit for minimum r_p value for pairs to be included. (Needed for measure_projected_correlation_multipoles)
		:param corr_type: Array with two entries. For first choose from [gg, g+, both], for second from [w, multipoles]
		:return:
		"""
		if corr_type[0] == "both":
			data = [corr_type[1] + "_g_plus", corr_type[1] + "_gg"]
			corr_type_suff = ["_g_plus", "_gg"]
		elif corr_type[0] == "g+":
			data = [corr_type[1] + "_g_plus"]
			corr_type_suff = ["_g_plus"]
		elif corr_type[0] == "gg":
			data = [corr_type[1] + "_gg"]
			corr_type_suff = ["_gg"]
		else:
			raise KeyError("Unknown value for first entry of corr_type. Choose from [g+, gg, both]")
		if corr_type[1] == "multipoles":
			bin_var_names = ["r", "mu_r"]
		elif corr_type[1] == "w":
			bin_var_names = ["rp", "pi"]
		else:
			raise KeyError("Unknown value for second entry of corr_type. Choose from [multipoles, w_g_plus]")
		L_sub = self.boxsize / L_subboxes
		if twoD:
			z_L_sub = 1
		else:
			z_L_sub = L_subboxes
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		num_box = 0
		args_xi_g_plus, args_multipoles, tree_args = [], [], []
		for i in np.arange(0, L_subboxes):
			for j in np.arange(0, L_subboxes):
				for k in np.arange(0, z_L_sub):
					x_bounds = [i * L_sub, (i + 1) * L_sub]
					y_bounds = [j * L_sub, (j + 1) * L_sub]
					if twoD:
						z_bounds = [0.0, self.boxsize]
					else:
						z_bounds = [k * L_sub, (k + 1) * L_sub]
					x_mask = (self.data["Position"][:, 0] > x_bounds[0]) * (self.data["Position"][:, 0] < x_bounds[1])
					y_mask = (self.data["Position"][:, 1] > y_bounds[0]) * (self.data["Position"][:, 1] < y_bounds[1])
					z_mask = (self.data["Position"][:, 2] > z_bounds[0]) * (self.data["Position"][:, 2] < z_bounds[1])
					mask_position = np.invert(
						x_mask * y_mask * z_mask
					)  # mask that is True for all positions not in the subbox
					if self.Num_position == self.Num_shape:
						mask_shape = mask_position
					else:
						x_mask = (self.data["Position_shape_sample"][:, 0] > x_bounds[0]) * (
								self.data["Position_shape_sample"][:, 0] < x_bounds[1]
						)
						y_mask = (self.data["Position_shape_sample"][:, 1] > y_bounds[0]) * (
								self.data["Position_shape_sample"][:, 1] < y_bounds[1]
						)
						z_mask = (self.data["Position_shape_sample"][:, 2] > z_bounds[0]) * (
								self.data["Position_shape_sample"][:, 2] < z_bounds[1]
						)
						mask_shape = np.invert(
							x_mask * y_mask * z_mask
						)  # mask that is True for all positions not in the subbox
					if tree_saved:
						if masks != None:
							indices_shape = np.where(mask_shape[masks["Position_shape_sample"]])[0]
							mask_not_position = np.invert(mask_position[masks["Position"]])
							indices_not_position = np.where(mask_not_position)[0]
							masks_total = {
								"Position": masks["Position"],
								"Position_shape_sample": masks["Position_shape_sample"],
								"Axis_Direction": masks["Position_shape_sample"],
								"q": masks["Position_shape_sample"],
							}
						else:
							indices_shape = np.where(mask_shape)[0]
							mask_not_position = np.invert(mask_position)
							indices_not_position = np.where(mask_not_position)[0]
							masks_total = None
						tree_input = [indices_not_position, indices_shape]
					else:
						if masks != None:
							mask_position = mask_position * masks["Position"]
							mask_shape = mask_shape * masks["Position_shape_sample"]
						tree_input = None
						masks_total = {
							"Position": mask_position,
							"Position_shape_sample": mask_shape,
							"Axis_Direction": mask_shape,
							"q": mask_shape,
						}
					if corr_type[1] == "multipoles":
						tree_args.append(tree_input)
						args_xi_g_plus.append(
							(
								masks_total,
								rp_cut,
								dataset_name + "_" + str(num_box),
								True,
								False,
								f"m_{self.simname}_tree_{figname_dataset_name}",
								False,
								file_tree_path,
							)
						)
					else:
						tree_args.append(tree_input)
						args_xi_g_plus.append(
							(
								masks_total,
								dataset_name + "_" + str(num_box),
								True,
								False,
								f"w_{self.simname}_tree_{figname_dataset_name}",
								False,
								file_tree_path,
							)
						)
					args_multipoles.append([corr_type[0], dataset_name + "_" + str(num_box)])

					num_box += 1
		args_xi_g_plus = np.array(args_xi_g_plus)
		args_multipoles = np.array(args_multipoles)
		multiproc_chuncks = np.array_split(np.arange(num_box), np.ceil(num_box / num_nodes))
		for chunck in multiproc_chuncks:
			chunck = np.array(chunck, dtype=int)
			if corr_type[1] == "multipoles":
				if tree:
					result = ProcessingPool(nodes=len(chunck)).map(
						self.measure_projected_correlation_multipoles_tree,
						tree_args[min(chunck):max(chunck) + 1],
						args_xi_g_plus[chunck][:, 0],
						args_xi_g_plus[chunck][:, 1],
						args_xi_g_plus[chunck][:, 2],
						args_xi_g_plus[chunck][:, 3],
						args_xi_g_plus[chunck][:, 4],
						args_xi_g_plus[chunck][:, 5],
						args_xi_g_plus[chunck][:, 6],
						args_xi_g_plus[chunck][:, 7],
					)
				else:
					result = ProcessingPool(nodes=len(chunck)).map(
						self.measure_projected_correlation_multipoles,
						args_xi_g_plus[chunck][:, 0],
						args_xi_g_plus[chunck][:, 1],
						args_xi_g_plus[chunck][:, 2],
						args_xi_g_plus[chunck][:, 3],
						args_xi_g_plus[chunck][:, 4],
					)
			else:
				if tree:
					result = ProcessingPool(nodes=len(chunck)).map(
						self.measure_projected_correlation_tree,
						tree_args[min(chunck):max(chunck) + 1],
						args_xi_g_plus[chunck][:, 0],
						args_xi_g_plus[chunck][:, 1],
						args_xi_g_plus[chunck][:, 2],
						args_xi_g_plus[chunck][:, 3],
						args_xi_g_plus[chunck][:, 4],
						args_xi_g_plus[chunck][:, 5],
						args_xi_g_plus[chunck][:, 6],
					)

				else:
					result = ProcessingPool(nodes=len(chunck)).map(
						self.measure_projected_correlation,
						args_xi_g_plus[chunck][:, 0],
						args_xi_g_plus[chunck][:, 1],
						args_xi_g_plus[chunck][:, 2],
						args_xi_g_plus[chunck][:, 3],
					)

			output_file = h5py.File(self.output_file_name, "a")
			for i in np.arange(0, len(chunck)):
				for j, data_j in enumerate(data):
					group_xigplus = create_group_hdf5(
						output_file, f"Snapshot_{self.snapshot}/" + corr_type[1] + "/xi" + corr_type_suff[j]
					)
					write_dataset_hdf5(group_xigplus, f"{dataset_name}_{chunck[i]}", data=result[i][j])
					write_dataset_hdf5(
						group_xigplus, f"{dataset_name}_{chunck[i]}_{bin_var_names[0]}", data=result[i][2]
					)
					write_dataset_hdf5(
						group_xigplus, f"{dataset_name}_{chunck[i]}_{bin_var_names[1]}", data=result[i][3]
					)
					write_dataset_hdf5(group_xigplus, f"{dataset_name}_{chunck[i]}_sigmasq", data=result[i][3])
			output_file.close()
		if remove_tree_file and tree_saved:
			if corr_type[1] == "multipoles":
				os.remove(
					f"{file_tree_path}/m_{self.simname}_tree_{figname_dataset_name}.pickle")  # removes temp pickle file
			else:
				os.remove(
					f"{file_tree_path}/w_{self.simname}_tree_{figname_dataset_name}.pickle")  # removes temp pickle file
		for i in np.arange(0, num_box):
			if corr_type[1] == "multipoles":
				self.measure_multipoles(corr_type=args_multipoles[i][0], dataset_name=args_multipoles[i][1])
			else:
				self.measure_w_g_i(corr_type=args_multipoles[i][0], dataset_name=args_multipoles[i][1])
		covs, stds = [], []
		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file[f"Snapshot_{self.snapshot}/" + data[d]]
			# calculating mean of the datavectors
			mean_multipoles = np.zeros((self.num_bins_r))
			for b in np.arange(0, num_box):
				mean_multipoles += group_multipoles[dataset_name + "_" + str(b)]
			mean_multipoles /= num_box

			# calculation the covariance matrix (multipoles) and the standard deviation (wg+)
			cov = np.zeros((self.num_bins_r, self.num_bins_r))
			std = np.zeros(self.num_bins_r)
			for b in np.arange(0, num_box):
				std += (group_multipoles[dataset_name + "_" + str(b)] - mean_multipoles) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group_multipoles[dataset_name + "_" + str(b)] - mean_multipoles) * (
							group_multipoles[dataset_name + "_" + str(b)][i] - mean_multipoles[i]
					)
			std *= (num_box - 1) / num_box  # see Singh 2023
			std = np.sqrt(std)  # size of errorbars
			cov *= (num_box - 1) / num_box  # cov not sqrt so to get std, sqrt of diag would need to be taken
			data_file.close()
			if self.output_file_name != None:
				output_file = h5py.File(self.output_file_name, "a")
				group_multipoles = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
			else:
				covs.append(cov)
				stds.append(std)
		if self.output_file_name != None:
			return
		else:
			return covs, stds

	def measure_jackknife_realisations_obs(
			self, patches_pos, patches_shape, masks=None, corr_type=["both", "multipoles"], dataset_name="All_galaxies",
			rp_cut=None, over_h=False, cosmology=None,
	):
		"""
		Measures the errors in the projected correlation function using the jackknife method.
		The box is divided into L_subboxes^3 subboxes; the correlation function is calculated omitting one box at a time.
		Then the standard deviation is taken for wg+ and the covariance matrix is calculated for the multipoles.
		:param rp_cut: Limit for minimum r_p value for pairs to be included. (Needed for measure_projected_correlation_multipoles)
		:param corr_type: Array with two entries. For first choose from [gg, g+, both], for second from [w, multipoles]
		:param dataset_name: Name of the dataset
		:param L_subboxes: Integer by which the length of the side of the mox should be divided.
		:return:
		"""
		if corr_type[0] == "both":
			data = [corr_type[1] + "_g_plus", corr_type[1] + "_gg"]
		elif corr_type[0] == "g+":
			data = [corr_type[1] + "_g_plus"]
		elif corr_type[0] == "gg":
			data = [corr_type[1] + "_gg"]
		else:
			raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")

		min_patch, max_patch = int(min(patches_pos)), int(max(patches_pos))
		if min(patches_shape) != min_patch:
			print(
				"Warning! Minimum patch number of shape sample is not equal to minimum patch number of position sample.")
		if max(patches_shape) != max_patch:
			print(
				"Warning! Maximum patch number of shape sample is not equal to maximum patch number of position sample.")
		print(
			f"Calculating jackknife realisations for {max_patch} patches for {dataset_name}.")

		for i in np.arange(min_patch, max_patch + 1):
			mask_position = (patches_pos != i)
			mask_shape = (patches_shape != i)
			if masks != None:
				mask_position = mask_position * masks["Redshift"]
				mask_shape = mask_shape * masks["Redshift_shape_sample"]
			masks_total = {
				"Redshift": mask_position,
				"Redshift_shape_sample": mask_shape,
				"RA": mask_position,
				"RA_shape_sample": mask_shape,
				"DEC": mask_position,
				"DEC_shape_sample": mask_shape,
				"e1": mask_shape,
				"e2": mask_shape,
			}
			if corr_type[1] == "multipoles":
				self.measure_projected_correlation_multipoles_obs_clusters(
					masks=masks_total,
					rp_cut=rp_cut,
					dataset_name=dataset_name + "_" + str(i),
					print_num=False,
					over_h=over_h,
					cosmology=cosmology,
				)

				self.measure_multipoles(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(i))
			else:
				self.measure_projected_correlation_obs_clusters(
					masks=masks_total,
					dataset_name=dataset_name + "_" + str(i),
					print_num=False,
					over_h=over_h,
					cosmology=cosmology,
				)
				self.measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(i))

		return

	def measure_jackknife_errors_obs(
			self, max_patch, min_patch=1, corr_type=["both", "multipoles"], dataset_name="All_galaxies",
			randoms_suf="_randoms"
	):
		"""
		Measures the errors in the projected correlation function using the jackknife method.
		The box is divided into L_subboxes^3 subboxes; the correlation function is calculated omitting one box at a time.
		Then the standard deviation is taken for wg+ and the covariance matrix is calculated for the multipoles.
		:param rp_cut: Limit for minimum r_p value for pairs to be included. (Needed for measure_projected_correlation_multipoles)
		:param corr_type: Array with two entries. For first choose from [gg, g+, both], for second from [w, multipoles]
		:param dataset_name: Name of the dataset
		:param L_subboxes: Integer by which the length of the side of the mox should be divided.
		:return:
		"""
		if corr_type[0] == "both":
			data = [corr_type[1] + "_g_plus", corr_type[1] + "_gg"]
		elif corr_type[0] == "g+":
			data = [corr_type[1] + "_g_plus"]
		elif corr_type[0] == "gg":
			data = [corr_type[1] + "_gg"]
		else:
			raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		num_patches = max_patch - min_patch + 1
		print(
			f"Calculating jackknife errors for {num_patches} patches for {dataset_name} with {dataset_name}{randoms_suf} as randoms.")

		covs, stds = [], []
		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file[f"Snapshot_{self.snapshot}/" + data[d]]
			# calculating mean of the datavectors
			mean_multipoles = np.zeros(self.num_bins_r)
			for b in np.arange(min_patch, max_patch + 1):
				mean_multipoles += (group_multipoles[f"{dataset_name}_{b}"][:] - group_multipoles[
																					 f"{dataset_name}{randoms_suf}_{b}"][
																				 :])
			mean_multipoles /= num_patches

			# calculation the covariance matrix (multipoles) and the standard deviation (sqrt of diag of cov)
			cov = np.zeros((self.num_bins_r, self.num_bins_r))
			std = np.zeros(self.num_bins_r)
			for b in np.arange(min_patch, max_patch + 1):
				correlation = group_multipoles[f"{dataset_name}_{b}"][:] - group_multipoles[
																			   f"{dataset_name}{randoms_suf}_{b}"][:]
				std += (correlation - mean_multipoles) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (correlation - mean_multipoles) * (correlation[i] - mean_multipoles[i])

			std *= (num_patches - 1) / num_patches  # see Singh 2023
			std = np.sqrt(std)  # size of errorbars
			cov *= (num_patches - 1) / num_patches  # cov not sqrt so to get std, sqrt of diag would need to be taken
			data_file.close()
			if self.output_file_name != None:
				output_file = h5py.File(self.output_file_name, "a")
				group_multipoles = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_mean_" + str(num_patches), data=mean_multipoles)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_patches), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_patches), data=cov)
				output_file.close()
			else:
				covs.append(cov)
				stds.append(std)
		if self.output_file_name != None:
			return
		else:
			return covs, stds

	def measure_jackknife_realisations_obs_multiprocessing(
			self, patches_pos, patches_shape, masks=None, corr_type=["both", "multipoles"], dataset_name="All_galaxies",
			rp_cut=None, over_h=False, num_nodes=4, cosmology=None
	):
		"""
		Measures the errors in the projected correlation function using the jackknife method, using multiple CPU cores.
		The box is divided into L_subboxes^3 subboxes; the correlation function is calculated omitting one box at a time.
		Then the standard deviation is taken for wg+ and the covariance matrix is calculated for the multipoles.
		:param twoD: Divide box into L_subboxes^2, no division in z-direction.
		:param num_nodes: Number of CPU nodes to use in multiprocessing.
		:param dataset_name: Name of the dataset
		:param L_subboxes: Integer by which the length of the side of the mox should be divided.
		:param rp_cut: Limit for minimum r_p value for pairs to be included. (Needed for measure_projected_correlation_multipoles)
		:param corr_type: Array with two entries. For first choose from [gg, g+, both], for second from [w, multipoles]
		:return:
		"""
		if corr_type[0] == "both":
			data = [corr_type[1] + "_g_plus", corr_type[1] + "_gg"]
			corr_type_suff = ["_g_plus", "_gg"]
		elif corr_type[0] == "g+":
			data = [corr_type[1] + "_g_plus"]
			corr_type_suff = ["_g_plus"]
		elif corr_type[0] == "gg":
			data = [corr_type[1] + "_gg"]
			corr_type_suff = ["_gg"]
		else:
			raise KeyError("Unknown value for first entry of corr_type. Choose from [g+, gg, both]")
		if corr_type[1] == "multipoles":
			bin_var_names = ["r", "mu_r"]
		elif corr_type[1] == "w":
			bin_var_names = ["rp", "pi"]
		else:
			raise KeyError("Unknown value for second entry of corr_type. Choose from [multipoles, w_g_plus]")
		min_patch, max_patch = int(min(patches_pos)), int(max(patches_pos))
		num_patches = max_patch - min_patch + 1
		if min(patches_shape) != min_patch:
			print(
				"Warning! Minimum patch number of shape sample is not equal to minimum patch number of position sample.")
		if max(patches_shape) != max_patch:
			print(
				"Warning! Maximum patch number of shape sample is not equal to maximum patch number of position sample.")
		args_xi_g_plus, args_multipoles, tree_args = [], [], []
		for i in np.arange(min_patch, max_patch + 1):
			mask_position = (patches_pos != i)
			mask_shape = (patches_shape != i)
			if masks != None:
				mask_position = mask_position * masks["Redshift"]
				mask_shape = mask_shape * masks["Redshift_shape_sample"]
			masks_total = {
				"Redshift": mask_position,
				"Redshift_shape_sample": mask_shape,
				"RA": mask_position,
				"RA_shape_sample": mask_shape,
				"DEC": mask_position,
				"DEC_shape_sample": mask_shape,
				"e1": mask_shape,
				"e2": mask_shape,
			}
			if corr_type[1] == "multipoles":
				args_xi_g_plus.append(
					(
						masks_total,
						dataset_name + "_" + str(i),
						True,
						False,
						over_h,
						cosmology,
						rp_cut,
					)
				)
			else:
				args_xi_g_plus.append(
					(
						masks_total,
						dataset_name + "_" + str(i),
						True,
						False,
						over_h,
						cosmology,
					)
				)
			args_multipoles.append([corr_type[0], dataset_name + "_" + str(i)])

		args_xi_g_plus = np.array(args_xi_g_plus)
		args_multipoles = np.array(args_multipoles)
		multiproc_chuncks = np.array_split(np.arange(num_patches), np.ceil(num_patches / num_nodes))
		for chunck in multiproc_chuncks:
			chunck = np.array(chunck, dtype=int)
			if corr_type[1] == "multipoles":
				result = ProcessingPool(nodes=len(chunck)).map(
					self.measure_projected_correlation_multipoles_obs_clusters,
					args_xi_g_plus[chunck][:, 0],
					args_xi_g_plus[chunck][:, 1],
					args_xi_g_plus[chunck][:, 2],
					args_xi_g_plus[chunck][:, 3],
					args_xi_g_plus[chunck][:, 4],
					args_xi_g_plus[chunck][:, 5],
					args_xi_g_plus[chunck][:, 4],
				)
			else:
				result = ProcessingPool(nodes=len(chunck)).map(
					self.measure_projected_correlation_obs_clusters,
					args_xi_g_plus[chunck][:, 0],
					args_xi_g_plus[chunck][:, 1],
					args_xi_g_plus[chunck][:, 2],
					args_xi_g_plus[chunck][:, 3],
					args_xi_g_plus[chunck][:, 4],
					args_xi_g_plus[chunck][:, 5],
				)

			output_file = h5py.File(self.output_file_name, "a")
			for i in np.arange(0, len(chunck)):
				for j, data_j in enumerate(data):
					group_xigplus = create_group_hdf5(
						output_file, f"Snapshot_{self.snapshot}/" + corr_type[1] + "/xi" + corr_type_suff[j]
					)
					write_dataset_hdf5(group_xigplus, f"{dataset_name}_{chunck[i] + min_patch}", data=result[i][j])
					write_dataset_hdf5(
						group_xigplus, f"{dataset_name}_{chunck[i] + min_patch}_{bin_var_names[0]}", data=result[i][2]
					)
					write_dataset_hdf5(
						group_xigplus, f"{dataset_name}_{chunck[i] + min_patch}_{bin_var_names[1]}", data=result[i][3]
					)
			# write_dataset_hdf5(group_xigplus, f"{dataset_name}_{chunck[i]}_sigmasq", data=result[i][3])
			output_file.close()

		for i in np.arange(0, num_patches):
			if corr_type[1] == "multipoles":
				self.measure_multipoles(corr_type=args_multipoles[i, 0], dataset_name=args_multipoles[i, 1])
			else:
				self.measure_w_g_i(corr_type=args_multipoles[i, 0], dataset_name=args_multipoles[i, 1])
		return

	def measure_covariance_multiple_datasets(self, corr_type, dataset_names, num_box=3, return_output=False):
		"""
		Combines the jackknife measurements for different datasets into one covariance matrix.
		Author: Marta Garcia Escobar (starting from measure_jackknife_errors code); updated
		:param corr_type: Takes "w_g_plus" or "multipoles_g_plus".
		:param dataset_names: List of the dataset names. If there is only one value, it calculates the covariance matrix with itself.
		:param num_box: Number of boxes.
		"""
		# check if corr_type is valid
		valid_corr_types = ["w_g_plus", "multipoles_g_plus"]
		if corr_type not in valid_corr_types:
			raise ValueError("corr_type must be 'w_g_plus' or 'multipoles_g_plus'.")

		data_file = h5py.File(self.output_file_name, "a")
		group = data_file[f"Snapshot_{self.snapshot}/" + corr_type]

		mean_list = []  # list of arrays

		for dataset_name in dataset_names:
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
			for b in np.arange(0, num_box):
				std += (group[dataset_name + "_" + str(b)] - mean_list[0]) ** 2
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group[dataset_name + "_" + str(b)] - mean_list[0]) * (
							group[dataset_name + "_" + str(b)][i] - mean_list[0][i]
					)
		elif len(dataset_names) == 2:
			for b in np.arange(0, num_box):
				std += (group[dataset_names[0] + "_" + str(b)] - mean_list[0]) * (
						group[dataset_names[1] + "_" + str(b)] - mean_list[1])
				for i in np.arange(self.num_bins_r):
					cov[:, i] += (group[dataset_names[0] + "_" + str(b)] - mean_list[0]) * (
							group[dataset_names[1] + "_" + str(b)][i] - mean_list[1][i]
					)
		else:
			raise KeyError("Too many datasets given, choose either 1 or 2")

		std *= (num_box - 1) / num_box  # see Singh 2023
		std = np.sqrt(std)  # size of errorbars
		cov *= (num_box - 1) / num_box  # cov not sqrt so to get std, sqrt of diag would need to be taken

		data_file.close()

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"Snapshot_{self.snapshot}/" + corr_type)
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
										   retun_output=False):
		'''
		Function that creates the full covariance matrix for all 3 projections by combining previously obtained jackknife information.
		Generalised from Marta Garcia Escobar's code.
		:param corr_type:
		:param dataset_names:
		:param num_box:
		:return:
		'''
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[0], dataset_names[1]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[0], dataset_names[2]], num_box=num_box)
		self.measure_covariance_multiple_datasets(corr_type=corr_type,
												  dataset_names=[dataset_names[1], dataset_names[2]], num_box=num_box)

		# import needed datasets
		output_file = h5py.File(self.output_file_name, "a")
		group = output_file[f"Snapshot_{self.snapshot}/{corr_type}"]

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

		if retun_output:
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
