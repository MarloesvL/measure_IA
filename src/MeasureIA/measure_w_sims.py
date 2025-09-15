import numpy as np
import h5py
import pickle
from pathos.multiprocessing import ProcessingPool
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureWBox(MeasureIABase):
	"""Class that contains all methods for the measurements of xi_gg and xi_g+ for w_gg and w_g+ with carthesian
	simulation data.

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
	):
		"""
		The __init__ method of the MeasureWSimulations class.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.

		"""
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		return

	def _measure_xi_rp_pi_sims_brute(self, masks=None, dataset_name="All_galaxies", return_output=False,
									 print_num=True,
									 jk_group_name=""):
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
		jk_group_name :
		     (Default value = "")

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
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		Num_position = len(positions)
		Num_shape = len(positions_shape_sample)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
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
			with np.errstate(invalid='ignore'):
				separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
				del projected_sep
				phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			del separation_dir
			e_plus, e_cross = self.get_ellipticity(e, phi)
			del phi
			e_plus[np.isnan(e_plus)] = 0.0
			e_cross[np.isnan(e_cross)] = 0.0

			# get the indices for the binning
			mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * (
					LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(self.r_bins[0]) / sub_box_len_logrp
			)
			del separation_len
			ind_r = np.array(ind_r, dtype=int)
			ind_pi = np.floor(
				LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
			)  # need length of LOS, so only positive values
			del LOS
			ind_pi = np.array(ind_pi, dtype=int)
			np.add.at(Splus_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_pi), (weight[n] * weight_shape[mask] * e_cross[mask]) / (2 * R))
			np.add.at(variance, (ind_r, ind_pi), ((weight[n] * weight_shape[mask] * e_plus[mask]) / (2 * R)) ** 2)
			del e_plus, e_cross
			np.add.at(DD, (ind_r, ind_pi), weight[n] * weight_shape[mask])

		# if Num_position == Num_shape:
		# 	corrtype = "auto"
		# 	DD = DD / 2.0  # auto correlation, all pairs are double
		# else:

		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins, Splus_D, DD, RR_g_plus

	def _measure_xi_rp_pi_sims_tree(self, tree_input=None, masks=None, dataset_name="All_galaxies",
									return_output=False, print_num=True, dataset_name_tree=None, save_tree=False,
									file_tree_path=None, jk_group_name=""):
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
		tree_input :
		     (Default value = None)
		print_num :
		     (Default value = True)
		dataset_name_tree :
		     (Default value = None)
		save_tree :
		     (Default value = False)
		file_tree_path :
		     (Default value = None)
		jk_group_name :
		     (Default value = "")

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
			weight = self.data["weight"]
			weight_shape = self.data["weight_shape_sample"]
		else:
			positions = self.data["Position"][masks["Position"]]
			positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			weight = self.data["weight"][masks["weight"]]
			weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		# masking changes the number of galaxies
		Num_position = len(positions)  # number of halos in position sample
		Num_shape = len(positions_shape_sample)  # number of halos in shape sample
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		del q
		R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape)
		# R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
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
			weight_shape_i = weight_shape[i:i2]
			if tree_input != None:
				ind_rbin = pickle.load(tree_file)
				indices_shape_i = indices_shape[(indices_shape >= i) * (indices_shape < i2)] - i
				ind_rbin_i = self.setdiff_omit(ind_rbin, indices_not_position, indices_shape_i)
				positions_shape_sample_i = positions_shape_sample_i[indices_shape_i]
				axis_direction_i = axis_direction_i[indices_shape_i]
				e_i = e_i[indices_shape_i]
				weight_shape_i = weight_shape_i[indices_shape_i]
			else:
				shape_tree = KDTree(positions_shape_sample_i[:, not_LOS], boxsize=self.boxsize)
				ind_min_i = shape_tree.query_ball_tree(pos_tree, self.r_min)
				ind_max_i = shape_tree.query_ball_tree(pos_tree, self.r_max)
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
					with np.errstate(invalid='ignore'):
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
					mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * (
							LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / sub_box_len_logrp - np.log10(
							self.r_bins[0]) / sub_box_len_logrp
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_pi = np.floor(
						LOS[mask] / sub_box_len_pi - self.pi_bins[0] / sub_box_len_pi
					)  # need length of LOS, so only positive values
					ind_pi = np.array(ind_pi, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_pi),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_plus[mask]) / (2 * R))
					np.add.at(Scross_D, (ind_r, ind_pi),
							  (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_cross[mask]) / (2 * R))
					del e_plus, e_cross, separation_len
					np.add.at(DD, (ind_r, ind_pi), weight[ind_rbin_i[n]][mask] * weight_shape_i[n])

		if tree_input != None:
			tree_file.close()
		# if Num_position == Num_shape:
		# 	corrtype = "auto"
		# 	DD = DD / 2.0  # auto correlation, all pairs are double
		# else:
		corrtype = "cross"
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins, Splus_D, DD, RR_g_plus

	def _measure_xi_rp_pi_sims_batch(self, indices):
		"""

		Parameters
		----------
		indices :
		    

		Returns
		-------

		"""
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
			weight_shape_i = self.weight_shape[i:i2]

			shape_tree = KDTree(positions_shape_sample_i[:, self.not_LOS], boxsize=self.boxsize)
			ind_min_i = shape_tree.query_ball_tree(self.pos_tree, self.r_min)
			ind_max_i = shape_tree.query_ball_tree(self.pos_tree, self.r_max)
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
					with np.errstate(invalid='ignore'):
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
					mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * (
							LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
					ind_r = np.floor(
						np.log10(separation_len[mask]) / self.sub_box_len_logrp - np.log10(
							self.r_bins[0]) / self.sub_box_len_logrp
					)
					ind_r = np.array(ind_r, dtype=int)
					ind_pi = np.floor(
						LOS[mask] / self.sub_box_len_pi - self.pi_bins[0] / self.sub_box_len_pi
					)  # need length of LOS, so only positive values
					ind_pi = np.array(ind_pi, dtype=int)
					np.add.at(Splus_D, (ind_r, ind_pi),
							  (self.weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_plus[mask]) / (2 * self.R))
					np.add.at(Scross_D, (ind_r, ind_pi),
							  (self.weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_cross[mask]) / (2 * self.R))
					del e_plus, e_cross, separation_len
					# np.add.at(variance, (ind_r, ind_pi), (e_plus[mask] / (2 * R)) ** 2)
					np.add.at(DD, (ind_r, ind_pi), self.weight[ind_rbin_i[n]][mask] * weight_shape_i[n])

		return Splus_D, Scross_D, DD, variance

	def _measure_xi_rp_pi_sims_multiprocessing(self, num_nodes=9, masks=None,
											   dataset_name="All_galaxies", return_output=False,
											   print_num=True, jk_group_name=""):
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
		num_nodes :
		     (Default value = 9)
		print_num :
		     (Default value = True)
		jk_group_name :
		     (Default value = "")

		Returns
		-------
		type
		    xi_g_plus, xi_gg, separation_bins, pi_bins if no output file is specified

		"""

		if masks == None:
			self.positions = self.data["Position"]
			self.positions_shape_sample = self.data["Position_shape_sample"]
			axis_direction_v = self.data["Axis_Direction"]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"]
			self.weight = self.data["weight"]
			self.weight_shape = self.data["weight_shape_sample"]
		else:
			self.positions = self.data["Position"][masks["Position"]]
			self.positions_shape_sample = self.data["Position_shape_sample"][masks["Position_shape_sample"]]
			axis_direction_v = self.data["Axis_Direction"][masks["Axis_Direction"]]
			axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
			self.axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
			q = self.data["q"][masks["q"]]
			try:
				weight_mask = masks["weight"]
			except:
				masks["weight"] = np.ones(self.Num_position, dtype=bool)
				masks["weight"][sum(masks["Position"]):self.Num_position] = 0
			try:
				weight_mask = masks["weight_shape_sample"]
			except:
				masks["weight_shape_sample"] = np.ones(self.Num_shape, dtype=bool)
				masks["weight_shape_sample"][sum(masks["Position_shape_sample"]):self.Num_shape] = 0
			self.weight = self.data["weight"][masks["weight"]]
			self.weight_shape = self.data["weight_shape_sample"][masks["weight_shape_sample"]]
		# masking changes the number of galaxies
		Num_position = len(self.positions)  # number of halos in position sample
		Num_shape = len(self.positions_shape_sample)  # number of halos in shape sample
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

		self.LOS_ind = self.data["LOS"]  # eg 2 for z axis
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], self.LOS_ind, invert=True)]  # eg 0,1 for x&y
		self.e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		self.R = sum(self.weight_shape * (1 - self.e ** 2 / 2.0)) / sum(self.weight_shape)
		# self.R = 1 - np.mean(self.e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		self.sub_box_len_logrp = (np.log10(self.r_max) - np.log10(self.r_min)) / self.num_bins_r
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
			self._measure_xi_rp_pi_sims_batch,
			self.multiproc_chuncks,
		)
		for i in np.arange(num_nodes):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]
			variance += result[i][3]

		# if Num_position == Num_shape:
		# 	corrtype = "auto"
		# 	DD = DD / 2.0  # auto correlation, all pairs are double
		# else:
		corrtype = "cross"

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross",
					Num_position, Num_shape)
				RR_gg[i, p] = self.get_random_pairs(
					self.r_bins[i + 1], self.r_bins[i], self.pi_bins[p + 1], self.pi_bins[p], L3, corrtype,
					Num_position, Num_shape)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_plus/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_g_cross/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/w/xi_gg/{jk_group_name}")
			write_dataset_hdf5(group, dataset_name, data=(DD / RR_gg) - 1)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, (DD / RR_gg) - 1, separation_bins, pi_bins, Splus_D, DD, RR_g_plus


if __name__ == "__main__":
	pass
