import numpy as np
import h5py
import pyccl as ccl
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from . import pair_kernel


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
		if data_suffix == "_SplusD":
			DD_suff = "_DD"
			Scross_suff = "_ScrossD"
		elif data_suffix == "_SplusR":
			DD_suff = "_SR"
			Scross_suff = "_ScrossR"
		else:
			raise ValueError("data_suffix must be _SplusD or _SplusR")
		sample_set = pair_kernel.prepare_lightcone_samples(
			self.data, masks, shapes=True, cosmology=cosmology, over_h=over_h,
			responsivity_correction=getattr(self, "responsivity_correction", False),
			base=self, print_num=print_num,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=True,
									   chunk_axis="position", chunk_size_outer=100, backend="brute")
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
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
		if data_suffix == "_SplusD":
			DD_suff = "_DD"
			Scross_suff = "_ScrossD"
		elif data_suffix == "_SplusR":
			DD_suff = "_SR"
			Scross_suff = "_ScrossR"
		else:
			raise ValueError("data_suffix must be _SplusD or _SplusR")
		sample_set = pair_kernel.prepare_lightcone_samples(
			self.data, masks, shapes=True, cosmology=cosmology, over_h=over_h,
			responsivity_correction=getattr(self, "responsivity_correction", False),
			base=self, print_num=print_num,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=True,
									   chunk_axis="position", chunk_size_outer=100, backend="tree")
		DD = grids.DD
		Splus_D = grids.Splus_D
		Scross_D = grids.Scross_D
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
		sample_set = pair_kernel.prepare_lightcone_samples(
			self.data, masks, shapes=False, cosmology=cosmology, over_h=over_h,
			responsivity_correction=getattr(self, "responsivity_correction", False),
			base=self, print_num=print_num,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="position", chunk_size_outer=100, backend="brute")
		DD = grids.DD
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
		sample_set = pair_kernel.prepare_lightcone_samples(
			self.data, masks, shapes=False, cosmology=cosmology, over_h=over_h,
			responsivity_correction=getattr(self, "responsivity_correction", False),
			base=self, print_num=print_num,
		)
		Num_position = len(sample_set.pos)
		Num_shape = len(sample_set.pos_shape)
		if print_num:
			print(
				f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="position", chunk_size_outer=100, backend="tree")
		DD = grids.DD
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
