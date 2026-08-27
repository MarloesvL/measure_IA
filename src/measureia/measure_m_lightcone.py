import numpy as np
import h5py
import pyccl as ccl
from scipy.spatial import KDTree
from .write_data import write_dataset_hdf5, create_group_hdf5
from .measure_IA_base import MeasureIABase
from . import pair_kernel
import os
from multiprocessing import shared_memory
from .read_data import ReadData
from . import worker_pool


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


	def _measure_xi_r_mur_lightcone_batch(self, i):
		r"""(sky) full-sample position-sample batch worker. Reads the shared arrays and
		delegates the counting to pair_kernel.accumulate (chunk_axis="position", reusing the
		parent's shared self.shape_tree). The jackknife twin of this is
		_measure_xi_r_mur_lightcone_jk_batch; this one carries no patch indices and no
		jk grids. Support function for the mp method."""
		if i + self.chunk_size > self.Num_position_masked:
			i2 = self.Num_position_masked
		else:
			i2 = i + self.chunk_size

		shms = []
		shared_data = {}
		for name, shape, dtype in self.shm_infos:
			shm = shared_memory.SharedMemory(name=name)
			shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
			shms.append(shm)

		sample_set = pair_kernel.SampleSet(
			pos=shared_data[f"s_pos_{self.ID_shm}"][i:i2],
			pos_shape=shared_data[f"s_shape_{self.ID_shm}"],
			weight=shared_data[f"weight_{self.ID_shm}"][i:i2],
			weight_shape=shared_data[f"weight_shape_{self.ID_shm}"],
			e=shared_data[f"e_{self.ID_shm}"],
			east=shared_data[f"east_{self.ID_shm}"][i:i2],
			north=shared_data[f"north_{self.ID_shm}"][i:i2],
		)
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=True,
									   chunk_axis="position", chunk_size_outer=100, backend="tree",
									   shape_tree=self.shape_tree)
		for shm in shms:
			shm.close()
		return grids.Splus_D, grids.Scross_D, grids.DD

	def _measure_xi_r_mur_lightcone_multiprocessing(self, dataset_name, temp_file_path, masks=None,
													 return_output=False, over_h=False, cosmology=None,
													 chunk_size=1000, num_nodes=1, data_suffix="_SplusD"):
		r"""Full-sample (no jackknife) sky measurement of $\xi_{g+}$ and $\xi_{gg}$ across
		``num_nodes`` processes.

		The single-process twins are _measure_xi_r_mur_lightcone_tree /
		_..._brute; this mirrors _measure_xi_r_mur_lightcone_jk_multiprocessing
		with the jackknife machinery removed. It exists because ``num_nodes`` used to be
		accepted and then silently ignored on this path -- only the jackknife branch had a
		multiprocessing implementation (benchmarks/FINDINGS.md F4).

		Results are identical to the tree backend; only the work distribution differs.
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
			base=self, print_num=True,
		)
		s_pos = sample_set.pos
		s_shape = sample_set.pos_shape
		e = sample_set.e
		east = sample_set.east
		north = sample_set.north
		weight = sample_set.weight
		weight_shape = sample_set.weight_shape
		self.Num_position_masked = len(s_pos)
		self.Num_shape_masked = len(s_shape)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")
		self.shape_tree = KDTree(s_shape)
		indices = np.arange(0, self.Num_position_masked, chunk_size)
		self.chunk_size = chunk_size

		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		file_temp = h5py.File(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5", "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
		file_temp.close()
		self.ID_shm = np.random.randint(100000)
		shm_blocks = []
		try:
			shared_data = {
				f"s_pos_{self.ID_shm}": s_pos,
				f"s_shape_{self.ID_shm}": s_shape,
				f"e_{self.ID_shm}": e,
				f"east_{self.ID_shm}": east,
				f"north_{self.ID_shm}": north,
				f"weight_{self.ID_shm}": weight,
				f"weight_shape_{self.ID_shm}": weight_shape,
			}
			for k in shared_data.keys():
				try:
					old = shared_memory.SharedMemory(name=k)
					old.unlink()
				except FileNotFoundError:
					pass
			self.shm_infos = []
			for k in shared_data.keys():
				shm = shared_memory.SharedMemory(name=k, create=True, size=shared_data[k].nbytes)
				shared_arr = np.ndarray(shared_data[k].shape, dtype=shared_data[k].dtype, buffer=shm.buf)
				np.copyto(shared_arr, shared_data[k])
				shm_blocks.append(shm)
				self.shm_infos.append([k, shared_data[k].shape, shared_data[k].dtype])
			self.data = {}
			if masks is not None:
				masks = {}
			del shared_data, shared_arr
			del weight, weight_shape, s_pos, e, s_shape, east, north
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._measure_xi_r_mur_lightcone_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			# restore self.data from the temp file even if a worker failed
			if os.path.exists(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"m_temp_data_{figname_dataset_name}", None,
										   data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				os.remove(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		for i in np.arange(len(result)):
			Splus_D += result[i][0]
			Scross_D += result[i][1]
			DD += result[i][2]
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		pi_bins = self.mu_r_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_plus/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_g_cross/")
			write_dataset_hdf5(group, dataset_name + Scross_suff, data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=pi_bins)
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + DD_suff, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=pi_bins)
			output_file.close()
			return
		else:
			return Splus_D, DD, separation_bins, pi_bins

	def _count_pairs_xi_r_mur_lightcone_batch(self, i):
		r"""(sky) DD-only full-sample position-sample batch worker. As
		_measure_xi_r_mur_lightcone_batch but with shapes=False, so no ellipticity
		arrays are shared and only the pair count is returned."""
		if i + self.chunk_size > self.Num_position_masked:
			i2 = self.Num_position_masked
		else:
			i2 = i + self.chunk_size

		shms = []
		shared_data = {}
		for name, shape, dtype in self.shm_infos:
			shm = shared_memory.SharedMemory(name=name)
			shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
			shms.append(shm)

		sample_set = pair_kernel.SampleSet(
			pos=shared_data[f"s_pos_{self.ID_shm}"][i:i2],
			pos_shape=shared_data[f"s_shape_{self.ID_shm}"],
			weight=shared_data[f"weight_{self.ID_shm}"][i:i2],
			weight_shape=shared_data[f"weight_shape_{self.ID_shm}"],
		)
		binning = pair_kernel.SkyRMuR(self)
		grids = pair_kernel.accumulate(sample_set, binning, base=self, shapes=False,
									   chunk_axis="position", chunk_size_outer=100, backend="tree",
									   shape_tree=self.shape_tree)
		for shm in shms:
			shm.close()
		return grids.DD

	def _count_pairs_xi_r_mur_lightcone_multiprocessing(self, dataset_name, temp_file_path, masks=None,
														 return_output=False, over_h=False, cosmology=None,
														 chunk_size=1000, num_nodes=1, data_suffix="_DD"):
		r"""Full-sample (no jackknife) sky pair count across ``num_nodes`` processes.

		DD-only twin of _measure_xi_r_mur_lightcone_multiprocessing; see that method
		for why this path exists.
		"""
		sample_set = pair_kernel.prepare_lightcone_samples(
			self.data, masks, shapes=False, cosmology=cosmology, over_h=over_h,
			responsivity_correction=getattr(self, "responsivity_correction", False),
			base=self, print_num=True,
		)
		s_pos = sample_set.pos
		s_shape = sample_set.pos_shape
		weight = sample_set.weight
		weight_shape = sample_set.weight_shape
		self.Num_position_masked = len(s_pos)
		self.Num_shape_masked = len(s_shape)
		print(
			f"There are {self.Num_shape_masked} galaxies in the shape sample and {self.Num_position_masked} galaxies in the position sample.")
		self.shape_tree = KDTree(s_shape)
		indices = np.arange(0, self.Num_position_masked, chunk_size)
		self.chunk_size = chunk_size

		figname_dataset_name = dataset_name
		if "/" in dataset_name:
			figname_dataset_name = figname_dataset_name.replace("/", "_")
		if "." in dataset_name:
			figname_dataset_name = figname_dataset_name.replace(".", "p")
		file_temp = h5py.File(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5", "w")
		keys = []
		for k in self.data.keys():
			if k != "LOS":
				write_dataset_hdf5(file_temp, k, self.data[k])
				if masks is not None:
					write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
				keys.append(k)
		file_temp.close()
		self.ID_shm = np.random.randint(100000)
		shm_blocks = []
		try:
			shared_data = {
				f"s_pos_{self.ID_shm}": s_pos,
				f"s_shape_{self.ID_shm}": s_shape,
				f"weight_{self.ID_shm}": weight,
				f"weight_shape_{self.ID_shm}": weight_shape,
			}
			for k in shared_data.keys():
				try:
					old = shared_memory.SharedMemory(name=k)
					old.unlink()
				except FileNotFoundError:
					pass
			self.shm_infos = []
			for k in shared_data.keys():
				shm = shared_memory.SharedMemory(name=k, create=True, size=shared_data[k].nbytes)
				shared_arr = np.ndarray(shared_data[k].shape, dtype=shared_data[k].dtype, buffer=shm.buf)
				np.copyto(shared_arr, shared_data[k])
				shm_blocks.append(shm)
				self.shm_infos.append([k, shared_data[k].shape, shared_data[k].dtype])
			self.data = {}
			if masks is not None:
				masks = {}
			del shared_data, shared_arr
			del weight, weight_shape, s_pos, s_shape
			with worker_pool.active_pool(num_nodes) as p:
				result = p.map(self._count_pairs_xi_r_mur_lightcone_batch, indices)

		finally:
			for shm in shm_blocks:
				shm.close()
				shm.unlink()
			if os.path.exists(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5"):
				temp_data_obj_m = ReadData(self.simname, f"m_temp_data_{figname_dataset_name}", None,
										   data_path=temp_file_path)
				for k in keys:
					self.data[k] = temp_data_obj_m.read_cat(k)
					if masks is not None:
						masks[k] = temp_data_obj_m.read_cat(f"mask_{k}")
				os.remove(f"{temp_file_path}/m_temp_data_{figname_dataset_name}.hdf5")

		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		for i in np.arange(len(result)):
			DD += result[i]
		dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
		separation_bins = self.r_bins[:-1] + abs(dsep)  # middle of bins
		dpi = (self.mu_r_bins[1:] - self.mu_r_bins[:-1]) / 2.0
		pi_bins = self.mu_r_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) and (return_output == False):
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, f"{self.snap_group}/multipoles/xi_gg/")
			write_dataset_hdf5(group, dataset_name + data_suffix, data=DD)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=pi_bins)
			output_file.close()
			return
		else:
			return DD, separation_bins, pi_bins


if __name__ == "__main__":
	pass
