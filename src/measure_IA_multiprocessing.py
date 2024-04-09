import numpy as np
import h5py
import time
from numpy.linalg import eig, inv
import sympy
from pathos.multiprocessing import ProcessingPool
from src.read_data_TNG import ReadTNGdata
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.Sim_info import SimInfo

KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureVariablesSnapshotMultiprocessing(SimInfo):
	"""
	Measures different variables using the particle snapshot data.
	WARNING: Currently set up for TNG only. Should work for other sims. Take care with velocity units.
	:param PT: Number indicating particle type
	:param project: Indicator of simulation. Choose from [TNG100, TNG300] for now.
	:param snapshot: Number of the snapshot
	:param output_file_name: Name and filepath of the file where the output should be stored.
	:param data_path: Start path to the raw data files
	"""

	def __init__(self, PT=4, project=None, snapshot=None, numnodes=30, output_file_name=None, data_path="./data/raw/",
				 exclude_wind=True):
		if project == None:
			raise KeyError("Input project name!")
		SimInfo.__init__(self, project, snapshot)
		self.PT = PT
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=data_path
		)
		try:
			mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
			self.Num_halos = len(mass_subhalo)
		except:
			self.Num_halos = 0
		self.exclude_wind = exclude_wind
		self.numnodes = numnodes
		self.output_file_name = output_file_name
		self.data_path = data_path
		return

	def create_self_arguments(self):
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		mass = TNG100_SubhaloPT.read_cat(self.mass_name)
		if self.snapshot == "50" and self.simname == "TNG300":
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only_snap50",
				self.snapshot,
				data_path=self.data_path,
			)
			print("Using " + self.simname + "_PT" + str(self.PT) + "_subhalos_only_snap50")
		else:
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only",
				self.snapshot,
				data_path=self.data_path,
			)
		# self.TNG100_SubhaloPT = TNG100_SubhaloPT
		if self.exclude_wind:
			self.TNG100_SubhaloPT = ReadTNGdata(
				self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
			)
		else:
			self.TNG100_SubhaloPT = ReadTNGdata(
				self.simname, "SubhaloPT_incl_wind", self.snapshot, sub_group="PT" + str(self.PT) + "/",
				data_path=self.data_path
			)
		self.off = off
		self.Len = Len
		self.mass = mass
		self.TNG100_snapshot = TNG100_snapshot
		self.multiproc_chuncks = np.array_split(np.arange(self.Num_halos), self.numnodes)
		return

	def measure_offsets(self, type="Subhalo"):
		"""
		Measures the Offsets for each subhalo or group needed to read snapshot files.
		Writes away data to SubhaloPT file in appropriate PT group.
		:param type: Determines if offsets for Subhalos or Groups are calculated (default is Subhalo).
		:return:
		"""
		TNG100_subhalo = ReadTNGdata(self.simname, type, self.snapshot, data_path=self.data_path)
		if type == "Subhalo":
			len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		elif self.simname == "EAGLE":
			len = TNG100_subhalo.read_cat(self.sub_len_name)
		else:
			len = TNG100_subhalo.read_subhalo(self.group_len_name)[:, self.PT]
		off = np.append(np.array([0]), np.cumsum(len)[:-1])

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Offset_" + type + "_all", data=off)
		output_file.close()
		return

	def omit_wind_only(self):
		"""
		Loops through all subhalos to give a flag (1) to the subhalos that consist only of wind particles.
		These can then be omitted in the 'select_nonzero_subhalos' method. Specific to IllustrisTNG simulations.
		:return:
		"""
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
		TNG100_subhalo = ReadTNGdata(self.simname, "Subhalo", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		Wind_Flag = []
		for n in np.arange(0, len(off)):
			Off = off[n]
			Len_n = Len[n]
			wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[Off, Off + Len_n])
			star_mask = wind_or_star > 0
			if sum(star_mask) > 0.0:
				Wind_Flag.append(0)
			else:
				Wind_Flag.append(1)
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Wind_Flag", Wind_Flag)
		output_file.close()
		return

	def omit_wind_only_single(self, indices):
		Wind_Flag = []
		for n in indices:
			Off = self.off_wind[n]
			Len_n = self.len_wind[n]
			wind_or_star = self.snap_wind.read_cat(self.wind_name, cut=[Off, Off + Len_n])
			star_mask = wind_or_star > 0
			if sum(star_mask) > 0.0:
				Wind_Flag.append(0)
			else:
				Wind_Flag.append(1)

		return Wind_Flag

	def omit_wind_only_multiprocessing(self, numnodes=30):
		"""
		Loops through all subhalos to give a flag (1) to the subhalos that consist only of wind particles.
		These can then be omitted in the 'select_nonzero_subhalos' method. Specific to IllustrisTNG simulations.
		:return:
		"""
		if self.snapshot == "50" and self.simname == "TNG300":
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only_snap50",
				self.snapshot,
				data_path=self.data_path,
			)
		else:
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only",
				self.snapshot,
				data_path=self.data_path,
			)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
		TNG100_subhalo = ReadTNGdata(self.simname, "Subhalo", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		Wind_Flag = []
		multiproc_chuncks = np.array_split(np.arange(len(off)), numnodes)
		self.snap_wind = TNG100_snapshot
		self.off_wind = off
		self.len_wind = Len
		result = ProcessingPool(nodes=numnodes).map(
			self.omit_wind_only_single,
			multiproc_chuncks,
		)
		for i in np.arange(numnodes):
			Wind_Flag.extend(result[i])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Wind_Flag", Wind_Flag)
		output_file.close()
		return

	def select_nonzero_subhalos(self):
		"""
		Selects the subhalos that have nonzero length and SubhaloFlag. Saves selected data in output file, including
		the IDs for the original file. Specific to IllustrisTNG simulations.
		:return:
		"""
		TNG100_subhalo = ReadTNGdata(self.simname, "Subhalo", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		mass_subhalo = TNG100_subhalo.read_subhalo(self.mass_name)[:, self.PT]
		flag = TNG100_subhalo.read_subhalo(self.flag_name)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
		if self.PT == 4:
			wind_flag = TNG100_SubhaloPT.read_cat("Wind_Flag")
			mask = (Len > 0.0) * (flag == 1) * (wind_flag == 0)
		else:
			mask = (Len > 0.0) * (flag == 1)
		IDs = np.where(mask)
		self.Num_halos = len(mass_subhalo[mask])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, self.sub_len_name, Len[mask])
		write_dataset_hdf5(group, self.offset_name, off[mask])
		write_dataset_hdf5(group, self.mass_name, mass_subhalo[mask])
		write_dataset_hdf5(group, self.ID_name, IDs)
		if self.PT == 4:
			photo_mag = TNG100_subhalo.read_subhalo(self.photo_name)
			SFR = TNG100_subhalo.read_subhalo(self.SFR_name)
			write_dataset_hdf5(group, self.photo_name, photo_mag[mask])
			write_dataset_hdf5(group, self.SFR_name, SFR[mask])
		output_file.close()
		return

	def select_nonzero_subhalos_EAGLE(self):
		"""
		Selects the subhalos that have nonzero length and SubhaloFlag. Saves selected data in output file, including
		the IDs for the original file. Specific to IllustrisTNG simulations.
		:return:
		"""
		TNG100_subhalo = ReadTNGdata(self.simname, "Subhalo_cat", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_cat(self.sub_len_name)
		mass_subhalo = TNG100_subhalo.read_cat(self.mass_name)
		gn = TNG100_subhalo.read_cat("GroupNumber")
		sn = TNG100_subhalo.read_cat("SubGroupNumber")  # less
		galaxyIDs = TNG100_subhalo.read_cat("GalaxyID")
		file = h5py.File(f"{self.data_path}/EAGLE/diff_gnsn.hdf5", 'a')
		indices_gnsn_in_sub = file['indices_gnsn_in_sub'][:, 0]
		indices_sub_in_gnsn = file["indices_sub_in_gnsn"][:, 0]  # dees
		file.close()
		flag = TNG100_subhalo.read_cat(self.flag_name)
		mask = (mass_subhalo > 0.0) * (flag == 0)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_cat_all")
		mass_subhalo = mass_subhalo[mask]
		galaxyIDs = galaxyIDs[mask]
		gn = gn[mask]
		sn = sn[mask]

		indices = np.arange(0, len(mass_subhalo))
		gnsn = np.array([indices, gn, sn]).transpose()
		sorted_gnsn = sorted(gnsn, key=lambda x: (x[1], x[2]))
		sorted_gnsn = np.array(sorted_gnsn)
		sorted_indices = sorted_gnsn[:, 0]
		sorted_indices = np.array(sorted_indices, dtype=int)

		mass_subhalo = mass_subhalo[sorted_indices]
		galaxyIDs = galaxyIDs[sorted_indices]
		gn = gn[sorted_indices]
		sn = sn[sorted_indices]

		# indices_snap = np.arange(0, len(Len))
		# gnsn_snap_ind = np.array([indices, gnsn[:, 0], gnsn[:, 1]]).transpose()
		# sorted_gnsn_snap = sorted(gnsn_snap_ind, key=lambda x: (x[1], x[2]))
		# sorted_indices_snap = sorted_gnsn_snap[:, 0]

		# IDs = np.where(mask)

		self.Num_halos = len(mass_subhalo[indices_gnsn_in_sub])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, self.sub_len_name, Len[indices_sub_in_gnsn])
		write_dataset_hdf5(group, self.offset_name, off[indices_sub_in_gnsn])
		write_dataset_hdf5(group, self.mass_name, mass_subhalo)
		write_dataset_hdf5(group, self.ID_name, galaxyIDs)
		if self.PT == 4:
			# photo_mag = TNG100_subhalo.read_subhalo(self.photo_name)
			SFR = TNG100_subhalo.read_cat(self.SFR_name)
			SFR = SFR[mask]
			SFR = SFR[sorted_indices]
			# write_dataset_hdf5(group, self.photo_name, photo_mag[mask])
			write_dataset_hdf5(group, self.SFR_name, SFR)
		output_file.close()
		return

	def save_number_of_particles(self):
		"""
		Saves the number of particles used in the snapshot calculations. This is usually equal to the 'Len' parameter,
		except when wind particles are omitted from the stellar particles.
		:return:
		"""
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		if self.snapshot == "50" and self.simname == "TNG300":
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only_snap50",
				self.snapshot,
				data_path=self.data_path,
			)
		else:
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only",
				self.snapshot,
				data_path=self.data_path,
			)
		number_of_particles_list = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			if self.PT == 4 and "TNG" in self.simname:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				number_of_particles = sum(star_mask)
			else:
				number_of_particles = len_n
			number_of_particles_list.append(number_of_particles)
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Number_of_particles", data=np.array(number_of_particles_list))
		output_file.close()
		return

	def measure_masses_excl_wind(self):
		"""
		Saves the number of particles used in the snapshot calculations. This is usually equal to the 'Len' parameter,
		except when wind particles are omitted from the stellar particles.
		:return:
		"""
		if self.PT == 4 and "TNG" in self.simname:
			pass
		else:
			print('Use given mass for this input')
			exit()
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		if self.snapshot == "50" and self.simname == "TNG300":
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only_snap50",
				self.snapshot,
				data_path=self.data_path,
			)
		else:
			TNG100_snapshot = ReadTNGdata(
				self.simname,
				self.simname + "_PT" + str(self.PT) + "_subhalos_only",
				self.snapshot,
				data_path=self.data_path,
			)
		mass_list = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
			star_mask = wind_or_star > 0
			masses = TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])[star_mask]
			mass = sum(masses)
			mass_list.append(mass)

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "StellarMass", data=np.array(mass_list))
		output_file.close()
		return


	def measure_velocities_single(self, indices):
		velocities = []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			mass_n = self.mass[n]
			if self.PT == 1:
				mass_particles = self.DM_part_mass
				velocity_particles = self.TNG100_snapshot.read_cat(
					self.velocities_name, cut=[off_n, off_n + len_n]
				) * np.sqrt(self.scalefactor)
			elif self.PT == 4 and "TNG" in self.simname:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass_particles = self.TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])[star_mask]
				velocity_particles = self.TNG100_snapshot.read_cat(self.velocities_name, cut=[off_n, off_n + len_n])[
										 star_mask
									 ] * np.sqrt(self.scalefactor)
			else:
				mass_particles = self.TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])
				velocity_particles = self.TNG100_snapshot.read_cat(
					self.velocities_name, cut=[off_n, off_n + len_n]
				) * np.sqrt(self.scalefactor)

			mass_vel = (velocity_particles.transpose() * mass_particles).transpose()
			velocity = np.sum(mass_vel, axis=0) / mass_n
			velocities.append(velocity)
		return velocities

	def measure_velocities_multiprocessing(self):
		"""
		Measures the velocity (km/s * a) of each subhalo for the initalised PT, weighted by
		particle mass. Writes data to SubhaloPT file in appropriate PT group.
		:return:
		"""
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		# mass = TNG100_SubhaloPT.read_cat(self.mass_name)
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )

		velocities = []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_velocities_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			velocities.extend(result[i])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Velocity", data=np.array(velocities))
		output_file.close()
		return

	def measure_COM_single(self, indices):
		COM = []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			mass_n = self.mass[n]
			if self.PT == 1:
				mass_particles = self.DM_part_mass
				coordinates_particles = self.TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])
			elif self.PT == 4 and "TNG" in self.simname and self.exclude_wind:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass_particles = self.TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])[star_mask]
				coordinates_particles = \
					self.TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])[
						star_mask
					]
			else:
				mass_particles = self.TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])
				coordinates_particles = self.TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])

			# account for periodicity of the box
			min_coord = np.min(coordinates_particles, axis=0)
			coordinates_particles[(coordinates_particles - min_coord) > self.L_0p5] -= self.boxsize
			coordinates_particles[(coordinates_particles - min_coord) < -self.L_0p5] += self.boxsize

			mass_coord = (coordinates_particles.transpose() * mass_particles).transpose()
			COM_n = np.sum(mass_coord, axis=0) / mass_n
			COM_n[COM_n < 0.0] += self.boxsize  # if negative: COM is on other side of box.
			COM.append(COM_n)
		return COM

	def measure_COM_multiprocessing(self):
		"""
		Measures the centre of mass (ckpc/h) of each subhalo for the initalised PT. Coordinates weighted by
		particle mass. Writes data to SubhaloPT file in appropriate PT group.
		:return:
		"""
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		# mass = TNG100_SubhaloPT.read_cat(self.mass_name)
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		COM = []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_COM_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			COM.extend(result[i])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "COM", data=np.array(COM))
		output_file.close()
		return

	def measure_inertia_tensor_single(self, indices):
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			mass = self.mass[n]

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
			elif self.PT == 4 and "TNG" in self.simname and self.exclude_wind:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_position = (
						self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
						self.COM[n]
				)
			else:
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			if mass == 0.0:
				raise AssertionError("Mass is 0. Something went wrong in pre-selection.")
			if len_n == 1.0:
				I_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
				value_list.append([0.0, 0.0, 0.0])
				vectors_list.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
				v0.append([0.0, 0.0, 0.0])
				v1.append([0.0, 0.0, 0.0])
				v2.append([0.0, 0.0, 0.0])
			else:
				I = self.measure_inertia_tensor_eq(mass, particle_mass, rel_position, self.reduced)
				if sum(np.isnan(I.flatten())) > 0 or sum(np.isinf(I.flatten())) > 0:
					print(f"NaN of inf found in galaxy {indices[n]}, I is {I}, mass {mass}, len {len_n}. Appending zeros.")
					I_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
					value_list.append([0.0, 0.0, 0.0])
					vectors_list.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
					v0.append([0.0, 0.0, 0.0])
					v1.append([0.0, 0.0, 0.0])
					v2.append([0.0, 0.0, 0.0])
				else:
					if self.eigen_v:
						values, vectors = eig(I)
						value_list.append(values)
						vectors_list.append(vectors)
						v0.append(vectors[:, 0])
						v1.append(vectors[:, 1])
						v2.append(vectors[:, 2])
					I_list.append(I.reshape(9))
		return I_list, value_list, v0, v1, v2, vectors_list

	@staticmethod
	def measure_inertia_tensor_eq(mass, particle_mass, rel_position, reduced=False):
		"""
		Calculates the inertia tensor values for a single galaxy.
		:param mass: Mass of galaxy.
		:param particle_mass: Mass of particles in galaxy.
		:param rel_position: Position of particles relative to centre of mass of galaxy.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:return: (reduced) inertia tensor
		"""
		if reduced:
			r = np.sqrt(np.sum(rel_position ** 2, axis=1))
			rel_position = (rel_position.transpose() / r).transpose()  # normalise the positions
		I = np.zeros((3, 3))
		for i in np.arange(0, 3):
			for j in np.arange(0, 3):
				I[i, j] = 1.0 / mass * np.sum(particle_mass * rel_position[:, i] * rel_position[:, j])
		return I

	def measure_inertia_tensor_multiprocessing(self, eigen_v=True, sorted=True, reduced=False):
		"""
		Measures the inertia tensor for given dataset. Either saved or returned.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:param sorted: Sorts eigen values and eigen vectors from lowest to highest if True.
		:param eigen_v: Also returns eigen values and vectors if True.
		:return: The inertia tensor, eigen values and vectors if no output file is specified.
		"""

		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		# mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		self.reduced = reduced
		self.eigen_v = eigen_v
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_inertia_tensor_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			I_list.extend(result[i][0])
			value_list.extend(result[i][1])
			v0.extend(result[i][2])
			v1.extend(result[i][3])
			v2.extend(result[i][4])
			vectors_list.extend(result[i][5])
		if sorted:
			eigen_values = np.array(value_list)
			sort_ind = np.argsort(eigen_values, axis=1)  # indices to sort from low to high values
			eigen_values_sorted = []
			eigen_vectors_sorted = {"0": [], "1": [], "2": []}
			for i in np.arange(0, self.Num_halos):
				ind = sort_ind[i]
				eigen_values_sorted.append(eigen_values[i][ind])
				vector_sorted = vectors_list[i][:, ind]
				eigen_vectors_sorted["0"].append(vector_sorted[:, 0])
				eigen_vectors_sorted["1"].append(vector_sorted[:, 1])
				eigen_vectors_sorted["2"].append(vector_sorted[:, 2])
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			if reduced:
				write_dataset_hdf5(group, "Reduced_Inertia_Tensor", data=np.array(I_list))
				if eigen_v:
					write_dataset_hdf5(group, "Eigen_values_reduced", data=np.array(value_list))
					write_dataset_hdf5(group, "Eigen_vector0_reduced", data=np.array(v0))
					write_dataset_hdf5(group, "Eigen_vector1_reduced", data=np.array(v1))
					write_dataset_hdf5(group, "Eigen_vector2_reduced", data=np.array(v2))
					if sorted:
						write_dataset_hdf5(
							group, "Axis_Lengths_reduced", data=np.sqrt(np.array(eigen_values_sorted, dtype=np.float64))
						)  # low to high
						write_dataset_hdf5(
							group, "Minor_Axis_Direction_reduced", data=np.array(eigen_vectors_sorted["0"])
						)
						write_dataset_hdf5(
							group, "Intermediate_Axis_Direction_reduced", data=np.array(eigen_vectors_sorted["1"])
						)
						write_dataset_hdf5(
							group, "Major_Axis_Direction_reduced", data=np.array(eigen_vectors_sorted["2"])
						)
			else:
				write_dataset_hdf5(group, "Inertia_Tensor", data=np.array(I_list))
				if eigen_v:
					write_dataset_hdf5(group, "Eigen_values", data=np.array(value_list))
					write_dataset_hdf5(group, "Eigen_vector0", data=np.array(v0))
					write_dataset_hdf5(group, "Eigen_vector1", data=np.array(v1))
					write_dataset_hdf5(group, "Eigen_vector2", data=np.array(v2))
					if sorted:
						write_dataset_hdf5(
							group, "Axis_Lengths", data=np.sqrt(np.array(eigen_values_sorted, dtype=np.float64))
						)  # low to high
						write_dataset_hdf5(group, "Minor_Axis_Direction", data=np.array(eigen_vectors_sorted["0"]))
						write_dataset_hdf5(
							group, "Intermediate_Axis_Direction", data=np.array(eigen_vectors_sorted["1"])
						)
						write_dataset_hdf5(group, "Major_Axis_Direction", data=np.array(eigen_vectors_sorted["2"]))
			print('closing file, reduced is ',reduced)
			print(output_file,group)
			output_file.close()
			print('file closed')
			print(output_file, group)
			return
		else:
			return np.array(I_list)

	def measure_projected_inertia_tensor_single(self, indices):
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			mass = self.mass[n]

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
			elif self.PT == 4 and "TNG" in self.simname and self.exclude_wind:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_position = (
						self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
						self.COM[n]
				)
			else:
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			rel_position = rel_position[:, self.not_LOS]
			if mass == 0.0:
				raise AssertionError("Mass is 0. Something went wrong in pre-selection.")
			if len_n == 1.0:
				I_list.append([0.0, 0.0, 0.0, 0.0])
				value_list.append([0.0, 0.0])
				vectors_list.append(np.array([[0.0, 0.0], [0.0, 0.0]]))
				v0.append([0.0, 0.0])
				v1.append([0.0, 0.0])
				v2.append([0.0, 0.0])
			else:
				I = self.measure_projected_inertia_tensor_eq(mass, particle_mass, rel_position, self.reduced)
				if self.eigen_v:
					values, vectors = eig(I)
					value_list.append(values)
					vectors_list.append(vectors)
					v0.append(vectors[:, 0])
					v1.append(vectors[:, 1])
				I_list.append(I.reshape(4))
		return I_list, value_list, v0, v1, v2, vectors_list

	@staticmethod
	def measure_projected_inertia_tensor_eq(mass, particle_mass, rel_position, reduced=False):
		"""
		Calculates the projected inertia tensor values for a single galaxy.
		:param mass: Mass of galaxy.
		:param particle_mass: Mass of particles in galaxy.
		:param rel_position: Position of particles relative to centre of mass of galaxy.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:return: (reduced) inertia tensor
		"""
		if reduced:
			r = np.sqrt(np.sum(rel_position ** 2, axis=1))
			rel_position = (rel_position.transpose() / r).transpose()  # normalise the positions
		I = np.zeros((2, 2))
		for i in np.arange(0, 2):
			for j in np.arange(0, 2):
				I[i, j] = 1.0 / mass * np.sum(particle_mass * rel_position[:, i] * rel_position[:, j])
		return I

	def measure_projected_inertia_tensor_multiprocessing(self, eigen_v=True, sorted=True, reduced=False, LOS_ind=2):
		"""
		Measures the inertia tensor for given dataset. Either saved or returned.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:param sorted: Sorts eigen values and eigen vectors from lowest to highest if True.
		:param eigen_v: Also returns eigen values and vectors if True.
		:return: The inertia tensor, eigen values and vectors if no output file is specified.
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			write_output = True
		else:
			write_output = False
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		# mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		self.reduced = reduced
		self.eigen_v = eigen_v
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]
		LOS_axis = {0: 'x', 1: 'y', 2: 'z'}
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_projected_inertia_tensor_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			I_list.extend(result[i][0])
			value_list.extend(result[i][1])
			v0.extend(result[i][2])
			v1.extend(result[i][3])
			v2.extend(result[i][4])
			vectors_list.extend(result[i][5])

		if sorted:
			eigen_values = np.array(value_list)
			sort_ind = np.argsort(eigen_values, axis=1)  # indices to sort from low to high values
			eigen_values_sorted = []
			eigen_vectors_sorted = {"0": [], "1": []}
			for i in np.arange(0, self.Num_halos):
				ind = sort_ind[i]
				eigen_values_sorted.append(eigen_values[i][ind])
				vector_sorted = vectors_list[i][:, ind]
				eigen_vectors_sorted["0"].append(vector_sorted[:, 0])
				eigen_vectors_sorted["1"].append(vector_sorted[:, 1])

		if write_output:
			if reduced:
				write_dataset_hdf5(group, f"Reduced_Projected_{LOS_axis[LOS_ind]}_Inertia_Tensor",
								   data=np.array(I_list))
				if eigen_v:
					write_dataset_hdf5(group, f"Eigen_values_reduced_Projected_{LOS_axis[LOS_ind]}",
									   data=np.array(value_list))
					write_dataset_hdf5(group, f"Eigen_vector0_reduced_Projected_{LOS_axis[LOS_ind]}", data=np.array(v0))
					write_dataset_hdf5(group, f"Eigen_vector1_reduced_Projected_{LOS_axis[LOS_ind]}", data=np.array(v1))
					if sorted:
						write_dataset_hdf5(
							group, f"Projected_{LOS_axis[LOS_ind]}_Axis_Lengths_reduced",
							data=np.sqrt(np.array(eigen_values_sorted, dtype=np.float64))
						)  # low to high
						write_dataset_hdf5(
							group, f"Semiminor_Axis_Direction_reduced_{LOS_axis[LOS_ind]}",
							data=np.array(eigen_vectors_sorted["0"])
						)
						write_dataset_hdf5(
							group, f"Semimajor_Axis_Direction_reduced_{LOS_axis[LOS_ind]}",
							data=np.array(eigen_vectors_sorted["1"])
						)

			else:
				write_dataset_hdf5(group, f"Projected_{LOS_axis[LOS_ind]}_Inertia_Tensor", data=np.array(I_list))
				if eigen_v:
					write_dataset_hdf5(group, f"Projected_{LOS_axis[LOS_ind]}_Eigen_values", data=np.array(value_list))
					write_dataset_hdf5(group, f"Projected_{LOS_axis[LOS_ind]}_Eigen_vector0", data=np.array(v0))
					write_dataset_hdf5(group, f"Projected_{LOS_axis[LOS_ind]}_Eigen_vector1", data=np.array(v1))
					if sorted:
						write_dataset_hdf5(
							group, f"Projected_{LOS_axis[LOS_ind]}_Axis_Lengths",
							data=np.sqrt(np.array(eigen_values_sorted, dtype=np.float64))
						)  # low to high
						write_dataset_hdf5(group, f"Semiminor_Axis_Direction_{LOS_axis[LOS_ind]}",
										   data=np.array(eigen_vectors_sorted["0"]))
						write_dataset_hdf5(
							group, f"Semimajor_Axis_Direction_{LOS_axis[LOS_ind]}",
							data=np.array(eigen_vectors_sorted["1"])
						)
			output_file.close()
			return
		else:
			return np.array(I_list)

	def measure_projected_axes(self, LOS_ind=2, reduced=False, catalogue="Shapes"):
		"""
		Measures the projected axes direcitons and lengths.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:param LOS_ind: Index of LOS (=2 for z-axis)
		:return:
		"""
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]
		TNG100_Shapes = ReadTNGdata(
			self.simname, catalogue, self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		if reduced:
			I_name = "Reduced_Inertia_Tensor"
			suffix = "_reduced"
		else:
			I_name = "Inertia_Tensor"
			suffix = ""
		inertia_tensor = TNG100_Shapes.read_cat(I_name)
		value_list, v0, v1, vector_list = [], [], [], []
		for n in np.arange(0, self.Num_halos):
			I = inertia_tensor[n].reshape((3, 3))
			I_projected = I[not_LOS]
			I_projected = I_projected[:, not_LOS]
			values, vectors = eig(I_projected)
			value_list.append(values)
			vector_list.append(vectors)
			v0.append(vectors[:, 0])
			v1.append(vectors[:, 1])
		eigen_values = np.array(value_list)
		sort_ind = np.argsort(eigen_values, axis=1)
		eigen_values_sorted = []
		eigen_vectors_sorted = {"0": [], "1": []}
		for i in np.arange(0, len(eigen_values[:, 0])):
			ind = sort_ind[i]
			eigen_values_sorted.append(eigen_values[i][ind])
			vectors_sorted = vector_list[i][:, ind]
			eigen_vectors_sorted["0"].append(vectors_sorted[:, 0])
			eigen_vectors_sorted["1"].append(vectors_sorted[:, 1])

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(
			group, f"Projected_Axis_Lengths{suffix}", data=np.sqrt(np.array(eigen_values_sorted, dtype=np.float64))
		)
		write_dataset_hdf5(group, f"Semiminor_Axis_Direction{suffix}", data=np.array(eigen_vectors_sorted["0"]))
		write_dataset_hdf5(group, f"Semimajor_Axis_Direction{suffix}", data=np.array(eigen_vectors_sorted["1"]))
		output_file.close()
		return

	def measure_spin_single(self, indices):
		spin_list = []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			if self.PT == 1:
				particle_mass = self.DM_part_mass * 1e10
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)
			elif self.PT == 4 and "TNG" in self.simname:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[
									star_mask] * 1e10
				rel_position = (
						self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
						self.COM[n]
				)
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- self.velocity[n]
				)

			else:
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n]) * 1e10
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize

			spin = np.sum((particle_mass * np.cross(rel_position, rel_velocity).transpose()).transpose(), axis=0)
			spin_list.append(spin)
		return spin_list

	def measure_spin_multiprocessing(self):
		"""
		Measures spin (angular momentum) of galaxies in kpc km s^-1 M_sun.
		:return: Spin if no output file name is specified.
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			write_output = True
		else:
			write_output = False
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		spin_list = []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_spin_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			spin_list.extend(result[i])
		if write_output:
			write_dataset_hdf5(group, "Spin", data=np.array(spin_list))
			output_file.close()
			return
		else:
			return np.array(spin_list)

	def measure_rotational_velocity_single(self, indices):
		avg_rot_vel, vel_disp, vel_z, vel_disp_cyl, vel_z_abs = [], [], [], [], []
		if self.calc_basis:
			basis = []
		for n in indices:
			# print(n)
			off_n = self.off[n]
			len_n = self.Len[n]
			mass = self.mass[n]
			# print(n,'1')
			if len_n < 2:
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if self.calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			spin_n = self.spin[n]
			if all(spin_n == 0.0):
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if self.calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			# print(n,'2')
			if self.calc_basis:
				transform_mat = self.find_cartesian_basis(spin_n)  # new orthonormal basis with Spin as z-axis
				basis_n = np.array(transform_mat)
				basis_n.resize(9)
				basis.append(basis_n)
			else:
				transform_mat = self.transform_mats[n]
				transform_mat.resize((3, 3))
				if np.shape(transform_mat) == ():
					transform_mat = [[None]]
			if transform_mat[0][0] == None:
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if self.calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			# print(n,'3')
			transform_mat_inv = inv(transform_mat)

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
			elif self.PT == 4 and "TNG" in self.simname:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- self.velocity[n]
				)
				rel_position = (
						self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
						self.COM[n]
				)
			else:
				particle_mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
			# print(n,'4')
			# get coords relative to galactic COM, velocity

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			# transform to new coords
			rel_velocity_transform = np.dot(transform_mat_inv, rel_velocity.transpose()).transpose()
			rel_position_transform = np.dot(transform_mat_inv, rel_position.transpose()).transpose()
			theta = np.arctan2(rel_position_transform[:, 1], rel_position_transform[:, 0])  # +np.pi  # arctan(y/x)
			# + np.pi  # +pi changes the sign of v_theta. theta: [-pi,pi] or [0,2pi]
			vel_theta = -rel_velocity_transform[:, 0] * np.sin(theta) + rel_velocity_transform[:, 1] * np.cos(theta)
			vel_theta_mean = (
					np.sum((vel_theta.transpose() * particle_mass).transpose(), axis=0) / mass
			)  # mass-weighted mean
			avg_rot_vel.append(vel_theta_mean)
			# print(n, '5')
			if self.measure_dispersion:
				vel_r_n = rel_velocity_transform[:, 0] * np.cos(theta) + rel_velocity_transform[:, 1] * np.sin(theta)
				vel_z_n = rel_velocity_transform[:, 2]
				rel_velocity_cyl = np.array([vel_r_n, vel_theta, vel_z_n]).transpose()
				mean_vel_cyl = (
						np.sum((rel_velocity_cyl.transpose() * particle_mass).transpose(), axis=0) / mass
				)  # mass-weighted mean
				dvel_cyl = (rel_velocity_cyl - mean_vel_cyl) ** 2
				dvel_mean_cyl = np.sum((dvel_cyl.transpose() * particle_mass).transpose(), axis=0) / mass
				mean_vel_z = np.sum(vel_z_n * particle_mass) / mass
				vel_z.append(mean_vel_z)
				vel_z_abs.append(np.sum(abs(vel_z_n) * particle_mass) / mass)
				vel_disp_cyl.append(dvel_mean_cyl)
				vel_disp.append(np.sqrt(np.sum(dvel_mean_cyl) / 3.0))
		# print(n, '6')
		return avg_rot_vel, vel_disp, vel_z, vel_disp_cyl, vel_z_abs, basis

	@staticmethod
	def find_cartesian_basis(vector):
		"""
		Find a new orthonormal carthesian basis with the given vector pointing in the z-direction.
		The x- and y-axis orientation is random.
		:param vector: Direction of z-axis in new system in old coordinates.
		:return: Matrix of transformation. Invert and multiply with vectors to obtain vector coordinates
								in the new system.
		"""
		l1, l2, l3 = vector[0], vector[1], vector[2]
		L = 1.0 / np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2) * np.array([l1, l2, l3])  # normalise vector (z direction)
		z1, z2, z3 = L[0], L[1], L[2]
		y3 = 0.0  # free parameter
		x1, x2, x3, y1, y2 = sympy.symbols("x1 x2 x3 y1 y2", real=True)
		eq1 = sympy.Eq(x1 * z1 + x2 * z2 + x3 * z3, 0.0)  # orthogonality
		eq2 = sympy.Eq(y1 * z1 + y2 * z2 + y3 * z3, 0.0)  # orthogonality
		eq3 = sympy.Eq(y1 * x1 + y2 * x2 + y3 * x3, 0.0)  # orthogonality
		eq4 = sympy.Eq(x1 ** 2 + x2 ** 2 + x3 ** 2, 1.0)  # length = 1
		eq5 = sympy.Eq(y1 ** 2 + y2 ** 2 + y3 ** 2, 1.0)  # length = 1
		try:
			sol = sympy.solve([eq1, eq2, eq3, eq4, eq5])[0]  # find solution that satisfies all eqs
		except:
			print("coord fail", sympy.solve([eq1, eq2, eq3, eq4, eq5]), vector)
			return [[None]]

		v1 = np.array([sol[x1], sol[x2], sol[x3]])
		v2 = np.array([sol[y1], sol[y2], y3])
		v3 = np.array([z1, z2, z3])

		return np.array([v1, v2, v3], dtype=np.float64).transpose()  # new orthonormal basis with z=L

	def measure_rotational_velocity_multiprocessing(self, measure_dispersion=True, calc_basis=True):
		"""
		Measures the average of the rotational component of the velocity in each galaxy, and optionally
		the velocity dispersion.
		Velocity and positions are transformed to frame of reference with spin as z direction.
		Velocity is transformed to cylindrical coordinates and the angular component is taken.
		Velocity dispersion is calculated in cylindrical components (not transformation invariant).
		See Dubois et al. 2016 for description of procedure. See Lohmann et al. 2023, Pulsoni et al. 2020 for equations
		for dispersion/ average velocity. [Extrapolated, also using Dubois et al. 2014]
		:param calc_basis: If basis based on spin is already saved, set this to False to save time.
		:param measure_dispersion: Measure the dispersion of the velocity. Default is True.
		:return: average tangential velocity of each galaxy (if no output file name is given)
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			write_output = True
		else:
			write_output = False
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		self.spin = self.TNG100_SubhaloPT.read_cat("Spin")
		avg_rot_vel, vel_disp, vel_z, vel_disp_cyl, vel_z_abs = [], [], [], [], []
		if calc_basis:
			basis = []
		else:
			try:
				transform_mat = group["Cart_basis_L_is_z"][0]
			except:
				print("Cart_basis_L_is_z does not exist but calc_basis is False. Changing calc_basis to True.")
				calc_basis = True
				basis = []
		if not calc_basis:
			self.transform_mats = group["Cart_basis_L_is_z"]
		self.calc_basis = calc_basis
		self.measure_dispersion = measure_dispersion
		# print('startup')
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_rotational_velocity_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			avg_rot_vel.extend(result[i][0])
			vel_disp.extend(result[i][1])
			vel_z.extend(result[i][2])
			vel_disp_cyl.extend(result[i][3])
			vel_z_abs.extend(result[i][4])
			if calc_basis:
				basis.extend(result[i][5])

		if write_output:
			write_dataset_hdf5(group, "Average_Tangential_Velocity", data=np.array(avg_rot_vel))
			if measure_dispersion:
				write_dataset_hdf5(group, "Velocity_Dispersion", data=np.array(vel_disp))
				write_dataset_hdf5(group, "Average_Velocity_L_direction", data=np.array(vel_z))
				write_dataset_hdf5(group, "Average_Absolute_Velocity_L_direction", data=np.array(vel_z_abs))
				write_dataset_hdf5(group, "Velocity_Dispersion_Cylindrical_Components", data=np.array(vel_disp_cyl))
			if calc_basis:
				write_dataset_hdf5(group, "Cart_basis_L_is_z", data=np.array(basis))
			output_file.close()
			return
		else:
			if measure_dispersion:
				return np.array(avg_rot_vel), np.array(vel_disp)
			return np.array(avg_rot_vel)

	def measure_krot_single(self, indices):
		krot = []
		for n in indices:
			off_n = self.off[n]
			len_n = self.Len[n]
			spin_n = self.spin[n]
			if sum(spin_n) == 0.0:
				krot.append(0.0)
				continue
			if len_n == 1.0:
				krot.append(1.0)
				continue
			spin_dir = spin_n / np.sqrt(sum(spin_n ** 2))
			if self.PT == 1:
				mass = self.DM_part_mass
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]
			elif self.PT == 4 and "TNG" in self.simname:
				wind_or_star = self.TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- self.velocity[n]
				)
				rel_position = (
						self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
						self.COM[n]
				)
			else:
				mass = self.TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_velocity = (
						self.TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- self.velocity[n]
				)
				rel_position = self.TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - self.COM[
					n]

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			position_dir = (rel_position.transpose() / np.sqrt(np.sum(rel_position ** 2, axis=1))).transpose()

			krot.append(
				sum(0.5 * mass * np.sum(np.cross(spin_dir, position_dir) * rel_velocity, axis=1) ** 2)
				/ (np.sum(0.5 * mass * np.sum(rel_velocity ** 2, axis=1)))
			)
		return krot

	def measure_krot_multiporcessing(self):
		"""
		Measure the K_rot parameter for each galaxy as defined in Shi et al. 2016.
		:return: K_rot if output filename is not given.
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			write_output = True
		else:
			write_output = False
		# TNG100_snapshot = ReadTNGdata(
		# 	self.simname,
		# 	self.simname + "_PT" + str(self.PT) + "_subhalos_only",
		# 	self.snapshot,
		# 	data_path=self.data_path,
		# )
		# TNG100_SubhaloPT = ReadTNGdata(
		# 	self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		# )
		# off = TNG100_SubhaloPT.read_cat(self.offset_name)
		# Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		self.spin = self.TNG100_SubhaloPT.read_cat("Spin")
		krot = []
		result = ProcessingPool(nodes=self.numnodes).map(
			self.measure_krot_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.numnodes):
			krot.extend(result[i])
		if write_output:
			write_dataset_hdf5(group, "Krot", data=np.array(krot))
			output_file.close()
			return
		else:
			return np.array(krot)

	def measure_sphericity_triaxality(self, only_S=False, only_T=False):
		"""
		Calculates the sphericity and triaxality of the galaxies given the axis lengths
		:param only_S: Only sphericity is calculated.
		:param only_T: Only triaxality is calculated.
		:return: Sphericity and triaxality unless output file name is given.
		"""
		TNG100_Shapes = ReadTNGdata(
			self.simname, "Shapes", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		axis_lengths = TNG100_Shapes.read_cat("Axis_Lengths")
		a = axis_lengths[:, 2]  # largest
		b = axis_lengths[:, 1]
		c = axis_lengths[:, 0]
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
			write_output = True
		else:
			write_output = False

		if only_T:
			T = (a ** 2 - b ** 2) / (a ** 2 - c ** 2)
			T[np.isnan(T)] = 0.0
			if write_output:
				write_dataset_hdf5(group, "Triaxality", data=T)
				output_file.close()
			else:
				return T
		elif only_S:
			S = c / a
			S[np.isnan(S)] = 0.0
			if write_output:
				write_dataset_hdf5(group, "Sphericity", data=S)
				output_file.close()
			else:
				return S
		else:
			T = (a ** 2 - b ** 2) / (a ** 2 - c ** 2)
			S = c / a
			T[np.isnan(T)] = 0.0
			S[np.isnan(S)] = 0.0
			if write_output:
				write_dataset_hdf5(group, "Sphericity", data=S)
				write_dataset_hdf5(group, "Triaxality", data=T)
				output_file.close()
			else:
				return S, T
		return
