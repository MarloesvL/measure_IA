import numpy as np
import h5py
import time
from numpy.linalg import eig, inv
import sympy
import sys
# sys.set_int_max_str_digits(8600)
from pathos.multiprocessing import ProcessingPool
from src.read_data import ReadData
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.Sim_info import SimInfo

KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureSnapshotVariables(SimInfo):
	"""
	Measures different galaxy variables using the particle snapshot data and galaxy catalogues of hydrodynamical simulations.
	WARNING: Currently set up for TNG and EAGLE only. Should work for other sims, if added to SimInfo.
	Take care with units. Make sure data files are in the specified format. Likely need to add a selection function.
	:param PT: Number indicating particle type
	:param project: Indicator of simulation. Choose from [TNG100, TNG100_2, TNG300, EAGLE] unless added to SimInfo.
	:param snapshot: Number of the snapshot in the simulation
	:param num_nodes: Number of nodes to be used in multiprocessing.
	:param output_file_name: Name and filepath of the file where the output should be stored.
	:param data_path: Start path to the raw data files. Assumes next folder to be self.simname (=project).
	:param snap_data_path: Start path to snapshot data files. If None (default) the data_path will be used.
	:param exclude_wind: If simulation is in the TNG project, True, will exclude wind particles from galaxies.
	:param update: If True, any SimInfo variables can be updated by calling the SimInfo methods they are created in.
	"""

	def __init__(self, PT=4, project=None, snapshot=None, num_nodes=30, output_file_name=None, data_path="./data/raw/",
				 snap_data_path=None, exclude_wind=True, update=False):
		if project == None:
			raise KeyError("Input project name!")
		try:
			self.numPT = len(PT)
			self.PT = PT
			self.PT_group = f"{PT[0]}"
			for p in PT[1:]:
				self.PT_group += f"_PT{p}"
		except TypeError:
			self.PT = [PT]
			self.numPT = 1
			self.PT_group = PT
		SimInfo.__init__(self, project, snapshot, self.PT, update=update)
		TNG100_SubhaloPT = ReadData(
			self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/", data_path=data_path
		)
		try:
			IDs = TNG100_SubhaloPT.read_cat(self.ID_name)
			self.Num_halos = len(IDs)
		except:
			self.Num_halos = 0
		self.exclude_wind = exclude_wind
		self.num_nodes = num_nodes
		self.output_file_name = output_file_name
		self.data_path = data_path
		if snap_data_path != None:
			self.data_path_snap = snap_data_path
		else:
			self.data_path_snap = data_path
		print(
			f"MeasureSnapshotVariables object initialised with:\
			\n simulation {project}, snapshot {snapshot}, parttype(s) {PT} \n \
			excluding wind is {exclude_wind}\n \
			Catalogues are named {self.subhalo_cat}, {self.shapes_cat} and found in {data_path}{project}.\n \
			Snapshot data is in file {self.snap_cat}, found in {self.data_path_snap}.\n \
			{num_nodes} cores are being used in mulitprocessing.")
		return

	def create_self_arguments(self):
		'''
		Creates the reading objects and data variables most widely used in multiprocessing calculations.
		:return:
		'''
		self.TNG100_SubhaloPT = ReadData(
			self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/",
			data_path=self.data_path)
		try:
			IDs = self.TNG100_SubhaloPT.read_cat(self.ID_name)
			self.Num_halos = len(IDs)
		except:
			self.Num_halos = 0
		print(f"There are {self.Num_halos} galaxies/halos in the sample.")
		if self.numPT == 1:
			self.off = [self.TNG100_SubhaloPT.read_cat(self.offset_name)]
			self.Len = [self.TNG100_SubhaloPT.read_cat(self.sub_len_name)]
			if self.exclude_wind:
				self.mass = [self.TNG100_SubhaloPT.read_cat(
					self.mass_name)]
			else:
				self.mass = [self.TNG100_SubhaloPT.read_cat("SubhaloMassType")]
			self.TNG100_snapshot = [ReadData(
				self.simname,
				self.snap_cat,
				self.snapshot,
				data_path=self.data_path_snap,
			)]
		else:
			self.off = self.TNG100_SubhaloPT.read_cat(self.offset_name)
			self.Len = self.TNG100_SubhaloPT.read_cat(self.sub_len_name)
			self.mass = self.TNG100_SubhaloPT.read_cat("Mass")
			self.TNG100_snapshot = []
			for p in np.arange(0, self.numPT):
				try:
					self.TNG100_snapshot.append(ReadData(
						self.simname,
						self.snap_cat[p],
						self.snapshot,
						data_path=self.data_path_snap))
				except TypeError:
					print("Update self.snap_cat to become a list of all snap_cats per given PT")
					exit()

		self.multiproc_chuncks = np.array_split(np.arange(self.Num_halos), self.num_nodes)
		return

	def measure_offsets(self, type="Subhalo"):
		"""
		Measures the Offsets for each subhalo or group needed to read snapshot files.
		EAGLE data differs from TNG and therefore EAGLE is given a separate case.
		Writes away data to output_file in appropriate PT group.
		:param type: Determines if offsets for Subhalos or Groups are calculated (default is Subhalo).
		:return:
		"""
		TNG100_subhalo = ReadData(self.simname, type, self.snapshot, data_path=self.data_path)
		if type == "Subhalo":
			len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		elif self.simname == "EAGLE":
			len = TNG100_subhalo.read_cat(self.sub_len_name)
		else:
			len = TNG100_subhalo.read_subhalo(self.group_len_name)[:, self.PT]
		off = []
		for p in np.arange(0, self.numPT):
			off.append([0])
			off[p].extend(np.cumsum(len[:, p])[:-1])
		off = np.array(off).transpose()

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
		write_dataset_hdf5(group, "Offset_" + type + "_all", data=off)
		output_file.close()
		return

	def omit_wind_only_single(self, indices):
		'''
		Creates a flag for a chunck of galaxies that is 1 if the 'galaxy' contains only wind particles and 0 otherwise.
		The pure wind particle galaxies can then be omitted in the 'select_nonzero_subhalos' method.
		Specific to IllustrisTNG simulations.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
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

	def omit_wind_only(self):
		"""
		Wrapper function for omit_wind_onlu_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results and writes to the output_file.
		:return:
		"""
		TNG100_snapshot = ReadData(self.simname, self.snap_cat, self.snapshot, data_path=self.data_path_snap)
		TNG100_SubhaloPT = ReadData(
			self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
		TNG100_subhalo = ReadData(self.simname, "Subhalo", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		Wind_Flag = []
		multiproc_chuncks = np.array_split(np.arange(len(off)), self.num_nodes)
		self.snap_wind = TNG100_snapshot
		self.off_wind = off
		self.len_wind = Len
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.omit_wind_only_single,
			multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			Wind_Flag.extend(result[i])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
		write_dataset_hdf5(group, "Wind_Flag", Wind_Flag)
		output_file.close()
		return

	def select_nonzero_subhalos(self, IDs=None, wind_flag=None):
		"""
		Selects the subhalos that have nonzero length, mass and SubhaloFlag. Saves selected data in output file, including
		the IDs for the original file. Specific to IllustrisTNG simulations.
		:return:
		"""
		TNG100_subhalo = ReadData(self.simname, "Subhalo", self.snapshot, data_path=self.data_path)
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)[:, self.PT]
		mass_subhalo = TNG100_subhalo.read_subhalo("SubhaloMassType")[:, self.PT]
		subhalo_pos = TNG100_subhalo.read_subhalo("SubhaloPos")
		SFR = TNG100_subhalo.read_subhalo(self.SFR_name)
		photo_mag = TNG100_subhalo.read_subhalo(self.photo_name)
		if self.numPT == 1:
			if self.PT == 4:
				flag = TNG100_subhalo.read_subhalo(self.flag_name)
				TNG100_SubhaloPT = ReadData(
					self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/",
					data_path=self.data_path
				)
				off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
				if self.PT == 4:
					wind_flag = TNG100_SubhaloPT.read_cat("Wind_Flag")
					mask = (Len > 0.0) * (flag == 1) * (wind_flag == 0)
				else:
					mask = (Len > 0.0) * (flag == 1)
				IDs = np.where(mask)[0]
				self.Num_halos = len(mass_subhalo[mask])
				output_file = h5py.File(self.output_file_name, "a")
				group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
				write_dataset_hdf5(group, self.sub_len_name, Len[mask])
				write_dataset_hdf5(group, self.offset_name, off[mask])
				write_dataset_hdf5(group, "SubhaloMassType", mass_subhalo[mask])
				write_dataset_hdf5(group, "SubhaloPos", subhalo_pos[mask])
				write_dataset_hdf5(group, self.ID_name, IDs)
				if self.PT == 4:
					photo_mag = TNG100_subhalo.read_subhalo(self.photo_name)
					write_dataset_hdf5(group, self.photo_name, photo_mag[mask])
					write_dataset_hdf5(group, self.SFR_name, SFR[mask])
				output_file.close()
			elif self.PT == 0:
				if IDs == None:
					try:
						TNG100_SubhaloPT4 = ReadData(
							self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT4/", data_path=self.data_path
						)
						IDs = TNG100_SubhaloPT4.read_cat(self.ID_name)
					except:
						print(f"SubhaloIDs not found in {self.subhalo_cat} group PT4, add manually")
				TNG100_SubhaloPT = ReadData(
					self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/",
					data_path=self.data_path
				)
				off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
				mask = (Len[IDs] > 0.0)
				IDs_mask = IDs[mask]

				self.Num_halos = len(mass_subhalo[IDs_mask])
				output_file = h5py.File(self.output_file_name, "a")
				group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
				write_dataset_hdf5(group, self.sub_len_name, Len[IDs_mask])
				write_dataset_hdf5(group, self.offset_name, off[IDs_mask])
				write_dataset_hdf5(group, "GasMass", mass_subhalo[IDs_mask])
				write_dataset_hdf5(group, "SubhaloPos", subhalo_pos[IDs_mask])
				write_dataset_hdf5(group, self.ID_name, IDs_mask)
				write_dataset_hdf5(group, self.SFR_name, SFR[IDs_mask])
				output_file.close()
			else:
				raise KeyError('No version of select_nonzero_subhalos exists for your chosen PT')
		else:
			TNG100_SubhaloPT = ReadData(
				self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/",
				data_path=self.data_path
			)
			off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")
			flag = TNG100_subhalo.read_subhalo(self.flag_name)
			mask = np.ones(np.shape(SFR), dtype=bool)
			for p, PT in enumerate(self.PT):
				if PT == 4:
					if wind_flag.any() == None:
						try:
							TNG100_SubhaloPT_4 = ReadData(
								self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT4/",
								data_path=self.data_path
							)
							wind_flag = TNG100_SubhaloPT_4.read_cat("Wind_Flag")
						except:
							print(f"Wind_Flag not found in {self.subhalo_cat} group PT4, add manually")
					mask = mask * (Len[:, p] > 0.0) * (flag == 1) * (wind_flag == 0)

				elif PT == 0:
					mask = mask * (Len[:, p] > 0.0) * (flag == 1)
				else:
					raise KeyError('No version of select_nonzero_subhalos exists for your chosen PT')
			IDs = np.where(mask)[0]
			mass_subhalo = np.sum(mass_subhalo[mask], axis=1)
			self.Num_halos = len(mass_subhalo)
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			write_dataset_hdf5(group, self.photo_name, photo_mag[mask])
			write_dataset_hdf5(group, self.sub_len_name, Len[mask])
			write_dataset_hdf5(group, self.offset_name, off[mask])
			write_dataset_hdf5(group, "SubhaloMassType", mass_subhalo)
			write_dataset_hdf5(group, "SubhaloPos", subhalo_pos[mask])
			write_dataset_hdf5(group, self.ID_name, IDs)
			write_dataset_hdf5(group, self.SFR_name, SFR[mask])
			output_file.close()
		return

	def select_nonzero_subhalos_EAGLE(self):
		"""
		Selects the subhalos that have nonzero length, mass and SubhaloFlag. Furthermore, this method sorts the galaxies
		according to their group number and subgroup number so the offsets and lengths can be used to read the correct
		particles in the snapshot files.
		Saves selected data in output file, including the IDs for the original file. Specific to EAGLE simulations.
		:return:
		"""
		EAGLE_subhalo = ReadData(self.simname, "Subhalo_cat", self.snapshot, data_path=self.data_path)
		Len = EAGLE_subhalo.read_cat(self.sub_len_name)
		mass_subhalo = EAGLE_subhalo.read_cat(self.mass_name)
		gn = EAGLE_subhalo.read_cat("GroupNumber")
		sn = EAGLE_subhalo.read_cat("SubGroupNumber")  # less
		galaxyIDs = EAGLE_subhalo.read_cat("GalaxyID")
		file = h5py.File(f"{self.data_path}/EAGLE/diff_gnsn.hdf5", 'r')
		group = file[f"Snapshot_{self.snapshot}"]
		indices_gnsn_in_sub = group['indices_gnsn_in_sub'][:, 0]
		indices_sub_in_gnsn = group["indices_sub_in_gnsn"][:, 0]  # dees
		file.close()
		flag = EAGLE_subhalo.read_cat(self.flag_name)
		mask = (mass_subhalo > 0.0) * (flag == 0)
		TNG100_SubhaloPT = ReadData(
			self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/", data_path=self.data_path
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
		self.Num_halos = len(mass_subhalo[indices_gnsn_in_sub])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
		write_dataset_hdf5(group, self.sub_len_name, Len[indices_sub_in_gnsn])
		write_dataset_hdf5(group, self.offset_name, off[indices_sub_in_gnsn])
		write_dataset_hdf5(group, self.mass_name, mass_subhalo * self.h / 1e10)  # 10^10 M_sun/h
		write_dataset_hdf5(group, self.ID_name, galaxyIDs)
		write_dataset_hdf5(group, "GroupNumber", gn)
		write_dataset_hdf5(group, "SubGroupNumber", sn)
		if self.PT == 4:
			SFR = EAGLE_subhalo.read_cat(self.SFR_name)
			SFR = SFR[mask]
			SFR = SFR[sorted_indices]
			write_dataset_hdf5(group, self.SFR_name, SFR)
		output_file.close()
		return

	def save_number_of_particles_single(self, indices):
		'''
		Saves the number of particles for a chunck of galaxies used in the snapshot calculations.
		This is usually equal to the 'Len' parameter,
		except when wind particles are omitted from the stellar particles, as in IllustrisTNG simulations.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		number_of_particles_list = []
		for n in indices:
			number_of_particles = 0
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					number_of_particles += sum(star_mask)
				else:
					number_of_particles += len_n
			number_of_particles_list.append(number_of_particles)
		return number_of_particles_list

	def save_number_of_particles(self):
		"""
		Wrapper function for save_number_of_particles_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results and writes to the output_file.
		:return:
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		number_of_particles_list = []
		multiproc_chuncks = np.array_split(np.arange(len(self.off)), self.num_nodes)
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.save_number_of_particles_single,
			multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			number_of_particles_list.extend(result[i])

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
		write_dataset_hdf5(group, "Number_of_particles", data=np.array(number_of_particles_list))
		output_file.close()
		return

	def measure_masses_excl_wind_single(self, indices):
		'''
		Measures the stellar masses of a chunck of the galaxies used in the snapshot calculations.
		This is usually equal to the 'SubhaloMassType' parameter,
		except when wind particles are omitted from the stellar particles, as in the IllutrisTNG simulations.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		mass_list = []
		for n in indices:
			masses_n = []
			for p in np.arange(0, self.numPT):
				off_n = self.off_mass[n, p]
				len_n = self.Len_mass[n, p]
				if self.PT[p] == 4:
					wind_or_star = self.TNG100_snapshot_mass[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					masses = self.TNG100_snapshot_mass[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])[
						star_mask]
				else:
					masses = self.TNG100_snapshot_mass[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])
				masses_n.extend(masses)
			mass = sum(masses_n)
			mass_list.append(mass)
		return mass_list

	def measure_masses_excl_wind(self):
		"""
		Wrapper function for measure_masses_excl_wind_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results and writes to the output_file.
		:return:
		"""
		if (self.PT == 4 and "TNG" in self.simname and self.exclude_wind) or self.numPT > 1:
			pass
		else:
			print('Use given mass for this input')
			exit()
		TNG100_SubhaloPT = ReadData(
			self.simname, self.subhalo_cat, self.snapshot, sub_group=f"PT{self.PT_group}/", data_path=self.data_path
		)
		if self.numPT == 1:
			self.off_mass = [TNG100_SubhaloPT.read_cat(self.offset_name)]
			self.Len_mass = [TNG100_SubhaloPT.read_cat(self.sub_len_name)]
			self.TNG100_snapshot_mass = [ReadData(
				self.simname,
				self.snap_cat,
				self.snapshot,
				data_path=self.data_path_snap,
			)]
		else:
			self.off_mass = TNG100_SubhaloPT.read_cat(self.offset_name)
			self.Len_mass = TNG100_SubhaloPT.read_cat(self.sub_len_name)
			self.TNG100_snapshot_mass = []
			for p in np.arange(0, self.numPT):
				try:
					self.TNG100_snapshot_mass.append(ReadData(
						self.simname,
						self.snap_cat[p],
						self.snapshot,
						data_path=self.data_path_snap))
				except TypeError:
					print("Update self.snap_cat to become a list of all snap_cats per given PT")
					exit()
		mass_list = []
		multiproc_chuncks = np.array_split(np.arange(len(self.off_mass)), self.num_nodes)
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_masses_excl_wind_single,
			multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			mass_list.extend(result[i])

		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
		if self.numPT > 1:
			write_dataset_hdf5(group, "Mass", data=np.array(mass_list))
		else:
			write_dataset_hdf5(group, "StellarMass", data=np.array(mass_list))
		output_file.close()
		return

	def measure_velocities_single(self, indices):
		'''
		Measures the galaxy velocity (km/s * a) for a chunck of galaxies for the initalised PT, weighted by
		particle mass.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		velocities = []
		for n in indices:
			mass_vel_sum = np.array([0., 0., 0.])
			mass_n = self.mass[n]
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 1:
					mass_particles = self.DM_part_mass
					velocity_particles = self.TNG100_snapshot[p].read_cat(
						self.velocities_name, cut=[off_n, off_n + len_n]
					) * np.sqrt(self.scalefactor)
				elif self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					mass_particles = self.TNG100_snapshot[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])[
						star_mask]
					velocity_particles = \
						self.TNG100_snapshot[p].read_cat(self.velocities_name, cut=[off_n, off_n + len_n])[
							star_mask
						] * np.sqrt(self.scalefactor)
				else:
					mass_particles = self.TNG100_snapshot[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])
					velocity_particles = self.TNG100_snapshot[p].read_cat(
						self.velocities_name, cut=[off_n, off_n + len_n]
					) * np.sqrt(self.scalefactor)

				mass_vel = (velocity_particles.transpose() * mass_particles).transpose()
				mass_vel_sum += np.sum(mass_vel, axis=0)
			velocity = mass_vel_sum / mass_n
			velocities.append(velocity)
		return velocities

	def measure_velocities(self):
		"""
		Wrapper function for measure_velocities_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results and writes to the output_file.
		:return:
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		velocities = []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_velocities_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			velocities.extend(result[i])
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			write_dataset_hdf5(group, "Velocity", data=np.array(velocities))
			output_file.close()
			return
		else:
			return np.array(velocities)

	def measure_COM_single(self, indices):
		'''
		Measures the centre of mass (ckpc/h) of each galaxy for a chunck of galaxies for the initalised PT.
		Coordinates weighted by particle mass. Periodicity of the box is accounted for.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		COM = []
		for n in indices:
			mass_n = self.mass[n]
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 1:
					mass_particles = self.DM_part_mass
					coordinates_particles = self.TNG100_snapshot[p].read_cat(self.coordinates_name,
																			 cut=[off_n, off_n + len_n])
				elif self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					mass_particles = self.TNG100_snapshot[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])[
						star_mask]
					coordinates_particles = \
						self.TNG100_snapshot[p].read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])[
							star_mask
						]
				else:
					mass_particles = self.TNG100_snapshot[p].read_cat(self.masses_name, cut=[off_n, off_n + len_n])
					coordinates_particles = self.TNG100_snapshot[p].read_cat(self.coordinates_name,
																			 cut=[off_n, off_n + len_n])
				if p == 0:
					coordinates_particles_n = coordinates_particles
					mass_particles_n = mass_particles
				else:
					coordinates_particles_n = np.append(coordinates_particles_n, coordinates_particles, axis=0)
					mass_particles_n = np.append(mass_particles_n, mass_particles)

				assert min(coordinates_particles[:, 0]) >= 0. and min(coordinates_particles[:, 1]) >= 0. and min(
					coordinates_particles[:, 2]) >= 0., "Minimum coordinates particles negative."
				assert max(coordinates_particles[:, 0]) <= self.boxsize and max(
					coordinates_particles[:, 1]) <= self.boxsize and max(
					coordinates_particles[:, 2]) <= self.boxsize, "Maximum coordinates particles larger than boxsize."

			# account for periodicity of the box
			min_coord = np.min(coordinates_particles_n, axis=0)
			coordinates_particles_n[(coordinates_particles_n - min_coord) > self.L_0p5] -= self.boxsize
			coordinates_particles_n[(coordinates_particles_n - min_coord) < -self.L_0p5] += self.boxsize

			if self.PT[0] == 4 and self.numPT == 1:
				try:
					assert np.isclose(np.sum(mass_particles_n), mass_n,
									  rtol=1e-5), f"Sum particle masses unequal to mass for galaxy {n} with len {len_n}. sum: {np.sum(mass_particles_n)}, mass: {mass_n}"
				except AssertionError as ass_err:
					if "TNG" in self.simname:
						wind_or_star = self.TNG100_snapshot[0].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
						star_mask = wind_or_star > 0
						mass_particles_nw = \
							self.TNG100_snapshot[0].read_cat(self.masses_name, cut=[off_n, off_n + len_n])[
								star_mask]
						try:
							assert np.isclose(np.sum(mass_particles_nw), mass_n,
											  rtol=1e-5), f"Sum particle masses without wind unequal to mass for galaxy {n} with len {len_n}. sum: {np.sum(mass_particles_nw)}, mass: {mass_n}"
							print(f"Using particles without wind for galaxy {n}.")
							mass_particles_n = mass_particles_nw
							coordinates_particles = \
								self.TNG100_snapshot[0].read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])[
									star_mask
								]
							coordinates_particles[coordinates_particles > self.L_0p5] -= self.boxsize
							coordinates_particles[coordinates_particles < -self.L_0p5] += self.boxsize
						except AssertionError as exc:
							print(exc)
							exit()
					else:
						print(ass_err)
						exit()
			mass_coord = (coordinates_particles_n.transpose() * mass_particles_n).transpose()
			COM_n = np.sum(mass_coord, axis=0) / mass_n
			COM_n[COM_n < 0.0] += self.boxsize  # if negative: COM is on other side of box.
			assert (COM_n < self.boxsize).all() and (COM_n > 0.0).all(), "COM coordinate not inside of box"
			COM.append(COM_n)
		return COM

	def measure_COM(self):
		"""
		Wrapper function for measure_COM_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results and writes to the output_file.
		:return:
		"""
		try:
			Len = self.Len  # exists if self.create_self_arguments() has been run
		except:
			self.create_self_arguments()
		COM = []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_COM_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			COM.extend(result[i])
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			write_dataset_hdf5(group, "COM", data=np.array(COM))
			output_file.close()
			return
		else:
			return np.array(COM)

	def measure_inertia_tensor_single(self, indices):
		'''
		Measures the inertia tensor, eigen vectors and eigen values for the galaxies within a given chunck of galaxies.
		Periodicity of the box is accounted for.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		for n in indices:
			I = np.zeros((3, 3))
			mass = self.mass[n]
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]

				if self.PT[p] == 1:
					particle_mass = self.DM_part_mass
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
				elif self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])[
						star_mask]
					rel_position = (
							self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
							self.COM[n]
					)
				else:
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
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
					I = self.measure_inertia_tensor_eq(mass, particle_mass, rel_position, self.reduced, I)
					if sum(np.isnan(I.flatten())) > 0 or sum(np.isinf(I.flatten())) > 0:
						print(
							f"NaN or inf found in galaxy {n}, I is {I}, mass {mass}, len {len_n}. Appending zeros.")
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
	def measure_inertia_tensor_eq(mass, particle_mass, rel_position, reduced=False, I=np.zeros((3, 3))):
		"""
		Calculates the simple or reduced inertia tensor values for a single galaxy.
		:param mass: Mass of galaxy.
		:param particle_mass: Mass of particles in galaxy.
		:param rel_position: Position of particles relative to centre of mass of galaxy.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:return: (reduced) inertia tensor
		"""
		if reduced:
			r = np.sqrt(np.sum(rel_position ** 2, axis=1))
			rel_position = (rel_position.transpose() / r).transpose()  # normalise the positions
		for i in np.arange(0, 3):
			for j in np.arange(0, 3):
				I[i, j] += 1.0 / mass * np.sum(particle_mass * rel_position[:, i] * rel_position[:, j])
		return I

	def measure_inertia_tensor(self, eigen_v=True, sorted=True, reduced=False):
		"""
		Wrapper function for measure_inertia_tensor_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results, sorts the eigenvectors and eigen values in ascending
		order and writes to the output_file. Note: this function assumes that the centre of mass was calculated using
		the method measure_COM. If this is not the case, make sure your galaxy catalogue contains the centre of masses,
		named 'COM'.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:param sorted: Sorts eigen values and eigen vectors from lowest to highest if True.
		:param eigen_v: Also returns eigen values and vectors if True.
		:return: The inertia tensor, eigen values and vectors if no output file is specified.
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		self.reduced = reduced
		self.eigen_v = eigen_v
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_inertia_tensor_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
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
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
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
			output_file.close()
			return
		else:
			return np.array(I_list)

	def measure_projected_inertia_tensor_single(self, indices):
		'''
		Measures the projected inertia tensor, eigen vectors and eigen values for the galaxies within a given chunck of
		galaxies. Periodicity of the box is accounted for.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		I_list, value_list, v0, v1, vectors_list = [], [], [], [], []
		for n in indices:
			I = np.zeros((2, 2))
			mass = self.mass[n]
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]

				if self.PT[p] == 1:
					particle_mass = self.DM_part_mass
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
				elif self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])[
						star_mask]
					rel_position = (
							self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
							self.COM[n]
					)
				else:
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
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
				else:
					I = self.measure_projected_inertia_tensor_eq(mass, particle_mass, rel_position, self.reduced, I)
					if sum(np.isnan(I.flatten())) > 0 or sum(np.isinf(I.flatten())) > 0:
						print(
							f"NaN or inf found in galaxy {n}, I is {I}, mass {mass}, len {len_n}. Appending zeros.")
						I_list.append([0.0, 0.0, 0.0, 0.0])
						value_list.append([0.0, 0.0])
						vectors_list.append(np.array([[0.0, 0.0], [0.0, 0.0]]))
						v0.append([0.0, 0.0])
						v1.append([0.0, 0.0])
					else:
						if self.eigen_v:
							values, vectors = eig(I)
							value_list.append(values)
							vectors_list.append(vectors)
							v0.append(vectors[:, 0])
							v1.append(vectors[:, 1])
						I_list.append(I.reshape(4))
		return I_list, value_list, v0, v1, vectors_list

	@staticmethod
	def measure_projected_inertia_tensor_eq(mass, particle_mass, rel_position, reduced=False, I=np.zeros((2, 2))):
		"""
		Calculates the simple or reduced projected inertia tensor values for a single galaxy.
		:param mass: Mass of galaxy.
		:param particle_mass: Mass of particles in galaxy.
		:param rel_position: Position of particles relative to centre of mass of galaxy.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:return: (reduced) inertia tensor
		"""
		if reduced:
			r = np.sqrt(np.sum(rel_position ** 2, axis=1))
			rel_position = (rel_position.transpose() / r).transpose()  # normalise the positions
		for i in np.arange(0, 2):
			for j in np.arange(0, 2):
				I[i, j] += 1.0 / mass * np.sum(particle_mass * rel_position[:, i] * rel_position[:, j])
		return I

	def measure_projected_inertia_tensor(self, eigen_v=True, sorted=True, reduced=False, LOS_ind=2):
		"""
		Wrapper function for measure_projected_inertia_tensor_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results, sorts the eigenvectors and eigen values in ascending
		order and writes to the output_file. Note: this function assumes that the centre of mass was calculated using
		the method measure_COM. If this is not the case, make sure your galaxy catalogue contains the centre of masses,
		named 'COM'.
		:param reduced: Calculate reduced (True) or simple (False) inertia tensor.
		:param sorted: Sorts eigen values and eigen vectors from lowest to highest if True.
		:param eigen_v: Also returns eigen values and vectors if True.
		:param LOS_ind: Index of the line of sight coordinate, over which the shape is projected.
		:return: The inertia tensor, eigen values and vectors if no output file is specified.
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		self.reduced = reduced
		self.eigen_v = eigen_v
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]
		LOS_axis = {0: 'x', 1: 'y', 2: 'z'}
		I_list, value_list, v0, v1, vectors_list = [], [], [], [], []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_projected_inertia_tensor_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			I_list.extend(result[i][0])
			value_list.extend(result[i][1])
			v0.extend(result[i][2])
			v1.extend(result[i][3])
			vectors_list.extend(result[i][4])

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

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
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
							group, f"Projected_Axis_Lengths_reduced_{LOS_axis[LOS_ind]}",
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
							group, f"Projected_Axis_Lengths_{LOS_axis[LOS_ind]}",
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

	def measure_spin_single(self, indices):
		'''
		Measures spin (angular momentum) for the galaxies within a given chunck of
		galaxies in kpc km s^-1 M_sun. Periodicity of the box is accounted for.
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		'''
		spin_list = []
		for n in indices:
			spin = np.array([0., 0., 0.])
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 1:
					particle_mass = self.DM_part_mass * 1e10
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)
				elif self.PT[p] == 4 and "TNG" in self.simname and self.exclude_wind:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])[
										star_mask] * 1e10
					rel_position = (
							self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
							self.COM[n]
					)
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
							* np.sqrt(self.scalefactor)
							- self.velocity[n]
					)

				else:
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n]) * 1e10
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)

				rel_position[rel_position > self.L_0p5] -= self.boxsize
				rel_position[rel_position < -self.L_0p5] += self.boxsize

				spin += np.sum((particle_mass * np.cross(rel_position, rel_velocity).transpose()).transpose(), axis=0)
			spin_list.append(spin)
		return spin_list

	def measure_spin(self):
		"""
		Wrapper function for measure_spin_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results, and writes to the output_file.
		Note: this function assumes that the centre of mass was calculated using
		the method measure_COM. If this is not the case, make sure your galaxy catalogue contains the centre of masses,
		named 'COM'. The same goes for the galaxy velocity (method: measure_velocity; name: 'Velocity').
		Measures spin (angular momentum) of galaxies in kpc km s^-1 M_sun.
		:return: Spin if no output file name is specified.
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			write_output = True
		else:
			write_output = False
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		spin_list = []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_spin_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
			spin_list.extend(result[i])
		if write_output:
			write_dataset_hdf5(group, "Spin", data=np.array(spin_list))
			output_file.close()
			return
		else:
			return np.array(spin_list)

	def measure_rotational_velocity_single(self, indices):
		"""
		Measures the average of the rotational component of the velocity in each galaxy for the galaxies within a given
		chunck of galaxies, and optionally the velocity dispersion.
		Velocity and positions are transformed to frame of reference with spin as z direction.
		Velocity is transformed to cylindrical coordinates and the angular component is taken.
		Velocity dispersion is calculated in cylindrical components (not transformation invariant).
		See Dubois et al. 2016 for description of procedure. See Lohmann et al. 2023, Pulsoni et al. 2020 for equations
		for dispersion/ average velocity. [Extrapolated, also using Dubois et al. 2014]
		:param indices: indices for the chunck of galaxies to be calculated.
		:return:
		"""
		avg_rot_vel, vel_disp, vel_z, vel_disp_cyl, vel_z_abs = [], [], [], [], []
		if self.calc_basis:
			basis = []
		for n in indices:
			vel_theta_mean = 0.
			mean_vel_z = 0.
			abs_vel_z = 0.
			mass = self.mass[n]
			if sum(self.Len[n]) < 2:
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
			transform_mat_inv = inv(transform_mat)
			assert np.allclose(transform_mat @ transform_mat_inv, np.eye(3, 3),
							   atol=1e-10), "Inverse of basis @ basis not within 1e-10 of identity."

			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 1:
					particle_mass = self.DM_part_mass
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
				elif self.PT[p] == 4 and "TNG" in self.simname:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])[
						star_mask]
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
							* np.sqrt(self.scalefactor)
							- self.velocity[n]
					)
					rel_position = (
							self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
							self.COM[n]
					)
				else:
					particle_mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]

				# get coords relative to galactic COM, velocity
				rel_position[rel_position > self.L_0p5] -= self.boxsize
				rel_position[rel_position < -self.L_0p5] += self.boxsize
				# transform to new coords
				rel_velocity_transform = np.dot(transform_mat_inv, rel_velocity.transpose()).transpose()
				rel_position_transform = np.dot(transform_mat_inv, rel_position.transpose()).transpose()
				del rel_position, rel_velocity
				theta = np.arctan2(rel_position_transform[:, 1], rel_position_transform[:, 0])  # +np.pi  # arctan(y/x)
				del rel_position_transform
				# + np.pi  # +pi changes the sign of v_theta. theta: [-pi,pi] or [0,2pi]
				vel_theta = -rel_velocity_transform[:, 0] * np.sin(theta) + rel_velocity_transform[:, 1] * np.cos(theta)
				vel_theta_mean += (
						np.sum((vel_theta.transpose() * particle_mass).transpose(), axis=0) / mass
				)  # mass-weighted mean

				if self.measure_dispersion:
					vel_r_n = rel_velocity_transform[:, 0] * np.cos(theta) + rel_velocity_transform[:, 1] * np.sin(
						theta)
					vel_z_n = rel_velocity_transform[:, 2]
					del rel_velocity_transform
					if p == 0:
						rel_velocity_cyl = np.array([vel_r_n, vel_theta, vel_z_n])
						particle_mass_n = particle_mass
					else:
						rel_velocity_cyl = np.append(rel_velocity_cyl,
													 np.array([vel_r_n, vel_theta, vel_z_n]), axis=1)
						particle_mass_n = np.append(particle_mass_n, particle_mass)
					mean_vel_z += np.sum(vel_z_n * particle_mass) / mass
					abs_vel_z += np.sum(abs(vel_z_n) * particle_mass) / mass

			mean_vel_cyl = np.sum((rel_velocity_cyl * particle_mass_n), axis=1) / mass
			dvel_cyl = (rel_velocity_cyl.transpose() - mean_vel_cyl) ** 2
			dvel_mean_cyl = np.sum((dvel_cyl.transpose() * particle_mass_n), axis=1) / mass
			vel_z.append(mean_vel_z)
			vel_z_abs.append(abs_vel_z)
			del particle_mass
			vel_disp_cyl.append(dvel_mean_cyl)
			vel_disp.append(np.sqrt(np.sum(dvel_mean_cyl) / 3.0))
			avg_rot_vel.append(vel_theta_mean)

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
		try:
			v1 = np.array([sol[x1], sol[x2], sol[x3]], dtype=np.float64)
			v2 = np.array([sol[y1], sol[y2], y3], dtype=np.float64)
			v3 = np.array([z1, z2, z3], dtype=np.float64)
		except TypeError:
			print(sol)
			print("Solving again with y2=0.0")
			y2 = 0.0  # another free parameter
			x1, x2, x3, y1 = sympy.symbols("x1 x2 x3 y1", real=True)
			eq1 = sympy.Eq(x1 * z1 + x2 * z2 + x3 * z3, 0.0)  # orthogonality
			eq2 = sympy.Eq(y1 * z1 + y2 * z2 + y3 * z3, 0.0)  # orthogonality
			eq3 = sympy.Eq(y1 * x1 + y2 * x2 + y3 * x3, 0.0)  # orthogonality
			eq4 = sympy.Eq(x1 ** 2 + x2 ** 2 + x3 ** 2, 1.0)  # length = 1
			eq5 = sympy.Eq(y1 ** 2 + y2 ** 2 + y3 ** 2, 1.0)  # length = 1
			try:
				sol = sympy.solve([eq1, eq2, eq3, eq4, eq5])[0]  # find solution that satisfies all eqs
				v1 = np.array([sol[x1], sol[x2], sol[x3]], dtype=np.float64)
				v2 = np.array([sol[y1], y2, y3], dtype=np.float64)
				v3 = np.array([z1, z2, z3], dtype=np.float64)
			except:
				print("coord fail", sympy.solve([eq1, eq2, eq3, eq4, eq5]), vector)
				return [[None]]
		assert np.isclose(np.sum(v1 ** 2), 1.0, atol=1e-5), f"V1 not unit length. {np.sum(v1 ** 2)}"
		assert np.isclose(np.sum(v2 ** 2), 1.0, atol=1e-5), f"V2 not unit length. {np.sum(v2 ** 2)}"
		assert np.isclose(np.sum(v3 ** 2), 1.0, atol=1e-5), f"V3 not unit length. {np.sum(v3 ** 2)}"
		try:
			assert np.isclose(np.dot(v1, v2), 0.0,
							  atol=1e-5), f"V1, V2 not orthogonal. (dot product, v1, v2, v3, vector): {np.dot(v1, v2), v1, v2, v3, vector}"
			assert np.isclose(np.dot(v1, v3), 0.0,
							  atol=1e-5), f"V1, V3 not orthogonal. (dot product, v1, v2, v3, vector): {np.dot(v1, v3), v1, v2, v3, vector}"
			assert np.isclose(np.dot(v3, v2), 0.0,
							  atol=1e-5), f"V3, V2 not orthogonal. (dot product, v1, v2, v3, vector): {np.dot(v3, v2), v1, v2, v3, vector}"
		except AssertionError:
			if max(np.dot(v1, v2), np.dot(v1, v3), np.dot(v3, v2)) > 1e-3:
				print('ERROR', np.dot(v1, v2), np.dot(v1, v3), np.dot(v3, v2))

		return np.array([v1, v2, v3], dtype=np.float64).transpose()  # new orthonormal basis with z=L

	def measure_rotational_velocity(self, measure_dispersion=True, calc_basis=True):
		"""
		Wrapper function for measure_rotational_velocity_single method. This function reads the data and creates the
		multiprocessing pool. Finally, it gathers the results, and writes to the output_file.
		:param calc_basis: If basis based on spin is already saved, set this to False to save time.
		:param measure_dispersion: Measure the dispersion of the velocity. Default is True.
		:return: average tangential velocity of each galaxy (if no output file name is given)
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		self.spin = self.TNG100_SubhaloPT.read_cat("Spin")
		avg_rot_vel, vel_disp, vel_z, vel_disp_cyl, vel_z_abs = [], [], [], [], []
		if calc_basis:
			basis = []
		else:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			try:
				transform_mat = group["Cart_basis_L_is_z"][0]
			except:
				print("Cart_basis_L_is_z does not exist but calc_basis is False. Changing calc_basis to True.")
				calc_basis = True
				basis = []
			output_file.close()
		if not calc_basis:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			self.transform_mats = group["Cart_basis_L_is_z"]
			output_file.close()
		self.calc_basis = calc_basis
		self.measure_dispersion = measure_dispersion

		multiproc_chuncks = np.array_split(np.arange(self.Num_halos), self.num_nodes * 3)  # high memory usage
		for j in [0, 3, 6]:
			result = ProcessingPool(nodes=self.num_nodes).map(
				self.measure_rotational_velocity_single,
				multiproc_chuncks[j:j + 3],
			)
			for i in np.arange(self.num_nodes):
				avg_rot_vel.extend(result[i][0])
				vel_disp.extend(result[i][1])
				vel_z.extend(result[i][2])
				vel_disp_cyl.extend(result[i][3])
				vel_z_abs.extend(result[i][4])
				if calc_basis:
					basis.extend(result[i][5])

		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
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
			spin_n = self.spin[n]
			if sum(spin_n) == 0.0:
				krot.append(0.0)
				continue
			spin_dir = spin_n / np.sqrt(sum(spin_n ** 2))
			krot_top = 0
			krot_bottom = 0
			for p in np.arange(0, self.numPT):
				off_n = self.off[n, p]
				len_n = self.Len[n, p]
				if self.PT[p] == 1:
					mass = self.DM_part_mass
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]
				elif self.PT[p] == 4 and "TNG" in self.simname:
					wind_or_star = self.TNG100_snapshot[p].read_cat(self.wind_name, cut=[off_n, off_n + len_n])
					star_mask = wind_or_star > 0
					mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
							* np.sqrt(self.scalefactor)
							- self.velocity[n]
					)
					rel_position = (
							self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] -
							self.COM[n]
					)
				else:
					mass = self.TNG100_snapshot[p].read_cat(self.masses_name, [off_n, off_n + len_n])
					rel_velocity = (
							self.TNG100_snapshot[p].read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
						self.scalefactor)
							- self.velocity[n]
					)
					rel_position = self.TNG100_snapshot[p].read_cat(self.coordinates_name, [off_n, off_n + len_n]) - \
								   self.COM[
									   n]

				rel_position[rel_position > self.L_0p5] -= self.boxsize
				rel_position[rel_position < -self.L_0p5] += self.boxsize
				position_dir = (rel_position.transpose() / np.sqrt(np.sum(rel_position ** 2, axis=1))).transpose()

				krot_top += sum(0.5 * mass * np.sum(np.cross(spin_dir, position_dir) * rel_velocity, axis=1) ** 2)
				krot_bottom += np.sum(0.5 * mass * np.sum(rel_velocity ** 2, axis=1))
			krot.append(krot_top / krot_bottom)
		return krot

	def measure_krot(self):
		"""
		Measure the K_rot parameter for each galaxy as defined in Shi et al. 2016.
		:return: K_rot if output filename is not given.
		"""
		try:
			Len = self.Len
		except:
			self.create_self_arguments()
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
			write_output = True
		else:
			write_output = False

		self.COM = self.TNG100_SubhaloPT.read_cat("COM")
		self.velocity = self.TNG100_SubhaloPT.read_cat("Velocity")
		self.spin = self.TNG100_SubhaloPT.read_cat("Spin")
		krot = []
		result = ProcessingPool(nodes=self.num_nodes).map(
			self.measure_krot_single,
			self.multiproc_chuncks,
		)
		for i in np.arange(self.num_nodes):
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
		TNG100_Shapes = ReadData(
			self.simname, self.shapes_cat, self.snapshot, sub_group=f"PT{self.PT_group}/", data_path=self.data_path
		)
		axis_lengths = TNG100_Shapes.read_cat("Axis_Lengths")
		a = axis_lengths[:, 2]  # largest
		b = axis_lengths[:, 1]
		c = axis_lengths[:, 0]
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT_group))
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
