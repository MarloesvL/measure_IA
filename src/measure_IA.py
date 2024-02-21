import math
import numpy as np
import h5py
import time
from numpy.linalg import eig, inv
from scipy.special import lpmn
import sympy
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.stats import binned_statistic
from src.read_data_TNG import ReadTNGdata
from src.write_data import write_dataset_hdf5, create_group_hdf5
from src.Sim_info import SimInfo

KPC_TO_KM = 3.086e16  # 1 kpc is 3.086e16 km


class MeasureVariablesSnapshot(SimInfo):
	"""
	Measures different variables using the particle snapshot data.
	WARNING: Currently set up for TNG only. Should work for other sims. Take care with velocity units.
	:param PT: Number indicating particle type
	:param project: Indicator of simulation. Choose from [TNG100, TNG300] for now.
	:param snapshot: Number of the snapshot
	:param output_file_name: Name and filepath of the file where the output should be stored.
	:param data_path: Start path to the raw data files
	"""

	def __init__(self, PT=4, project=None, snapshot=None, output_file_name=None, data_path="./data/raw/"):
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
		self.output_file_name = output_file_name
		self.data_path = data_path
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
		Len = TNG100_subhalo.read_subhalo(self.sub_len_name)
		mass_subhalo = TNG100_subhalo.read_subhalo(self.mass_name)
		# gn = TNG100_subhalo.read_subhalo("GroupNumber")
		# sn = TNG100_subhalo.read_subhalo("SubGroupNumber")  # less
		galaxyIDs = TNG100_subhalo.read_subhalo("GalaxyID")
		file = h5py.File(f"{self.data_path}/EAGLE/diff_gnsn.hdf5", 'a')
		indices_gnsn_in_sub = file['indices_gnsn_in_sub'][:]
		file.close()
		flag = TNG100_subhalo.read_subhalo(self.flag_name)
		mask = (mass_subhalo > 0.0) * (flag == 0)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat("Offset_Subhalo_all")

		# indices = np.arange(0, len(mass_subhalo))
		# gnsn = np.array([indices, gn, sn]).transpose()
		# sorted_gnsn = sorted(gnsn, key=lambda x: (x[1], x[2]))
		# sorted_nonzero_indices = sorted_gnsn[:, 0][mask]
		#
		# indices_snap = np.arange(0, len(Len))
		# gnsn_snap_ind = np.array([indices, gnsn[:, 0], gnsn[:, 1]]).transpose()
		# sorted_gnsn_snap = sorted(gnsn_snap_ind, key=lambda x: (x[1], x[2]))
		# sorted_indices_snap = sorted_gnsn_snap[:, 0]

		# IDs = np.where(mask)

		self.Num_halos = len(mass_subhalo[mask][indices_gnsn_in_sub])
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, self.sub_len_name, Len)
		write_dataset_hdf5(group, self.offset_name, off)
		write_dataset_hdf5(group, self.mass_name, mass_subhalo[mask][indices_gnsn_in_sub])
		write_dataset_hdf5(group, self.ID_name, galaxyIDs[mask][indices_gnsn_in_sub])
		if self.PT == 4:
			# photo_mag = TNG100_subhalo.read_subhalo(self.photo_name)
			SFR = TNG100_subhalo.read_subhalo(self.SFR_name)
			# write_dataset_hdf5(group, self.photo_name, photo_mag[mask])
			write_dataset_hdf5(group, self.SFR_name, SFR[mask][indices_gnsn_in_sub])
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
			if self.PT == 4:
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

	def measure_velocities(self):
		"""
		Measures the velocity (km/s * a) of each subhalo for the initalised PT, weighted by
		particle mass. Writes data to SubhaloPT file in appropriate PT group.
		:return:
		"""
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		mass = TNG100_SubhaloPT.read_cat(self.mass_name)
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		velocities = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			mass_n = mass[n]
			if self.PT == 1:
				mass_particles = self.DM_part_mass
				velocity_particles = TNG100_snapshot.read_cat(
					self.velocities_name, cut=[off_n, off_n + len_n]
				) * np.sqrt(self.scalefactor)
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass_particles = TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])[star_mask]
				velocity_particles = TNG100_snapshot.read_cat(self.velocities_name, cut=[off_n, off_n + len_n])[
										 star_mask
									 ] * np.sqrt(self.scalefactor)
			else:
				mass_particles = TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])
				velocity_particles = TNG100_snapshot.read_cat(
					self.velocities_name, cut=[off_n, off_n + len_n]
				) * np.sqrt(self.scalefactor)

			mass_vel = (velocity_particles.transpose() * mass_particles).transpose()
			velocity = np.sum(mass_vel, axis=0) / mass_n
			velocities.append(velocity)
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "Velocity", data=np.array(velocities))
		output_file.close()
		return

	def measure_COM(self):
		"""
		Measures the centre of mass (ckpc/h) of each subhalo for the initalised PT. Coordinates weighted by
		particle mass. Writes data to SubhaloPT file in appropriate PT group.
		:return:
		"""
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		mass = TNG100_SubhaloPT.read_cat(self.mass_name)
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		COM = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			mass_n = mass[n]
			if self.PT == 1:
				mass_particles = self.DM_part_mass
				coordinates_particles = TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass_particles = TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])[star_mask]
				coordinates_particles = TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])[
					star_mask
				]
			else:
				mass_particles = TNG100_snapshot.read_cat(self.masses_name, cut=[off_n, off_n + len_n])
				coordinates_particles = TNG100_snapshot.read_cat(self.coordinates_name, cut=[off_n, off_n + len_n])

			# account for periodicity of the box
			min_coord = np.min(coordinates_particles, axis=0)
			coordinates_particles[(coordinates_particles - min_coord) > self.L_0p5] -= self.boxsize
			coordinates_particles[(coordinates_particles - min_coord) < -self.L_0p5] += self.boxsize

			mass_coord = (coordinates_particles.transpose() * mass_particles).transpose()
			COM_n = np.sum(mass_coord, axis=0) / mass_n
			COM_n[COM_n < 0.0] += self.boxsize  # if negative: COM is on other side of box.
			COM.append(COM_n)
		output_file = h5py.File(self.output_file_name, "a")
		group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/PT" + str(self.PT))
		write_dataset_hdf5(group, "COM", data=np.array(COM))
		output_file.close()
		return

	@staticmethod
	def measure_inertia_tensor_single(mass, particle_mass, rel_position, reduced=False):
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

	def measure_inertia_tensor(self, eigen_v=True, sorted=True, reduced=False):
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
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		COM = TNG100_SubhaloPT.read_cat("COM")
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			mass = mass_subhalo[n]

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_position = (
						TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] - COM[n]
				)
			else:
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]

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
				I = self.measure_inertia_tensor_single(mass, particle_mass, rel_position, reduced)
				if eigen_v:
					values, vectors = eig(I)
					value_list.append(values)
					vectors_list.append(vectors)
					v0.append(vectors[:, 0])
					v1.append(vectors[:, 1])
					v2.append(vectors[:, 2])
				I_list.append(I.reshape(9))
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

		if write_output:
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

	@staticmethod
	def measure_projected_inertia_tensor_single(mass, particle_mass, rel_position, reduced=False):
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

	def measure_projected_inertia_tensor(self, eigen_v=True, sorted=True, reduced=False, LOS_ind=2):
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
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		COM = TNG100_SubhaloPT.read_cat("COM")
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]
		LOS_axis = {0: 'x', 1: 'y', 2: 'z'}
		I_list, value_list, v0, v1, v2, vectors_list = [], [], [], [], [], []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			mass = mass_subhalo[n]

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_position = (
						TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] - COM[n]
				)
			else:
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			rel_position = rel_position[:, not_LOS]
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
				I = self.measure_projected_inertia_tensor_single(mass, particle_mass, rel_position, reduced)
				if eigen_v:
					values, vectors = eig(I)
					value_list.append(values)
					vectors_list.append(vectors)
					v0.append(vectors[:, 0])
					v1.append(vectors[:, 1])
				I_list.append(I.reshape(4))
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

	def measure_spin(self):
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
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		COM = TNG100_SubhaloPT.read_cat("COM")
		velocity = TNG100_SubhaloPT.read_cat("Velocity")
		spin_list = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			if self.PT == 1:
				particle_mass = self.DM_part_mass * 1e10
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask] * 1e10
				rel_position = (
						TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] - COM[n]
				)
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- velocity[n]
				)

			else:
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n]) * 1e10
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize

			spin = np.sum((particle_mass * np.cross(rel_position, rel_velocity).transpose()).transpose(), axis=0)
			spin_list.append(spin)
		if write_output:
			write_dataset_hdf5(group, "Spin", data=np.array(spin_list))
			output_file.close()
			return
		else:
			return np.array(spin_list)

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

	def measure_rotational_velocity(self, measure_dispersion=True, calc_basis=True):
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
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		mass_subhalo = TNG100_SubhaloPT.read_cat(self.mass_name)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		COM = TNG100_SubhaloPT.read_cat("COM")
		velocity = TNG100_SubhaloPT.read_cat("Velocity")
		spin = TNG100_SubhaloPT.read_cat("Spin")
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
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			mass = mass_subhalo[n]
			if len_n < 2:
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			spin_n = spin[n]
			if all(spin_n == 0.0):
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			if calc_basis:
				transform_mat = self.find_cartesian_basis(spin_n)  # new orthonormal basis with Spin as z-axis
				basis_n = np.array(transform_mat)
				basis_n.resize(9)
				basis.append(basis_n)
			else:
				transform_mat = group["Cart_basis_L_is_z"][n]
				transform_mat.resize((3, 3))
				if np.shape(transform_mat) == ():
					transform_mat = [[None]]
			if transform_mat[0][0] == None:
				avg_rot_vel.append(0)
				vel_disp.append(0)
				vel_z.append(0)
				vel_disp_cyl.append([0, 0, 0])
				vel_z_abs.append(0)
				if calc_basis:
					basis.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
				continue
			transform_mat_inv = inv(transform_mat)

			if self.PT == 1:
				particle_mass = self.DM_part_mass
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- velocity[n]
				)
				rel_position = (
						TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] - COM[n]
				)
			else:
				particle_mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]

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
			if measure_dispersion:
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

	def measure_krot(self):
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
		TNG100_snapshot = ReadTNGdata(
			self.simname,
			self.simname + "_PT" + str(self.PT) + "_subhalos_only",
			self.snapshot,
			data_path=self.data_path,
		)
		TNG100_SubhaloPT = ReadTNGdata(
			self.simname, "SubhaloPT", self.snapshot, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
		)
		off = TNG100_SubhaloPT.read_cat(self.offset_name)
		Len = TNG100_SubhaloPT.read_cat(self.sub_len_name)
		COM = TNG100_SubhaloPT.read_cat("COM")
		velocity = TNG100_SubhaloPT.read_cat("Velocity")
		spin = TNG100_SubhaloPT.read_cat("Spin")
		krot = []
		for n in np.arange(0, self.Num_halos):
			off_n = off[n]
			len_n = Len[n]
			spin_n = spin[n]
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
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]
			elif self.PT == 4:
				wind_or_star = TNG100_snapshot.read_cat(self.wind_name, cut=[off_n, off_n + len_n])
				star_mask = wind_or_star > 0
				mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])[star_mask]
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n])[star_mask]
						* np.sqrt(self.scalefactor)
						- velocity[n]
				)
				rel_position = (
						TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n])[star_mask] - COM[n]
				)
			else:
				mass = TNG100_snapshot.read_cat(self.masses_name, [off_n, off_n + len_n])
				rel_velocity = (
						TNG100_snapshot.read_cat(self.velocities_name, [off_n, off_n + len_n]) * np.sqrt(
					self.scalefactor)
						- velocity[n]
				)
				rel_position = TNG100_snapshot.read_cat(self.coordinates_name, [off_n, off_n + len_n]) - COM[n]

			rel_position[rel_position > self.L_0p5] -= self.boxsize
			rel_position[rel_position < -self.L_0p5] += self.boxsize
			position_dir = (rel_position.transpose() / np.sqrt(np.sum(rel_position ** 2, axis=1))).transpose()

			krot.append(
				sum(0.5 * mass * np.sum(np.cross(spin_dir, position_dir) * rel_velocity, axis=1) ** 2)
				/ (np.sum(0.5 * mass * np.sum(rel_velocity ** 2, axis=1)))
			)

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
			self.simname, "Shapes", 99, sub_group="PT" + str(self.PT) + "/", data_path=self.data_path
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
	):
		if type(simulation) == str:  # simulation is a tag that is hardcoded into SimInfo
			SimInfo.__init__(self, simulation, snapshot)
			self.boxsize /= 1000.0  # ckpc -> cMpc
			self.L_0p5 /= 1000.0
		elif boxsize is not None:  # boxsize is given manually
			self.boxsize = boxsize
			self.L_0p5 = boxsize / 2.0
		else:
			SimInfo.__init__(self, simulation,
							 snapshot)  # simulation is a SimInfo object created in the file that calls this class
		self.data = data
		self.output_file_name = output_file_name
		self.PT = PT
		try:
			self.Num_position = len(data["Position"])  # number of halos in position sample
			self.Num_shape = len(data["Position_shape_sample"])  # number of halos in shape sample
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
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/3D_correlations")
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

	def get_random_pairs(self, rp_max, rp_min, pi_max, pi_min, L3, corrtype):
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
					(self.Num_position - 1.0)
					/ 2.0
					* self.Num_shape
					* np.pi
					* (rp_max ** 2 - rp_min ** 2)
					* abs(pi_max - pi_min)
					/ L3
			)  # volume is cylindrical pi*dr^2 * height
		elif corrtype == "cross":
			RR = self.Num_position * self.Num_shape * np.pi * (rp_max ** 2 - rp_min ** 2) * abs(pi_max - pi_min) / L3
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

	def get_random_pairs_r_mur(self, r_max, r_min, mur_max, mur_min, L3, corrtype):
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
					(self.Num_position - 1.0)
					/ 2.0
					* self.Num_shape
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
					(self.Num_position - 1.0)
					* self.Num_shape
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

	def measure_projected_correlation(self, masks=None, dataset_name="All_galaxies", return_output=False):
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

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = self.boxsize / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		for n in np.arange(0, len(positions)):
			# for Splus_D (calculate ellipticities around position sample)
			separation = positions_shape_sample - positions[n]
			separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
			separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			separation_dir = (projected_sep.transpose() / separation_len).transpose()  # normalisation of rp
			phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			e_plus, e_cross = self.get_ellipticity(e, phi)
			if self.Num_position == self.Num_shape:
				e_plus[n], e_cross[n] = 0.0, 0.0

			# get the indices for the binning
			mask = (separation_len >= self.bin_edges[0]) * (separation_len <= self.bin_edges[-1])
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

		DD = DD / 2.0  # auto correlation, all pairs are double

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross"
				)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "auto"
				)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=DD / RR_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			output_file.close()
			return
		else:
			return correlation, DD / RR_gg, separation_bins, pi_bins

	def measure_projected_correlation_tree(self, masks=None, dataset_name="All_galaxies", return_output=False):
		"""
		Measures the projected correlation function (xi_g_plus, xi_gg) for given coordinates of the position sample using trees.
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

		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logrp = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_pi = self.boxsize / self.num_bins_pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)  # check this
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		variance = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		dist_bin_min = np.log10(self.separation_min)
		dist_bin_max = np.log10(self.separation_max)

		tree_positions = KDTree(positions, boxsize=self.boxsize)
		tree_shape = KDTree(positions_shape_sample, boxsize=self.boxsize)
		indices = tree_positions.query_ball_tree(tree_shape, r=100)

		cos_phi_array = []
		dist = []
		j = 0

		for index_list in tqdm(indices):

			index_list.remove(j)

			if len(index_list) == 0:
				j += 1
				continue

			separation = positions_shape_sample[index_list].T - positions[j][:, None]
			separation = (separation + 0.5 * self.boxsize) % self.boxsize - 0.5 * self.boxsize
			projected_sep = separation[not_LOS, :]
			LOS = separation[LOS_ind, :]
			separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=0))
			separation_dir = (projected_sep / separation_len)  # normalisation of rp
			cos_phi = np.abs(np.dot(separation_dir.T,
									axis_direction[j]))  # get some which are > 1., but maybe precision err 1.00000002.

			cos_phi[np.where(cos_phi > 1.0)] = 1.0
			phi = np.arccos(cos_phi)  # [0,pi]
			dist.extend(separation_len)
			cos_phi_array.extend(cos_phi)
			e_plus, e_cross = self.get_ellipticity(e[index_list], phi)
			# if self.Num_position == self.Num_shape:
			#     e_plus, e_cross = 0.0, 0.0        # what was this for?
			mask = (separation_len >= self.bin_edges[0]) * (separation_len <= self.bin_edges[-1])
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
			j += 1

		dist = np.array(dist)
		cos_phi_array = np.array(cos_phi_array)
		DD = DD / 2.0  # auto correlation, all pairs are double

		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "cross"
				)
				RR_gg[i, p] = self.get_random_pairs(
					self.bin_edges[i + 1], self.bin_edges[i], self.pi_bins[p + 1], self.pi_bins[p], L3, "auto"
				)
		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		sigsq = variance / RR_g_plus ** 2
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
		pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins

		# bin up cos phi
		cos_phi_means, cos_phi_edges, _ = binned_statistic(dist, cos_phi_array, statistic='mean',
														   bins=np.logspace(dist_bin_min, dist_bin_max,
																			self.num_bins_r))
		cos_phi_std, _, _ = binned_statistic(dist, cos_phi_array, statistic='std',
											 bins=np.logspace(dist_bin_min, dist_bin_max, self.num_bins_r))
		cos_phi_counts, _, _ = binned_statistic(dist, cos_phi_array, statistic='count',
												bins=np.logspace(dist_bin_min, dist_bin_max, self.num_bins_r))

		if (self.output_file_name != None) & return_output == False:
			print('Saving ...')
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_g_cross")
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/w/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=DD / RR_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_sigmasq", data=sigsq)
			write_dataset_hdf5(group, dataset_name + "_rp", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_pi", data=pi_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/Angles/cos_phi")
			write_dataset_hdf5(group, dataset_name + "_R", data=(cos_phi_edges[:-1] + cos_phi_edges[1:]) / 2.0)
			write_dataset_hdf5(group, dataset_name + "_cos_phi", data=cos_phi_means)
			write_dataset_hdf5(group, dataset_name + "_cos_phi_sig", data=cos_phi_std / np.sqrt(cos_phi_counts))
			output_file.close()
			return
		else:
			return correlation, DD / RR_gg, separation_bins, pi_bins

	def measure_projected_correlation_multipoles(
			self, masks=None, rp_cut=2.0, dataset_name="All_galaxies", return_output=False
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

		if rp_cut == None:
			rp_cut = 0.0
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
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
			separation[separation > self.L_0p5] -= self.boxsize  # account for periodicity of box
			separation[separation < -self.L_0p5] += self.boxsize
			projected_sep = separation[:, not_LOS]
			LOS = separation[:, LOS_ind]
			projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			separation_dir = (projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
			separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
			mu_r = LOS / separation_len
			phi = np.arccos(self.calculate_dot_product_arrays(separation_dir, axis_direction))  # [0,pi]
			e_plus, e_cross = self.get_ellipticity(e, phi)
			if self.Num_position == self.Num_shape:
				e_plus[n], e_cross[n] = 0.0, 0.0
				mu_r[n] = 0.0

			# get the indices for the binning
			mask = (
					(projected_separation_len > rp_cut)
					* (separation_len >= self.bin_edges[0])
					* (separation_len <= self.bin_edges[-1])
			)
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(self.bin_edges[0]) / sub_box_len_logr
			)
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.bins_mu_r[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			ind_mu_r = np.array(ind_mu_r, dtype=int)

			np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask] / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask] / (2 * R))
			np.add.at(DD, (ind_r, ind_mu_r), 1.0)

		DD = DD / 2.0  # auto correlation, all pairs are double

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "cross"
				)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "auto"
				)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=DD / RR_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, DD / RR_gg, separation_bins, mu_r_bins

	def measure_projected_correlation_multipoles_tree(
			self, masks=None, rp_cut=2.0, dataset_name="All_galaxies", return_output=False
	):
		"""
		Measures the projected correlation function (xi_g_plus) for given coordinates of the position sample using trees
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

		if rp_cut == None:
			rp_cut = 0.0
		LOS_ind = self.data["LOS"]  # eg 2 for z axis
		not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]  # eg 0,1 for x&y
		e = (1 - q ** 2) / (1 + q ** 2)  # size of ellipticity
		R = 1 - np.mean(e ** 2) / 2.0  # responsitivity factor
		L3 = self.boxsize ** 3  # box volume
		sub_box_len_logr = (np.log10(self.separation_max) - np.log10(self.separation_min)) / self.num_bins_r
		sub_box_len_mu_r = 2.0 / self.num_bins_pi  # mu_r ranges from -1 to 1. Same number of bins as pi
		DD = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Splus_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		Scross_D = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_g_plus = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)
		RR_gg = np.array([[0.0] * self.num_bins_pi] * self.num_bins_r)

		tree_positions = KDTree(positions, boxsize=self.boxsize)
		tree_shape = KDTree(positions_shape_sample, boxsize=self.boxsize)
		indices = tree_positions.query_ball_tree(tree_shape, r=100)

		j = 0

		for index_list in tqdm(indices):

			index_list.remove(j)

			if len(index_list) == 0:
				j += 1
				continue

			separation = positions_shape_sample[index_list].T - positions[j][:, None]
			separation = (separation + 0.5 * self.boxsize) % self.boxsize - 0.5 * self.boxsize
			projected_sep = separation[not_LOS, :]
			LOS = separation[LOS_ind, :]
			projected_separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
			separation_dir = (projected_sep.transpose() / projected_separation_len).transpose()  # normalisation of rp
			separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
			mu_r = LOS / separation_len
			cos_phi = np.abs(np.dot(separation_dir.T,
									axis_direction[j]))  # get some which are > 1., but maybe precision err 1.00000002.

			cos_phi[np.where(cos_phi > 1.0)] = 1.0
			phi = np.arccos(cos_phi)  # [0,pi]

			e_plus, e_cross = self.get_ellipticity(e[index_list], phi)
			# get the indices for the binning
			mask = (
					(projected_separation_len > rp_cut)
					* (separation_len >= self.bin_edges[0])
					* (separation_len <= self.bin_edges[-1])
			)
			ind_r = np.floor(
				np.log10(separation_len[mask]) / sub_box_len_logr - np.log10(self.bin_edges[0]) / sub_box_len_logr
			)
			ind_r = np.array(ind_r, dtype=int)
			ind_mu_r = np.floor(
				mu_r[mask] / sub_box_len_mu_r - self.bins_mu_r[0] / sub_box_len_mu_r
			)  # need length of LOS, so only positive values
			ind_mu_r = np.array(ind_mu_r, dtype=int)

			np.add.at(Splus_D, (ind_r, ind_mu_r), e_plus[mask] / (2 * R))
			np.add.at(Scross_D, (ind_r, ind_mu_r), e_cross[mask] / (2 * R))
			np.add.at(DD, (ind_r, ind_mu_r), 1.0)

		DD = DD / 2.0  # auto correlation, all pairs are double

		# analytical calc is much more difficult for (r,mu_r) bins
		for i in np.arange(0, self.num_bins_r):
			for p in np.arange(0, self.num_bins_pi):
				RR_g_plus[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "cross"
				)
				RR_gg[i, p] = self.get_random_pairs_r_mur(
					self.bin_edges[i + 1], self.bin_edges[i], self.bins_mu_r[p + 1], self.bins_mu_r[p], L3, "auto"
				)

		correlation = Splus_D / RR_g_plus  # (Splus_D - Splus_R) / RR_g_plus
		xi_g_cross = Scross_D / RR_g_plus  # (Scross_D - Scross_R) / RR_g_plus
		dsep = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
		separation_bins = self.bin_edges[:-1] + abs(dsep)  # middle of bins
		dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
		mu_r_bins = self.bins_mu_r[:-1] + abs(dmur)  # middle of bins

		if (self.output_file_name != None) & return_output == False:
			print('Saving ...')
			output_file = h5py.File(self.output_file_name, "a")
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_g_plus")
			write_dataset_hdf5(group, dataset_name, data=correlation)
			write_dataset_hdf5(group, dataset_name + "_SplusD", data=Splus_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_plus", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_g_cross")
			write_dataset_hdf5(group, dataset_name, data=xi_g_cross)
			write_dataset_hdf5(group, dataset_name + "_ScrossD", data=Scross_D)
			write_dataset_hdf5(group, dataset_name + "_RR_g_cross", data=RR_g_plus)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/multipoles/xi_gg")
			write_dataset_hdf5(group, dataset_name, data=DD / RR_gg)
			write_dataset_hdf5(group, dataset_name + "_DD", data=DD)
			write_dataset_hdf5(group, dataset_name + "_RR_gg", data=RR_gg)
			write_dataset_hdf5(group, dataset_name + "_r", data=separation_bins)
			write_dataset_hdf5(group, dataset_name + "_mu_r", data=mu_r_bins)
			output_file.close()
			return
		else:
			return correlation, DD / RR_gg, separation_bins, mu_r_bins

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
			group = correlation_data_file["Snapshot_" + self.snapshot + "/w/" + xi_data[i]]
			correlation_data = group[dataset_name][:]
			pi = group[dataset_name + "_pi"]
			rp = group[dataset_name + "_rp"]
			dpi = (self.pi_bins[1:] - self.pi_bins[:-1]) / 2.0
			pi_bins = self.pi_bins[:-1] + abs(dpi)  # middle of bins
			variance = group[dataset_name + "_sigmasq"][:]
			if sum(np.isin(pi, pi_bins)) == len(pi):
				dpi = np.array([dpi] * len(correlation_data[:, 0]))
				correlation_data = correlation_data * abs(dpi)
				sigsq_el = variance * dpi ** 2
			else:
				raise ValueError("Update pi bins in initialisation of object to match xi_g_plus dataset.")
			w_g_i = np.sum(correlation_data, axis=1)  # sum over pi values
			sigsq = np.sum(sigsq_el, axis=1)
			if return_output:
				output_data = np.array([rp, w_g_i]).transpose()
				correlation_data_file.close()
				return output_data
			else:
				group_out = create_group_hdf5(correlation_data_file, "Snapshot_" + self.snapshot + "/" + wg_data[i])
				write_dataset_hdf5(group_out, dataset_name + "_rp", data=rp)
				write_dataset_hdf5(group_out, dataset_name, data=w_g_i)
				write_dataset_hdf5(group_out, dataset_name + "_sigma", data=np.sqrt(sigsq))
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
			group = correlation_data_file["Snapshot_" + self.snapshot + "/multipoles/xi_g_plus"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of r,mur
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [2]
			l_list = sab_list
			corr_type_list = ["g_plus"]
		elif corr_type == "gg":
			group = correlation_data_file["Snapshot_" + self.snapshot + "/multipoles/xi_gg"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			sab_list = [0]
			l_list = sab_list
			corr_type_list = ["gg"]
		elif corr_type == "both":
			group = correlation_data_file["Snapshot_" + self.snapshot + "/multipoles/xi_g_plus"]
			correlation_data_list = [group[dataset_name][:]]  # xi_g+ in grid of rp,pi
			r_list = [group[dataset_name + "_r"][:]]
			mu_r_list = [group[dataset_name + "_mu_r"][:]]
			group = correlation_data_file["Snapshot_" + self.snapshot + "/multipoles/xi_gg"]
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
			dmur = (self.bins_mu_r[1:] - self.bins_mu_r[:-1]) / 2.0
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
					correlation_data_file, "Snapshot_" + self.snapshot + "/multipoles_" + corr_type_i
				)
				write_dataset_hdf5(group_out, dataset_name + "_r", data=separation)
				write_dataset_hdf5(group_out, dataset_name, data=multipoles)
		correlation_data_file.close()
		return

	def measure_jackknife_errors(
			self, corr_type=["both", "multipoles"], dataset_name="All_galaxies", L_subboxes=3, rp_cut=2.0
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
					if corr_type[1] == "multipoles":
						self.measure_projected_correlation_multipoles(
							{
								"Position": mask_position,
								"Position_shape_sample": mask_shape,
								"Axis_Direction": mask_shape,
								"q": mask_shape,
							},
							rp_cut=rp_cut,
							dataset_name=dataset_name + "_" + str(num_box),
						)
						self.measure_multipoles(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(num_box))
					else:
						self.measure_projected_correlation(
							{
								"Position": mask_position,
								"Position_shape_sample": mask_shape,
								"Axis_Direction": mask_shape,
								"q": mask_shape,
							},
							dataset_name=dataset_name + "_" + str(num_box),
						)
						self.measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name + "_" + str(num_box))

					num_box += 1
		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file["Snapshot_" + self.snapshot + "/" + data[d]]
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
				group_multipoles = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
				return
			else:
				return cov, std

	def measure_jackknife_errors_multiprocessing(
			self,
			corr_type=["both", "multipoles"],
			dataset_name="All_galaxies",
			L_subboxes=3,
			rp_cut=2.0,
			num_nodes=4,
			twoD=False,
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
		num_box = 0
		args_xi_g_plus, args_multipoles = [], []
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
					if corr_type[1] == "multipoles":
						args_xi_g_plus.append(
							(
								{
									"Position": mask_position,
									"Position_shape_sample": mask_shape,
									"Axis_Direction": mask_shape,
									"q": mask_shape,
								},
								rp_cut,
								dataset_name + "_" + str(num_box),
								True,
							)
						)
					else:
						args_xi_g_plus.append(
							(
								{
									"Position": mask_position,
									"Position_shape_sample": mask_shape,
									"Axis_Direction": mask_shape,
									"q": mask_shape,
								},
								dataset_name + "_" + str(num_box),
								True,
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
				result = ProcessingPool(nodes=len(chunck)).map(
					self.measure_projected_correlation_multipoles,
					args_xi_g_plus[chunck][:, 0],
					args_xi_g_plus[chunck][:, 1],
					args_xi_g_plus[chunck][:, 2],
					args_xi_g_plus[chunck][:, 3],
				)
			else:
				result = ProcessingPool(nodes=len(chunck)).map(
					self.measure_projected_correlation,
					args_xi_g_plus[chunck][:, 0],
					args_xi_g_plus[chunck][:, 1],
					args_xi_g_plus[chunck][:, 2],
				)
			output_file = h5py.File(self.output_file_name, "a")
			for i in np.arange(0, len(chunck)):
				for j, data_j in enumerate(data):
					group_xigplus = create_group_hdf5(
						output_file, "Snapshot_" + self.snapshot + "/" + corr_type[1] + "/xi" + corr_type_suff[j]
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
		for i in np.arange(0, num_box):
			if corr_type[1] == "multipoles":
				self.measure_multipoles(corr_type=args_multipoles[i][0], dataset_name=args_multipoles[i][1])
			else:
				self.measure_w_g_i(corr_type=args_multipoles[i][0], dataset_name=args_multipoles[i][1])

		for d in np.arange(0, len(data)):
			data_file = h5py.File(self.output_file_name, "a")
			group_multipoles = data_file["Snapshot_" + self.snapshot + "/" + data[d]]
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
				group_multipoles = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/" + data[d])
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_" + str(num_box), data=std)
				write_dataset_hdf5(group_multipoles, dataset_name + "_jackknife_cov_" + str(num_box), data=cov)
				output_file.close()
				return
			else:
				return cov, std

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
			group = create_group_hdf5(output_file, "Snapshot_" + self.snapshot + "/Misalignment_angels")
			write_dataset_hdf5(group, vector1_name + "_" + vector2_name, data=misalignment_angle)
			output_file.close()
		else:
			return misalignment_angle
		return


if __name__ == "__main__":
	pass
