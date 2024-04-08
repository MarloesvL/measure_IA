class SimInfo:
	"""
	Class that stores simulation information in an object to be inherited by other classes.
	Simulation information is hard coded and therefore uses are limited. However, can easily be expanded,
	given the file structures are the same.
	:param sim_name: identifier of the simulation, allowing for correct information to be obtained. [TNG100, TNG300]
	:param snapshot: number of the snapshot, influences which folders are located.
	"""

	def __init__(self, sim_name, snapshot, manual=False):
		if type(sim_name) == str:
			self.simname = sim_name
			self.snapshot = str(snapshot)
			self.manual = manual
			if not self.manual:
				self.get_specs()
				self.get_variable_names()
				self.get_folders()
				self.get_scalefactor()
		else:
			self.sim_name = sim_name
			self.snapshot = str(snapshot)
			self.manual = manual
		return

	def __getattr__(self, name):
		try:
			return getattr(self.sim_name, name)
		except:
			raise AttributeError("Child' object has no attribute '%s'" % name)

	def get_specs(self, boxsize=None, h=None, DM_part_mass=None, N_files=None):
		"""
		Creates attributes describing the simulation specs, e.g. boxsize, DM particle mass.
		:return:
		"""
		if self.manual:
			self.boxsize = boxsize  # ckpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = h
			self.DM_part_mass = DM_part_mass  # 10^10 M_sun/h
			self.N_files = N_files
		elif self.simname == "TNG100":
			self.boxsize = 75000.0  # ckpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6774
			self.DM_part_mass = 0.000505574296436975  # 10^10 M_sun/h
			self.N_files = 448
		elif self.simname == "TNG300":
			self.boxsize = 205000.0  # ckpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6774
			self.DM_part_mass = 0.00398342749867548  # 10^10 M_sun/h
			self.N_files = 600
		elif self.simname == "EAGLE":
			self.boxsize = 100000.0/0.6777  # ckpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6777
			self.DM_part_mass = 0.000970  # 10^10 M_sun (*h?)
			self.N_files = 256
		else:
			raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300, EAGLE].")
		return

	def get_variable_names(
			self,
			mass_name=None,
			ID_name=None,
			offset_name=None,
			sub_len_name=None,
			group_len_name=None,
			photo_name=None,
			SFR_name=None,
			flag_name=None,
			wind_name=None,
			velocities_name=None,
			masses_name=None,
			coordinates_name=None,
	):
		"""
		Creates attributes describing the variable names for different datasets.
		:return:
		"""
		if self.manual:
			self.mass_name = mass_name
			self.ID_name = ID_name
			self.offset_name = offset_name
			self.sub_len_name = sub_len_name
			self.group_len_name = group_len_name
			self.photo_name = photo_name
			self.SFR_name = SFR_name
			self.flag_name = flag_name
			self.wind_name = wind_name

			self.velocities_name = velocities_name
			self.masses_name = masses_name
			self.coordinates_name = coordinates_name
		elif "TNG" in self.simname:
			self.mass_name = "SubhaloMassType"
			self.ID_name = "SubhaloIDs"
			self.offset_name = "Offset_Subhalo"
			self.sub_len_name = "SubhaloLenType"
			self.group_len_name = "GroupLenType"
			self.photo_name = "SubhaloStellarPhotometrics"
			self.SFR_name = "SubhaloSFR"
			self.flag_name = "SubhaloFlag"
			self.wind_name = "GFM_StellarFormationTime"  # >0 for star particles

			self.velocities_name = "Velocities"
			self.masses_name = "Masses"
			self.coordinates_name = "Coordinates"
		elif self.simname == "EAGLE":
			self.mass_name = "MassType_Star"
			self.ID_name = "GalaxyID"
			self.offset_name = "Offset_Subhalo"
			self.sub_len_name = "Len"
			self.group_len_name = None  # "GroupLenType"
			self.photo_name = None  # "SubhaloStellarPhotometrics"
			self.SFR_name = "StarFormationRate"
			self.flag_name = "Spurious"
			self.wind_name = None  # "GFM_StellarFormationTime"  # >0 for star particles

			self.velocities_name = "Velocity"
			self.masses_name = "Mass"
			self.coordinates_name = "Coordinates"
		else:
			raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
		return

	def get_folders(self, fof_folder=None, snap_folder=None, snap_group=None):
		"""
		Creates attributes for subpaths leading to datafiles. Assumes that data is always in the same format.
		:return:
		"""
		if self.manual:
			self.fof_folder = fof_folder
			self.snap_folder = snap_folder
			self.snap_group = snap_group
		elif "TNG" in self.simname:
			self.fof_folder = f"/fof_subhalo_tab_0{self.snapshot}/fof_subhalo_tab_0{self.snapshot}"
			self.snap_folder = f"/snap_0{self.snapshot}/snap_0{self.snapshot}"
			self.snap_group = f"Snapshot_{self.snapshot}/"
		elif self.simname == "EAGLE":
			self.snap_folder = f"/snap_0{self.snapshot}/RefL0100N1504/snapshot_0{self.snapshot}_z000p000/snap_0{self.snapshot}_z000p000"  # update for different z?
			self.snap_group = f"Snapshot_{self.snapshot}/"
			self.fof_folder = None
		else:
			raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
		return

	def get_scalefactor(self, redshifts=None):
		if self.manual:
			pass
		elif "TNG" in self.simname:
			redshifts = {"40": 1.5, "50": 1.0, "67": 0.5, "78": 0.3, "59": 0.7, "99": 0.0}
		elif self.simname == "EAGLE":
			redshifts = {"28": 0.0}
		else:
			raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
		try:
			self.scalefactor = 1.0 / (1.0 + redshifts[self.snapshot])
		except ValueError:
			raise KeyError(f"Snapshot {self.snapshot} not in redshift directory for {self.simname}.")
		return
