class SimInfo:
	"""Class that stores simulation information in an object to be inherited by other classes.
		Simulation information is hard coded and therefore uses are limited. However, can easily be expanded.
		Currently, these simulations are available: [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1,
		FLAMINGO_L2p8].

	Attributes
	----------
	simname : str or NoneType
		Identifier of the simulation, allowing for correct information to be obtained.
	snapshot : int or str or NoneType
		Number of the snapshot.
	snap_group : str
		Name of group in output file. Equal to 'Snapshot_[snapshot]' if snapshot is given, otherwise emtpy string.
	boxsize :  int or float, default=None
		Size of simulation box. If simname is in SimInfo, units are cMpc/h. Otherwise, manual input.
	L_0p5 : int or float, default=None
		Half of the boxsize.
	h : float, default=None
		Value of cosmological h parameter, for easy access to convert units.

	"""

	def __init__(self, sim_name, snapshot, boxsize=None, h=None):
		"""
		The __init__ method of SimInfo class.
		Creates all attributes and obtains information that is hardcoded in the class.

		Parameters
		----------
		sim_name : str or NoneType
			Identifier of the simulation, allowing for correct information to be obtained.
			Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8].
			If None, no information will be returned that is not already given as input.
		snapshot : int or str or NoneType
			Number of the snapshot, which, if given, will ensure that the output file to contains a group
			'Snapshot_[snapshot]'.
			If None, the group is omitted from the output file structure.
		boxsize : int or float, default=None
			Size of simulation box. Use if your simulation information is not provided by SimInfo.
			Make sure that the boxsize is in the same units as your position coordinates.
		h : float, default=None
			Value of cosmological h parameter, for easy access to convert units.
		"""
		self.simname = sim_name
		if snapshot is None:
			self.snapshot = None
			self.snap_group = ""
		else:
			self.snapshot = str(snapshot)
			self.snap_group = f"Snapshot_{self.snapshot}/"
		if type(sim_name) == str:
			self.get_specs()
		else:
			self.boxsize = boxsize
			self.h = h
			if boxsize is None:
				self.L_0p5 = None
			else:
				self.L_0p5 = boxsize / 2.
		return

	def get_specs(self):
		"""Obtains the boxsize, L_0p5 and h parameters that are stored for [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN,
		FLAMINGO_L1, FLAMINGO_L2p8].

		Raises
		------
		KeyError
			If unknown simname is given.

		"""
		if self.simname == "TNG100":
			self.boxsize = 75.0  # cMpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6774
		elif self.simname == "TNG100_2":
			self.boxsize = 75.0  # cMpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6774
		elif self.simname == "TNG300":
			self.boxsize = 205.0  # cMpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6774
		elif self.simname == "EAGLE":
			self.boxsize = 100.0 * 0.6777  # cMpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.6777
		elif self.simname == "HorizonAGN":
			self.boxsize = 100.0  # cMpc/h
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.704
		elif "FLAMINGO" in self.simname:
			if "L1" in self.simname:
				self.boxsize = 1000.0 * 0.681  # cMpc/h
			elif "L2p8" in self.simname:
				self.boxsize = 2800.0 * 0.681  # cMpc/h
			else:
				raise KeyError("Add an L1 or L2p8 suffix to your simname to specify which boxsize is used")
			self.L_0p5 = self.boxsize / 2.0
			self.h = 0.681
		else:
			raise KeyError(
				"Simulation name not recognised. Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, "
				"FLAMINGO_L2p8].")
		return
