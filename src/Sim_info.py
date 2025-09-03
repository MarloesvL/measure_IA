class SimInfo:
	"""Class that stores simulation information in an object to be inherited by other classes.
		Simulation information is hard coded and therefore uses are limited. However, can easily be expanded,
		given the file structures are the same.

	Parameters
	----------
	sim_name :
		identifier of the simulation, allowing for correct information to be obtained.
		[TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8]
	snapshot :
		number of the snapshot, influences which folders are located.

	Returns
	-------

	"""

	def __init__(self, sim_name, snapshot, boxsize=None, h=None):
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

	def __getattr__(self, name):
		try:
			return getattr(self.sim_name, name)
		except:
			raise AttributeError("Child' object has no attribute '%s'" % name)

	def get_specs(self):
		"""Creates attributes describing the simulation specs, e.g. boxsize, DM particle mass.
		:return:

		Parameters
		----------
		boxsize :
			 (Default value = None)
		h :
			 (Default value = None)

		Returns
		-------

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
				"Simulation name not recognised. Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8].")
		return
