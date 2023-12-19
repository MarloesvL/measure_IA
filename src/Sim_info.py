class SimInfo:
    """
    Class that stores simulation information in an object to be inherited by other classes.
    Simulation information is hard coded and therefore uses are limited. However, can easily be expanded,
    given the file structures are the same.
    :param sim_name: identifier of the simulation, allowing for correct information to be obtained. [TNG100, TNG300]
    :param snapshot: number of the snapshot, influences which folders are located.
    """

    def __init__(self, sim_name, snapshot):
        self.simname = sim_name
        self.snapshot = str(snapshot)
        self.get_specs()
        self.get_variable_names()
        self.get_folders()
        self.get_scalefactor()
        return

    def get_specs(self):
        """
        Creates attributes describing the simulation specs, e.g. boxsize, DM particle mass.
        :return:
        """
        if self.simname == "TNG100":
            self.boxsize = 75000.0  # ckpc/h
            self.L_0p5 = self.boxsize / 2.0
            self.DM_part_mass = 0.000505574296436975  # 10^10 M_sun/h
            self.N_files = 448
        elif self.simname == "TNG300":
            self.boxsize = 205000.0  # ckpc/h
            self.L_0p5 = self.boxsize / 2.0
            self.DM_part_mass = 0.00398342749867548  # 10^10 M_sun/h
            self.N_files = 600
        else:
            raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
        return

    def get_variable_names(self):
        """
        Creates attributes describing the variable names for different datasets.
        :return:
        """
        if "TNG" in self.simname:
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
        else:
            raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
        return

    def get_folders(self):
        """
        Creates attributes for subpaths leading to datafiles. Assumes that data is always in the same format.
        :return:
        """
        if "TNG" in self.simname:
            self.fof_folder = f"/fof_subhalo_tab_0{self.snapshot}/fof_subhalo_tab_0{self.snapshot}"
            self.snap_folder = f"/snap_0{self.snapshot}/snap_0{self.snapshot}"
            self.snap_group = f"Snapshot_{self.snapshot}/"
        else:
            raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
        return

    def get_scalefactor(self):
        if "TNG" in self.simname:
            redshifts = {"40": 1.5, "50": 1.0, "99": 0.0}
        else:
            raise KeyError("Simulation name not recognised. Choose from [TNG100, TNG300].")
        self.scalefactor = 1.0 / (1.0 + redshifts[self.snapshot])
        return
