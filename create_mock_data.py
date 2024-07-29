import numpy as np
import h5py
from src.write_data import create_group_hdf5, write_dataset_hdf5

path_raw = ("./tests/data/raw/")
sim = "TNG100"
snap = 99

# mock Subhalo data
mock_SubhaloPT = h5py.File(f"{path_raw}/{sim}/SubhaloPT.hdf5", "a")
groupPT = create_group_hdf5(mock_SubhaloPT, f"Snapshot_{snap}/PT4")
off = np.array([0, 2, 5])
Len = np.array([2, 3, 4])
mass = np.array([3., 4., 5.])
write_dataset_hdf5(groupPT, "Offset_Subhalo", off)
write_dataset_hdf5(groupPT, "SubhaloLenType", Len)
write_dataset_hdf5(groupPT, "StellarMass", mass)
mock_SubhaloPT.close()

# mock Snap data
mock_Snap = h5py.File(f"{path_raw}/{sim}/TNG100_PT4_subhalos_only.hdf5", "a")
groupsnap = create_group_hdf5(mock_Snap, f"Snapshot_{snap}")
coordinates = np.array(
	[[1000., 1000., 1000.], [2500., 2500., 2500.], [1000, 1000, 1000], [74000, 74000, 74000], [2000, 2000, 2000],
	 [1000, 1000, 1000],
	 [4000, 4000, 4000], [73000, 2000, 2000], [74000, 2500, 2500]])  # 75000 ckpc/h boxsize
windflag = np.array([1., 1., 1., 1., 1., 1., 0., 1., 1.])
masses = np.array([1., 2., 1., 1., 2., 1., 1., 2., 2.])
# gal1 1 2
# gal2 1 1 2
# gal3 1 (1) 2 2
velocities = np.array(
	[[1, 1, 1], [2.5, 2.5, 2.5], [1, 1, 1], [-1, -1, -1], [2, 2, 2], [1, 1, 1], [4, 4, 4], [-2, 2, 2], [-1, 2.5, 2.5]])
write_dataset_hdf5(groupsnap, "Coordinates", coordinates)
write_dataset_hdf5(groupsnap, "Velocities", velocities)
write_dataset_hdf5(groupsnap, "GFM_StellarFormationTime", windflag)
write_dataset_hdf5(groupsnap, "Masses", masses)
mock_Snap.close()

# COM1 2000,2000,2000
# COM2 1000,1000,1000
# COM3 74000,2000,2000

# Vel1 2.,2.,2.
# Vel2 1.,1.,1.
# Vel3 -1,2.,2.

# mock Shape data
# mock_Shapes = h5py.File(f"{path_raw}/{sim}/Shapes.hdf5", "a")
# group_shapes = create_group_hdf5(mock_Shapes, f"Snapshot_{snap}/PT4")
#
#
# mock_Shapes.close()
