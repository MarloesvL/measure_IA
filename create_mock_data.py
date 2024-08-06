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


# mock IA data
path_proc = './tests/data/processed/'
mock_IA = h5py.File(f"{path_proc}/{sim}/IA_correlations.hdf5", "a")
group_wgplus = create_group_hdf5(mock_IA, f"Snapshot_99/w_g_plus")
group_multi = create_group_hdf5(mock_IA, f"Snapshot_99/multipoles_g_plus")
data1 = np.array([0, 1, 0, 3])
data2 = np.array([1, 3, 2, 1])
data3 = np.array([2, 2, 1, 2])
# means 1,2,1,2
# dev1 -1,-1,-1,1
# dev2 0,1,1,-1
# dev3 1,0,0,0
write_dataset_hdf5(group_wgplus, "set1_0", data1)
write_dataset_hdf5(group_wgplus, "set1_1", data2)
write_dataset_hdf5(group_wgplus, "set1_2", data3)
# cov (*2/3)
# 2		1	1	-1
# 1		2	2	-2
# 1		2	2	-2
# -1   -2  -2	 2

data1 = np.array([2, 2, 3, 0])
data2 = np.array([1, 0, 1, 2])
data3 = np.array([3, 1, 2, 1])
# means 2,1,2,1
# dev1 0, 1, 1, -1
# dev2 -1,-1,-1,1
# dev3 1,0,0,0
write_dataset_hdf5(group_wgplus, "set2_0", data1)
write_dataset_hdf5(group_wgplus, "set2_1", data2)
write_dataset_hdf5(group_wgplus, "set2_2", data3)
# cov (*2/3)
#  2	1	1	-1
#  1	2	2	-2
#  1	2	2	-2
# -1	-2	-2	2

# comb1&2
# dev1 -1,-1,-1, 1
# dev1 0, 1,  1, -1

# dev2 0, 1,  1,-1
# dev2 -1,-1,-1, 1

# dev3 1,0,0,0
# dev3 1,0,0,0

# comb cov (*2/3)
# 1	-1	-1	1
# -1-2	-2	2
# -1 -2	-2	2
# 1	2	2	-2


data1 = 2 * np.array([2, 2, 3, 0])
data2 = 2 * np.array([1, 0, 1, 2])
data3 = 2 * np.array([3, 1, 2, 1])
# means 4,2,4,2
# dev1 0, 2, 2, -2
# dev2 -2,-2,-2,2
# dev3 2,0,0,0
write_dataset_hdf5(group_wgplus, "set3_0", data1)
write_dataset_hdf5(group_wgplus, "set3_1", data2)
write_dataset_hdf5(group_wgplus, "set3_2", data3)


# comb 1&3
# dev1 -1,-1,-1,1
# dev1 0, 2, 2, -2

# dev2 0,1,  1,-1
# dev2 -2,-2,-2,2

# dev3 1,0,0,0
# dev3 2,0,0,0

# 2	-2	-2	2
# -2 -4	-4	4
# -2 -4	-4	4
# 2	4	4	-4

# comb 2&3

# dev2 0,1,  1,-1
# dev1 0, 2, 2, -2

# dev1 -1,-1,-1,1
# dev2 -2,-2,-2,2

# dev3 1,0,0,0
# dev3 2,0,0,0

#  4	2	2	-2
#  2	4	4	-4
#  2	4	4	-4
# -2	-4	-4	4

mock_IA.close()
