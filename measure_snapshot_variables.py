import time
from src.measure_SV import MeasureSnapshotVariables

data_path_raw = "raw data path"
data_path_snap = "data path to snapshot data"

PT = 4  # particle type
snapshot = 99
simulation = "TNG100"  # simulation name
num_nodes = 9  # number of cores to use

start_time = time.time()

TNG100_Variable_snapshot = MeasureSnapshotVariables(project=simulation,
													PT=PT, snapshot=snapshot, numnodes=num_nodes,
													output_file_name=data_path_raw + f"{simulation}/SubhaloPT_cat.hdf5",
													data_path=data_path_raw, update=True,
													snap_data_path=data_path_snap, exclude_wind=True
													)
TNG100_Variable_snapshot.subhalo_cat = "SubhaloPT_cat"
TNG100_Variable_snapshot.measure_offsets(type="Subhalo")
TNG100_Variable_snapshot.measure_offsets(type="Group")
print("off", time.time() - start_time)
TNG100_Variable_snapshot.omit_wind_only()
print('wind done', time.time() - start_time)
TNG100_Variable_snapshot.select_nonzero_subhalos()
print("selected", time.time() - start_time)
TNG100_Variable_snapshot.measure_masses_excl_wind()
print('Mass', time.time() - start_time)
TNG100_Variable_snapshot.save_number_of_particles()
print('NoP', time.time() - start_time)
TNG100_Variable_snapshot.measure_COM()
print("COM", time.time() - start_time)

Subtime = time.time()
TNG100_Variable_snapshot_shapes = MeasureSnapshotVariables(project=simulation,
														   PT=PT, snapshot=snapshot, numnodes=num_nodes,
														   output_file_name=data_path_raw + f"{simulation}/Shapes_cat.hdf5",
														   data_path=data_path_raw, update=True,
														   snap_data_path=data_path_snap, exclude_wind=True
														   )
print('start shapes')
TNG100_Variable_snapshot_shapes.subhalo_cat = "SubhaloPT_cat"
TNG100_Variable_snapshot_shapes.shapes_cat = "Shapes_cat"
TNG100_Variable_snapshot_shapes.measure_inertia_tensor(reduced=False)
TNG100_Variable_snapshot_shapes.measure_inertia_tensor(reduced=True)
TNG100_Variable_snapshot_shapes.measure_projected_inertia_tensor(reduced=False)
TNG100_Variable_snapshot_shapes.measure_projected_inertia_tensor(reduced=True)
TNG100_Variable_snapshot_shapes.measure_sphericity_triaxality()
print("shapes done")
print(time.time() - Subtime)

TNG100_Variable_snapshot = MeasureSnapshotVariables(project=simulation,
													PT=PT, snapshot=snapshot, numnodes=num_nodes,
													output_file_name=data_path_raw + f"{simulation}/SubhaloPT_cat.hdf5",
													data_path=data_path_raw, update=True,
													snap_data_path=data_path_snap, exclude_wind=True
													)
Subtime = time.time()
TNG100_Variable_snapshot.subhalo_cat = "SubhaloPT_cat"
TNG100_Variable_snapshot.shapes_cat = "Shapes_cat"
TNG100_Variable_snapshot.measure_velocities()
print("vel")
TNG100_Variable_snapshot.measure_spin()
TNG100_Variable_snapshot.measure_krot()
print("spin, krot")
TNG100_Variable_snapshot.measure_rotational_velocity(calc_basis=True)
print("rot vel")
print(time.time() - Subtime)
