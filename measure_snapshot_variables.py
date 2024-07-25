import time
import configparser
from src.measure_IA_sims.measure_IA import MeasureVariablesSnapshot

config = configparser.RawConfigParser()
config.read("./configs/measure_variables.cfg")
config_dict = dict(config.items("input_vals"))
data_path_raw = "raw data path"

PT = int(config_dict["pt"])  # 1 #
snapshot = int(config_dict["snapshot"])
simulation = str(config_dict["simulation"])

start_time = time.time()

TNG100_Variable_snapshot = MeasureVariablesSnapshot(
    PT=PT, snapshot=snapshot, output_file_name=data_path_raw + f"{simulation}/SubhaloPT.hdf5", data_path=data_path_raw
)
TNG100_Variable_snapshot.measure_offsets(type='Subhalo')
TNG100_Variable_snapshot.measure_offsets(type='Group')
TNG100_Variable_snapshot.omit_wind_only()
TNG100_Variable_snapshot.select_nonzero_subhalos()
TNG100_Variable_snapshot.save_number_of_particles()
TNG100_Variable_snapshot.measure_COM()
TNG100_Variable_snapshot.measure_velocities()
TNG100_Variable_snapshot.measure_spin()
TNG100_Variable_snapshot.measure_krot()
krot_time = time.time()
TNG100_Variable_snapshot.measure_rotational_velocity(calc_basis=True)
vsig_time = time.time()
print("rot vel done", vsig_time - krot_time)
subPT_time = time.time()
print("SubhaloPT in ", subPT_time - start_time)


TNG100_Variable_snapshot = MeasureVariablesSnapshot(
    PT=PT, snapshot=snapshot, output_file_name=data_path_raw + f"{simulation}/Shapes.hdf5", data_path=data_path_raw
)

TNG100_Variable_snapshot.measure_inertia_tensor(reduced=False)
TNG100_Variable_snapshot.measure_inertia_tensor(reduced=True)
intertia_time = time.time()
print("inertia tensor done", intertia_time - subPT_time)

TNG100_Variable_snapshot.measure_projected_axes(reduced=False)
TNG100_Variable_snapshot.measure_projected_axes(reduced=True)
projected_time = time.time()
print("projected axes done", projected_time - intertia_time)

TNG100_Variable_snapshot.measure_sphericity_triaxality()
ST_time = time.time()
print("ST done", ST_time - vsig_time)
print("total time", time.time() - start_time)
