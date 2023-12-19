import numpy as np
from src.read_data_TNG import ReadTNGdata
from src.measure_IA import MeasureIA

# read data
COM = "center of mass data"
Semimajor_Axis_Direction = "eigenvectors"
q = "b/a"

# create directory and parameters
data_IA_projected = {
	"Position": COM,
	"Position_shape_sample": COM,
	"Axis_Direction": Semimajor_Axis_Direction,
	"LOS": 2,  # Line of sight
	"q": q,
}
num_bins_r = 10
num_bins_pi = 8
separation_limits = [2.0, 20.0]  # cMpc/h
LOS_lims = None
data_path_out = "data path for output data"
file_name = "name of output file" # needs to be hdf5
dataset_name = "specific dataset name"
corr_type = ['g+', 'w']  # there are multiple choices here
L_subboxes = 3  # for jk errors -> 3^3 boxes
rp_cut = None

IA_Projected = MeasureIA(
	data_IA_projected,
	num_bins_r=num_bins_r,
	num_bins_pi=num_bins_pi,
	separation_limits=separation_limits,
	LOS_lim=LOS_lims,
	output_file_name=data_path_out + file_name,
)
# wg+,wgg
IA_Projected.measure_projected_correlation(dataset_name=dataset_name)
IA_Projected.measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name)
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=9,
	rp_cut=rp_cut,
)
corr_type = ['g+', 'multipoles']
# multipoles
IA_Projected.measure_projected_correlation_multipoles(dataset_name=dataset_name, rp_cut=rp_cut)
IA_Projected.measure_multipoles(dataset_name=dataset_name, corr_type=corr_type[0])
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=9,
	rp_cut=rp_cut,
)
