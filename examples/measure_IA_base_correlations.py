import numpy as np
from measure_IA_base import MeasureIABase

# creating random sample data
positions = np.random.rand(100, 3) * 100  # in cMpc, so boxlength is 100 cMpc here
positions_shape = np.random.rand(50, 3) * 100  # shape sample positions
x_dir = np.random.rand(50)
y_dir = np.sqrt(1. - x_dir ** 2)
axis_dir = np.array([x_dir, y_dir]).transpose()
q = np.random.rand(50)

# read data
COM = positions  # "center of mass data"
COM_shape = positions_shape  # "center of mass for shape data"
Semimajor_Axis_Direction = axis_dir  # "eigenvectors"
q = q  # "b/a"

# create directory and parameters
data_IA_projected = {
	"Position": COM,  # this can also be COM_shape, if the shape and position samples are the same
	"Position_shape_sample": COM_shape,
	"Axis_Direction": Semimajor_Axis_Direction,
	"LOS": 2,  # Line of sight
	"q": q,
}
num_bins_r = 10
num_bins_pi = 8
separation_limits = [0.1, 20.0]  # cMpc/h
boxsize = 100.  # boxsize in cMpc/h
LOS_lims = None
data_path_out = "data path for output data"
file_name = "name of output file"  # needs to be hdf5
dataset_name = "specific dataset name"
tree_path = "path to where you want to save the tree info where the error method can access it"
corr_type = ['g+', 'w']  # there are multiple choices here: [0]: g+, gg or both and [1]: w or multipoles
L_subboxes = 3  # for jk errors -> 3^3 boxes
rp_cut = None

IA_Projected = MeasureIABase(
	data_IA_projected,
	simulation="TNG100",
	num_bins_r=num_bins_r,
	num_bins_pi=num_bins_pi,
	separation_limits=separation_limits,
	pi_max=LOS_lims,
	output_file_name=data_path_out + file_name,
	boxsize=boxsize,
)

'''
There are multiple options for calculations. 
Below, you can find the methods you need to calculate everything using a single core.
Using the 'save tree' option will speed your results up by about 1.3. If you do not wish to use this,
set the 'save_tree' input value to False and do not add a file_tree_path.
'''

# wg+,wgg
IA_Projected.measure_projected_correlation_tree(dataset_name=dataset_name, save_tree=True, file_tree_path=tree_path)
IA_Projected._measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name)
IA_Projected.measure_jackknife_errors(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	rp_cut=rp_cut,
	tree_saved=True,
	file_tree_path=tree_path
)
corr_type = ['g+', 'multipoles']
# multipoles
IA_Projected.measure_projected_correlation_multipoles_tree(dataset_name=dataset_name, rp_cut=rp_cut, save_tree=True,
														   file_tree_path=tree_path)
IA_Projected._measure_multipoles(dataset_name=dataset_name, corr_type=corr_type[0])
IA_Projected.measure_jackknife_errors(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	rp_cut=rp_cut,
	tree_saved=True,
	file_tree_path=tree_path
)

'''
If you want to use more than one core in your calculations, you have two options. 
Which is faster depends on your number of cores available and the number of jackknife regions.
If your number of available cores is < the number of jackknife regions, or close to it, this method is fastest:
'''
num_nodes = 9
L_subboxes = 3  # 27 jk regions
# wg+,wgg
IA_Projected.measure_projected_correlation_tree(dataset_name=dataset_name, save_tree=True, file_tree_path=tree_path)
IA_Projected._measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name)
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=num_nodes,
	rp_cut=rp_cut,
	tree_saved=True,
	file_tree_path=tree_path
)
corr_type = ['g+', 'multipoles']
# multipoles
IA_Projected.measure_projected_correlation_multipoles_tree(dataset_name=dataset_name, rp_cut=rp_cut, save_tree=True,
														   file_tree_path=tree_path)
IA_Projected._measure_multipoles(dataset_name=dataset_name, corr_type=corr_type[0])
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=num_nodes,
	rp_cut=rp_cut,
	tree_saved=True,
	file_tree_path=tree_path
)

'''
If your number of available cores is >> the number of jackknife regions (e.g. a factor of 2 or more), 
this method becomes faster:
Due to the nature of the methods, they cannot be combined with the 'save tree' option.
'''
num_nodes = 30
L_subboxes = 2  # 8 jk regions
# wg+,wgg
IA_Projected.measure_projected_correlation_multiprocessing(num_nodes=num_nodes, dataset_name=dataset_name)
IA_Projected._measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name)
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=num_nodes,
	rp_cut=rp_cut,
	tree_saved=False
)
corr_type = ['g+', 'multipoles']
# multipoles
IA_Projected.measure_projected_correlation_multipoles_multiprocessing(num_nodes=num_nodes,
																	  dataset_name=dataset_name, rp_cut=rp_cut)
IA_Projected._measure_multipoles(dataset_name=dataset_name, corr_type=corr_type[0])
IA_Projected.measure_jackknife_errors_multiprocessing(
	corr_type=corr_type,
	dataset_name=dataset_name,
	L_subboxes=L_subboxes,
	num_nodes=num_nodes,
	rp_cut=rp_cut,
	tree_saved=False
)
