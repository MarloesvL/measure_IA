import numpy as np
from measureia import MeasureIABox

# parameters for MeasureIA object
simulation = "TNG300"  # Indicator of simulation.
# Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8] for now.
# If your simulation is not included: input None and make sure to add the boxsize input parameter in the object creation.
# If this is used, the boxsize is in Mpc/h so the positions and r/rp limits must also be in these units.
snapshot = 99  # Number of snapshot - for saving the output in group 'Snapshot_99'
outfile = f'./testing_IA.hdf5'  # file path to output data and datafile name (must be hdf5)
num_nodes = 3  # number of CPU cores available for calculation
r_lims = [0.1, 20.]  # r or rp bin edges. Must be in same units as position and boxsize.
num_r = 10  # number of r or rp bins
num_pi = 8  # number of pi bins. (trivial for wg+, not too little for multipoles)
LOS_lim = None  # pi max value
periodicity = True  # take periodic boundary conditions into account

# parameters for methods
tree_path = f"../"  # path where tree can be temporarily stored. For large samples this file can grow large.
# If tree_path=None, no trees will be used (slower calculation)
calc_errors = True  # If true, jackknife errors are calculated (Default is True)
num_jk = 27  # number of jackknife regions. Must be x^3 with x and int.
corr_type = "g+"  # type of correlation to be calculated, choose g+, gg or both
remove_tree_file = True  # if True (default), tree file is removed at the end of the calculation.
masks = None  # optional directory in form of data_dir containing masks to be placed over data in data_dir
rp_cut = None  # optional minimum cut on rp for multipoles calculation

q = np.array([])  # 1D array containing all values of q (= b/a, projected axis lengths)
Axis_direction = np.array(
	[[], []]).transpose()  # 2D array with eigen vectors of each galaxy shape with vector elements in columns
COM = np.array([[], [], []]).transpose()  # 2D array of positions of galaxies with x,y,z as columns

data_dir = {
	"Position": COM,  # positions of the position (D) sample
	"Position_shape_sample": COM,  # positions of the shape (S) sample
	"Axis_Direction": Axis_direction,
	"LOS": 2,  # column index of the line of sight parameter (2 assumes the shapes are projected over the z-axis)
	"q": q,
}

dataset_name = "test"

MeasureIA_validation = MeasureIABox(data_dir, simulation=simulation, snapshot=snapshot,
									separation_limits=r_lims, pi_max=LOS_lim,
									num_bins_r=num_r, num_nodes=num_nodes,
									num_bins_pi=num_pi, output_file_name=outfile, periodicity=periodicity)
# calculate wg+
MeasureIA_validation.measure_xi_w(dataset_name, corr_type, num_jk, file_tree_path=tree_path)
# calculate multipoles
MeasureIA_validation.measure_xi_multipoles(dataset_name, corr_type, num_jk, file_tree_path=tree_path)
