"""Measure w_gg, w_g+ and the multipoles on a periodic simulation box.

The example runs on the synthetic radial-alignment mock from
``measureia.mocks``: centrals placed uniformly in a periodic box
with satellites scattered around them, whose projected major axes point at
their own central. That gives a strong, seeded (i.e. reproducible) w_g+ and
w_gg signal, so the run below produces a real measurement in ~1 s without
needing any simulation data. Swap the ``data_dir`` entries for your own
catalogue to measure your own data.

Run it from this directory:

    python example_measure_IA_box.py

It writes ./example_IA_box.hdf5, which example_read_and_plot.py reads back.
"""
from measureia import MeasureIABox
from measureia.mocks import radial_alignment_box_mock

# parameters for MeasureIA object
simulation = None  # Indicator of simulation.
# Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8,
# COLIBRE_L400, COLIBRE_L200] for now, which sets the
# boxsize automatically. If your simulation is not included (as for this mock): input None and make sure to add the
# boxsize input parameter in the object creation.
# If this is used, the boxsize is in Mpc/h so the positions and r/rp limits must also be in these units.
boxsize = 205.0  # size of the periodic box, in the same units as the positions. Only used if simulation is None.
snapshot = None  # Number of snapshot - if given, the output is saved in group 'Snapshot_[snapshot]'
outfile = "./example_IA_box.hdf5"  # file path to output data and datafile name (must be hdf5)
num_nodes = 2  # number of CPU cores available for calculation
r_lims = [0.3, 8.0]  # r or rp bin edges. Must be in same units as position and boxsize.
num_r = 8  # number of r or rp bins
num_pi = 8  # number of pi bins. (trivial for wg+, not too little for multipoles)
pi_max = None  # pi max value. If None, half the boxsize is used.
periodicity = True  # take periodic boundary conditions into account

# parameters for methods
tree_path = "./"  # path where tree can be temporarily stored. For large samples this file can grow large.
# If tree_path=None, no trees will be used (slower calculation)
num_jk = 27  # number of jackknife regions. Must be x^3 with x and int. Use 0 for no covariance.
corr_type = "both"  # type of correlation to be calculated, choose g+, gg or both
masks = None  # optional directory in form of data_dir containing masks to be placed over data in data_dir
rp_cut = None  # optional minimum cut on rp for multipoles calculation

# Mock catalogue with a known radial-alignment signal (seeded, so the result is reproducible).
# n_centrals centrals + n_centrals * n_sat satellites in a periodic box of `boxsize`.
mock = radial_alignment_box_mock(n_centrals=600, n_sat=8, boxsize=boxsize)

q = mock["q"]  # 1D array containing all values of q (= b/a, projected axis lengths)
# 2D array with eigen vectors of each galaxy shape with vector elements in columns
Axis_direction = mock["Axis_Direction"]
COM = mock["Position"]  # 2D array of positions of galaxies with x,y,z as columns
COM_shapes = mock["Position_shape_sample"]  # positions of the galaxies that have a measured shape

data_dir = {
	"Position": COM,  # positions of the position (D) sample
	"Position_shape_sample": COM_shapes,  # positions of the shape (S) sample
	"Axis_Direction": Axis_direction,
	"LOS": 2,  # column index of the line of sight parameter (2 assumes the shapes are projected over the z-axis)
	"q": q,
}

dataset_name = "mock"

if __name__ == "__main__":  # when using multiprocessing, this statement is needed.
	MeasureIA_mock = MeasureIABox(data_dir, simulation=simulation, snapshot=snapshot, boxsize=boxsize,
								  separation_limits=r_lims, pi_max=pi_max,
								  num_bins_r=num_r, num_nodes=num_nodes,
								  num_bins_pi=num_pi, output_file_name=outfile, periodicity=periodicity)
	# calculate wgg, wg+
	MeasureIA_mock.measure_xi_w(dataset_name, corr_type, num_jk, temp_file_path=tree_path, masks=masks)
	# calculate multipoles
	MeasureIA_mock.measure_xi_multipoles(dataset_name, corr_type, num_jk, temp_file_path=tree_path, masks=masks,
										 rp_cut=rp_cut)
	print(f"wrote {outfile}")
