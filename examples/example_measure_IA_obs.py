import numpy as np
from measure_IA import MeasureIA

# data
RA = np.array([])  # RA of clusters
DEC = np.array([])  # DEC of clusters
e1 = np.array([])  # e1 of clusters
e2 = np.array([])  # e2 of clusters
z = np.array([])  # redshift of clusters

RA_rand = np.array([])  # RA of randoms
DEC_rand = np.array([])  # DEC of randoms
z_rand = np.array([])  # redshift of randoms

jk_patches_randoms = np.array([])  # jackknife patch index for each random point
jk_patches_shape = np.array([])  # jackknife patch index for each cluster
# directory of jackknife information for the position (D), shape (S) and randoms (R) samples
patches = {"position": jk_patches_shape, "shape": jk_patches_shape,
		   "randoms": jk_patches_randoms}

# parameters for MeasureIA object
h = 0.7  # value of hubble parameter
num_bins_r = 10  # number of r or rp bins
num_bins_pi = 20  # number of pi bins. (trivial for wg+, not too little for multipoles)
separation_limits = [2.5 / h, 140.0 / h]  # Mpc
LOS_lims = 100. / h  # pi max value
data_path_out = "../"  # file path to output data
file_name = "test_IA.hdf5"  # datafile name (must be hdf5)
num_nodes = 5  # number of CPU cores available for calculation

IA_estimator = "clusters"  # type of estimator to be used. Choose "clusters" or "galaxies"
# Definition of the estimator will be printed when a method is called

# parameters for methods
cosmology = None  # pyccl cosmology to be used. WMAP9 is default if None
over_h = False  # if True, units are changed from Mpc -> Mpc/h
calc_errors = True  # If true, jackknife errors are calculated (Default is True)
num_jk = 27  # number of jackknife regions. Must be x^3 with x and int.
corr_type = "g+"  # type of correlation to be calculated, choose g+, gg or both
masks = None  # optional directory in form of data_dir containing masks to be placed over data in data_dir
masks_randoms = None  # same as masks, but for the randoms
rp_cut = None  # optional minimum cut on rp for multipoles calculation

#  directory of randoms position data and cluster shape data (for S+R term)
data_r = {"Redshift": z_rand,  # redshift of R_D sample
		  "Redshift_shape_sample": z,  # redshift of R_S sample (optional)
		  "RA": RA_rand,  # RA of R_D sample
		  "RA_shape_sample": RA,  # RA of R_S sample (optional)
		  "DEC": DEC_rand,  # DEC of R_D sample
		  "DEC_shape_sample": DEC,  # DEC of R_S sample (optional)
		  }
# If only the Redshift, RA and DEC are provided, the code will use this random sample for both positions and shape clustering

# directory of cluster data (for S+D term)
data = {"Redshift": z,  # redshift of D sample
		"Redshift_shape_sample": z,  # redshift of S sample
		"RA": RA,  # RA of D sample
		"RA_shape_sample": RA,  # RA of S sample
		"DEC": DEC,  # DEC of D sample
		"DEC_shape_sample": DEC,  # DEC of S sample
		"e1": e1,  # e1 of S sample
		"e2": e2}  # e2 of S sample

MeasureIA_validation_obs = MeasureIA(data, simulation=False,
									 separation_limits=separation_limits, pi_max=LOS_lims,
									 num_bins_r=num_bins_r, num_nodes=num_nodes,
									 num_bins_pi=num_bins_pi, output_file_name=data_path_out + file_name)
dataset_name = "test"
# measure wg+
MeasureIA_validation_obs.measure_xi_w_obs("clusters", dataset_name, corr_type, jk_patches=patches, randoms_data=data_r,
										  measure_cov=calc_errors, masks=masks, masks_randoms=masks_randoms,
										  cosmology=cosmology,
										  over_h=over_h)
# measure multipoles
MeasureIA_validation_obs.measure_xi_multipoles_obs("clusters", dataset_name, corr_type, jk_patches=patches,
												   randoms_data=data_r,
												   calc_errors=calc_errors, rp_cut=rp_cut, masks=masks,
												   masks_randoms=masks_randoms,
												   cosmology=cosmology, over_h=over_h)
