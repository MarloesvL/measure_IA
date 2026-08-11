"""Measure w_gg, w_g+ and the multipoles on lightcone (sky-coordinate) data.

The example runs on the synthetic radial-alignment lightcone mock from
``validation/mock_catalogues.py``: centrals uniform in comoving volume inside a
cone section, with satellites scattered around them whose ellipticities e1/e2
point at their own central. The position (density) sample is the centrals and
the shape sample is the satellites, giving a strong, seeded (i.e. reproducible)
w_g+ signal, so the run below produces a real measurement in ~1 s without
needing any survey data. Swap the ``data`` / ``data_r`` entries for your own
catalogue to measure your own data.

The mock is generated in comoving distance ("r_com" keys); MeasureIALightcone
takes redshifts, so the distances are inverted to redshift with the *same*
cosmology that is passed to the measurement methods below.

Run it from this directory:

    python example_measure_IA_lightcone.py

It writes ./example_IA_lightcone.hdf5.
"""
import os
import sys

import pyccl as ccl

from measureia import MeasureIALightcone

# The mock generator lives in the repository's validation/ directory, next to examples/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
from mock_catalogues import radial_alignment_lightcone_mock

# parameters for MeasureIA object
h = 0.7  # value of hubble parameter
num_bins_r = 8  # number of r or rp bins
num_bins_pi = 6  # number of pi bins. (trivial for wg+, not too little for multipoles)
separation_limits = [1.0, 20.0]  # Mpc
LOS_lims = 30.0  # pi max value
data_path_out = "./"  # file path to output data
file_name = "example_IA_lightcone.hdf5"  # datafile name (must be hdf5)
num_nodes = 1  # number of CPU cores available for calculation

IA_estimator = "galaxies"  # type of estimator to be used. Choose "clusters" or "galaxies"
# Definition of the estimator will be printed when a method is called. "galaxies" uses the randoms (RR) in the
# denominator, "clusters" normalises by the data pairs instead; use the latter when the position sample is a
# cluster catalogue.

# parameters for methods
cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=h, sigma8=0.8, n_s=0.96)  # pyccl cosmology to be used.
# If None, a default cosmology is used (Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0).
over_h = False  # if True, units are changed from Mpc -> Mpc/h
calc_errors = True  # If true, jackknife errors are calculated (Default is True)
num_jk = 8  # number of jackknife patches, assigned internally on the sky with kmeans
corr_type = "both"  # type of correlation to be calculated, choose g+, gg or both
masks = None  # optional directory in form of data_dir containing masks to be placed over data in data_dir
masks_randoms = None  # same as masks, but for the randoms


def redshift_of_comoving_distance(chi):
	"""Invert the CCL comoving distance to get the redshift of each object."""
	return 1.0 / ccl.scale_factor_of_chi(cosmology, chi) - 1.0


# Mock catalogue with a known radial-alignment signal (seeded, so the result is reproducible).
# `data` holds the D and S samples, `data_r` the randoms for both; n_randoms_factor sets how many
# randoms per data object (enough randoms are needed to fill every rp/pi bin, or the estimator is NaN there).
data, data_r, info = radial_alignment_lightcone_mock(n_centrals=400, n_sat=8, n_randoms_factor=10)
for _catalogue in (data, data_r):
	_catalogue["Redshift"] = redshift_of_comoving_distance(_catalogue.pop("r_com"))
	_catalogue["Redshift_shape_sample"] = redshift_of_comoving_distance(_catalogue.pop("r_com_shape_sample"))

#  dictionary of randoms position data and shape-sample randoms (for the S+R term).
# data_r = {"Redshift": ...,  # redshift of R_D sample
#           "Redshift_shape_sample": ...,  # redshift of R_S sample (optional)
#           "RA": ...,  # RA of R_D sample
#           "RA_shape_sample": ...,  # RA of R_S sample (optional)
#           "DEC": ...,  # DEC of R_D sample
#           "DEC_shape_sample": ...,  # DEC of R_S sample (optional)
#           }
# If only the Redshift, RA and DEC are provided, the code will use this random sample for both positions and shapes.

# dictionary of the data (for the S+D term):
# {"Redshift": redshift of D sample, "Redshift_shape_sample": redshift of S sample,
#  "RA": RA of D sample, "RA_shape_sample": RA of S sample,
#  "DEC": DEC of D sample, "DEC_shape_sample": DEC of S sample,
#  "e1": e1 of S sample, "e2": e2 of S sample}

# Optional: pre-assigned jackknife patches. If left as None and num_jk is an int, the patches are assigned
# internally with kmeans on the sky. To supply your own, pass a dictionary of patch indices per sample:
# patches = {"position": ..., "shape": ..., "randoms": ...}
patches = None

dataset_name = "mock"

if __name__ == "__main__":  # when using multiprocessing, this statement is needed.
	MeasureIA_mock_obs = MeasureIALightcone(data, data_r,
											separation_limits=separation_limits, pi_max=LOS_lims,
											num_bins_r=num_bins_r, num_nodes=num_nodes,
											num_bins_pi=num_bins_pi, output_file_name=data_path_out + file_name)
	# measure wgg, wg+
	MeasureIA_mock_obs.measure_xi_w(IA_estimator, dataset_name, corr_type, jk_patches=patches, num_jk=num_jk,
									measure_cov=calc_errors, masks=masks, masks_randoms=masks_randoms,
									cosmology=cosmology, over_h=over_h)
	# measure multipoles
	MeasureIA_mock_obs.measure_xi_multipoles(IA_estimator, dataset_name, corr_type, jk_patches=patches,
											 num_jk=num_jk, measure_cov=calc_errors, masks=masks,
											 masks_randoms=masks_randoms,
											 cosmology=cosmology, over_h=over_h)
	print(f"wrote {data_path_out + file_name}")
