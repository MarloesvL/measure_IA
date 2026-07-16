"""Jackknife covariance validation: MeasureIALightcone vs treecorr.

The identical kmeans patch assignment (from measureia's
assign_jackknife_patches, seeded) is supplied to both codes on the
lightcone radial-alignment mock, so both compute the same deterministic
delete-one-patch statistic:

- w_gg: treecorr's built-in patch-based jackknife — NNCorrelation with
  patch labels, calculateXi(rr=, dr=, rd=) per signed pi slab, and
  treecorr.estimate_multi_cov over the slabs; the w_gg covariance follows
  by the linear dpi-summation map, cov_w = M cov_xi M^T.
- w_g+: measureia's RR-normalised estimator (S+D - S+R)/RR is not what
  treecorr's compensated NG calculateXi computes, so treecorr's built-in
  covariance machinery cannot reproduce it directly. Instead the
  jackknife loop is made explicit: for each patch, the catalogues minus
  that patch are re-processed with treecorr and the estimator is rebuilt
  from raw counts (exactly as in run_lightcone_treecorr.py), then the
  standard (N-1)/N delete-one formula is applied — the same definition
  measureia uses internally.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lightcone_treecorr as base
from run_lightcone_treecorr import (COSMOLOGY, RP_LIMS, NUM_BINS_RP, PI_MAX,
									NUM_BINS_PI, build_catalogues, run_treecorr)

from measureia import MeasureIALightcone

NUM_JK = 9
PATCH_SEED = 123
DATASET = "lc_treecorr_cov_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs", "lightcone_treecorr_cov.hdf5"
)


def make_measureia(data, randoms, output_file):
	return MeasureIALightcone(
		data={k: v for k, v in data.items()},
		randoms_data={k: v for k, v in randoms.items()},
		output_file_name=output_file,
		separation_limits=RP_LIMS,
		num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI,
		pi_max=PI_MAX,
		num_nodes=1,
	)


def run_measureia_jk(data, randoms, patches, output_file, temp_path):
	ia = make_measureia(data, randoms, output_file)
	ia.measure_xi_w("galaxies", DATASET, "both", jk_patches=patches, num_jk=NUM_JK,
					measure_cov=True, tree=True, cosmology=COSMOLOGY, over_h=False,
					temp_file_path=temp_path)
	out = {}
	with h5py.File(output_file, "r") as f:
		for grp, key in [("w_g_plus", "gp"), ("w_gg", "gg")]:
			out[f"w_{key}"] = f[f"{grp}/{DATASET}"][:]
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
	return ia, out


def run_treecorr_jackknife(data, randoms, dist, patches, r_bins, pi_bins):
	"""Explicit delete-one-patch jackknife with the validated treecorr
	count reconstruction; returns (cov_gp, cov_gg) with the (N-1)/N factor."""
	w_gp, w_gg = [], []
	labels = {
		"D": np.asarray(patches["position"], dtype=int),
		"S": np.asarray(patches["shape"], dtype=int),
		"R_D": np.asarray(patches["randoms_position"], dtype=int),
		"R_S": np.asarray(patches["randoms_shape"], dtype=int),
	}
	for p in range(NUM_JK):
		keep = {k: labels[k] != p for k in labels}
		data_p = {
			"RA": data["RA"][keep["D"]], "DEC": data["DEC"][keep["D"]],
			"weight": data["weight"][keep["D"]],
			"RA_shape_sample": data["RA_shape_sample"][keep["S"]],
			"DEC_shape_sample": data["DEC_shape_sample"][keep["S"]],
			"weight_shape_sample": data["weight_shape_sample"][keep["S"]],
			"e1": data["e1"][keep["S"]], "e2": data["e2"][keep["S"]],
		}
		randoms_p = {
			"RA": randoms["RA"][keep["R_D"]], "DEC": randoms["DEC"][keep["R_D"]],
			"weight": randoms["weight"][keep["R_D"]],
			"RA_shape_sample": randoms["RA_shape_sample"][keep["R_S"]],
			"DEC_shape_sample": randoms["DEC_shape_sample"][keep["R_S"]],
			"weight_shape_sample": randoms["weight_shape_sample"][keep["R_S"]],
		}
		dist_p = {k: dist[{"D": "D", "S": "S", "R_D": "R_D", "R_S": "R_S"}[k]][keep[k]]
				  for k in labels}
		wgp_p, wgg_p = run_treecorr(data_p, randoms_p, dist_p, r_bins, pi_bins)
		w_gp.append(wgp_p)
		w_gg.append(wgg_p)

	def jk_cov(samples):
		samples = np.array(samples)
		mean = samples.mean(axis=0)
		d = samples - mean
		return (NUM_JK - 1) / NUM_JK * (d.T @ d)

	return jk_cov(w_gp), jk_cov(w_gg)


def main():
	import treecorr

	data, randoms, info, dist = build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	ia = make_measureia(data, randoms, scratch)
	patches = ia.assign_jackknife_patches(data, randoms, NUM_JK, seed=PATCH_SEED)
	if "randoms_position" not in patches:
		patches["randoms_position"] = patches["randoms"]
		patches["randoms_shape"] = patches["randoms"]
	if os.path.exists(scratch):
		os.remove(scratch)

	scratch = os.path.join(here, f"{DATASET}_mia_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	ia, mia = run_measureia_jk(data, randoms, patches, scratch, here + "/")
	os.remove(scratch)

	cov_gp_tc, cov_gg_tc = run_treecorr_jackknife(data, randoms, dist, patches,
												  ia.r_bins, ia.pi_bins)

	os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
	with h5py.File(REFERENCE_FILE, "w") as f:
		f.attrs["treecorr_version"] = treecorr.__version__
		f.attrs["num_jk"] = NUM_JK
		f.attrs["patch_seed"] = PATCH_SEED
		f.attrs["mock_seed"] = info["seed"]
		f["cov_w_g_plus"] = cov_gp_tc
		f["cov_w_gg"] = cov_gg_tc

	std_gp_tc = np.sqrt(np.diag(cov_gp_tc))
	std_gg_tc = np.sqrt(np.diag(cov_gg_tc))
	print("--- w_g+ jackknife std (measureia / treecorr) ---")
	print(mia["std_gp"] / std_gp_tc)
	print("--- w_gg jackknife std (measureia / treecorr) ---")
	print(mia["std_gg"] / std_gg_tc)


if __name__ == "__main__":
	main()
