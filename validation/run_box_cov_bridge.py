"""Box jackknife covariance validation: delete-one identity + bridge.

The box jackknife partitions the periodic box into L^3 subboxes and
reconstructs each delete-one realisation by count subtraction (a pair is
removed if EITHER member lies in the deleted subbox), with the analytic RR
rescaled to the retained counts and volume. Two complementary checks:

1. DELETE-ONE IDENTITY (the rigorous machinery test): every reconstructed
   realisation must equal an independent direct measurement on the
   physically deleted catalogue — same estimator, same analytic RR formula.
   The retained S+D and DD grids match to floating-point precision, the
   jackknife RR equals the direct-run RR times the exact volume factor
   V/V_del = L^3/(L^3-1), and the per-realisation w vectors follow. This
   locks the count-subtraction reconstruction and the per-realisation
   responsivity with no external approximation involved.

2. COVARIANCE BRIDGE (cross-pipeline sanity check): the identical subbox
   partition is supplied as jk_patches to the treecorr-validated lightcone
   jackknife on the plane-parallel embedding of the same catalogue
   (responsivity=True so the estimators match). The covariances agree only
   at the tens-of-percent level (std ratios ~0.7-1.05 with 8 patches), and
   this is the EXPECTED outcome, fully attributed (2026-07-16 forensics):

   a. Rebuilding the box-style estimator from the lightcone's own retained
	  counts reproduces the LIGHTCONE covariance (std ratios ~0.9-1.1,
	  printed below as the 'box-style estimator from lightcone counts'
	  row) — so estimator machinery is consistent; what differs between
	  the pipelines are the raw retained counts themselves: plane-parallel
	  vs sky geometry migrates ~0.1-1% of pairs between bins per
	  realisation, and with 8 patches the delete-one deviations are only a
	  few % of the mean, so tiny realisation-dependent count differences
	  distort individual deviations by tens of percent.
   b. The estimator definitions genuinely differ (box: natural DD/RR-1 and
	  S+D/RR_analytic with per-realisation responsivity; lightcone:
	  LS-compensated (DD-RD-SR)/RR+1 and (S+D-S+R)/RR_empirical with
	  full-sample responsivity). Different estimators have different
	  covariances.
   c. The analytic-RR-under-deletion approximation: the count/volume
	  rescale amplitude is exact, but the hole-boundary suppression varies
	  ~2% across rp bins (printed below), moving stds by <~15%.

   Correlation matrices with 8 realisations are additionally very noisy,
   so element-level differences up to ~0.7 arise from the same causes.

The rigorous covariance validation of the jackknife machinery is therefore
check 1 (machine precision) plus the lightcone-vs-treecorr comparison in
run_lightcone_treecorr_cov.py (<=5e-5); this bridge documents the expected
level of cross-pipeline agreement and quantifies the analytic-RR
approximation.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_plane_parallel as pp
from measureia.mocks import embed_box_mock_on_lightcone, subbox_labels
from run_lightcone_treecorr import COSMOLOGY, _redshift_of_chi
from measureia import MeasureIABox, MeasureIALightcone

NUM_JK = 8  # 2x2x2 subboxes; box jackknife requires x^3
VOLUME_FACTOR = NUM_JK / (NUM_JK - 1.0)  # V_box / V_delete-one
DATASET = "box_cov_bridge_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs", "box_cov_bridge.hdf5"
)


def _box_data(mock):
	return {k: mock[k] for k in
			["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}


def _make_box(data, output_file, mock):
	return MeasureIABox(data, output_file, simulation=None, snapshot=None,
						separation_limits=pp.RP_LIMS, num_bins_r=pp.NUM_BINS_RP,
						num_bins_pi=pp.NUM_BINS_PI, pi_max=pp.PI_MAX,
						boxsize=mock["boxsize"], num_nodes=1)


def _read_cov(fname):
	out = {}
	with h5py.File(fname, "r") as f:
		for grp, key in [("w_g_plus", "gp"), ("w_gg", "gg")]:
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
	return out


def run_box_jk(mock, output_file, temp_path):
	"""Box jackknife run; returns covariance plus all per-realisation pieces."""
	ia = _make_box(_box_data(mock), output_file, mock)
	ia.measure_xi_w(DATASET, "both", num_jk=NUM_JK, temp_file_path=temp_path)
	out = _read_cov(output_file)
	jk = f"{DATASET}_jk{NUM_JK}"
	with h5py.File(output_file, "r") as f:
		out["w_gp_real"] = np.array(
			[f[f"w_g_plus/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
		out["w_gg_real"] = np.array(
			[f[f"w_gg/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
		out["SpD_real"] = np.array(
			[f[f"w/xi_g_plus/{jk}/{DATASET}_{i}_SplusD"][:] for i in range(NUM_JK)])
		out["DD_real"] = np.array(
			[f[f"w/xi_gg/{jk}/{DATASET}_{i}_DD"][:] for i in range(NUM_JK)])
		out["RR_real"] = np.array(
			[f[f"w/xi_g_plus/{jk}/{DATASET}_{i}_RR"][:] for i in range(NUM_JK)])
	return out


def run_delete_one_direct(mock, output_file, temp_path):
	"""Independent delete-one measurements: physically remove each subbox
	and measure with the plain (non-jackknife) box estimator."""
	ia_full = _make_box(_box_data(mock), output_file, mock)
	lab_pos, lab_shape = ia_full._get_jackknife_region_indices(
		None, round(NUM_JK ** (1 / 3)))
	results = []
	for i in range(NUM_JK):
		keep_p, keep_s = lab_pos != i, lab_shape != i
		data_i = {
			"Position": mock["Position"][keep_p],
			"Position_shape_sample": mock["Position_shape_sample"][keep_s],
			"Axis_Direction": mock["Axis_Direction"][keep_s],
			"q": mock["q"][keep_s],
			"LOS": mock["LOS"],
		}
		name = f"{DATASET}_direct_del{i}"
		ia = _make_box(data_i, output_file, mock)
		ia.measure_xi_w(name, "both", 0, temp_file_path=temp_path)
		with h5py.File(output_file, "r") as f:
			results.append({
				"SpD": f[f"w/xi_g_plus/{name}_SplusD"][:],
				"DD": f[f"w/xi_gg/{name}_DD"][:],
				"RR": f[f"w/xi_g_plus/{name}_RR_g_plus"][:],
				"w_gp": f[f"w_g_plus/{name}"][:],
				"w_gg": f[f"w_gg/{name}"][:],
			})
	return results


def delete_one_identity(box, direct):
	"""Max relative differences between the jackknife's reconstructed
	realisations and the independent direct delete-one measurements.
	All should be at floating-point level (<~1e-12)."""
	total_pi = 2.0 * pp.PI_MAX

	def maxrel(a, b):
		scale = np.maximum(np.abs(b), 1e-8 * np.max(np.abs(b)))
		return np.max(np.abs(a - b) / scale)

	res = {k: 0.0 for k in ["DD", "SplusD", "RR", "w_g_plus", "w_gg"]}
	for i in range(NUM_JK):
		d = direct[i]
		res["DD"] = max(res["DD"], maxrel(box["DD_real"][i], d["DD"]))
		res["SplusD"] = max(res["SplusD"], maxrel(box["SpD_real"][i], d["SpD"]))
		# analytic RR: identical formula, volume V_del = V * (N-1)/N
		res["RR"] = max(res["RR"], maxrel(box["RR_real"][i], d["RR"] * VOLUME_FACTOR))
		# xi_jk = xi_direct / VOLUME_FACTOR (g+); xi_jk+1 = (xi_direct+1)/VF (gg)
		res["w_g_plus"] = max(res["w_g_plus"], maxrel(
			box["w_gp_real"][i], d["w_gp"] / VOLUME_FACTOR))
		res["w_gg"] = max(res["w_gg"], maxrel(
			box["w_gg_real"][i], (d["w_gg"] + total_pi) / VOLUME_FACTOR - total_pi))
	return res


def run_lightcone_jk(mock, output_file, temp_path):
	"""Lightcone jackknife on the embedded catalogue with the identical
	subbox partition; returns covariance plus per-realisation raw counts
	and the sample sizes needed to renormalise them."""
	data, randoms, cube = embed_box_mock_on_lightcone(
		mock, distance=pp.DISTANCE, n_randoms_factor=pp.N_RANDOMS_FACTOR)
	L = round(NUM_JK ** (1 / 3))
	labels = {
		"position": subbox_labels(mock["Position"], mock["boxsize"], L),
		"shape": subbox_labels(mock["Position_shape_sample"], mock["boxsize"], L),
		"randoms_position": subbox_labels(cube["randoms_position"], mock["boxsize"], L),
		"randoms_shape": subbox_labels(cube["randoms_shape"], mock["boxsize"], L),
	}
	# D-S overlap exactly as measure_xi_w computes it (matching RA/DEC pairs)
	cD = np.column_stack((data["RA"], data["DEC"]))
	cS = np.column_stack((data["RA_shape_sample"], data["DEC_shape_sample"]))
	_, ind_D, _ = np.intersect1d(cD.view([("", cD.dtype)] * 2),
								 cS.view([("", cS.dtype)] * 2), return_indices=True)
	key_map = {"D": "position", "S": "shape",
			   "R_D": "randoms_position", "R_S": "randoms_shape"}
	counts = {k: len(labels[v]) for k, v in key_map.items()}
	counts["D_S"] = len(ind_D)
	counts_i = [{
		**{k: counts[k] - int(np.sum(labels[v] == i)) for k, v in key_map.items()},
		"D_S": counts["D_S"] - int(np.sum(labels["position"][ind_D] == i)),
	} for i in range(NUM_JK)]

	for d in (data, randoms):
		d["Redshift"] = _redshift_of_chi(d.pop("r_com"))
		d["Redshift_shape_sample"] = _redshift_of_chi(d.pop("r_com_shape_sample"))
	ia = MeasureIALightcone(data=data, randoms_data=randoms,
							output_file_name=output_file,
							separation_limits=pp.RP_LIMS, num_bins_r=pp.NUM_BINS_RP,
							num_bins_pi=pp.NUM_BINS_PI, pi_max=pp.PI_MAX, num_nodes=1)
	ia.measure_xi_w("galaxies", DATASET, "both", jk_patches=labels, num_jk=NUM_JK,
					tree=True, cosmology=COSMOLOGY, over_h=False,
					temp_file_path=temp_path, responsivity=True)
	out = _read_cov(output_file)
	out["counts"], out["counts_i"] = counts, counts_i
	jk = f"{DATASET}_jk{NUM_JK}"
	with h5py.File(output_file, "r") as f:
		for k, grp, suff in [("SpD", "xi_g_plus", "_SplusD"), ("DD", "xi_gg", "_DD"),
							 ("RR", "xi_gg", "_RR")]:
			out[k] = np.array(
				[f[f"w/{grp}/{jk}/{DATASET}_{i}{suff}"][:] for i in range(NUM_JK)])
			out[k + "_full"] = f[f"w/{grp}/{DATASET}{suff}"][:]
	return out


def _jk_std(ws):
	d = ws - ws.mean(axis=0)
	return np.sqrt((NUM_JK - 1) / NUM_JK * np.sum(d ** 2, axis=0))


def boxstyle_from_lc_counts(lc, mock):
	"""Rebuild the BOX-style estimator (numerator-only, analytic-RR rescale,
	per-realisation responsivity) from the lightcone run's retained counts.
	Its covariance matching the lightcone one shows the cross-pipeline
	discrepancy lives in the retained counts (geometry), not the jackknife
	machinery."""
	n, ni = lc["counts"], lc["counts_i"]
	L = round(NUM_JK ** (1 / 3))
	lab_shape = subbox_labels(mock["Position_shape_sample"], mock["boxsize"], L)
	q = mock["q"]
	e = (1 - q ** 2) / (1 + q ** 2)
	R_full = np.mean(1 - e ** 2 / 2.0)
	dpi = 2.0 * pp.PI_MAX / pp.NUM_BINS_PI
	w_gp, w_gg = [], []
	for i in range(NUM_JK):
		R_i = np.mean(1 - e[lab_shape != i] ** 2 / 2.0)
		# analytic-style RR: full-sample empirical RR shape, exact
		# count/volume amplitude rescale, in data-pair-count units
		rr_ana = (lc["RR_full"] * (ni[i]["D"] * ni[i]["S"]) / (n["R_D"] * n["R_S"])
				  * VOLUME_FACTOR)
		w_gp.append(np.sum(lc["SpD"][i] * (R_full / R_i) / rr_ana, axis=1) * dpi)
		w_gg.append(np.sum(lc["DD"][i] / rr_ana - 1, axis=1) * dpi)
	return {"std_gp": _jk_std(np.array(w_gp)), "std_gg": _jk_std(np.array(w_gg))}


def rr_shape_error(lc):
	"""Bin-dependence of the empirical delete-one RR that the analytic
	rescale misses (hole-boundary effect): max over realisations of
	(max/min - 1) across rp bins of the normalised RR ratio."""
	n, ni = lc["counts"], lc["counts_i"]
	full = lc["RR_full"] / (n["R_D"] * n["R_S"])
	worst = 0.0
	for i in range(NUM_JK):
		ratio = (lc["RR"][i] / (ni[i]["R_D"] * ni[i]["R_S"]) / full).mean(axis=1)
		worst = max(worst, ratio.max() / ratio.min() - 1)
	return worst


def corrmat(cov):
	s = np.sqrt(np.diag(cov))
	return cov / np.outer(s, s)


def main():
	mock = pp.build_mock()
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_box_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	box = run_box_jk(mock, scratch, here + "/")
	os.remove(scratch)

	scratch = os.path.join(here, f"{DATASET}_direct_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	direct = run_delete_one_direct(mock, scratch, here + "/")
	os.remove(scratch)
	identity = delete_one_identity(box, direct)

	scratch = os.path.join(here, f"{DATASET}_lc_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	lc = run_lightcone_jk(mock, scratch, here + "/")
	os.remove(scratch)

	boxstyle = boxstyle_from_lc_counts(lc, mock)
	rr_err = rr_shape_error(lc)

	os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
	with h5py.File(REFERENCE_FILE, "w") as f:
		f.attrs["num_jk"] = NUM_JK
		f.attrs["mock_seed"] = mock["seed"]
		f.attrs["rr_shape_error"] = rr_err
		for k, v in identity.items():
			f.attrs[f"identity_{k}"] = v
		for k in ["cov_gp", "std_gp", "cov_gg", "std_gg"]:
			f[f"box_{k}"] = box[k]
			f[f"lightcone_{k}"] = lc[k]
		f["boxstyle_std_gp"] = boxstyle["std_gp"]
		f["boxstyle_std_gg"] = boxstyle["std_gg"]

	print("\n--- 1. delete-one identity: jackknife reconstruction vs direct")
	print("       measurement on the physically deleted catalogue (max rel diff) ---")
	for k, v in identity.items():
		print(f"{k:10s}: {v:.2e}")

	print("\n--- 2. covariance bridge: box vs lightcone with the same partition ---")
	print("jackknife std ratios (box / lightcone):")
	print(f"w_g+: {box['std_gp'] / lc['std_gp']}")
	print(f"w_gg: {box['std_gg'] / lc['std_gg']}")
	print("corr-matrix max |diff|:")
	print(f"w_g+: {np.max(np.abs(corrmat(box['cov_gp']) - corrmat(lc['cov_gp']))):.4f}")
	print(f"w_gg: {np.max(np.abs(corrmat(box['cov_gg']) - corrmat(lc['cov_gg']))):.4f}")

	print("\n--- 3. attribution ---")
	print("box-style estimator rebuilt from the lightcone's own retained counts,")
	print("std ratio vs lightcone (near 1 -> the bridge residual above comes from")
	print("the retained counts/geometry + estimator definitions, not the box jk):")
	print(f"w_g+: {boxstyle['std_gp'] / lc['std_gp']}")
	print(f"w_gg: {boxstyle['std_gg'] / lc['std_gg']}")
	print(f"analytic-RR missed hole-boundary bin-shape (max/min - 1): {rr_err:.3%}")


if __name__ == "__main__":
	main()
