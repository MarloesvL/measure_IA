"""Lightcone jackknife covariance validation against corr_pc's built-in
delete-one jackknife (do_jk=1).

The identical seeded kmeans patch assignment (measureia's
assign_jackknife_patches) is written into corr_pc's per-galaxy jk-region
files for all four samples (shapes, density, both randoms). In sky mode
corr_pc's jackknife is deterministic and has the same deletion semantics
as measureia: every pair is tallied into the regions of both members
(once if they share a region), and the region tallies are subtracted from
the full-sample counts — union (two-sided) pair deletion. Each delete-one
sample then goes through corr_pc's own compensated estimator, giving a
fully independent end-to-end jackknife covariance of the same statistic.

Three comparison levels:

1. Retained pair counts per realisation (summed over the signed-pi axis,
   which cancels corr_pc's internal signed-pi mirroring): corr_pc's
   count-subtraction tallies vs measureia's, for all six ingredients
   (S+D, S+R, SD, SR, RD, RR). This locks the deletion semantics
   externally; the agreement is limited only by the known separation-
   definition differences (bin-edge pair migration), not by the jackknife.
2. Delete-one w per realisation and its jackknife covariance with the
   NORMALISATION CONVENTION HELD FIXED: corr_pc normalises every
   delete-one sample by the full-sample weight products, measureia by the
   retained sample sizes. corr_pc's convention is rebuilt from measureia's
   own retained counts and compared to corr_pc's per-region estimator
   output — a tight statement that covariances agree when the (documented)
   normalisation choice is matched.
3. Each code's own end-to-end jackknife std (measureia: retained-sample
   normalisation; corr_pc: full-sample normalisation) — the honest
   two-independent-implementations comparison; the residual band contains
   the normalisation-convention difference.

The mock, configuration, patches (9 kmeans patches, seed 123) match the
treecorr covariance leg (run_lightcone_treecorr_cov.py), so all three
codes' covariances are mutually comparable. Requires the DRs-patched
corr_pc binary (validation/corrpc_patches/, see README). Set CORR_PC_BIN
to (re)generate the reference outputs.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lightcone_corrpc as lcpc
import run_lightcone_treecorr as lc
import run_lightcone_treecorr_cov as tcov

NUM_JK = tcov.NUM_JK  # 9
PATCH_SEED = tcov.PATCH_SEED  # 123
DATASET = "lc_corrpc_cov_mock"
COUNT_KEYS = ["SpD", "SpR", "DD", "RD", "SR", "RR"]
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"lightcone_corrpc_cov.hdf5"
)


def sample_sizes(data, randoms):
	return {"S": len(data["RA_shape_sample"]), "D": len(data["RA"]),
			"R_S": len(randoms["RA_shape_sample"]), "R_D": len(randoms["RA"])}


def run_measureia_jk(data, randoms, patches, output_file, temp_path):
	"""measureia jackknife run; returns covariance, per-realisation w and
	the per-realisation retained raw count grids."""
	ia = tcov.make_measureia(data, randoms, output_file)
	ia.measure_xi_w("galaxies", DATASET, "both", jk_patches=patches,
					num_jk=NUM_JK, measure_cov=True, tree=True,
					cosmology=lc.COSMOLOGY, over_h=False,
					temp_file_path=temp_path)
	out = {}
	jk = f"{DATASET}_jk{NUM_JK}"
	with h5py.File(output_file, "r") as f:
		for grp, key in [("w_g_plus", "gp"), ("w_gg", "gg")]:
			out[f"w_{key}_full"] = f[f"{grp}/{DATASET}"][:]
			out[f"w_{key}"] = np.array(
				[f[f"{grp}/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
		for key, grp, suff in [("SpD", "xi_g_plus", "_SplusD"),
							   ("SpR", "xi_g_plus", "_SplusR"),
							   ("DD", "xi_gg", "_DD"), ("RD", "xi_gg", "_RD"),
							   ("SR", "xi_gg", "_SR"), ("RR", "xi_gg", "_RR")]:
			out[key] = np.array(
				[f[f"w/{grp}/{jk}/{DATASET}_{i}{suff}"][:] for i in range(NUM_JK)])
			out[f"{key}_full"] = f[f"w/{grp}/{DATASET}{suff}"][:]
	return out


def run_corrpc_jk(data, randoms, patches, workdir, binary):
	"""corr_pc with do_jk=1; returns its per-region estimator w and the
	per-region retained raw count grids parsed from the term files."""
	inp, out_pref = lcpc.corrpc_write_inputs(data, randoms, workdir,
											 jk_patches=patches, n_jk=NUM_JK)
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	shape2d = (lc.NUM_BINS_RP, lc.NUM_BINS_PI)
	dpi = 2.0 * lc.PI_MAX / lc.NUM_BINS_PI
	N = sample_sizes(data, randoms)
	# full-sample weight products entering corr_pc's term normalisation
	SR_wt = N["S"] * N["R_D"]
	RR_wt = N["R_S"] * N["R_D"]

	def grids(path, cols):
		raw = np.loadtxt(path)
		return [raw[:, c].reshape(shape2d) for c in cols]

	def parse_terms(pref):
		"""Raw retained counts from one (full or per-region) set of term
		files. Columns: 2 = npairs, 5 = data[1].val, 17 = weighted npairs.
		After corr_pc's final_calc mutation the SD file's data[1] holds the
		raw Sum(w e+) while the SR file's holds it normalised by the
		retained RR and the full-sample weight ratio — inverted here."""
		out = {}
		sd_cnt, sd_ep = grids(pref + "_SD.dat", [17, 5])
		out["DD"], out["SpD"] = sd_cnt, sd_ep
		(out["RR"],) = grids(pref + "_RR.dat", [17])
		sr_cnt, sr_ep_norm = grids(pref + "_SR.dat", [17, 5])
		out["SR"] = sr_cnt
		with np.errstate(invalid="ignore"):
			out["SpR"] = np.where(out["RR"] > 0,
								  np.nan_to_num(sr_ep_norm) * out["RR"]
								  * SR_wt / RR_wt, 0.0)
		(out["RD"],) = grids(pref + "_DR.dat", [17])
		return out

	out = {"counts_full": parse_terms(out_pref + "bins2D_cross"),
		   "counts": [], "w_gp_own": [], "w_gg_own": []}
	xi_gg, xi_gp = grids(out_pref + "bins2D_cross_final.dat", [3, 5])
	out["w_gp_full"] = np.sum(xi_gp, axis=1) * dpi
	out["w_gg_full"] = np.sum(xi_gg, axis=1) * dpi
	for i in range(NUM_JK):
		pref = out_pref + f"bins2D_cross_jk{i}"
		out["counts"].append(parse_terms(pref))
		xi_gg, xi_gp = grids(pref + "_final.dat", [3, 5])
		out["w_gp_own"].append(np.sum(xi_gp, axis=1) * dpi)
		out["w_gg_own"].append(np.sum(xi_gg, axis=1) * dpi)
	out["w_gp_own"] = np.array(out["w_gp_own"])
	out["w_gg_own"] = np.array(out["w_gg_own"])
	out["counts"] = {k: np.array([c[k] for c in out["counts"]])
					 for k in COUNT_KEYS}
	return out


def pc_convention_w(counts, N):
	"""corr_pc's estimator with its full-sample-weight normalisation AND its
	empty-cell policy, built from raw count grids (per realisation or full
	sample). corr_pc zeroes a xi_g+ (rp, pi) cell whenever the SD, SR or RR
	term has zero raw pairs there (final_calc_bins: sd_num==0 || sr_num==0
	|| rr_num==0), whereas measureia only needs RR > 0 — in sparse cells of
	delete-one realisations this is a real estimator-definition difference,
	so it is reproduced here to keep the comparison exact."""
	dpi = 2.0 * lc.PI_MAX / lc.NUM_BINS_PI
	SDn = counts["DD"] / (N["S"] * N["D"])
	SpDn = counts["SpD"] / (N["S"] * N["D"])
	SpRn = counts["SpR"] / (N["S"] * N["R_D"])
	SRn = counts["SR"] / (N["S"] * N["R_D"])
	RDn = counts["RD"] / (N["D"] * N["R_S"])
	RRn = counts["RR"] / (N["R_D"] * N["R_S"])
	live = (counts["DD"] > 0) & (counts["SR"] > 0) & (counts["RR"] > 0)
	with np.errstate(invalid="ignore", divide="ignore"):
		xi_gp = np.where(live, (SpDn - SpRn) / RRn, 0.0)
		xi_gg = np.where(RRn > 0, (SDn - SRn - RDn) / RRn + 1, 0.0)
	return (np.sum(xi_gp, axis=-1) * dpi, np.sum(xi_gg, axis=-1) * dpi)


def jk_cov(samples):
	d = samples - samples.mean(axis=0)
	return (NUM_JK - 1.0) / NUM_JK * (d.T @ d)


def maxrel(a, b):
	scale = np.maximum(np.abs(b), 1e-3 * np.max(np.abs(b)))
	return np.max(np.abs(a - b) / scale)


def main():
	data, randoms, info, dist = lc.build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))
	N = sample_sizes(data, randoms)

	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	ia = tcov.make_measureia(data, randoms, scratch)
	patches = ia.assign_jackknife_patches(data, randoms, NUM_JK,
										  seed=PATCH_SEED)
	if "randoms_position" not in patches:
		patches["randoms_position"] = patches["randoms"]
		patches["randoms_shape"] = patches["randoms"]
	if os.path.exists(scratch):
		os.remove(scratch)

	mia = run_measureia_jk(data, randoms, patches, scratch, here + "/")
	os.remove(scratch)

	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if binary and os.path.exists(binary):
		import tempfile
		with tempfile.TemporaryDirectory() as workdir:
			pc = run_corrpc_jk(data, randoms, patches, workdir, binary)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["source"] = "corr_pc github.com/sukhdeep2/corr_pc"
			f.attrs["num_jk"] = NUM_JK
			f.attrs["patch_seed"] = PATCH_SEED
			f.attrs["mock_seed"] = info["seed"]
			f.attrs["pi_max"] = lc.PI_MAX
			for k, v in N.items():
				f.attrs[f"n_{k}"] = v
			f["w_gp_full"] = pc["w_gp_full"]
			f["w_gg_full"] = pc["w_gg_full"]
			f["w_gp_own"] = pc["w_gp_own"]
			f["w_gg_own"] = pc["w_gg_own"]
			for k in COUNT_KEYS:
				f[f"counts_full/{k}"] = pc["counts_full"][k]
				f[f"counts/{k}"] = pc["counts"][k]
		print(f"corr_pc results written to {REFERENCE_FILE}")
	else:
		print("corr_pc binary not found (set CORR_PC_BIN); comparing against "
			  "committed reference outputs")
		if not os.path.exists(REFERENCE_FILE):
			print(f"No reference file at {REFERENCE_FILE} — build corr_pc and rerun.")
			return
		pc = {"counts": {}, "counts_full": {}}
		with h5py.File(REFERENCE_FILE, "r") as f:
			for k in ["w_gp_full", "w_gg_full", "w_gp_own", "w_gg_own"]:
				pc[k] = f[k][:]
			for k in COUNT_KEYS:
				pc["counts_full"][k] = f[f"counts_full/{k}"][:]
				pc["counts"][k] = f[f"counts/{k}"][:]

	print("\n--- 1. retained pair counts, pi-summed (max rel diff over "
		  "realisations) ---")
	for k in COUNT_KEYS:
		print(f"{k:4s}: full {maxrel(mia[f'{k}_full'].sum(axis=1), pc['counts_full'][k].sum(axis=1)):.2e}"
			  f"   delete-one {maxrel(mia[k].sum(axis=2), pc['counts'][k].sum(axis=2)):.2e}")

	print("\n--- 2. matched normalisation (corr_pc convention from measureia "
		  "counts vs corr_pc) ---")
	wgp_conv, wgg_conv = pc_convention_w(mia, N)
	wgp_conv_full, wgg_conv_full = pc_convention_w(
		{k: mia[f"{k}_full"] for k in COUNT_KEYS}, N)
	print(f"full w_g+ ratio : {wgp_conv_full / pc['w_gp_full']}")
	print(f"full w_gg ratio : {wgg_conv_full / pc['w_gg_full']}")
	print(f"realisations w_g+ max rel diff: {maxrel(wgp_conv, pc['w_gp_own']):.2e}")
	print(f"realisations w_gg max rel diff: {maxrel(wgg_conv, pc['w_gg_own']):.2e}")
	std_gp_conv = np.sqrt(np.diag(jk_cov(wgp_conv)))
	std_gg_conv = np.sqrt(np.diag(jk_cov(wgg_conv)))
	std_gp_pc = np.sqrt(np.diag(jk_cov(pc["w_gp_own"])))
	std_gg_pc = np.sqrt(np.diag(jk_cov(pc["w_gg_own"])))
	print(f"std w_g+ ratio  : {std_gp_conv / std_gp_pc}")
	print(f"std w_gg ratio  : {std_gg_conv / std_gg_pc}")

	print("\n--- 3. each code's own convention (measureia retained-sample vs "
		  "corr_pc full-sample normalisation) ---")
	print(f"std w_g+ ratio  : {mia['std_gp'] / std_gp_pc}")
	print(f"std w_gg ratio  : {mia['std_gg'] / std_gg_pc}")


if __name__ == "__main__":
	main()
