"""Lightcone multipole jackknife covariance validation against corr_pc's
built-in delete-one jackknife (do_jk=1, sky r-mu mode, coordinates=1).

Same design as the w-level leg (run_lightcone_corrpc_cov.py, see its
docstring for the deletion-semantics, normalisation-convention and
empty-cell-policy findings): the identical seeded kmeans patches are fed
to corr_pc's per-galaxy jk-region files, corr_pc tallies union (two-sided)
pair deletion per region and runs its own compensated estimator on every
delete-one sample. measureia's associated-Legendre integration is applied
to each per-region corr_pc grid, and — because the Legendre weights are
even in mu — corr_pc's internal signed-mu mirroring cancels, so the
comparison is enforced at the multipole level:

1. Retained pair counts per realisation (mu-summed): external lock of the
   union-deletion count subtraction.
2. Matched normalisation: corr_pc's convention (full-sample-weight
   normalisation + empty-cell policy) rebuilt from measureia's retained
   counts vs corr_pc's own per-region estimator, at the multipole level,
   per realisation and as jackknife std.
3. Each code's own convention (documented band; contains the
   retained-vs-full-sample normalisation difference).

Mock and configuration follow the multipole signal leg
(run_lightcone_multipoles_corrpc.py): r in [2, 20] Mpc, 25x randoms —
the (r, mu) grid needs denser randoms so no empirical-RR cell of any
delete-one sample is empty. Requires the DRs-patched corr_pc binary; set
CORR_PC_BIN to (re)generate the reference outputs.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lightcone_treecorr as lc
import run_lightcone_multipoles_corrpc as mp
from run_box_multipoles_corrpc import legendre_multipole
from run_lightcone_corrpc import corrpc_write_inputs
from run_lightcone_corrpc_cov import (NUM_JK, PATCH_SEED, COUNT_KEYS,
									  sample_sizes, jk_cov, maxrel)

from measureia import MeasureIALightcone

DATASET = "lc_multipoles_corrpc_cov_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"lightcone_multipoles_corrpc_cov.hdf5"
)


def make_measureia(data, randoms, output_file):
	return MeasureIALightcone(
		data={k: v for k, v in data.items()},
		randoms_data={k: v for k, v in randoms.items()},
		output_file_name=output_file,
		separation_limits=mp.R_LIMS,
		num_bins_r=mp.NUM_BINS_R,
		num_bins_pi=mp.NUM_BINS_MU,
		pi_max=mp.R_LIMS[1],
		num_nodes=1,
	)


def run_measureia_jk(data, randoms, patches, output_file, temp_path):
	"""measureia lightcone multipole jackknife run; returns covariance,
	per-realisation multipoles and retained raw count grids."""
	ia = make_measureia(data, randoms, output_file)
	ia.measure_xi_multipoles("galaxies", DATASET, "both", jk_patches=patches,
							 num_jk=NUM_JK, tree=True,
							 cosmology=lc.COSMOLOGY, over_h=False,
							 temp_file_path=temp_path)
	out = {}
	jk = f"{DATASET}_jk{NUM_JK}"
	with h5py.File(output_file, "r") as f:
		out["mu"] = f[f"multipoles/xi_g_plus/{DATASET}_mu_r"][:]
		for grp, key in [("multipoles_g_plus", "gp"), ("multipoles_gg", "gg")]:
			out[f"mult_{key}_full"] = f[f"{grp}/{DATASET}"][:]
			out[f"mult_{key}"] = np.array(
				[f[f"{grp}/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
		for key, grp, suff in [("SpD", "xi_g_plus", "_SplusD"),
							   ("SpR", "xi_g_plus", "_SplusR"),
							   ("DD", "xi_gg", "_DD"), ("RD", "xi_gg", "_RD"),
							   ("SR", "xi_gg", "_SR"), ("RR", "xi_gg", "_RR")]:
			out[key] = np.array(
				[f[f"multipoles/{grp}/{jk}/{DATASET}_{i}{suff}"][:]
				 for i in range(NUM_JK)])
			out[f"{key}_full"] = f[f"multipoles/{grp}/{DATASET}{suff}"][:]
	return out


def run_corrpc_jk(data, randoms, patches, workdir, binary):
	"""corr_pc sky r-mu mode with do_jk=1; returns per-region grids, counts
	and its own per-region multipoles."""
	inp, out_pref = corrpc_write_inputs(data, randoms, workdir,
										jk_patches=patches, n_jk=NUM_JK)
	# switch the shared input file to r-mu coordinates and this leg's bins
	with open(inp) as f:
		text = f.read()
	for old, new in [
		("coordinates    0", "coordinates    1"),
		(f"binR_min    {lc.RP_LIMS[0]}", f"binR_min    {mp.R_LIMS[0]}"),
		(f"binR_max    {lc.RP_LIMS[1]}", f"binR_max    {mp.R_LIMS[1]}"),
		(f"n_bins    {lc.NUM_BINS_RP}", f"n_bins    {mp.NUM_BINS_R}"),
		(f"n_p_bin    {lc.NUM_BINS_PI}", f"n_p_bin    {mp.NUM_BINS_MU}"),
		(f"pmin    {-lc.PI_MAX}", "pmin    -1"),
		(f"pmax    {lc.PI_MAX}", "pmax    1"),
	]:
		assert old in text, old
		text = text.replace(old, new)
	with open(inp, "w") as f:
		f.write(text)
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	shape2d = (mp.NUM_BINS_R, mp.NUM_BINS_MU)
	N = sample_sizes(data, randoms)
	SR_wt = N["S"] * N["R_D"]
	RR_wt = N["R_S"] * N["R_D"]

	def grids(path, cols):
		raw = np.loadtxt(path)
		return [raw[:, c].reshape(shape2d) for c in cols]

	def parse_terms(pref):
		"""Raw retained counts (see run_lightcone_corrpc_cov.parse_terms)."""
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
		   "counts": [], "mult_gp_own": [], "mult_gg_own": []}
	raw = np.loadtxt(out_pref + "bins2D_cross_final.dat")
	out["mu"] = raw[:, 1].reshape(shape2d)[0]
	xi_gg = raw[:, 3].reshape(shape2d)
	xi_gp = raw[:, 5].reshape(shape2d)
	out["mult_gp_full"] = legendre_multipole(xi_gp, out["mu"], 2, 2)
	out["mult_gg_full"] = legendre_multipole(xi_gg, out["mu"], 0, 0)
	for i in range(NUM_JK):
		pref = out_pref + f"bins2D_cross_jk{i}"
		out["counts"].append(parse_terms(pref))
		xi_gg, xi_gp = grids(pref + "_final.dat", [3, 5])
		out["mult_gp_own"].append(legendre_multipole(xi_gp, out["mu"], 2, 2))
		out["mult_gg_own"].append(legendre_multipole(xi_gg, out["mu"], 0, 0))
	out["mult_gp_own"] = np.array(out["mult_gp_own"])
	out["mult_gg_own"] = np.array(out["mult_gg_own"])
	out["counts"] = {k: np.array([c[k] for c in out["counts"]])
					 for k in COUNT_KEYS}
	return out


def pc_convention_multipoles(counts, N, mu):
	"""corr_pc's estimator (full-sample-weight normalisation + empty-cell
	policy, see run_lightcone_corrpc_cov.pc_convention_w) on xi(r, mu)
	grids, then measureia's Legendre integration."""
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
	if xi_gp.ndim == 2:
		return (legendre_multipole(xi_gp, mu, 2, 2),
				legendre_multipole(xi_gg, mu, 0, 0))
	return (np.array([legendre_multipole(x, mu, 2, 2) for x in xi_gp]),
			np.array([legendre_multipole(x, mu, 0, 0) for x in xi_gg]))


def main():
	data, randoms, info = mp.build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))
	N = sample_sizes(data, randoms)

	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	ia = make_measureia(data, randoms, scratch)
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
			f.attrs["r_lims"] = mp.R_LIMS
			for k, v in N.items():
				f.attrs[f"n_{k}"] = v
			# Store the patches this reference was built with: the covariance is only
			# comparable against these exact regions, so the test reads them back rather
			# than regenerating them.
			for key, labels in patches.items():
				f.create_dataset(f"patches/{key}", data=np.asarray(labels, dtype=np.int16),
										 compression="gzip", compression_opts=9)
			f["mu"] = pc["mu"]
			for k in ["mult_gp_full", "mult_gg_full", "mult_gp_own",
					  "mult_gg_own"]:
				f[k] = pc[k]
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
			pc["mu"] = f["mu"][:]
			for k in ["mult_gp_full", "mult_gg_full", "mult_gp_own",
					  "mult_gg_own"]:
				pc[k] = f[k][:]
			for k in COUNT_KEYS:
				pc["counts_full"][k] = f[f"counts_full/{k}"][:]
				pc["counts"][k] = f[f"counts/{k}"][:]

	print("\n--- 1. retained pair counts, mu-summed (max rel diff over "
		  "realisations) ---")
	for k in COUNT_KEYS:
		print(f"{k:4s}: full {maxrel(mia[f'{k}_full'].sum(axis=1), pc['counts_full'][k].sum(axis=1)):.2e}"
			  f"   delete-one {maxrel(mia[k].sum(axis=2), pc['counts'][k].sum(axis=2)):.2e}")

	print("\n--- 2. matched normalisation (corr_pc convention from measureia "
		  "counts vs corr_pc), multipole level ---")
	mgp_conv, mgg_conv = pc_convention_multipoles(
		{k: mia[k] for k in COUNT_KEYS}, N, pc["mu"])
	mgp_conv_full, mgg_conv_full = pc_convention_multipoles(
		{k: mia[f"{k}_full"] for k in COUNT_KEYS}, N, pc["mu"])
	print(f"full xi_g+,2 ratio: {mgp_conv_full / pc['mult_gp_full']}")
	print(f"full xi_gg,0 ratio: {mgg_conv_full / pc['mult_gg_full']}")
	print(f"realisations g+ max rel diff: {maxrel(mgp_conv, pc['mult_gp_own']):.2e}")
	print(f"realisations gg max rel diff: {maxrel(mgg_conv, pc['mult_gg_own']):.2e}")
	std_gp_conv = np.sqrt(np.diag(jk_cov(mgp_conv)))
	std_gg_conv = np.sqrt(np.diag(jk_cov(mgg_conv)))
	std_gp_pc = np.sqrt(np.diag(jk_cov(pc["mult_gp_own"])))
	std_gg_pc = np.sqrt(np.diag(jk_cov(pc["mult_gg_own"])))
	print(f"std g+ ratio      : {std_gp_conv / std_gp_pc}")
	print(f"std gg ratio      : {std_gg_conv / std_gg_pc}")

	print("\n--- 3. each code's own convention ---")
	print(f"std g+ ratio      : {mia['std_gp'] / std_gp_pc}")
	print(f"std gg ratio      : {mia['std_gg'] / std_gg_pc}")


if __name__ == "__main__":
	main()
