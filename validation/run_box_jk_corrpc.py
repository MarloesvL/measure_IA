"""corr_pc's built-in periodic-box jackknife vs the cross-pair schemes (diagnostic 1f).

corr_pc's box mode (do_jk=1, periodic_box=1) implements a *stochastic* cross-pair rule
(read_dat.cpp, jk.cpp):

- after a fixed srand(124693517) (initialization.cpp -> jk.cpp seed()), every galaxy gets
  jk_prob = rand() / (RAND_MAX + 1.), one draw per galaxy in file-read order: the shape
  catalogue first, then the density catalogue (do_corr.cpp);
- a pair with both members in region k is removed from k once (jk_check_PB);
- a cross pair is removed from member i's region iff jk_prob_i >= 0.5, and independently
  from member j's region iff jk_prob_j >= 0.5 (corel_jk). In expectation that is weight
  1/2 per region, but all pairs of one galaxy share its draw.

Because the seed is fixed, the draws are reproducible: a 5-line C helper compiled against
the same libc regenerates them, the patch-pair counter (jk_cross_pairs_lib) applies them as
per-galaxy flags, and corr_pc's per-region retained counts are compared realisation by
realisation. The deterministic-1/2 scheme is the expectation over draws.

Requires CORR_PC_BIN (see README, "Building corr_pc without MPI").
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jk_cross_pairs_lib as lib
import run_box_jk_cross_pairs as diag

RAND_C = r"""
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  long n = atol(argv[1]);
  srand(124693517);
  for (long i = 0; i < n; i++) printf("%.17g\n", rand() / (RAND_MAX + 1.));
  return 0;
}
"""


def corrpc_draws(n, workdir):
	"""corr_pc's jk_prob sequence (libc rand after srand(124693517))."""
	src = os.path.join(workdir, "draws.c")
	exe = os.path.join(workdir, "draws")
	with open(src, "w") as f:
		f.write(RAND_C)
	subprocess.run(["cc", "-O0", "-o", exe, src], check=True)
	out = subprocess.run([exe, str(n)], check=True, capture_output=True, text=True).stdout
	return np.array(out.split(), dtype=float)


def corrpc_ellipticities(mock):
	"""corr_pc periodic-box chirality (as run_box_corrpc_cov.py): e1 = e cos 2phi,
	e2 = -e sin 2phi."""
	e = diag.distortion(mock["q"])
	ax = mock["Axis_Direction"]
	phi = np.arctan2(ax[:, 1], ax[:, 0])
	return e * np.cos(2 * phi), -e * np.sin(2 * phi)


def run_corrpc_jk(mock, lab_pos, lab_shape, n_jk, workdir, binary):
	"""corr_pc periodic box, rp-pi (coordinates=6), built-in jackknife with our labels.
	Returns full and per-region retained raw DD / S+D grids, plus the sort orders."""
	order_d = np.argsort(mock["Position"][:, 2], kind="stable")
	order_s = np.argsort(mock["Position_shape_sample"][:, 2], kind="stable")
	dens = mock["Position"][order_d]
	shape = mock["Position_shape_sample"][order_s]
	e1, e2 = corrpc_ellipticities(mock)
	pref_s = os.path.join(workdir, "shape")
	pref_d = os.path.join(workdir, "density")
	np.savetxt(pref_s + "_pos.dat", shape[:, :2], fmt="%.17g")
	np.savetxt(pref_s + "_z.dat", shape[:, 2], fmt="%.17g")
	np.savetxt(pref_s + "_e.dat", np.column_stack((e1[order_s], e2[order_s])), fmt="%.17g")
	np.savetxt(pref_s + "_jk.dat", lab_shape[order_s], fmt="%i")
	np.savetxt(pref_d + "_pos.dat", dens[:, :2], fmt="%.17g")
	np.savetxt(pref_d + "_z.dat", dens[:, 2], fmt="%.17g")
	np.savetxt(pref_d + "_jk.dat", lab_pos[order_d], fmt="%i")
	out_dir = os.path.join(workdir, "corr_data_out")
	os.makedirs(out_dir, exist_ok=True)
	out_pref = os.path.join(out_dir, "mia_")
	box = diag.BOXSIZE
	lines = [
		("which_corr", 1), ("coordinates", 6), ("estimator", 0),
		("data_sorted", 1), ("use_comoving", 1), ("do_jk", 1),
		("sig_crit", 0),
		("shape_pos", pref_s + "_pos.dat"), ("shape_z", pref_s + "_z.dat"),
		("shape_e", pref_s + "_e.dat"), ("shape_wt", 0),
		("shape_jk", pref_s + "_jk.dat"),
		("density_patch", 0),
		("density_pos", pref_d + "_pos.dat"), ("density_z", pref_d + "_z.dat"),
		("density_wt", 0), ("density_jk", pref_d + "_jk.dat"),
		("density_e", 0),
		("Srandom_pos", 0), ("Srandom_z", 0), ("Srandoms_wt", 0), ("Srand_jk", 0),
		("Drandom_patch", 0), ("Drandom_pos", 0), ("Drandom_z", 0),
		("Drandoms_wt", 0), ("drand_jk", 0),
		("distances", 0), ("patch_file", 0),
		("out_file", out_pref), ("n_threads", 1),
		("n_shape", len(shape)), ("n_density", len(dens)),
		("n_Srand", 0), ("n_Drand", 0), ("rand_subsample", 0),
		("n_jk", n_jk), ("n_patch", 0),
		("binR_min", diag.RP_LIMS[0]), ("binR_max", diag.RP_LIMS[1]),
		("n_bins", diag.NUM_BINS_RP), ("lin_bin", 0),
		("n_p_bin", diag.NUM_BINS_PI), ("pmin", -diag.PI_MAX), ("pmax", diag.PI_MAX),
		("z_min", 0), ("z_max", 2 * box), ("dz", 0.0001),
		("z_sep_min", -2 * box), ("z_sep_max", 2 * box),
		("periodic_box", 1), ("box_size", box),
	]
	inp = os.path.join(workdir, "mia.inp")
	with open(inp, "w") as f:
		for key, val in lines:
			f.write(f"{key}    {val}\n")
	res = subprocess.run([binary, inp], cwd=workdir, stdout=subprocess.PIPE,
						 stderr=subprocess.STDOUT, text=True)
	if res.returncode != 0:
		print(res.stdout[-3000:])
		raise RuntimeError("corr_pc failed")
	return out_pref, order_d, order_s, res.stdout


def parse_sd(path):
	"""(weighted pair count, raw sum w e+) grids from a corr_pc *_SD.dat term file."""
	raw = np.loadtxt(path)
	shape2d = (diag.NUM_BINS_RP, diag.NUM_BINS_PI)
	return raw[:, 17].reshape(shape2d), raw[:, 5].reshape(shape2d)


def compare(n_jk=8, seed=5, binary=None):
	"""Exact per-realisation comparison: corr_pc's retained DD / S+D vs the patch-pair
	counter with corr_pc's own per-galaxy draws as flags. Also returns the deterministic
	1/2 and union reductions for context."""
	mock = diag.clustered_mock(seed)
	L = round(n_jk ** (1 / 3))
	lab_p = lib.subbox_labels_measureia(mock["Position"], diag.BOXSIZE, L)
	lab_s = lib.subbox_labels_measureia(mock["Position_shape_sample"], diag.BOXSIZE, L)
	n_p, n_s = len(lab_p), len(lab_s)
	with tempfile.TemporaryDirectory() as tmp:
		out_pref, order_d, order_s, _ = run_corrpc_jk(mock, lab_p, lab_s, n_jk, tmp, binary)
		pc_full = parse_sd(out_pref + "bins2D_cross_SD.dat")
		pc_ret = [parse_sd(out_pref + f"bins2D_cross_jk{i}_SD.dat") for i in range(n_jk)]
		draws = corrpc_draws(n_s + n_p, tmp)
	# read order: shape catalogue first, then density, each in the written (sorted) order
	flag_s = np.zeros(n_s, np.int64)
	flag_p = np.zeros(n_p, np.int64)
	flag_s[order_s] = draws[:n_s] >= 0.5
	flag_p[order_d] = draws[n_s:] >= 0.5
	M_DD, M_SpD, _, _ = diag.counter(mock, n_jk, flag_pos=flag_p, flag_shape=flag_s)
	out = {"full_DD": maxrel(pc_full[0], lib.full_counts(M_DD)),
		   "full_SpD": maxrel(pc_full[1], lib.full_counts(M_SpD))}
	ours = {s: {"DD": lib.full_counts(M_DD) - lib.removed_counts(M_DD, *a),
				"SpD": lib.full_counts(M_SpD) - lib.removed_counts(M_SpD, *a)}
			for s, a in [("corrpc_gal", ("flagged",)), ("half", ("weight", 0.5)),
						 ("union", ("weight", 1.0))]}
	pc_DD = np.array([r[0] for r in pc_ret])
	pc_SpD = np.array([r[1] for r in pc_ret])
	for s in ours:
		out[f"{s}_DD"] = maxrel(pc_DD, ours[s]["DD"])
		out[f"{s}_SpD"] = maxrel(pc_SpD, ours[s]["SpD"])
	return out


def dense_one(mock_index, n_jk_list=(8, 27), n_centrals=3750, seed0=1000, binary=None,
			  outfile=None):
	"""One realistic-density mock (same seeds as run_box_jk_cross_pairs `terms`): corr_pc's
	own box jackknife (its per-galaxy draws; per realisation it normalises the retained
	DD / S+D by the FULL N_p N_s and the full-box RR), and the same draws in the
	patch-pair counter with three per-realisation normalisations: 'fixed' (corr_pc's, a
	check), 'cur' (measureia's current analytic RR_jk) and 'pair' (pair-consistent, c=1/2)."""
	mock = diag.radial_alignment_box_mock(n_centrals=n_centrals, n_sat=8, boxsize=diag.BOXSIZE,
										  seed=seed0 + mock_index)
	n_p, n_s = len(mock["Position"]), len(mock["Position_shape_sample"])
	e = diag.distortion(mock["q"])
	R_full = np.mean(1 - 0.5 * e ** 2)
	Vshell = lib.shell_volumes(diag.R_BINS, diag.PI_BINS)
	V = diag.BOXSIZE ** 3
	dpi = 2.0 * diag.PI_MAX / diag.NUM_BINS_PI
	shape2d = (diag.NUM_BINS_RP, diag.NUM_BINS_PI)
	store = {}
	for n_jk in n_jk_list:
		L = round(n_jk ** (1 / 3))
		lab_p = lib.subbox_labels_measureia(mock["Position"], diag.BOXSIZE, L)
		lab_s = lib.subbox_labels_measureia(mock["Position_shape_sample"], diag.BOXSIZE, L)
		with tempfile.TemporaryDirectory() as tmp:
			out_pref, order_d, order_s, _ = run_corrpc_jk(mock, lab_p, lab_s, n_jk, tmp, binary)
			fin = [np.loadtxt(out_pref + f"bins2D_cross_jk{i}_final.dat") for i in range(n_jk)]
			fin_full = np.loadtxt(out_pref + "bins2D_cross_final.dat")
			draws = corrpc_draws(n_s + n_p, tmp)
		store[f"{n_jk}/corrpc/w_gp"] = np.array([f[:, 5].reshape(shape2d).sum(1) * dpi for f in fin]) / (2 * R_full)
		store[f"{n_jk}/corrpc/w_gg"] = np.array([f[:, 3].reshape(shape2d).sum(1) * dpi for f in fin])
		store[f"{n_jk}/corrpc_full/w_gp"] = fin_full[:, 5].reshape(shape2d).sum(1) * dpi / (2 * R_full)
		flag_s = np.zeros(n_s, np.int64)
		flag_p = np.zeros(n_p, np.int64)
		flag_s[order_s] = draws[:n_s] >= 0.5
		flag_p[order_d] = draws[n_s:] >= 0.5
		M_DD, M_SpD, _, _ = diag.counter(mock, n_jk, flag_pos=flag_p, flag_shape=flag_s, chunk=200)
		DD = lib.full_counts(M_DD) - lib.removed_counts(M_DD, "flagged")
		SpD = lib.full_counts(M_SpD) - lib.removed_counts(M_SpD, "flagged")
		Wp_k = lib.region_weight_sums(np.ones(n_p), lab_p, n_jk)
		Ws_k = lib.region_weight_sums(np.ones(n_s), lab_s, n_jk)
		ov, ov_k = diag.overlap(mock, np.ones(n_p), np.ones(n_s), lab_s, n_jk)
		R_k = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, 0.5)["retained"][:, None, None]
		RRs = {
			# corr_pc: full N_p N_s (corr_pc treats the samples as independent: no overlap)
			"fixed": np.broadcast_to(n_p * n_s * Vshell / V, DD.shape),
			"cur": lib.rr_current(Vshell, V, n_jk, n_p, n_s, Wp_k, Ws_k, ov, ov_k),
			"pair": lib.rr_pair_consistent(Vshell, V, 0.5, n_p, n_s, Wp_k, Ws_k, ov, ov_k),
		}
		for name, RR in RRs.items():
			Rr = R_full if name == "fixed" else R_k
			store[f"{n_jk}/draws_{name}/w_gp"] = lib.w_from_xi(SpD / (RR * 2 * Rr), diag.PI_BINS)
			store[f"{n_jk}/draws_{name}/w_gg"] = lib.w_from_xi(DD / RR - 1, diag.PI_BINS)
		print(f"mock {mock_index} N_jk {n_jk}: corr_pc vs counter-with-fixed-norm max rel diff "
			  f"w_gp {maxrel(store[f'{n_jk}/corrpc/w_gp'], store[f'{n_jk}/draws_fixed/w_gp']):.1e}, "
			  f"w_gg {maxrel(store[f'{n_jk}/corrpc/w_gg'], store[f'{n_jk}/draws_fixed/w_gg']):.1e}",
			  flush=True)
	if outfile:
		np.savez_compressed(outfile, **store)
	return store


def dense_analyse(corrpc_files, terms_files, n_jk_list=(8, 27)):
	"""sigma_jk(corr_pc-style options) / sigma_jk(randoms reference) on the same dense mocks."""
	C = [np.load(p) for p in corrpc_files]
	parts = [np.load(p) for p in terms_files]
	T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
	rp_mid = np.sqrt(diag.R_BINS[1:] * diag.R_BINS[:-1])
	print(f"\n=== corr_pc draws on {len(C)} dense mocks (n = 1e-2) vs the randoms reference ===")
	for n_jk in n_jk_list:
		jk = diag._estimators(T, n_jk, "jk", "RR")
		var = lambda w: np.mean([np.diag(lib.jk_cov(r)) for r in w], axis=0)
		for s, stat in enumerate(["w_gp", "w_gg"]):
			ref = var(jk["randoms/union"][s])
			print(f"\n[{stat}] N_jk = {n_jk}   rp: " + "".join(f"{x:>7.2f}" for x in rp_mid))
			rows = [("corr_pc itself", "corrpc"), ("draws + fixed norm", "draws_fixed"),
					("draws + current RR", "draws_cur"), ("draws + pair-consistent RR", "draws_pair")]
			for label, key in rows:
				a = var(np.array([c[f"{n_jk}/{key}/{stat}"] for c in C]))
				print(f"  {label:28s}" + "".join(f"{v:7.2f}" for v in np.sqrt(a / ref)))
			a = var(jk["analytic/union/cur"][s])
			print(f"  {'today (union, current RR)':28s}" + "".join(f"{v:7.2f}" for v in np.sqrt(a / ref)))


def maxrel(a, b):
	scale = np.maximum(np.abs(b), 1e-8 * np.max(np.abs(b)))
	return float(np.max(np.abs(a - b) / scale))


def main():
	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if not binary:
		print("set CORR_PC_BIN")
		return
	print("=== corr_pc built-in box jackknife (1f) vs patch-pair reductions ===")
	print("max |rel diff| of corr_pc's retained counts against each reduction; only the\n"
		  "reduction with corr_pc's own per-galaxy draws is expected to match exactly")
	for n_jk in (8, 27):
		res = compare(n_jk=n_jk, binary=binary)
		print(f"N_jk = {n_jk}")
		for k, v in res.items():
			print(f"   {k:16s} {v:.2e}")


if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("command", nargs="?", default="compare", choices=["compare", "dense", "dense_analyse"])
	ap.add_argument("--mock", type=int, default=0)
	ap.add_argument("--out", default=None)
	ap.add_argument("--corrpc-files", nargs="+")
	ap.add_argument("--terms-files", nargs="+")
	args = ap.parse_args()
	if args.command == "compare":
		main()
	elif args.command == "dense":
		dense_one(args.mock, binary=os.environ.get("CORR_PC_BIN"), outfile=args.out)
	else:
		dense_analyse(args.corrpc_files, args.terms_files)
