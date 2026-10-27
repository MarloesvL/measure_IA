"""Box jackknife: how should a pair that straddles two sub-boxes be removed?

measureia's box jackknife (analytic RR, periodic box) removes a pair from delete-one
realisation k whenever EITHER member lies in sub-box k ("union" deletion). With analytic
randoms this (1) biases the realisations, because the retained pair count no longer
follows the analytic RR shape: without clustering E[DD_k] / RR_k = (V - 2V_k + V_ov(r)) /
(V - V_k), and (2) over-removes cross pairs at large r, which is expected to inflate the
jackknife errors. The proposed fix (cf. Mohammad & Percival 2022; treecorr
cross_patch_weight='mean'; corr_pc's stochastic box jackknife) removes a cross pair with
weight 1/2 from each of its two regions.

This script is the diagnostic phase. Sub-commands:

  selfcheck  the independent patch-pair counter (jk_cross_pairs_lib) reproduces
             measureia's full DD / S+D and every union-deletion realisation exactly
  predict    (1c) Poisson box: mean_k DD_k / RR_k against the zero-parameter prediction
             (V - 2c V_k - (1-2c) V_ov) / (V - V_k), for every scheme
"""

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jk_cross_pairs_lib as lib
from measureia import MeasureIABox
from measureia.mocks import radial_alignment_box_mock

# ------------------------------------------------------------------ configuration
BOXSIZE = 150.0
RP_LIMS = [1.0, 60.0]
NUM_BINS_RP = 10
PI_MAX = 30.0
NUM_BINS_PI = 6
DATASET = "jk_cross"
R_BINS = np.logspace(np.log10(RP_LIMS[0]), np.log10(RP_LIMS[1]), NUM_BINS_RP + 1)
PI_BINS = np.linspace(-PI_MAX, PI_MAX, NUM_BINS_PI + 1)


def clustered_mock(seed, n_centrals=400, n_sat=8):
	return radial_alignment_box_mock(n_centrals=n_centrals, n_sat=n_sat, boxsize=BOXSIZE,
									 seed=seed)


def poisson_mock(seed, n_pos=4000, n_shape=4000):
	"""Uniform positions, independent position and shape samples, random orientations."""
	rng = np.random.default_rng(seed)
	phi = rng.uniform(0, np.pi, n_shape)
	return {
		"Position": rng.uniform(0, BOXSIZE, (n_pos, 3)),
		"Position_shape_sample": rng.uniform(0, BOXSIZE, (n_shape, 3)),
		"Axis_Direction": np.column_stack([np.cos(phi), np.sin(phi)]),
		"q": rng.uniform(0.3, 0.9, n_shape),
		"LOS": 2,
		"boxsize": BOXSIZE,
	}


def box_data(mock):
	return {k: mock[k] for k in ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}


def distortion(q):
	return (1 - q ** 2) / (1 + q ** 2)


def overlap(mock, w_pos, w_shape, lab_shape, n_patch):
	"""Weighted self-pair overlap (objects with identical coordinates in both samples),
	total and per shape-side region, as measureia's overlap_weight / overlap_jk_counts."""
	P, S = mock["Position"], mock["Position_shape_sample"]
	vp = np.ascontiguousarray(P).view([("", P.dtype)] * 3).ravel()
	vs = np.ascontiguousarray(S).view([("", S.dtype)] * 3).ravel()
	_, ip, is_ = np.intersect1d(vp, vs, return_indices=True)
	ww = w_pos[ip] * w_shape[is_]
	return ww.sum(), np.bincount(lab_shape[is_], weights=ww, minlength=n_patch)


def counter(mock, n_jk, **kw):
	"""Patch-pair matrices for a mock with measureia's own sub-box labels."""
	L = round(n_jk ** (1 / 3))
	lab_p = lib.subbox_labels_measureia(mock["Position"], BOXSIZE, L)
	lab_s = lib.subbox_labels_measureia(mock["Position_shape_sample"], BOXSIZE, L)
	M_DD, M_SpD = _ppc(
		mock["Position"], mock["Position_shape_sample"], boxsize=BOXSIZE,
		r_bins=R_BINS, pi_bins=PI_BINS, lab_pos=lab_p, lab_shape=lab_s, n_patch=n_jk,
		axis=mock["Axis_Direction"], e=distortion(mock["q"]), **kw)
	return M_DD, M_SpD, lab_p, lab_s


def run_measureia(mock, n_jk, outdir):
	"""measureia's current box jackknife, responsivity off (raw S+D counts)."""
	fname = os.path.join(outdir, f"mia_{n_jk}.hdf5")
	ia = MeasureIABox(box_data(mock), fname, separation_limits=RP_LIMS,
					  num_bins_r=NUM_BINS_RP, num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX,
					  boxsize=BOXSIZE, num_nodes=1)
	ia.measure_xi_w(DATASET, "both", num_jk=n_jk, temp_file_path=outdir, responsivity=False)
	jk = f"{DATASET}_jk{n_jk}"
	with h5py.File(fname, "r") as f:
		return {
			"DD": f[f"w/xi_gg/{DATASET}_DD"][:],
			"SpD": f[f"w/xi_g_plus/{DATASET}_SplusD"][:],
			"DD_ret": np.array([f[f"w/xi_gg/{jk}/{DATASET}_{i}_DD"][:] for i in range(n_jk)]),
			"SpD_ret": np.array([f[f"w/xi_g_plus/{jk}/{DATASET}_{i}_SplusD"][:] for i in range(n_jk)]),
			"RR_ret": np.array([f[f"w/xi_g_plus/{jk}/{DATASET}_{i}_RR"][:] for i in range(n_jk)]),
		}


def maxrel(a, b):
	scale = np.maximum(np.abs(b), 1e-12 * np.max(np.abs(b)))
	return float(np.max(np.abs(a - b) / scale))


# ----------------------------------------------------------------------- selfcheck
def selfcheck(n_jk_list=(8, 27), seed=1):
	print("=== selfcheck: independent counter vs measureia's current (union) jackknife ===")
	mock = clustered_mock(seed)
	Vshell = lib.shell_volumes(R_BINS, PI_BINS)
	V = BOXSIZE ** 3
	ok = True
	with tempfile.TemporaryDirectory() as tmp:
		for n_jk in n_jk_list:
			mia = run_measureia(mock, n_jk, tmp)
			M_DD, M_SpD, lab_p, lab_s = counter(mock, n_jk)
			ret_DD = lib.full_counts(M_DD) - lib.removed_counts(M_DD, "weight", 1.0)
			ret_SpD = lib.full_counts(M_SpD) - lib.removed_counts(M_SpD, "weight", 1.0)
			w_p = np.ones(len(lab_p))
			w_s = np.ones(len(lab_s))
			ov, ov_k = overlap(mock, w_p, w_s, lab_s, n_jk)
			RR = lib.rr_current(Vshell, V, n_jk, w_p.sum(), w_s.sum(),
								lib.region_weight_sums(w_p, lab_p, n_jk),
								lib.region_weight_sums(w_s, lab_s, n_jk), ov, ov_k)
			res = {
				"full DD": maxrel(lib.full_counts(M_DD), mia["DD"]),
				"full S+D": maxrel(lib.full_counts(M_SpD), mia["SpD"]),
				"retained DD (all k)": maxrel(ret_DD, mia["DD_ret"]),
				"retained S+D (all k)": maxrel(ret_SpD, mia["SpD_ret"]),
				"RR_jk (all k)": maxrel(RR, mia["RR_ret"]),
			}
			print(f"N_jk = {n_jk}:  overlap = {ov:.0f}")
			for k, v in res.items():
				flag = "OK" if v < 1e-10 else "MISMATCH"
				ok &= v < 1e-10
				print(f"   {k:22s} max rel diff {v:.2e}  {flag}")
	print("selfcheck", "PASSED" if ok else "FAILED")
	return ok


# ------------------------------------------------------------------------- predict
def predict(n_jk_list=(8, 27), n_mocks=20, seed0=100):
	"""(1c) Poisson null: mean over realisations (and mocks) of DD_k / RR_k(current)."""
	print("=== predict (1c): Poisson box, mean_k DD_k / RR_k(current analytic RR) ===")
	Vshell = lib.shell_volumes(R_BINS, PI_BINS)
	V = BOXSIZE ** 3
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	results = {}
	for n_jk in n_jk_list:
		L = round(n_jk ** (1 / 3))
		ov_frac = lib.subbox_overlap_fraction(R_BINS, PI_BINS, BOXSIZE / L)
		schemes = {"union": 1.0, "match": lib.match_weight(n_jk), "half": 0.5}
		ratios = {s: [] for s in schemes}
		for m in range(n_mocks):
			mock = poisson_mock(seed0 + m)
			M_DD, _, lab_p, lab_s = counter(mock, n_jk)
			w_p, w_s = np.ones(len(lab_p)), np.ones(len(lab_s))
			RR = lib.rr_current(Vshell, V, n_jk, w_p.sum(), w_s.sum(),
								lib.region_weight_sums(w_p, lab_p, n_jk),
								lib.region_weight_sums(w_s, lab_s, n_jk))
			full = lib.full_counts(M_DD)
			for s, c in schemes.items():
				# always against the CURRENT analytic RR: the prediction is
				# E[retained_c] / RR_current = (V - 2c V_k - (1-2c) V_ov) / (V - V_k)
				ret = full - lib.removed_counts(M_DD, "weight", c)
				ratios[s].append(ret / RR)
		print(f"\nN_jk = {n_jk} (L_sub = {BOXSIZE / L:.1f}); {n_mocks} Poisson mocks; "
			  "pi-summed ratio per rp bin")
		print("   rp     " + "".join(f"{s:>17s}" for s in schemes))
		res_n = {}
		for s, c in schemes.items():
			r = np.array(ratios[s])  # (mocks, k, r, pi)
			# pi-sum numerator and denominator per (mock, k) before averaging
			pred = lib.predicted_mean_ratio(c, n_jk, ov_frac)
			res_n[s] = {
				"measured": r.mean(axis=(0, 1)),
				"err": r.std(axis=(0, 1)) / np.sqrt(r.shape[0] * r.shape[1]),
				"predicted": pred,
			}
		for i in range(NUM_BINS_RP):
			line = f"  {rp_mid[i]:6.2f} "
			for s in schemes:
				meas = res_n[s]["measured"][i].mean()
				pred = (res_n[s]["predicted"][i] * Vshell[i]).sum() / Vshell[i].sum()
				line += f"  {meas:6.4f} ({pred:6.4f})"
			print(line)
		print("   (measured, with the zero-parameter prediction in brackets; union should "
			  "fall below 1 at large rp, half should stay at 1)")
		results[n_jk] = res_n
	return results


# ------------------------------------------------------------------------ ensemble
SCHEMES = ["union", "match", "half", "corrpc_gal", "pair_draw"]


def _scheme_removed(M_gal, M_pair, scheme, n_jk):
	"""Removed grids per realisation for each scheme; M_gal carries per-galaxy flags
	(corr_pc-style), M_pair per-pair flags. Deterministic schemes use M_gal summed."""
	if scheme == "union":
		return lib.removed_counts(M_gal, "weight", 1.0)
	if scheme == "match":
		return lib.removed_counts(M_gal, "weight", lib.match_weight(n_jk))
	if scheme == "half":
		return lib.removed_counts(M_gal, "weight", 0.5)
	if scheme == "corrpc_gal":
		return lib.removed_counts(M_gal, "flagged")
	if scheme == "pair_draw":
		return lib.removed_counts(M_pair, "flagged")
	raise ValueError(scheme)


def _scheme_c(scheme, n_jk):
	return {"union": 1.0, "match": lib.match_weight(n_jk)}.get(scheme, 0.5)


def measure_one_mock(mock, n_jk, seed):
	"""Full-sample w and the per-realisation w for every scheme x RR option x R option."""
	rng = np.random.default_rng(seed)
	n_p, n_s = len(mock["Position"]), len(mock["Position_shape_sample"])
	flag_p = (rng.random(n_p) >= 0.5).astype(np.int64)
	flag_s = (rng.random(n_s) >= 0.5).astype(np.int64)
	M_DD, M_SpD, lab_p, lab_s = counter(mock, n_jk, flag_pos=flag_p, flag_shape=flag_s)
	P_DD, P_SpD, _, _ = counter(mock, n_jk, pair_flag_seed=seed + 7)
	Vshell = lib.shell_volumes(R_BINS, PI_BINS)
	V = BOXSIZE ** 3
	w_p, w_s = np.ones(n_p), np.ones(n_s)
	Wp, Ws = w_p.sum(), w_s.sum()
	Wp_k = lib.region_weight_sums(w_p, lab_p, n_jk)
	Ws_k = lib.region_weight_sums(w_s, lab_s, n_jk)
	ov, ov_k = overlap(mock, w_p, w_s, lab_s, n_jk)
	e = distortion(mock["q"])
	R_full = np.sum(w_s * (1 - 0.5 * e ** 2)) / Ws
	RR_full = (Wp * Ws - ov) * Vshell / V
	DD, SpD = lib.full_counts(M_DD), lib.full_counts(M_SpD)
	out = {"w_gp": lib.w_from_xi(SpD / (RR_full * 2 * R_full), PI_BINS),
		   "w_gg": lib.w_from_xi(DD / RR_full - 1, PI_BINS)}
	RR_cur = lib.rr_current(Vshell, V, n_jk, Wp, Ws, Wp_k, Ws_k, ov, ov_k)
	for scheme in SCHEMES:
		c = _scheme_c(scheme, n_jk)
		rem_DD = _scheme_removed(M_DD, P_DD, scheme, n_jk)
		rem_SpD = _scheme_removed(M_SpD, P_SpD, scheme, n_jk)
		RRs = {"cur": RR_cur,
			   "pair": lib.rr_pair_consistent(Vshell, V, c, Wp, Ws, Wp_k, Ws_k, ov, ov_k)}
		Rs = lib.responsivity_options(e, w_s, lab_s, n_jk, c)
		for rr_name, RR in RRs.items():
			out[f"{scheme}/{rr_name}/w_gg"] = lib.w_from_xi((DD - rem_DD) / RR - 1, PI_BINS)
			for r_name, R in Rs.items():
				xi = (SpD - rem_SpD) / (RR * 2 * R[:, None, None])
				out[f"{scheme}/{rr_name}/{r_name}/w_gp"] = lib.w_from_xi(xi, PI_BINS)
	return out


def ensemble(n_jk_list=(8, 27, 64), n_mocks=100, kind="clustered", seed0=1000, outfile=None):
	"""(1d) Truth covariance from independent mocks vs the jackknife of every scheme."""
	make = clustered_mock if kind == "clustered" else poisson_mock
	store = {}
	for n_jk in n_jk_list:
		rows = [measure_one_mock(make(seed0 + m), n_jk, seed0 + 10_000 + m)
				for m in range(n_mocks)]
		for key in rows[0]:
			store[f"{n_jk}/{key}"] = np.array([r[key] for r in rows])
		print(f"  N_jk = {n_jk}: {n_mocks} mocks done", flush=True)
	if outfile:
		np.savez_compressed(outfile, **store)
	summarise(store, n_jk_list, n_mocks, kind)
	return store


def summarise(store, n_jk_list, n_mocks, kind):
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	print(f"\n=== ensemble (1d), {kind} mocks, {n_mocks} mocks ===")
	print("sigma_jk / sigma_true per rp bin, sigma_jk = sqrt(mean over mocks of the jk "
		  "variance); sigma_true = std over mocks of the full-sample w. Statistical "
		  f"uncertainty of each ratio ~ {1 / np.sqrt(2 * (n_mocks - 1)):.2f}.")
	for stat in ["w_gp", "w_gg"]:
		for n_jk in n_jk_list:
			truth = store[f"{n_jk}/{stat}"].std(axis=0, ddof=1)
			print(f"\n[{stat}]  N_jk = {n_jk}")
			print("  rp     " + "".join(f"{x:>8.2f}" for x in rp_mid))
			for scheme in SCHEMES:
				for rr in ["cur", "pair"]:
					key = (f"{n_jk}/{scheme}/{rr}/retained/{stat}" if stat == "w_gp"
						   else f"{n_jk}/{scheme}/{rr}/{stat}")
					real = store[key]  # (mocks, k, r)
					var = np.mean([np.diag(lib.jk_cov(real[m])) for m in range(len(real))], axis=0)
					ratio = np.sqrt(var) / truth
					print(f"  {scheme:10s} RR={rr:4s} " + "".join(f"{x:8.3f}" for x in ratio))
			if stat == "w_gp":
				print("  responsivity options (half scheme, RR=pair):")
				for r_name in ["retained", "full", "pair"]:
					real = store[f"{n_jk}/half/pair/{r_name}/w_gp"]
					var = np.mean([np.diag(lib.jk_cov(real[m])) for m in range(len(real))], axis=0)
					print(f"  R={r_name:9s}      " + "".join(f"{x:8.3f}" for x in np.sqrt(var) / truth))
	print("\nMean bias of the realisations: <mean_k w_k> / <w_full> - 1 (w_g+), and "
		  "<mean_k w_k - w_full> / sigma_true (w_gg)")
	for n_jk in n_jk_list:
		gp_full = store[f"{n_jk}/w_gp"].mean(axis=0)
		gg_full = store[f"{n_jk}/w_gg"]
		gg_truth = gg_full.std(axis=0, ddof=1)
		print(f"  N_jk = {n_jk}")
		for scheme in SCHEMES:
			for rr in ["cur", "pair"]:
				gp = store[f"{n_jk}/{scheme}/{rr}/retained/w_gp"].mean(axis=(0, 1))
				gg = (store[f"{n_jk}/{scheme}/{rr}/w_gg"].mean(axis=1) - gg_full).mean(axis=0)
				print(f"  {scheme:10s} RR={rr:4s} g+ " + "".join(f"{x:7.3f}" for x in gp / gp_full - 1)
					  + "  | gg " + "".join(f"{x:6.2f}" for x in gg / gg_truth))


# ------------------------------------------------------------------ randoms (1a)
def _labels(points, n_jk):
	return lib.subbox_labels_measureia(points, BOXSIZE, round(n_jk ** (1 / 3)))


def _counts_multi(P, S, n_jk_list, axis=None, e=None):
	"""Patch-pair matrices (flag axes dropped) for several N_jk from one pair search."""
	M, Msp = _ppc(
		P, S, boxsize=BOXSIZE, r_bins=R_BINS, pi_bins=PI_BINS,
		lab_pos=[_labels(P, n) for n in n_jk_list], lab_shape=[_labels(S, n) for n in n_jk_list],
		n_patch=list(n_jk_list), axis=axis, e=e, chunk=500)
	strip = lambda X: None if X is None else X[:, :, 0, 0]
	return [strip(x) for x in M], [strip(x) for x in Msp]


def _retained(M, c):
	M6 = M[:, :, None, None]
	return lib.full_counts(M6), lib.full_counts(M6) - lib.removed_counts(M6, "weight", c)


def _norms(c, A, B, n_jk, ov=0.0, ov_k=None):
	"""Full and per-realisation pair normalisation for samples A, B (unit weights)."""
	wa, wb = np.ones(len(A)), np.ones(len(B))
	Wa_k = lib.region_weight_sums(wa, _labels(A, n_jk), n_jk)
	Wb_k = lib.region_weight_sums(wb, _labels(B, n_jk), n_jk)
	full = wa.sum() * wb.sum() - ov
	return full, lib.pair_amplitude(c, wa.sum(), wb.sum(), Wa_k, Wb_k, ov, ov_k)


def randoms_ensemble(n_jk_list=(8, 27), n_mocks=50, factor=10, seed0=1000, outfile=None):
	"""(1a, ensemble form) The randoms-based estimator with union deletion -- the paper's
	reference -- next to the analytic-RR schemes, on the same clustered mocks.

	gg: xi = (DS - D R_S - R_D S + R_D R_S) / R_D R_S;  g+: xi = (S+D - S+R_D) / R_D R_S,
	each count normalised by its own (retained) pair amplitude."""
	n_jk_list = list(n_jk_list)
	m0 = clustered_mock(seed0)
	n_p, n_s = len(m0["Position"]), len(m0["Position_shape_sample"])
	rng = np.random.default_rng(424242)
	R_D = rng.uniform(0, BOXSIZE, (factor * n_p, 3))
	R_S = rng.uniform(0, BOXSIZE, (factor * n_s, 3))
	print(f"randoms: R_D {len(R_D)}, R_S {len(R_S)}; counting R_D R_S once ...", flush=True)
	RR_m, _ = _counts_multi(R_D, R_S, n_jk_list)
	Vshell = lib.shell_volumes(R_BINS, PI_BINS)
	V = BOXSIZE ** 3
	store = {}
	for m in range(n_mocks):
		mock = clustered_mock(seed0 + m)
		P, S = mock["Position"], mock["Position_shape_sample"]
		e = distortion(mock["q"])
		ax = mock["Axis_Direction"]
		DS_m, SpD_m = _counts_multi(P, S, n_jk_list, axis=ax, e=e)
		RdS_m, SpRd_m = _counts_multi(R_D, S, n_jk_list, axis=ax, e=e)
		DRs_m, _ = _counts_multi(P, R_S, n_jk_list)
		for j, n_jk in enumerate(n_jk_list):
			lab_s = _labels(S, n_jk)
			ov, ov_k = overlap(mock, np.ones(n_p), np.ones(n_s), lab_s, n_jk)
			R_opts = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, 1.0)
			R_full = R_opts["full"][0]
			res = {}
			for scheme, c in [("rand_union", 1.0), ("rand_half", 0.5)]:
				N_DS = _norms(c, P, S, n_jk, ov, ov_k)
				N_RdS = _norms(c, R_D, S, n_jk)
				N_DRs = _norms(c, P, R_S, n_jk)
				N_RR = _norms(c, R_D, R_S, n_jk)
				R_k = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, c)["retained"]
				terms = {}
				for name, M in [("DS", DS_m[j]), ("SpD", SpD_m[j]), ("RdS", RdS_m[j]),
								("SpRd", SpRd_m[j]), ("DRs", DRs_m[j]), ("RR", RR_m[j])]:
					terms[name] = _retained(M, c)
				for idx, R in [(0, R_full), (1, R_k[:, None, None])]:
					nrm = lambda t, N: terms[t][idx] / (N[idx] if idx == 0 else N[idx][:, None, None])
					rr = nrm("RR", N_RR)
					xi_gp = (nrm("SpD", N_DS) - nrm("SpRd", N_RdS)) / rr / (2 * R)
					xi_gg = (nrm("DS", N_DS) - nrm("DRs", N_DRs) - nrm("RdS", N_RdS) + rr) / rr
					tag = "full" if idx == 0 else "jk"
					res[f"{scheme}/{tag}/w_gp"] = lib.w_from_xi(xi_gp, PI_BINS)
					res[f"{scheme}/{tag}/w_gg"] = lib.w_from_xi(xi_gg, PI_BINS)
			# analytic-RR estimators from the same D x S counts, for side-by-side
			Wp_k = lib.region_weight_sums(np.ones(n_p), _labels(P, n_jk), n_jk)
			Ws_k = lib.region_weight_sums(np.ones(n_s), lab_s, n_jk)
			RR_full = (n_p * n_s - ov) * Vshell / V
			DS_f, SpD_f = lib.full_counts(DS_m[j][:, :, None, None]), lib.full_counts(SpD_m[j][:, :, None, None])
			res["analytic/full/w_gp"] = lib.w_from_xi(SpD_f / (RR_full * 2 * R_full), PI_BINS)
			res["analytic/full/w_gg"] = lib.w_from_xi(DS_f / RR_full - 1, PI_BINS)
			for scheme, c, RRk in [
				("an_union_cur", 1.0, lib.rr_current(Vshell, V, n_jk, n_p, n_s, Wp_k, Ws_k, ov, ov_k)),
				("an_half_pair", 0.5, lib.rr_pair_consistent(Vshell, V, 0.5, n_p, n_s, Wp_k, Ws_k, ov, ov_k))]:
				R_k = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, c)["retained"]
				_, DSk = _retained(DS_m[j], c)
				_, SpDk = _retained(SpD_m[j], c)
				res[f"{scheme}/jk/w_gp"] = lib.w_from_xi(SpDk / (RRk * 2 * R_k[:, None, None]), PI_BINS)
				res[f"{scheme}/jk/w_gg"] = lib.w_from_xi(DSk / RRk - 1, PI_BINS)
			for k, v in res.items():
				store.setdefault(f"{n_jk}/{k}", []).append(v)
		print(f"  mock {m + 1}/{n_mocks}", flush=True)
	store = {k: np.array(v) for k, v in store.items()}
	if outfile:
		np.savez_compressed(outfile, **store)
	summarise_randoms(store, n_jk_list, n_mocks)
	return store


def summarise_randoms(store, n_jk_list, n_mocks):
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	print(f"\n=== randoms reference (1a), clustered mocks, {n_mocks} mocks ===")
	print("sigma_jk / sigma_true; sigma_true = std over mocks of the SAME estimator's "
		  f"full-sample w (+- {1 / np.sqrt(2 * (n_mocks - 1)):.2f})")
	rows = [("rand_union", "rand_union"), ("rand_half", "rand_union"),
			("an_union_cur", "analytic"), ("an_half_pair", "analytic")]
	for stat in ["w_gp", "w_gg"]:
		for n_jk in n_jk_list:
			print(f"\n[{stat}]  N_jk = {n_jk}")
			print("  rp              " + "".join(f"{x:>8.2f}" for x in rp_mid))
			for scheme, truth_of in rows:
				truth = store[f"{n_jk}/{truth_of}/full/{stat}"].std(axis=0, ddof=1)
				real = store[f"{n_jk}/{scheme}/jk/{stat}"]
				var = np.mean([np.diag(lib.jk_cov(r)) for r in real], axis=0)
				print(f"  {scheme:14s}  " + "".join(f"{x:8.3f}" for x in np.sqrt(var) / truth))
			t_r = store[f"{n_jk}/rand_union/full/{stat}"].std(axis=0, ddof=1)
			t_a = store[f"{n_jk}/analytic/full/{stat}"].std(axis=0, ddof=1)
			print("  truth ratio randoms/analytic " + "".join(f"{x:8.3f}" for x in t_r / t_a))


RULES = ["union", "half", "match", "corrpc_gal", "pair_draw"]


def _rule_c(rule, n_jk):
	"""Cross-pair weight c of a rule (its expectation for the stochastic ones)."""
	return {"union": 1.0, "match": lib.match_weight(n_jk)}.get(rule, 0.5)


def _flagged_counts(A, B, n_jk_list, seed, axis=None, e=None, flag_seed=None):
	"""For each N_jk: (M with per-galaxy flags, M with per-pair flags), for DD and S+D.
	flag_seed draws the per-galaxy flags of both samples (corr_pc-style)."""
	rng = np.random.default_rng(flag_seed)
	fa = (rng.random(len(A)) >= 0.5).astype(np.int64)
	fb = (rng.random(len(B)) >= 0.5).astype(np.int64)
	kw = dict(boxsize=BOXSIZE, r_bins=R_BINS, pi_bins=PI_BINS,
			  lab_pos=[_labels(A, n) for n in n_jk_list],
			  lab_shape=[_labels(B, n) for n in n_jk_list],
			  n_patch=list(n_jk_list), axis=axis, e=e, chunk=500)
	Mg, Mg_sp = lib.patch_pair_counts(A, B, flag_pos=fa, flag_shape=fb, **kw)
	Mp, Mp_sp = lib.patch_pair_counts(A, B, pair_flag_seed=seed, **kw)
	return {"gal": (Mg, Mg_sp), "pair": (Mp, Mp_sp)}


def _retained_rule(F, j, rule, n_jk, sp=False):
	"""(full, retained per k) under a rule, from _flagged_counts output F for N_jk index j."""
	idx = 1 if sp else 0
	if rule == "pair_draw":
		M = F["pair"][idx][j]
		return lib.full_counts(M), lib.full_counts(M) - lib.removed_counts(M, "flagged")
	M = F["gal"][idx][j]
	if rule == "corrpc_gal":
		return lib.full_counts(M), lib.full_counts(M) - lib.removed_counts(M, "flagged")
	return lib.full_counts(M), lib.full_counts(M) - lib.removed_counts(M, "weight", _rule_c(rule, n_jk))


def direct_ensemble(n_jk_list=(8, 27), n_mocks=100, factor=10, seed0=1000, outfile=None, mock_start=0):
	"""Every analytic option next to a randoms-catalogue jackknife with the SAME rule, on
	the same clustered mocks and the same fixed random catalogues (R_D, R_S at `factor`x).

	Randoms estimators: gg (DS - D R_S - R_D S + R_D R_S)/R_D R_S, g+ (S+D - S+R_D)/R_D R_S;
	every term (data-data, data-random, random-random) and every normalisation uses the
	rule. Per-galaxy draws are made for data galaxies per mock and for the random points
	once (the RR count is shared by all mocks)."""
	n_jk_list = list(n_jk_list)
	m0 = clustered_mock(seed0)
	n_p, n_s = len(m0["Position"]), len(m0["Position_shape_sample"])
	rng = np.random.default_rng(424242)
	R_D = rng.uniform(0, BOXSIZE, (factor * n_p, 3))
	R_S = rng.uniform(0, BOXSIZE, (factor * n_s, 3))
	print(f"randoms: R_D {len(R_D)}, R_S {len(R_S)}; counting R_D R_S once ...", flush=True)
	F_RR = _flagged_counts(R_D, R_S, n_jk_list, seed=11, flag_seed=12)
	Vshell = lib.shell_volumes(R_BINS, PI_BINS)
	V = BOXSIZE ** 3
	store = {}
	for m in range(mock_start, mock_start + n_mocks):
		mock = clustered_mock(seed0 + m)
		P, S = mock["Position"], mock["Position_shape_sample"]
		e = distortion(mock["q"])
		ax = mock["Axis_Direction"]
		s = seed0 + 100_000 * (m + 1)
		F_DS = _flagged_counts(P, S, n_jk_list, seed=s + 1, axis=ax, e=e, flag_seed=s + 2)
		# the R_D side of R_D S reuses the fixed per-point random flags of the RR count
		F_RdS = _flagged_counts(R_D, S, n_jk_list, seed=s + 3, axis=ax, e=e, flag_seed=s + 4)
		F_DRs = _flagged_counts(P, R_S, n_jk_list, seed=s + 5, flag_seed=s + 6)
		for j, n_jk in enumerate(n_jk_list):
			lab_s = _labels(S, n_jk)
			ov, ov_k = overlap(mock, np.ones(n_p), np.ones(n_s), lab_s, n_jk)
			Wp_k = lib.region_weight_sums(np.ones(n_p), _labels(P, n_jk), n_jk)
			Ws_k = lib.region_weight_sums(np.ones(n_s), lab_s, n_jk)
			R_full = np.mean(1 - 0.5 * e ** 2)
			res = {}
			for rule in RULES:
				c = _rule_c(rule, n_jk)
				R_k = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, c)["retained"][:, None, None]
				DS = _retained_rule(F_DS, j, rule, n_jk)
				SpD = _retained_rule(F_DS, j, rule, n_jk, sp=True)
				# randoms estimator under this rule
				RdS = _retained_rule(F_RdS, j, rule, n_jk)
				SpRd = _retained_rule(F_RdS, j, rule, n_jk, sp=True)
				DRs = _retained_rule(F_DRs, j, rule, n_jk)
				RR = _retained_rule(F_RR, j, rule, n_jk)
				N_DS = _norms(c, P, S, n_jk, ov, ov_k)
				N_RdS, N_DRs, N_RR = _norms(c, R_D, S, n_jk), _norms(c, P, R_S, n_jk), _norms(c, R_D, R_S, n_jk)
				for idx, R in [(0, R_full), (1, R_k)]:
					nrm = lambda t, N: t[idx] / (N[idx] if idx == 0 else N[idx][:, None, None])
					rr = nrm(RR, N_RR)
					xi_gp = (nrm(SpD, N_DS) - nrm(SpRd, N_RdS)) / rr / (2 * R)
					xi_gg = (nrm(DS, N_DS) - nrm(DRs, N_DRs) - nrm(RdS, N_RdS) + rr) / rr
					tag = "full" if idx == 0 else "jk"
					res[f"rand_{rule}/{tag}/w_gp"] = lib.w_from_xi(xi_gp, PI_BINS)
					res[f"rand_{rule}/{tag}/w_gg"] = lib.w_from_xi(xi_gg, PI_BINS)
				# analytic estimators from the same data counts
				RR_full = (n_p * n_s - ov) * Vshell / V
				res["analytic/full/w_gp"] = lib.w_from_xi(SpD[0] / (RR_full * 2 * R_full), PI_BINS)
				res["analytic/full/w_gg"] = lib.w_from_xi(DS[0] / RR_full - 1, PI_BINS)
				for rr_name, RRk in [
						("cur", lib.rr_current(Vshell, V, n_jk, n_p, n_s, Wp_k, Ws_k, ov, ov_k)),
						("pair", lib.rr_pair_consistent(Vshell, V, c, n_p, n_s, Wp_k, Ws_k, ov, ov_k))]:
					res[f"an_{rule}_{rr_name}/jk/w_gp"] = lib.w_from_xi(SpD[1] / (RRk * 2 * R_k), PI_BINS)
					res[f"an_{rule}_{rr_name}/jk/w_gg"] = lib.w_from_xi(DS[1] / RRk - 1, PI_BINS)
			for k, v in res.items():
				store.setdefault(f"{n_jk}/{k}", []).append(v)
		print(f"  mock {m + 1 - mock_start}/{n_mocks}", flush=True)
	store = {k: np.array(v) for k, v in store.items()}
	if outfile:
		np.savez_compressed(outfile, **store)
	summarise_direct(store, n_jk_list, n_mocks)
	return store


def summarise_direct(store, n_jk_list, n_mocks):
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	var = lambda key: np.mean([np.diag(lib.jk_cov(r)) for r in store[key]], axis=0)
	print(f"\n=== direct comparison, analytic vs randoms jackknife, {n_mocks} clustered mocks ===")
	print("Each entry: sqrt(<jk variance of A>) / sqrt(<jk variance of B>), averaged over mocks.")
	for stat in ["w_gp", "w_gg"]:
		for n_jk in n_jk_list:
			ref_u = var(f"{n_jk}/rand_union/jk/{stat}")
			truth = store[f"{n_jk}/rand_union/full/{stat}"].std(axis=0, ddof=1)
			print(f"\n[{stat}]  N_jk = {n_jk}      rp: " + "".join(f"{x:>7.2f}" for x in rp_mid))
			print("  randoms + rule, vs randoms + union:")
			for rule in RULES:
				x = var(f"{n_jk}/rand_{rule}/jk/{stat}")
				print(f"    rand_{rule:11s}            " + "".join(f"{v:7.2f}" for v in np.sqrt(x / ref_u)))
			for rr in ["cur", "pair"]:
				print(f"  analytic, RR={rr}:  vs randoms+union  |  vs randoms + SAME rule")
				for rule in RULES:
					a = var(f"{n_jk}/an_{rule}_{rr}/jk/{stat}")
					same = var(f"{n_jk}/rand_{rule}/jk/{stat}")
					print(f"    {rule:11s} " + "".join(f"{v:6.2f}" for v in np.sqrt(a / ref_u))
						  + "  | " + "".join(f"{v:6.2f}" for v in np.sqrt(a / same)))
			print("  (for scale) randoms + union vs mock-to-mock truth: "
				  + "".join(f"{v:6.2f}" for v in np.sqrt(ref_u) / truth))


# mock / random-catalogue knobs for the robustness runs (set from the CLI; defaults are the
# configuration used throughout: n_sat = 8, sigma_sat = 2 Mpc/h, randoms seed 424242)
MOCK_KW = {"n_sat": 8, "sigma_sat": 2.0}
BINNING = ["rppi"]  # "rppi" (w) or "rmu" (multipoles); set from the CLI
NUM_BINS_MU = 10


def _legendre(xi, l, sab):
	"""measureia's multipole integration (measure_IA_base._measure_multipoles) along the last
	(mu) axis: l=0, sab=0 for gg; l=2, sab=2 for g+."""
	import math
	from scipy.special import assoc_legendre_p
	mu_c = 0.5 * (PI_BINS[1:] + PI_BINS[:-1])
	Lp = assoc_legendre_p(sab, l, mu_c)[0]
	w = (2 * l + 1) / 2.0 * math.factorial(l - sab) / math.factorial(l + sab)
	return w * np.sum(xi * Lp * np.diff(PI_BINS), axis=-1)


def _ppc(*a, **k):
	return lib.patch_pair_counts(*a, binning=BINNING[0], **k)


def proj_gp(xi):
	return lib.w_from_xi(xi, PI_BINS) if BINNING[0] == "rppi" else _legendre(xi, 2, 2)


def proj_gg(xi):
	return lib.w_from_xi(xi, PI_BINS) if BINNING[0] == "rppi" else _legendre(xi, 0, 0)
RAND_SEED = [424242]


# ------------------------------------------------- term grids (density test + swaps)
DET_RULES = ["union", "half", "match"]


def exact_rr_normalised(c, n_jk, ov_frac, Vshell, V):
	"""Infinite-randoms limit of the normalised retained RR_k (uniform randoms, rule c):
	(Vshell/V) [1 - 2c v - (1-2c) v ov_frac] / [1 - 2c v + (2c-1) v^2], v = V_k/V.
	c = 1 at small r reduces to Vshell/(V - V_k), measureia's current analytic RR."""
	v = 1.0 / n_jk
	num = 1.0 - 2 * c * v - (1 - 2 * c) * v * ov_frac
	den = 1.0 - 2 * c * v + (2 * c - 1) * v ** 2
	return (Vshell / V)[None] * np.broadcast_to(num / den, Vshell.shape)[None]


def _randoms(n_centrals, factor, seed0=1000):
	m0 = radial_alignment_box_mock(n_centrals=n_centrals, boxsize=BOXSIZE, **MOCK_KW, seed=seed0)
	n_p, n_s = len(m0["Position"]), len(m0["Position_shape_sample"])
	rng = np.random.default_rng(RAND_SEED[0])
	return (rng.uniform(0, BOXSIZE, (factor * n_p, 3)), rng.uniform(0, BOXSIZE, (factor * n_s, 3)))


def rr_part(part, n_parts, n_jk_list=(8, 27), n_centrals=400, factor=10, outfile=None):
	"""Count R_D x R_S for every n_parts-th R_S point (slice `part`); the parts sum to RR."""
	R_D, R_S = _randoms(n_centrals, factor)
	lab_d = [_labels(R_D, n) for n in n_jk_list]
	sl = np.arange(part, len(R_S), n_parts)
	M, _ = _ppc(R_D, R_S[sl], boxsize=BOXSIZE, r_bins=R_BINS, pi_bins=PI_BINS,
								 lab_pos=lab_d, lab_shape=[_labels(R_S, n)[sl] for n in n_jk_list],
								 n_patch=list(n_jk_list), chunk=100)
	np.savez_compressed(outfile, **{str(n): x[:, :, 0, 0] for n, x in zip(n_jk_list, M)})
	print(f"RR part {part}/{n_parts} done", flush=True)


def term_grids(n_jk_list=(8, 27), n_mocks=5, mock_start=0, n_centrals=400, factor=10,
			   count_rr=True, seed0=1000, outfile=None, rr_files=None):
	"""Save, per mock / N_jk / deterministic rule, every NORMALISED term grid (full and
	retained per realisation): DS, S+D, R_D S, S+R_D, D R_S, RR (counted and/or the exact
	infinite-randoms limit), the two analytic RR amplitudes, and the responsivities.
	Every estimator (analytic options, randoms reference, term swaps) is then built
	offline by `analyse_terms`."""
	n_jk_list = list(n_jk_list)
	m0 = radial_alignment_box_mock(n_centrals=n_centrals, boxsize=BOXSIZE, **MOCK_KW, seed=seed0)
	n_p, n_s = len(m0["Position"]), len(m0["Position_shape_sample"])
	rng = np.random.default_rng(RAND_SEED[0])
	R_D = rng.uniform(0, BOXSIZE, (factor * n_p, 3))
	R_S = rng.uniform(0, BOXSIZE, (factor * n_s, 3))
	print(f"density {n_p / BOXSIZE ** 3:.2e}; D {n_p}, S {n_s}, R_D {len(R_D)}, R_S {len(R_S)}",
		  flush=True)
	Vshell = lib.shell_volumes(R_BINS, PI_BINS, binning=BINNING[0])
	V = BOXSIZE ** 3
	ov_frac = {n: lib.subbox_overlap_fraction(R_BINS, PI_BINS, BOXSIZE / round(n ** (1 / 3)), binning=BINNING[0])
			   for n in n_jk_list}
	if rr_files:
		parts = [np.load(p) for p in rr_files]
		RR_m = [sum(p[str(n)] for p in parts) for n in n_jk_list]
	else:
		RR_m = _counts_multi(R_D, R_S, n_jk_list)[0] if count_rr else None
	count_rr = RR_m is not None
	store = {}
	for m in range(mock_start, mock_start + n_mocks):
		mock = radial_alignment_box_mock(n_centrals=n_centrals, boxsize=BOXSIZE, **MOCK_KW,
										 seed=seed0 + m)
		P, S = mock["Position"], mock["Position_shape_sample"]
		e = distortion(mock["q"])
		ax = mock["Axis_Direction"]
		DS_m, SpD_m = _counts_multi(P, S, n_jk_list, axis=ax, e=e)
		RdS_m, SpRd_m = _counts_multi(R_D, S, n_jk_list, axis=ax, e=e)
		DRs_m, _ = _counts_multi(P, R_S, n_jk_list)
		for j, n_jk in enumerate(n_jk_list):
			lab_s = _labels(S, n_jk)
			ov, ov_k = overlap(mock, np.ones(n_p), np.ones(n_s), lab_s, n_jk)
			Wp_k = lib.region_weight_sums(np.ones(n_p), _labels(P, n_jk), n_jk)
			Ws_k = lib.region_weight_sums(np.ones(n_s), lab_s, n_jk)
			res = {"R_full": np.array([np.mean(1 - 0.5 * e ** 2)])}
			for rule in DET_RULES:
				c = _rule_c(rule, n_jk)
				res[f"{rule}/R_k"] = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, c)["retained"]
				N = {"DS": _norms(c, P, S, n_jk, ov, ov_k), "RdS": _norms(c, R_D, S, n_jk),
					 "DRs": _norms(c, P, R_S, n_jk), "RR": _norms(c, R_D, R_S, n_jk)}
				terms = [("DS", DS_m[j], "DS"), ("SpD", SpD_m[j], "DS"), ("RdS", RdS_m[j], "RdS"),
						 ("SpRd", SpRd_m[j], "RdS"), ("DRs", DRs_m[j], "DRs")]
				if count_rr:
					terms.append(("RR", RR_m[j], "RR"))
				for name, M, nk in terms:
					full, ret = _retained(M, c)
					res[f"{rule}/{name}/full"] = full / N[nk][0]
					res[f"{rule}/{name}/jk"] = ret / N[nk][1][:, None, None]
				res[f"{rule}/RRexact/full"] = Vshell / V
				res[f"{rule}/RRexact/jk"] = exact_rr_normalised(c, n_jk, ov_frac[n_jk], Vshell, V)[0]
				# analytic RR amplitudes, normalised by the data-pair amplitude of the rule
				res[f"{rule}/RRa_cur/jk"] = (lib.rr_current(Vshell, V, n_jk, n_p, n_s, Wp_k, Ws_k, ov, ov_k)
											 / N["DS"][1][:, None, None])
				res[f"{rule}/RRa_pair/jk"] = (lib.rr_pair_consistent(Vshell, V, c, n_p, n_s, Wp_k, Ws_k, ov, ov_k)
											  / N["DS"][1][:, None, None])
				res[f"{rule}/RRa/full"] = Vshell / V
			for k, v in res.items():
				store.setdefault(f"{n_jk}/{k}", []).append(v)
		print(f"  mock {m + 1 - mock_start}/{n_mocks}", flush=True)
	store = {k: np.array(v) for k, v in store.items()}
	if outfile:
		np.savez_compressed(outfile, **store)
	return store


def _estimators(T, n_jk, tag, rr_ref):
	"""All estimators of one realisation set (tag 'full' or 'jk') from saved terms T.
	rr_ref: 'RR' (counted randoms) or 'RRexact' (infinite-randoms limit) for the randoms
	reference. Returns {name: (w_gp, w_gg)} arrays over mocks."""
	g = lambda key: T[f"{n_jk}/{key}/{tag}"]
	R = (T[f"{n_jk}/R_full"][:, :, None] if tag == "full" else None)
	out = {}

	def emit(name, rule, spd, ds, rr, rdS=None, spRd=None, drs=None):
		Rr = R if tag == "full" else T[f"{n_jk}/{rule}/R_k"][:, :, None, None]
		if rdS is None:
			xi_gp, xi_gg = spd / rr / (2 * Rr), ds / rr - 1
		else:
			xi_gp = (spd - spRd) / rr / (2 * Rr)
			xi_gg = (ds - drs - rdS + rr) / rr
		out[name] = (proj_gp(xi_gp), proj_gg(xi_gg))

	for rule in DET_RULES:
		ds, spd = g(f"{rule}/DS"), g(f"{rule}/SpD")
		rdS, spRd, drs = g(f"{rule}/RdS"), g(f"{rule}/SpRd"), g(f"{rule}/DRs")
		rr_r = g(f"{rule}/{rr_ref}")
		emit(f"randoms/{rule}", rule, spd, ds, rr_r, rdS, spRd, drs)
		if tag == "full":
			emit("analytic", rule, spd, ds, T[f"{n_jk}/{rule}/RRa/full"])
			continue
		for rr in ["cur", "pair"]:
			ra = g(f"{rule}/RRa_{rr}")
			emit(f"analytic/{rule}/{rr}", rule, spd, ds, ra)
			# swap A: analytic RR, but the randoms' DR and S+R terms added
			emit(f"swapA_DR_SR/{rule}/{rr}", rule, spd, ds, ra, rdS, spRd, drs)
		# swap B: randoms RR shape, no DR / S+R terms (natural estimator on randoms RR)
		emit(f"swapB_RRonly/{rule}", rule, spd, ds, rr_r)
	return out


def analyse_terms(paths, n_jk_list=(8, 27), rr_ref="RR", label=""):
	parts = [np.load(p) for p in paths]
	T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
	n_mocks = len(T[f"{n_jk_list[0]}/R_full"])
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	print(f"\n=== {label}: {n_mocks} mocks; reference = randoms catalogue + union "
		  f"(RR from '{rr_ref}') ===")
	print("entry: sigma_jk(option) / sigma_jk(reference), sqrt of mean-over-mocks variances")
	for n_jk in n_jk_list:
		jk = _estimators(T, n_jk, "jk", rr_ref)
		var = lambda w: np.mean([np.diag(lib.jk_cov(r)) for r in w], axis=0)
		for s, stat in enumerate(["w_gp", "w_gg"]):
			ref = var(jk["randoms/union"][s])
			print(f"\n[{stat}] N_jk = {n_jk}   rp: " + "".join(f"{x:>7.2f}" for x in rp_mid))
			order = ([f"analytic/{r}/{rr}" for r in DET_RULES for rr in ["cur", "pair"]]
					 + [f"randoms/{r}" for r in ["half", "match"]]
					 + [f"swapA_DR_SR/union/{rr}" for rr in ["cur", "pair"]]
					 + ["swapB_RRonly/union"])
			for name in order:
				print(f"  {name:26s}" + "".join(f"{v:7.2f}" for v in np.sqrt(var(jk[name][s]) / ref)))
		full = _estimators(T, n_jk, "full", rr_ref)
		for s, stat in enumerate(["w_gp", "w_gg"]):
			t_r = full["randoms/union"][s].std(axis=0, ddof=1)
			t_a = full["analytic"][s].std(axis=0, ddof=1)
			ref = np.sqrt(np.mean([np.diag(lib.jk_cov(r)) for r in jk["randoms/union"][s]], axis=0))
			if n_mocks > 2:
				print(f"  [{stat}] reference sigma_jk / sigma_true " + "".join(f"{v:7.2f}" for v in ref / t_r)
					  + "   | truth randoms/analytic " + "".join(f"{v:6.2f}" for v in t_r / t_a))
	return T


# ------------------------------------------------- option 2: analytic DR / S+R terms
def analytic_terms_run(n_jk_list=(8, 27), n_mocks=20, mock_start=0, n_centrals=400, seed0=1000,
					   outfile=None):
	"""Analytic (infinite-randoms) union D R_S, R_D S, S+R_D per realisation for the same
	mocks (same seeds, same order) as `terms`, so they line up with the saved term files."""
	store = {}
	import time
	for m in range(mock_start, mock_start + n_mocks):
		t0 = time.time()
		mock = radial_alignment_box_mock(n_centrals=n_centrals, boxsize=BOXSIZE, **MOCK_KW,
										 seed=seed0 + m)
		P, S = mock["Position"], mock["Position_shape_sample"]
		for n_jk in n_jk_list:
			if BINNING[0] == "rmu":
				DRs, RdS, SpRd = lib.analytic_dr_terms_rmu(
					P, S, mock["Axis_Direction"], distortion(mock["q"]), _labels(P, n_jk), _labels(S, n_jk),
					n_jk, boxsize=BOXSIZE, r_bins=R_BINS, mu_bins=PI_BINS)
			else:
				DRs, RdS, SpRd = lib.analytic_dr_terms(
					P, S, mock["Axis_Direction"], distortion(mock["q"]), _labels(P, n_jk), _labels(S, n_jk),
					n_jk, boxsize=BOXSIZE, r_bins=R_BINS, pi_bins=PI_BINS)
			for name, v in [("DRs", DRs), ("RdS", RdS), ("SpRd", SpRd)]:
				store.setdefault(f"{n_jk}/an_{name}", []).append(v)
		print(f"  mock {m + 1 - mock_start}/{n_mocks} ({time.time() - t0:.0f} s)", flush=True)
	store = {k: np.array(v) for k, v in store.items()}
	if outfile:
		np.savez_compressed(outfile, **store)
	return store


def compare_option2(terms_files, analytic_files, n_jk_list=(8, 27), label=""):
	"""Option 2 (analytic DR/S+R + exact RR) and option 1 (randoms DR/S+R + exact RR) against
	the reference (randoms catalogue, counted RR, union), plus a term-level comparison."""
	parts = [np.load(p) for p in terms_files]
	T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
	parts = [np.load(p) for p in analytic_files]
	A = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
	n_mocks = len(A[f"{n_jk_list[0]}/an_DRs"])
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	print(f"\n=== option 2 vs option 1 vs reference: {label}, {n_mocks} mocks ===")
	for n_jk in n_jk_list:
		g = lambda k: T[f"{n_jk}/union/{k}/jk"][:n_mocks]
		R = T[f"{n_jk}/union/R_k"][:n_mocks, :, None, None]
		rr_x = np.broadcast_to(g("RRexact")[:, None], g("RR").shape)

		def est(rr, drs, rds, sprd):
			gp = (g("SpD") - sprd) / rr / (2 * R)
			gg = (g("DS") - drs - rds + rr) / rr
			return proj_gp(gp), proj_gg(gg)

		ref = est(g("RR"), g("DRs"), g("RdS"), g("SpRd"))
		opt1 = est(rr_x, g("DRs"), g("RdS"), g("SpRd"))
		opt2 = est(rr_x, A[f"{n_jk}/an_DRs"], A[f"{n_jk}/an_RdS"], A[f"{n_jk}/an_SpRd"])
		today = (proj_gp(g("SpD") / g("RRa_cur") / (2 * R)),
				 proj_gg(g("DS") / g("RRa_cur") - 1))
		var = lambda w: np.mean([np.diag(lib.jk_cov(r)) for r in w], axis=0)
		print(f"\nN_jk = {n_jk}: term level, analytic minus 10x-randoms term, in units of the "
			  "randoms term's own scatter across realisations (median |.| over bins)")
		for name in ["DRs", "RdS", "SpRd"]:
			diff = A[f"{n_jk}/an_{name}"] - g(name)
			scat = g(name).std(axis=1, keepdims=True) + 1e-30
			print(f"   {name:5s} median |diff|/scatter {np.median(np.abs(diff) / scat):.2f}")
		for s, stat in enumerate(["w_gp", "w_gg"]):
			r = var(ref[s])
			print(f"[{stat}] rp:                   " + "".join(f"{x:>7.2f}" for x in rp_mid))
			for nm, w in [("today (analytic, union)", today), ("option 1 (randoms DR)", opt1),
						  ("option 2 (analytic DR)", opt2)]:
				print(f"  {nm:27s}" + "".join(f"{v:7.2f}" for v in np.sqrt(var(w[s]) / r)))
			# realisation-level agreement option 2 vs option 1 (same mocks)
			d12 = np.sqrt(np.mean((opt2[s] - opt1[s]) ** 2, axis=(0, 1)))
			print(f"  rms(opt2 - opt1) / sigma_jk(ref)" + "".join(f"{v:7.2f}" for v in d12 / np.sqrt(r)))
			# against the mock-to-mock truth of the (unchanged, analytic) full-sample w
			Rf = T[f"{n_jk}/R_full"][:n_mocks, :, None]
			ds_f, spd_f = T[f"{n_jk}/union/DS/full"][:n_mocks], T[f"{n_jk}/union/SpD/full"][:n_mocks]
			rra = T[f"{n_jk}/union/RRa/full"][:n_mocks]
			full = (proj_gp(spd_f / rra / (2 * Rf)), proj_gg(ds_f / rra - 1))
			truth = full[s].std(axis=0, ddof=1)
			print(f"  vs mock-to-mock truth (+-{1 / np.sqrt(2 * (n_mocks - 1)):.2f}):")
			for nm, w in [("reference", ref), ("today", today), ("option 1", opt1), ("option 2", opt2)]:
				print(f"    {nm:25s}" + "".join(f"{v:7.2f}" for v in np.sqrt(var(w[s])) / truth))


def compare_refs(files_a, files_b, n_jk_list=(8, 27), label=""):
	"""Robustness of the reference: sigma_jk of the randoms-catalogue reference built with
	random catalogue set B (other seed, or other factor) over that with set A, same mocks."""
	def load(files):
		parts = [np.load(p) for p in files]
		return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
	TA, TB = load(files_a), load(files_b)
	n_mocks = min(len(TA[f"{n_jk_list[0]}/R_full"]), len(TB[f"{n_jk_list[0]}/R_full"]))
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	print(f"\n=== reference robustness: {label}, {n_mocks} mocks ===")
	for n_jk in n_jk_list:
		ests = []
		for T in (TA, TB):
			g = lambda k: T[f"{n_jk}/union/{k}/jk"][:n_mocks]
			R = T[f"{n_jk}/union/R_k"][:n_mocks, :, None, None]
			rr = g("RR")
			ests.append((proj_gp((g("SpD") - g("SpRd")) / rr / (2 * R)),
						 proj_gg((g("DS") - g("DRs") - g("RdS") + rr) / rr)))
		var = lambda w: np.mean([np.diag(lib.jk_cov(r)) for r in w], axis=0)
		for s, stat in enumerate(["w_gp", "w_gg"]):
			print(f"[{stat}] N_jk = {n_jk}  rp: " + "".join(f"{x:>7.2f}" for x in rp_mid))
			print("  sigma_jk(B) / sigma_jk(A)  " + "".join(f"{v:7.2f}" for v in np.sqrt(var(ests[1][s]) / var(ests[0][s]))))


def _measureia_counts(P, S, axis, q, n_jk, outdir, name):
	"""One measureia box jackknife run (responsivity off, so all S+D are raw sums);
	returns the full and retained DD and S+D grids."""
	fname = os.path.join(outdir, f"{name}.hdf5")
	data = {"Position": P, "Position_shape_sample": S, "Axis_Direction": axis, "q": q, "LOS": 2}
	ia = MeasureIABox(data, fname, separation_limits=RP_LIMS, num_bins_r=NUM_BINS_RP,
					  num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX, boxsize=BOXSIZE, num_nodes=1)
	ia.measure_xi_w(name, "both", num_jk=n_jk, temp_file_path=outdir, responsivity=False)
	jk = f"{name}_jk{n_jk}"
	with h5py.File(fname, "r") as f:
		out = {
			"DD": f[f"w/xi_gg/{name}_DD"][:],
			"SpD": f[f"w/xi_g_plus/{name}_SplusD"][:],
			"DD_k": np.array([f[f"w/xi_gg/{jk}/{name}_{i}_DD"][:] for i in range(n_jk)]),
			"SpD_k": np.array([f[f"w/xi_g_plus/{jk}/{name}_{i}_SplusD"][:] for i in range(n_jk)]),
			"w_gp": f[f"w_g_plus/{name}"][:],
			"w_gg": f[f"w_gg/{name}"][:],
			"w_gp_k": np.array([f[f"w_g_plus/{jk}/{name}_{i}"][:] for i in range(n_jk)]),
			"w_gg_k": np.array([f[f"w_gg/{jk}/{name}_{i}"][:] for i in range(n_jk)]),
		}
	os.remove(fname)
	return out


def randoms_measureia(n_jk=8, factor=10, seed=1000):
	"""(1a, literal) The randoms estimator built from four separate measureia runs.

	Runs: (D, S) -> S+D, DS;  (R_D, S) -> S+R_D, R_D S;  (D, R_S) -> D R_S;  (R_D, R_S) -> RR.
	R_S carries dummy shapes (never used). Each retained count is normalised by its own
	retained weight-sum product (union deletion), then
	  gg: (DS - D R_S - R_D S + R_D R_S) / R_D R_S,   g+: (S+D - S+R_D) / R_D R_S / 2R_k."""
	print(f"=== randoms estimator from four measureia runs (1a), N_jk = {n_jk} ===")
	mock = clustered_mock(seed)
	P, S = mock["Position"], mock["Position_shape_sample"]
	ax, q = mock["Axis_Direction"], mock["q"]
	n_p, n_s = len(P), len(S)
	rng = np.random.default_rng(424242)
	R_D = rng.uniform(0, BOXSIZE, (factor * n_p, 3))
	R_S = rng.uniform(0, BOXSIZE, (factor * n_s, 3))
	phi = rng.uniform(0, np.pi, len(R_S))
	ax_R = np.column_stack([np.cos(phi), np.sin(phi)])
	q_R = np.full(len(R_S), 0.5)
	with tempfile.TemporaryDirectory() as tmp:
		DS = _measureia_counts(P, S, ax, q, n_jk, tmp, "DS")
		RdS = _measureia_counts(R_D, S, ax, q, n_jk, tmp, "RdS")
		DRs = _measureia_counts(P, R_S, ax_R, q_R, n_jk, tmp, "DRs")
		print("  counting R_D x R_S with measureia ...", flush=True)
		RR = _measureia_counts(R_D, R_S, ax_R, q_R, n_jk, tmp, "RR")
	lab_s = _labels(S, n_jk)
	ov, ov_k = overlap(mock, np.ones(n_p), np.ones(n_s), lab_s, n_jk)
	N = {"DS": _norms(1.0, P, S, n_jk, ov, ov_k), "RdS": _norms(1.0, R_D, S, n_jk),
		 "DRs": _norms(1.0, P, R_S, n_jk), "RR": _norms(1.0, R_D, R_S, n_jk)}
	e = distortion(q)
	R_full = np.mean(1 - 0.5 * e ** 2)
	R_k = lib.responsivity_options(e, np.ones(n_s), lab_s, n_jk, 1.0)["retained"]

	def build(sel, idx, R):
		get = lambda run, key, n: (run[key] if idx == 0 else run[key + "_k"]) / (
			N[n][0] if idx == 0 else N[n][1][:, None, None])
		rr = get(RR, "DD", "RR")
		xi_gp = (get(DS, "SpD", "DS") - get(RdS, "SpD", "RdS")) / rr / (2 * R)
		xi_gg = (get(DS, "DD", "DS") - get(DRs, "DD", "DRs") - get(RdS, "DD", "RdS") + rr) / rr
		return proj_gp(xi_gp), proj_gg(xi_gg)

	w_gp, w_gg = build(None, 0, R_full)
	w_gp_k, w_gg_k = build(None, 1, R_k[:, None, None])
	# the same estimator from the independent counter
	DS_m, SpD_m = _counts_multi(P, S, [n_jk], axis=ax, e=e)
	RdS_m, SpRd_m = _counts_multi(R_D, S, [n_jk], axis=ax, e=e)
	DRs_m, _ = _counts_multi(P, R_S, [n_jk])
	RR_m, _ = _counts_multi(R_D, R_S, [n_jk])
	cnt = {}
	for nm, M in [("DS", DS_m[0]), ("SpD", SpD_m[0]), ("RdS", RdS_m[0]), ("SpRd", SpRd_m[0]),
				  ("DRs", DRs_m[0]), ("RR", RR_m[0])]:
		cnt[nm] = _retained(M, 1.0)
	print("  measureia-run counts vs independent counter (max rel diff, all realisations):")
	for nm, run, key in [("DS", DS, "DD"), ("SpD", DS, "SpD"), ("RdS", RdS, "DD"),
						 ("SpRd", RdS, "SpD"), ("DRs", DRs, "DD"), ("RR", RR, "DD")]:
		print(f"     {nm:5s} full {maxrel(run[key], cnt[nm][0]):.1e}   "
			  f"jk {maxrel(run[key + '_k'], cnt[nm][1]):.1e}")
	rp_mid = np.sqrt(R_BINS[1:] * R_BINS[:-1])
	sd = lambda real: np.sqrt(np.diag(lib.jk_cov(real)))
	print("\n  rp                        " + "".join(f"{x:>8.2f}" for x in rp_mid))
	print("  w_g+ randoms estimator    " + "".join(f"{x:8.3f}" for x in w_gp))
	print("  w_g+ analytic (measureia) " + "".join(f"{x:8.3f}" for x in DS["w_gp"] / (2 * R_full)))
	print("  sigma_jk randoms+union    " + "".join(f"{x:8.3f}" for x in sd(w_gp_k)))
	print("  sigma_jk analytic (now)   " + "".join(f"{x:8.3f}" for x in sd(DS["w_gp_k"])))
	print("  w_gg randoms estimator    " + "".join(f"{x:8.2f}" for x in w_gg))
	print("  w_gg analytic (measureia) " + "".join(f"{x:8.2f}" for x in DS["w_gg"]))
	print("  sigma_jk randoms+union    " + "".join(f"{x:8.3f}" for x in sd(w_gg_k)))
	print("  sigma_jk analytic (now)   " + "".join(f"{x:8.3f}" for x in sd(DS["w_gg_k"])))
	print("  (single mock: the sigma rows are one jackknife draw each; the ensemble "
		  "command gives the averaged comparison)")


# --------------------------------------------------------------------------- masks
def masks_check(n_jk=8, seed=7):
	"""(1g) measureia's box jackknife with masks == the same run on the pre-filtered
	catalogue, for every per-realisation output. Baseline the new scheme must keep."""
	print(f"=== masks (1g): masked run vs pre-filtered run, N_jk = {n_jk} ===")
	mock = clustered_mock(seed)
	rng = np.random.default_rng(seed)
	n_p, n_s = len(mock["Position"]), len(mock["Position_shape_sample"])
	L = round(n_jk ** (1 / 3))
	box0_p = lib.subbox_labels_measureia(mock["Position"], BOXSIZE, L) == 0
	box0_s = lib.subbox_labels_measureia(mock["Position_shape_sample"], BOXSIZE, L) == 0
	cases = {
		"position-only mask": (rng.random(n_p) > 0.3, np.ones(n_s, bool)),
		"shape-only mask": (np.ones(n_p, bool), rng.random(n_s) > 0.3),
		"both masked, sub-box 0 emptied": (~box0_p & (rng.random(n_p) > 0.1),
										   ~box0_s & (rng.random(n_s) > 0.1)),
	}
	keys = ["DD", "SpD", "RR", "w_gp", "w_gg"]

	def run(data, masks, tmp, name):
		fname = os.path.join(tmp, f"{name}.hdf5")
		ia = MeasureIABox(data, fname, separation_limits=RP_LIMS, num_bins_r=NUM_BINS_RP,
						  num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX, boxsize=BOXSIZE, num_nodes=1)
		ia.measure_xi_w(name, "both", num_jk=n_jk, temp_file_path=tmp, masks=masks)
		jk = f"{name}_jk{n_jk}"
		with h5py.File(fname, "r") as f:
			g = lambda p: np.array([f[p.format(jk=jk, i=i)][:] for i in range(n_jk)])
			out = {"DD": g("w/xi_gg/{jk}/" + name + "_{i}_DD"),
				   "SpD": g("w/xi_g_plus/{jk}/" + name + "_{i}_SplusD"),
				   "RR": g("w/xi_g_plus/{jk}/" + name + "_{i}_RR"),
				   "w_gp": g("w_g_plus/{jk}/" + name + "_{i}"),
				   "w_gg": g("w_gg/{jk}/" + name + "_{i}"),
				   "cov_gp": f[f"w_g_plus/{name}_jackknife_cov_{n_jk}"][:]}
		os.remove(fname)
		return out

	ok = True
	with tempfile.TemporaryDirectory() as tmp:
		for label, (mp, ms) in cases.items():
			masked = run(box_data(mock), {"Position": mp, "Position_shape_sample": ms,
									 "Axis_Direction": ms, "q": ms}, tmp, "m")
			pre = dict(box_data(mock))
			pre["Position"] = mock["Position"][mp]
			pre["Position_shape_sample"] = mock["Position_shape_sample"][ms]
			pre["Axis_Direction"] = mock["Axis_Direction"][ms]
			pre["q"] = mock["q"][ms]
			direct = run(pre, None, tmp, "p")
			diffs = {k: maxrel(masked[k], direct[k]) for k in keys + ["cov_gp"]}
			worst = max(diffs.values())
			ok &= worst < 1e-10
			print(f"  {label:32s} worst max rel diff {worst:.1e}  "
				  + ("OK" if worst < 1e-10 else "MISMATCH " + str(diffs)))
	print("masks check", "PASSED" if ok else "FAILED")
	return ok


# ------------------------------------------------------------------------ treecorr
def treecorr_check(n_jk_list=(8, 27), seed=3, n_r=10, r_lims=(1.0, 60.0)):
	"""(1e) treecorr (Periodic metric, our sub-box labels as patches): per-patch-pair DD in
	3D r bins must equal the patch-pair matrix, and treecorr's own jackknife pair weights
	for cross_patch_weight in {'simple', 'mean', 'match'} must equal our reductions with
	c = 1, 1/2 and MP22 alpha."""
	import treecorr
	print("=== treecorr (1e): per-realisation DD under each cross_patch_weight ===")
	mock = clustered_mock(seed)
	r_bins = np.logspace(np.log10(r_lims[0]), np.log10(r_lims[1]), n_r + 1)
	mu_bins = np.array([-1.0, 1.0])
	ok = True
	for n_jk in n_jk_list:
		L = round(n_jk ** (1 / 3))
		lab_p = lib.subbox_labels_measureia(mock["Position"], BOXSIZE, L)
		lab_s = lib.subbox_labels_measureia(mock["Position_shape_sample"], BOXSIZE, L)
		M, _ = lib.patch_pair_counts(mock["Position"], mock["Position_shape_sample"],
									 boxsize=BOXSIZE, r_bins=r_bins, pi_bins=mu_bins,
									 lab_pos=lab_p, lab_shape=lab_s, n_patch=n_jk,
									 binning="rmu")
		M = M[..., 0]  # (n, n, 1, 1, r)
		P, S = mock["Position"], mock["Position_shape_sample"]
		cat_p = treecorr.Catalog(x=P[:, 0], y=P[:, 1], z=P[:, 2], patch=lab_p)
		cat_s = treecorr.Catalog(x=S[:, 0], y=S[:, 1], z=S[:, 2], patch=lab_s)
		nn = treecorr.NNCorrelation(min_sep=r_lims[0], max_sep=r_lims[1], nbins=n_r,
									bin_slop=0, metric="Periodic", period=BOXSIZE)
		nn.process(cat_p, cat_s)
		T = np.zeros((n_jk, n_jk, n_r))
		for (i, j), res in nn.results.items():
			T[i, j] = res.weight
		mat = maxrel(T, M[:, :, 0, 0])
		print(f"N_jk = {n_jk}: patch-pair matrix vs treecorr max rel diff {mat:.2e}")
		ok &= mat < 1e-12
		full = lib.full_counts(M[..., None])[..., 0]
		for cpw, c in [("simple", 1.0), ("mean", 0.5), ("match", lib.match_weight(n_jk))]:
			ours = full - lib.removed_counts(M[..., None], "weight", c)[..., 0]
			theirs = np.zeros((n_jk, n_r))
			for k, it in enumerate(nn._jackknife_pairs(cpw)):
				for i, j, w in it:
					theirs[k] += w * nn.results[(i, j)].weight
			d = maxrel(ours, theirs)
			ok &= d < 1e-12
			print(f"   cross_patch_weight={cpw:6s} (c = {c:.4f}): retained DD max rel diff {d:.2e}")
	print("treecorr check", "PASSED" if ok else "FAILED")
	return ok


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("command", choices=["selfcheck", "predict", "ensemble", "treecorr", "randoms", "randoms_measureia", "masks", "direct", "terms", "analyse", "rr_part", "analytic_terms", "option2", "compare_refs"])
	ap.add_argument("--n-mocks", type=int, default=20)
	ap.add_argument("--kind", default="clustered", choices=["clustered", "poisson"])
	ap.add_argument("--n-jk", type=int, nargs="+", default=[8, 27, 64])
	ap.add_argument("--out", default=None)
	ap.add_argument("--mock-start", type=int, default=0)
	ap.add_argument("--merge", nargs="+", default=None)
	ap.add_argument("--n-centrals", type=int, default=400)
	ap.add_argument("--no-count-rr", action="store_true")
	ap.add_argument("--rr-ref", default="RR")
	ap.add_argument("--label", default="")
	ap.add_argument("--part", type=int, default=0)
	ap.add_argument("--n-sat", type=int, default=8)
	ap.add_argument("--sigma-sat", type=float, default=2.0)
	ap.add_argument("--rand-seed", type=int, default=424242)
	ap.add_argument("--factor", type=int, default=10)
	ap.add_argument("--binning", default="rppi", choices=["rppi", "rmu"])
	ap.add_argument("--n-parts", type=int, default=1)
	args = ap.parse_args()
	MOCK_KW.update(n_sat=args.n_sat, sigma_sat=args.sigma_sat)
	RAND_SEED[0] = args.rand_seed
	BINNING[0] = args.binning
	if args.binning == "rmu":
		PI_BINS = np.linspace(-1.0, 1.0, NUM_BINS_MU + 1)
	if args.command == "selfcheck":
		selfcheck()
	elif args.command == "predict":
		predict(n_mocks=args.n_mocks)
	elif args.command == "treecorr":
		treecorr_check()
	elif args.command == "terms":
		term_grids(n_jk_list=tuple(args.n_jk), n_mocks=args.n_mocks, mock_start=args.mock_start,
				   n_centrals=args.n_centrals, count_rr=not args.no_count_rr, outfile=args.out,
				   rr_files=args.merge, factor=args.factor)
	elif args.command == "analytic_terms":
		analytic_terms_run(n_jk_list=tuple(args.n_jk), n_mocks=args.n_mocks, mock_start=args.mock_start,
						   n_centrals=args.n_centrals, outfile=args.out)
	elif args.command == "compare_refs":
		half = len(args.merge) // 2
		compare_refs(args.merge[:half], args.merge[half:], n_jk_list=tuple(args.n_jk), label=args.label)
	elif args.command == "option2":
		half = len(args.merge) // 2
		compare_option2(args.merge[:half], args.merge[half:], n_jk_list=tuple(args.n_jk), label=args.label)
	elif args.command == "rr_part":
		rr_part(args.part, args.n_parts, n_jk_list=tuple(args.n_jk), n_centrals=args.n_centrals,
				factor=args.factor, outfile=args.out)
	elif args.command == "analyse":
		analyse_terms(args.merge, n_jk_list=tuple(args.n_jk), rr_ref=args.rr_ref, label=args.label)
	elif args.command == "direct":
		if args.merge:
			parts = [np.load(p) for p in args.merge]
			store = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
			summarise_direct(store, list(args.n_jk), len(store[next(iter(store))]))
		else:
			direct_ensemble(n_jk_list=tuple(args.n_jk), n_mocks=args.n_mocks, outfile=args.out,
							mock_start=args.mock_start)
	elif args.command == "masks":
		masks_check()
	elif args.command == "randoms_measureia":
		randoms_measureia(n_jk=args.n_jk[0])
	elif args.command == "randoms":
		randoms_ensemble(n_jk_list=tuple(args.n_jk), n_mocks=args.n_mocks, outfile=args.out)
	elif args.command == "ensemble":
		ensemble(n_jk_list=tuple(args.n_jk), n_mocks=args.n_mocks, kind=args.kind,
				 outfile=args.out)