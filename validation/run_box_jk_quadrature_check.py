"""How much does the r-mu quadrature error of the analytic box-jackknife terms change the
jackknife error bars?

One sparse clustered mock (the validation mock of box_jk_analytic_randoms.md), one N_jk per
run. measureia (unmodified) supplies the retained DD_k and S+D_k counts; the analytic DR, RD
and S+R terms come from the loop prototype (tests/box_jk_reference.py) with n_gl = 16 (the
prototype's setting) and with 64 and 256 nodes per segment as a more precise reference. The
realisations are combined with the exact union RR into xi_gg,0 and xi_g+,2, and the jackknife
sigma of each variant is compared bin by bin.

Usage:
	python validation/run_box_jk_quadrature_check.py N_JK OUTDIR
"""

import os
import sys
import time

import h5py
import numpy as np
from scipy.special import assoc_legendre_p
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
import box_jk_reference as ref  # noqa: E402
from measureia import MeasureIABox  # noqa: E402
from measureia.mocks import radial_alignment_box_mock  # noqa: E402

BOXSIZE = 150.0
R_LIMS = [1.0, 60.0]
NUM_BINS_R = 10
NUM_BINS_MU = 10
N_GL = (16, 64, 256)
NAME = "q"


def labels(points, L):
	i = np.floor(points / (BOXSIZE / L)).astype(int)
	return i[:, 0] * L ** 2 + i[:, 1] * L + i[:, 2]


def legendre(xi, mu_bins, l, sab):
	"""measureia's multipole sum over mu bins (last axis)."""
	mu_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])
	Lp = assoc_legendre_p(sab, l, mu_c)[0]
	w = (2 * l + 1) / 2.0 * math.factorial(l - sab) / math.factorial(l + sab)
	return w * np.sum(xi * Lp * np.diff(mu_bins), axis=-1)


def jk_sigma(x):
	n = len(x)
	return np.sqrt((n - 1) / n * np.sum((x - x.mean(axis=0)) ** 2, axis=0))


def dr_terms(P, S, theta, e, lab_p, lab_s, n_jk, r_bins, mu_bins, n_gl):
	V = BOXSIZE ** 3
	Vk = V / n_jk
	Vb = ref.shell_volumes(r_bins, mu_bins, "rmu")
	kw = dict(boxsize=BOXSIZE, r_bins=r_bins, mu_bins=mu_bins, n_gl=n_gl, chunk=64)
	sv_p, _ = ref.partial_shell_rmu(P, lab_p, n_jk, **kw)
	sv_s, sep_s = ref.partial_shell_rmu(S, lab_s, n_jk, reflect=True, theta=theta, e=e, **kw)
	Wp = len(P) - np.bincount(lab_p, minlength=n_jk)
	Ws = len(S) - np.bincount(lab_s, minlength=n_jk)
	DR = (Wp[:, None, None] * Vb[None] - sv_p) / (Wp[:, None, None] * (V - Vk))
	RD = (Ws[:, None, None] * Vb[None] - sv_s) / (Ws[:, None, None] * (V - Vk))
	SpR = -sep_s / (Ws[:, None, None] * (V - Vk))
	return DR, RD, SpR


def main(n_jk, outdir):
	os.makedirs(outdir, exist_ok=True)
	L = round(n_jk ** (1 / 3))
	mock = radial_alignment_box_mock(n_centrals=400, n_sat=8, sigma_sat=2.0, boxsize=BOXSIZE, seed=1000)
	P, S = mock["Position"], mock["Position_shape_sample"]
	data = {k: mock[k] for k in ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	fname = os.path.join(outdir, f"mia_{n_jk}.hdf5")
	ia = MeasureIABox(data, fname, separation_limits=R_LIMS, num_bins_r=NUM_BINS_R,
					  num_bins_pi=NUM_BINS_MU, boxsize=BOXSIZE, num_nodes=1)
	t0 = time.time()
	ia.measure_xi_multipoles(NAME, "both", num_jk=n_jk, temp_file_path=outdir + "/", responsivity=False)
	print(f"[N_jk={n_jk}] measureia {time.time() - t0:.0f} s; D {len(P)}, S {len(S)}", flush=True)
	r_bins, mu_bins = ia.r_bins, ia.mu_r_bins
	g = lambda stat: f"multipoles/{stat}/{NAME}_jk{n_jk}"
	with h5py.File(fname, "r") as f:
		get = lambda stat, suf: np.array([f[g(stat)][f"{NAME}_{i}{suf}"][:] for i in range(n_jk)])
		DD, SpD, RR_old = get("xi_gg", "_DD"), get("xi_g_plus", "_SplusD"), get("xi_gg", "_RR")
		xi_gg_old, xi_gp_old = get("xi_gg", ""), get("xi_g_plus", "")
	# measureia's current realisation estimators, to pin the count units used below
	print(f"[N_jk={n_jk}] check today's estimators: gg {np.abs(xi_gg_old - (DD / RR_old - 1)).max():.1e}, "
		  f"g+ {np.abs(xi_gp_old - SpD / RR_old).max():.1e}", flush=True)
	V = BOXSIZE ** 3
	Vb = ref.shell_volumes(r_bins, mu_bins, "rmu")
	N = (RR_old / Vb[None] * V * (n_jk - 1) / n_jk)[:, :1, :1]   # retained pairs per realisation
	RRx = ref.exact_union_rr_reference(L, boxsize=BOXSIZE, r_bins=r_bins, bins2=mu_bins, binning="rmu")[None]
	theta = np.arctan2(mock["Axis_Direction"][:, 1], mock["Axis_Direction"][:, 0])
	e = (1 - mock["q"] ** 2) / (1 + mock["q"] ** 2)
	lab_p, lab_s = labels(P, L), labels(S, L)
	out = {"r_bins": r_bins, "mu_bins": mu_bins}
	for n_gl in N_GL:
		t0 = time.time()
		DR, RD, SpR = dr_terms(P, S, theta, e, lab_p, lab_s, n_jk, r_bins, mu_bins, n_gl)
		xi_gg = (DD / N - DR - RD + RRx) / RRx
		xi_gp = (SpD / N - SpR) / RRx
		out[f"{n_gl}/DR"], out[f"{n_gl}/RD"], out[f"{n_gl}/SpR"] = DR, RD, SpR
		out[f"{n_gl}/xi_gg0"] = legendre(xi_gg, mu_bins, 0, 0)
		out[f"{n_gl}/xi_gp2"] = legendre(xi_gp, mu_bins, 2, 2)
		print(f"[N_jk={n_jk}] n_gl={n_gl}: {time.time() - t0:.0f} s", flush=True)
		np.savez_compressed(os.path.join(outdir, f"quadrature_{n_jk}.npz"), **out)
	ref_n = N_GL[-1]
	r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
	lines = [f"N_jk = {n_jk}   r: " + "".join(f"{x:8.2f}" for x in r_mid)]
	for stat in ["xi_gg0", "xi_gp2"]:
		s_ref = jk_sigma(out[f"{ref_n}/{stat}"])
		for n_gl in N_GL[:-1]:
			x = out[f"{n_gl}/{stat}"]
			ratio = jk_sigma(x) / s_ref
			shift = np.abs(x - out[f"{ref_n}/{stat}"]).max(axis=0) / s_ref
			lines.append(f"  {stat} sigma(n={n_gl})/sigma(n={ref_n}) - 1 [%]: "
						 + "".join(f"{100 * (v - 1):8.3f}" for v in ratio))
			lines.append(f"  {stat} max |xi_k(n={n_gl}) - xi_k(n={ref_n})| / sigma:  "
						 + "".join(f"{v:8.1e}" for v in shift))
	print("\n".join(lines), flush=True)


if __name__ == "__main__":
	main(int(sys.argv[1]), sys.argv[2])
