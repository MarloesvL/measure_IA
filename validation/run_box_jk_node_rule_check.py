"""Which r-mu quadrature rule for the analytic box-jackknife edge correction?

The r-mu overlaps integrate the projected radius rho numerically, on [rp_cut, r_i] and
[r_i, r_{i+1}] for bin i. This compares four rules against a 64-node reference:

    plain8, plain16   Gauss-Legendre with 8 / 16 nodes per segment (16 = the module before
                      this check)
    cos8, cos16       the same after the substitution rho = a + (b - a)(1 - cos(pi u)) / 2,
                      which crowds the nodes towards the segment ends, where the shell
                      heights sqrt(r^2 - rho^2) have a vertical tangent

Two checks:

  geometry   per-galaxy overlaps (subbox_overlaps) of random points with every sub-box,
             N_jk = 27, the test binning (5 r bins to 60, 6 mu bins): error of O and of the
             cos/sin 2phi moments, in units of V_b, where the overlap is partial. Also the
             full-bin volume (the overlaps summed over all sub-boxes, which should be V_b).
  dense      the dense benchmark mock (n = 1e-2, L = 150, 10 r bins to 60, 10 mu bins): one
             measureia run per N_jk (8, 27, 64), then for each rule the production
             _apply_box_jk_edge_correction is re-applied and the multipoles xi_gg,0 and
             xi_g+,2 of every realisation are formed. Reported: sigma_jk(rule) / sigma_jk(ref)
             - 1 and max_k |xi_k(rule) - xi_k(ref)| / sigma_jk(ref), per r bin.

Usage:
	python validation/run_box_jk_node_rule_check.py geometry OUTDIR
	python validation/run_box_jk_node_rule_check.py dense N_JK OUTDIR [NUM_NODES]
"""

import math
import os
import sys
import tempfile
import time

import h5py
import numpy as np
from scipy.special import assoc_legendre_p

from measureia import MeasureIABox
from measureia import box_jk_analytic as bja
from measureia.mocks import radial_alignment_box_mock

HERE = os.path.dirname(os.path.abspath(__file__))
BOXSIZE = 150.0
NAME = "d"
RULES = {"plain8": ("plain", 8), "plain16": ("plain", 16), "cos8": ("cos", 8), "cos16": ("cos", 16),
		 "ref64": ("plain", 64)}
PLAIN_SEGMENTS = bja._gl_segments


def cos_segments(segments):
	"""Cosine-substituted Gauss-Legendre nodes and weights (incl. the rho Jacobian)."""
	out = [bja._gl_cos(a, b, bja._N_GL_RMU) for a, b in segments]
	rho = np.concatenate([r for r, _ in out])
	return rho, np.concatenate([w for _, w in out]) * rho


def use_rule(rule):
	"""Patch the module to one rule at every N_jk (the module itself picks 16 nodes at N_jk = 8
	and 8 above, a choice made with this script's results)."""
	kind, n = RULES[rule]
	bja._N_GL_RMU = bja._N_GL_RMU_FEW_JK = n
	bja._gl_segments = PLAIN_SEGMENTS if kind == "plain" else cos_segments


def legendre(xi, mu_bins, l, sab):
	"""measureia's multipole sum over mu bins (last axis)."""
	mu_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])
	Lp = assoc_legendre_p(sab, l, mu_c)[0]
	w = (2 * l + 1) / 2.0 * math.factorial(l - sab) / math.factorial(l + sab)
	return w * np.sum(xi * Lp * np.diff(mu_bins), axis=-1)


def jk_sigma(x):
	n = len(x)
	return np.sqrt((n - 1) / n * np.sum((x - x.mean(axis=0)) ** 2, axis=0))


def geometry(outdir):
	r_bins = np.logspace(0, np.log10(60.0), 6)
	mu_bins = np.linspace(-1.0, 1.0, 7)
	Vb = (2 * np.pi / 3) * np.diff(r_bins ** 3)[:, None] * np.diff(mu_bins)[None, :]
	pts = np.random.default_rng(3).uniform(0, BOXSIZE, (60, 3))
	kw = dict(boxsize=BOXSIZE, r_bins=r_bins, bins2=mu_bins, binning="rmu")
	res = {}
	for rule in RULES:
		use_rule(rule)
		res[rule] = np.array([bja.subbox_overlaps(pts, k, 3, **kw) for k in range(27)])  # (27, 3, N, n_r, n_mu)
	ref = res["ref64"]
	partial = (ref[:, 0] > 1e-9 * Vb) & (ref[:, 0] < (1 - 1e-9) * Vb)
	lines = [f"per-galaxy overlap error / V_b where the overlap is partial ({partial.sum()} galaxy-subbox-bin cases)",
			 f"{'rule':8s} {'O max':>9s} {'O p95':>9s} {'O median':>9s} {'cos/sin max':>12s} {'full bin max':>13s}"]
	for rule in RULES:
		if rule == "ref64":
			continue
		dO = np.abs(res[rule][:, 0] - ref[:, 0]) / Vb
		dQ = np.abs(res[rule][:, 1:] - ref[:, 1:]).max(axis=1) / Vb
		full = np.abs(res[rule][:, 0].sum(axis=0) / Vb - 1)
		lines.append(f"{rule:8s} {dO[partial].max():9.1e} {np.percentile(dO[partial], 95):9.1e} "
					 f"{np.median(dO[partial]):9.1e} {dQ[partial].max():12.1e} {full.max():13.1e}")
	full_ref = np.abs(ref[:, 0].sum(axis=0) / Vb - 1).max()
	lines.append(f"(ref64 full-bin error: {full_ref:.1e})")
	text = "\n".join(lines)
	print(text, flush=True)
	with open(os.path.join(outdir, "geometry.txt"), "w") as f:
		f.write(text + "\n")


def dense(n_jk, outdir, num_nodes=1):
	# The rules are switched by patching the module in this process; worker processes import
	# it afresh and would silently use the default rule. The counting may use workers, the
	# re-applied correction must not.
	L = round(n_jk ** (1 / 3))
	mock = radial_alignment_box_mock(n_centrals=3750, n_sat=8, sigma_sat=2.0, boxsize=BOXSIZE, seed=1000)
	data = {k: mock[k] for k in ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	out = {}
	with tempfile.TemporaryDirectory() as tmp:
		ia = MeasureIABox(data, os.path.join(tmp, "b.hdf5"), separation_limits=[1.0, 60.0], num_bins_r=10,
						  num_bins_pi=10, pi_max=30.0, boxsize=BOXSIZE, num_nodes=num_nodes)
		t0 = time.time()
		ia.measure_xi_multipoles(NAME, "both", num_jk=n_jk, temp_file_path=tmp + "/", jk_edge_correction=True)
		print(f"[N_jk={n_jk}] measureia {time.time() - t0:.0f} s", flush=True)
		mu_bins = ia.mu_r_bins
		ia.num_nodes = 1
		group = lambda stat: f"{ia.snap_group}/multipoles/{stat}/{NAME}_jk{n_jk}"
		for rule in RULES:
			use_rule(rule)
			t0 = time.time()
			ia._apply_box_jk_edge_correction("multipoles", NAME, f"{NAME}_jk{n_jk}", L)
			with h5py.File(ia.output_file_name, "r") as f:
				xi_gg = np.array([f[group("xi_gg")][f"{NAME}_{i}"][:] for i in range(n_jk)])
				xi_gp = np.array([f[group("xi_g_plus")][f"{NAME}_{i}"][:] for i in range(n_jk)])
			out[f"{rule}/xi_gg0"] = legendre(xi_gg, mu_bins, 0, 0)
			out[f"{rule}/xi_gp2"] = legendre(xi_gp, mu_bins, 2, 2)
			print(f"[N_jk={n_jk}] {rule}: {time.time() - t0:.0f} s", flush=True)
		out["r_bins"] = ia.r_bins
	np.savez_compressed(os.path.join(outdir, f"dense_{n_jk}.npz"), **out)
	r_mid = np.sqrt(out["r_bins"][1:] * out["r_bins"][:-1])
	lines = [f"N_jk = {n_jk}   r:" + " " * 26 + "".join(f"{x:8.2f}" for x in r_mid)]
	for stat in ["xi_gg0", "xi_gp2"]:
		s_ref = jk_sigma(out[f"ref64/{stat}"])
		for rule in RULES:
			if rule == "ref64":
				continue
			x = out[f"{rule}/{stat}"]
			lines.append(f"  {stat} {rule:8s} sigma/sigma_ref - 1 [%]: "
						 + "".join(f"{100 * v:8.3f}" for v in jk_sigma(x) / s_ref - 1))
			lines.append(f"  {stat} {rule:8s} max shift / sigma_ref:    "
						 + "".join(f"{v:8.4f}" for v in np.abs(x - out[f'ref64/{stat}']).max(axis=0) / s_ref))
	text = "\n".join(lines)
	print(text, flush=True)
	with open(os.path.join(outdir, f"dense_{n_jk}.txt"), "w") as f:
		f.write(text + "\n")


if __name__ == "__main__":
	if sys.argv[1] == "geometry":
		os.makedirs(sys.argv[2], exist_ok=True)
		geometry(sys.argv[2])
	else:
		os.makedirs(sys.argv[3], exist_ok=True)
		dense(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
