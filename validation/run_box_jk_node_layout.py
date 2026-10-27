"""r-mu analytic jackknife terms: shared rho nodes (prototype) vs per-bin rho nodes.

Modes:
	accuracy          per-galaxy overlap error against adaptive quadrature split at every kink
	sigma OUTDIR      jackknife sigma of xi_gg,0 / xi_g+,2 on the sparse mock against the 256-node
	                  reference saved by run_box_jk_quadrature_check.py (same OUTDIR)
	time LAYOUT DENSITY N_JK
	                  wall time and peak memory of analytic_dr_terms (LAYOUT shared|per_bin,
	                  DENSITY sparse|dense); with LAYOUT=counting, measureia's own multipoles
	                  jackknife pair counting (tree, 1 core, no edge correction) for comparison
"""

import os
import resource
import sys
import tempfile
import time
import warnings

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
sys.path.insert(0, HERE)
import box_jk_reference as ref  # noqa: E402
from measureia import box_jk_analytic as bja  # noqa: E402
from measureia import MeasureIABox  # noqa: E402
from measureia.mocks import radial_alignment_box_mock  # noqa: E402
import run_box_jk_quadrature_check as qc  # noqa: E402

BOXSIZE = 150.0
LAYOUTS = ("shared", "per_bin")
_PER_BIN_TABLE = bja._rmu_node_table


def _shared_table(r_bins, rp_cut):
	"""The loop prototype's layout: one node set for all bins, 16-point Gauss-Legendre on
	[0, r_0], [r_0, r_1], ... (plus a break at rp_cut), repeated for every bin."""
	r_bins = np.asarray(r_bins, dtype=float)
	edges = np.concatenate([[0.0], r_bins])
	if 0.0 < rp_cut < edges[-1] and rp_cut not in edges:
		edges = np.sort(np.append(edges, rp_cut))
	rho, w = bja._gl_segments(list(zip(edges[:-1], edges[1:])))
	n_r = len(r_bins) - 1
	rho, w = np.tile(rho, (n_r, 1)), np.tile(w, (n_r, 1))
	return rho, np.where(rho >= rp_cut, w, 0.0)


def set_layout(name):
	"""measureia uses per-bin nodes; the shared layout is patched in for the comparison."""
	bja._rmu_node_table = _shared_table if name == "shared" else _PER_BIN_TABLE


def accuracy():
	from scipy.integrate import quad
	warnings.simplefilter("ignore")
	r_bins = np.logspace(0, np.log10(60.0), 11)
	mu_bins = np.linspace(-1, 1, 11)
	L = 3
	LS = BOXSIZE / L
	rng = np.random.default_rng(5)
	cases = []
	while len(cases) < 30:
		x = rng.uniform(0, BOXSIZE, 3)
		k = int(rng.integers(27))
		O, _, _ = bja.subbox_overlaps(x[None], k, L, boxsize=BOXSIZE, r_bins=r_bins, bins2=mu_bins, binning="rmu")
		if O.sum() > 0 and k != qc.labels(x[None], L)[0]:
			cases.append((x, k))
	err = {name: [] for name in LAYOUTS}
	for x, k in cases:
		ix, iy, iz = k // 9, (k // 3) % 3, k % 3
		rects = [(xl, xh, yl, yh) for xl, xh in ref._image_intervals(ix * LS - x[0], LS, BOXSIZE)
				 for yl, yh in ref._image_intervals(iy * LS - x[1], LS, BOXSIZE)]
		zs = ref._image_intervals(iz * LS - x[2], LS, BOXSIZE)
		a = np.array(rects)
		got = {}
		for name in LAYOUTS:
			set_layout(name)
			got[name] = bja.subbox_overlaps(x[None], k, L, boxsize=BOXSIZE, r_bins=r_bins, bins2=mu_bins,
											binning="rmu")[0][0]
		Vb = ref.shell_volumes(r_bins, mu_bins, "rmu")
		for i in range(10):
			for j in range(10):
				r1, r2, m1, m2 = r_bins[i], r_bins[i + 1], mu_bins[j], mu_bins[j + 1]
				g1, g2 = bja._mu_slopes([m1, m2])

				def f(rho):
					if rho <= 0:
						return 0.0
					I0 = ref.arc_integrals(rho, a[:, 0], a[:, 1], a[:, 2], a[:, 3])[0].sum()
					ah, bh = np.sqrt(max(r1 ** 2 - rho ** 2, 0)), np.sqrt(max(r2 ** 2 - rho ** 2, 0))
					lo_m = rho * g1 if np.isfinite(g1) else -np.inf
					hi_m = rho * g2 if np.isfinite(g2) else np.inf
					tot = 0.0
					for lo_s, hi_s in ((ah, bh), (-bh, -ah)):
						lo, hi = max(lo_s, lo_m), min(hi_s, hi_m)
						for zl, zh in zs:
							tot += max(0.0, min(hi, zh) - max(lo, zl))
					return rho * I0 * tot

				pts = {0.0, r1, r2}
				for r in (r1, r2):
					for mu in (m1, m2):
						if abs(mu) < 1:
							pts.add(r * np.sqrt(1 - mu ** 2))
					for zl, zh in zs:
						for zf in (zl, zh):
							if abs(zf) < r:
								pts.add(np.sqrt(r ** 2 - zf ** 2))
				for g in (g1, g2):
					if np.isfinite(g) and g != 0:
						for zl, zh in zs:
							for zf in (zl, zh):
								if zf / g > 0:
									pts.add(zf / g)
				for xl, xh, yl, yh in rects:
					pts |= {abs(xl), abs(xh), abs(yl), abs(yh)}
					pts |= {np.hypot(u, v) for u in (xl, xh) for v in (yl, yh)}
				pts = np.array(sorted(p for p in pts if 0 <= p <= r2))
				exact = sum(quad(f, u, v, epsabs=1e-12 * Vb[i, j], epsrel=1e-11, limit=300)[0]
							for u, v in zip(pts[:-1], pts[1:]))
				if exact == 0 and all(got[n][i, j] == 0 for n in LAYOUTS):
					continue
				for name in LAYOUTS:
					err[name].append((got[name][i, j] - exact) / Vb[i, j])
	print(f"per-galaxy overlap error / V_b ({len(cases)} galaxy-sub-box pairs next to the hole, "
		  f"10 r x 10 mu bins, {len(err['shared'])} non-zero bins)")
	for name in LAYOUTS:
		e = np.abs(err[name])
		print(f"  {name:8s} max {e.max():.1e}  99% {np.quantile(e, 0.99):.1e}  95% {np.quantile(e, 0.95):.1e}  "
			  f"median {np.median(e):.1e}  mean signed {np.mean(err[name]):+.1e}")


def sigma(outdir):
	for n_jk in (8, 27, 64):
		npz = os.path.join(outdir, f"quadrature_{n_jk}.npz")
		hdf = os.path.join(outdir, f"mia_{n_jk}.hdf5")
		if not (os.path.exists(npz) and "256/xi_gg0" in np.load(npz).files):
			print(f"N_jk = {n_jk}: no 256-node reference yet, skipped")
			continue
		saved = np.load(npz)
		r_bins, mu_bins = saved["r_bins"], saved["mu_bins"]
		L = round(n_jk ** (1 / 3))
		mock = radial_alignment_box_mock(n_centrals=400, n_sat=8, sigma_sat=2.0, boxsize=BOXSIZE, seed=1000)
		P, S = mock["Position"], mock["Position_shape_sample"]
		g = lambda stat: f"multipoles/{stat}/{qc.NAME}_jk{n_jk}"
		with h5py.File(hdf, "r") as f:
			get = lambda stat, suf: np.array([f[g(stat)][f"{qc.NAME}_{i}{suf}"][:] for i in range(n_jk)])
			DD, SpD, RR_old = get("xi_gg", "_DD"), get("xi_g_plus", "_SplusD"), get("xi_gg", "_RR")
		V = BOXSIZE ** 3
		Vb = ref.shell_volumes(r_bins, mu_bins, "rmu")
		N = (RR_old / Vb[None] * V * (n_jk - 1) / n_jk)[:, :1, :1]
		RRx = ref.exact_union_rr_reference(L, boxsize=BOXSIZE, r_bins=r_bins, bins2=mu_bins, binning="rmu")[None]
		theta = np.arctan2(mock["Axis_Direction"][:, 1], mock["Axis_Direction"][:, 0])
		e = (1 - mock["q"] ** 2) / (1 + mock["q"] ** 2)
		r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
		print(f"N_jk = {n_jk}   r: " + "".join(f"{x:8.2f}" for x in r_mid))
		for name in LAYOUTS:
			set_layout(name)
			DR, RD, SpR = bja.analytic_dr_terms(P, S, theta, e, qc.labels(P, L), qc.labels(S, L), L, boxsize=BOXSIZE,
												r_bins=r_bins, bins2=mu_bins, binning="rmu")
			xi = {"xi_gg0": qc.legendre((DD / N - DR - RD + RRx) / RRx, mu_bins, 0, 0),
				  "xi_gp2": qc.legendre((SpD / N - SpR) / RRx, mu_bins, 2, 2)}
			for stat in ("xi_gg0", "xi_gp2"):
				ratio = qc.jk_sigma(xi[stat]) / qc.jk_sigma(saved[f"256/{stat}"]) - 1
				print(f"  {name:8s} {stat} sigma / sigma(256 nodes) - 1 [%]: "
					  + "".join(f"{100 * v:8.3f}" for v in ratio))


def timing(layout, density, n_jk):
	n_centrals = 400 if density == "sparse" else 3750
	mock = radial_alignment_box_mock(n_centrals=n_centrals, n_sat=8, sigma_sat=2.0, boxsize=BOXSIZE, seed=1000)
	P, S = mock["Position"], mock["Position_shape_sample"]
	L = round(n_jk ** (1 / 3))
	r_bins = np.logspace(0, np.log10(60.0), 11)
	mu_bins = np.linspace(-1, 1, 11)
	t0 = time.time()
	if layout == "counting":
		data = {k: mock[k] for k in ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
		with tempfile.TemporaryDirectory() as tmp:
			ia = MeasureIABox(data, os.path.join(tmp, "t.hdf5"), separation_limits=[1.0, 60.0], num_bins_r=10,
							  num_bins_pi=10, boxsize=BOXSIZE, num_nodes=1)
			ia.measure_xi_multipoles("t", "both", num_jk=n_jk, temp_file_path=tmp + "/", jk_edge_correction=False)
	else:
		set_layout(layout)
		theta = np.arctan2(mock["Axis_Direction"][:, 1], mock["Axis_Direction"][:, 0])
		e = (1 - mock["q"] ** 2) / (1 + mock["q"] ** 2)
		bja.analytic_dr_terms(P, S, theta, e, qc.labels(P, L), qc.labels(S, L), L, boxsize=BOXSIZE, r_bins=r_bins,
							  bins2=mu_bins, binning="rmu")
	peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes
	print(f"{layout:8s} {density:6s} N_jk={n_jk:3d} D={len(P)} S={len(S)}: {time.time() - t0:7.1f} s, "
		  f"peak memory {peak:.2f} GB", flush=True)


if __name__ == "__main__":
	mode = sys.argv[1]
	if mode == "accuracy":
		accuracy()
	elif mode == "sigma":
		sigma(sys.argv[2])
	else:
		timing(sys.argv[2], sys.argv[3], int(sys.argv[4]))
