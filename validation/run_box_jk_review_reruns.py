"""Reruns for the external review of box_jk_analytic_randoms.md (section 8, points 1-3), with
the production module measureia.box_jk_analytic for option 2.

The randoms terms per mock come from ``run_box_jk_cross_pairs.py terms`` (same mocks, same
random catalogue, union deletion). This script adds the analytic terms and the analysis.

Commands:
	analytic --binning B --n-centrals N --n-mocks M --mock-start S --out FILE
		option 2's per-realisation DR, RD, S+R from measureia.box_jk_analytic
	analyse --binning B --terms F... --analytic F... [--label L]
		(3) mean +- s.e.m. over mocks of sigma_jk(option) / sigma_jk(reference), per bin
		(1) term swap: option 2 with the reference's counted RR_k, and option 1 (which differs
		    from the reference only in RR)
		(2) variance decomposition, Delta_k = ref_k - opt2_k:
		    sigma2_ref = sigma2_opt2 + sigma2_Delta + 2 cov(opt2, Delta), jackknife-scaled
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_box_jk_cross_pairs as rb  # noqa: E402
import jk_cross_pairs_lib as lib  # noqa: E402
from measureia import box_jk_analytic as bja  # noqa: E402
from measureia.mocks import radial_alignment_box_mock  # noqa: E402

N_JK = (8, 27, 64)


def set_binning(binning):
	rb.BINNING[0] = binning
	if binning == "rmu":
		rb.PI_BINS = np.linspace(-1.0, 1.0, rb.NUM_BINS_MU + 1)


def analytic(binning, n_centrals, n_mocks, mock_start, outfile):
	store = {}
	for m in range(mock_start, mock_start + n_mocks):
		mock = radial_alignment_box_mock(n_centrals=n_centrals, boxsize=rb.BOXSIZE, **rb.MOCK_KW, seed=1000 + m)
		P, S = mock["Position"], mock["Position_shape_sample"]
		theta = np.arctan2(mock["Axis_Direction"][:, 1], mock["Axis_Direction"][:, 0])
		e = rb.distortion(mock["q"])
		for n_jk in N_JK:
			L = round(n_jk ** (1 / 3))
			terms = bja.analytic_dr_terms(P, S, theta, e, rb._labels(P, n_jk), rb._labels(S, n_jk), L,
										  boxsize=rb.BOXSIZE, r_bins=rb.R_BINS, bins2=rb.PI_BINS, binning=binning)
			for name, v in zip(("DRs", "RdS", "SpRd"), terms):
				store.setdefault(f"{n_jk}/an_{name}", []).append(v)
		print(f"  analytic mock {m + 1 - mock_start}/{n_mocks}", flush=True)
	np.savez_compressed(outfile, **{k: np.array(v) for k, v in store.items()})


def _load(files):
	parts = [np.load(p) for p in files]
	return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def _jk_var(x):
	"""Jackknife-scaled variance per bin of realisations x[..., k, bin]."""
	n = x.shape[-2]
	return (n - 1) / n * np.sum((x - x.mean(axis=-2, keepdims=True)) ** 2, axis=-2)


def _jk_cov(x, y):
	n = x.shape[-2]
	return (n - 1) / n * np.sum((x - x.mean(axis=-2, keepdims=True)) * (y - y.mean(axis=-2, keepdims=True)),
								axis=-2)


def _row(label, values, fmt="{:7.3f}"):
	return f"  {label:34s}" + "".join(fmt.format(v) for v in values)


def analyse(binning, terms_files, analytic_files, label):
	T, A = _load(terms_files), _load(analytic_files)
	n_mocks = min(len(T[f"{N_JK[0]}/R_full"]), len(A[f"{N_JK[0]}/an_DRs"]))
	mid = np.sqrt(rb.R_BINS[1:] * rb.R_BINS[:-1])
	stats = ("g+", "gg") if binning == "rppi" else ("g+,2", "gg,0")
	print(f"\n=== {label}: {binning}, {n_mocks} mocks ===")
	for n_jk in N_JK:
		g = lambda k: T[f"{n_jk}/union/{k}/jk"][:n_mocks]
		an = lambda k: A[f"{n_jk}/an_{k}"][:n_mocks]
		R = T[f"{n_jk}/union/R_k"][:n_mocks, :, None, None]
		rr_x = np.broadcast_to(g("RRexact")[:, None], g("RR").shape)

		def est(rr, drs, rds, sprd):
			return (rb.proj_gp((g("SpD") - sprd) / rr / (2 * R)), rb.proj_gg((g("DS") - drs - rds + rr) / rr))

		ref = est(g("RR"), g("DRs"), g("RdS"), g("SpRd"))
		opts = {
			"today (analytic RR only)": (rb.proj_gp(g("SpD") / g("RRa_cur") / (2 * R)),
										 rb.proj_gg(g("DS") / g("RRa_cur") - 1)),
			"option 1 (randoms DR, exact RR)": est(rr_x, g("DRs"), g("RdS"), g("SpRd")),
			"option 2 (analytic DR, exact RR)": est(rr_x, an("DRs"), an("RdS"), an("SpRd")),
			"swap: option 2 with counted RR": est(g("RR"), an("DRs"), an("RdS"), an("SpRd")),
		}
		print(f"\nN_jk = {n_jk}    r: " + "".join(f"{x:7.2f}" for x in mid))
		for s, stat in enumerate(stats):
			v_ref = _jk_var(ref[s])                                   # (mocks, bins)
			print(f" [{stat}] sigma_jk(option) / sigma_jk(reference): mean over mocks, then +- s.e.m.")
			for name, o in opts.items():
				ratio = np.sqrt(_jk_var(o[s]) / v_ref)
				print(_row(name, ratio.mean(axis=0)))
				print(_row("  +- s.e.m.", ratio.std(axis=0, ddof=1) / np.sqrt(n_mocks)))
			# (2) variance decomposition against option 2
			o2 = opts["option 2 (analytic DR, exact RR)"][s]
			delta = ref[s] - o2
			v_o2, v_d, c = _jk_var(o2), _jk_var(delta), _jk_cov(o2, delta)
			print(f" [{stat}] decomposition, Delta = ref - opt2 (mean over mocks, in units of sigma2_ref):")
			print(_row("gap sigma2_ref - sigma2_opt2", ((v_ref - v_o2) / v_ref).mean(axis=0), "{:7.3f}"))
			print(_row("sigma2_Delta", (v_d / v_ref).mean(axis=0), "{:7.3f}"))
			print(_row("2 cov(opt2, Delta)", (2 * c / v_ref).mean(axis=0), "{:7.3f}"))
			print(_row("corr(opt2, Delta)", (c / np.sqrt(v_o2 * v_d)).mean(axis=0), "{:7.2f}"))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("command", choices=["analytic", "analyse"])
	ap.add_argument("--binning", default="rppi", choices=["rppi", "rmu"])
	ap.add_argument("--n-centrals", type=int, default=400)
	ap.add_argument("--n-mocks", type=int, default=1)
	ap.add_argument("--mock-start", type=int, default=0)
	ap.add_argument("--out")
	ap.add_argument("--terms", nargs="+")
	ap.add_argument("--analytic", nargs="+")
	ap.add_argument("--label", default="")
	a = ap.parse_args()
	set_binning(a.binning)
	if a.command == "analytic":
		analytic(a.binning, a.n_centrals, a.n_mocks, a.mock_start, a.out)
	else:
		analyse(a.binning, a.terms, a.analytic, a.label)
