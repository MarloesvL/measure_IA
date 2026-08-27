"""Microbenchmark of the candidate-selection step alone.

`profile_measureia.py` shows that the annulus set-difference is a large share
of a measurement (see FINDINGS.md F1), but it aggregates both KDTree queries
into one number and says nothing about memory. This script isolates the four
lines that matter (`pair_kernel.accumulate`, box branch):

    shape_tree = KDTree(binning.tree_coords(shape_chunk, not_LOS), boxsize=...)
    ind_min_i  = shape_tree.query_ball_tree(pos_tree, binning.r_min)
    ind_max_i  = shape_tree.query_ball_tree(pos_tree, binning.r_max)
    ind_rbin_i = base.setdiff2D(ind_max_i, ind_min_i)

and times and memory-profiles each part separately, plus the two candidate
remedies:

  A  np.setdiff1d(..., assume_unique=True)   — skip the two unique() sorts
  B  drop the inner query and the setdiff    — let bin_pairs' mask do the work

Nothing in src/measureia is imported except the real binning classes and a real
MeasureIABox, so the geometry, the tree construction and the query radii are
exactly what a measurement uses. It does not modify anything.

Correctness is checked, not assumed: A is compared elementwise against the
current output, and B's candidate lists are compared to report exactly how many
extra candidates the mask would have to absorb.

    python benchmarks/micro_candidate_selection.py
    python benchmarks/micro_candidate_selection.py --n-shape 100000
"""

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np
from scipy.spatial import KDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_lib
from measureia import MeasureIABox
from measureia import pair_kernel

CHUNK = 100  # pair_kernel.accumulate's chunk_size_outer for the box path


def _setdiff2D(a1, a2, assume_unique=False):
	"""measure_IA_base.setdiff2D, with the assume_unique switch exposed."""
	return [np.setdiff1d(a1[i], a2[i], assume_unique=assume_unique)
			for i in range(len(a1))]


# tracemalloc roughly doubles the cost of allocation-heavy steps, which would
# make the timings useless as absolutes. It is therefore off by default and
# enabled only for the memory pass (--memory).
TRACE = False


def _timed(label, fn):
	if TRACE:
		tracemalloc.start()
	t0 = time.perf_counter()
	out = fn()
	t = time.perf_counter() - t0
	peak = 0
	if TRACE:
		_, peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()
	return {"label": label, "t": t, "peak_mb": peak / 1024 ** 2, "out": out}


def run(binning_name, mock, base, repeats):
	"""Measure one binning's candidate selection over the whole shape sample."""
	positions = mock["Position"]
	shape = mock["Position_shape_sample"]
	not_LOS = np.array([0, 1])

	binning = (pair_kernel.BoxRpPi(base) if binning_name == "BoxRpPi"
			   else pair_kernel.BoxRMuR(base, 0.0))
	pos_tree = KDTree(binning.tree_coords(positions, not_LOS), boxsize=base.boxsize)

	totals = {k: 0.0 for k in
			  ("build", "inner", "outer", "setdiff", "setdiff_au", "asarray")}
	peaks = {k: 0.0 for k in totals}
	n_inner = n_outer = n_kept = 0
	mismatch_A = 0
	extra_B = 0

	for _ in range(repeats):
		for i in range(0, len(shape), CHUNK):
			chunk = shape[i:i + CHUNK]

			r = _timed("build", lambda: KDTree(
				binning.tree_coords(chunk, not_LOS), boxsize=base.boxsize))
			shape_tree = r["out"]
			totals["build"] += r["t"]; peaks["build"] = max(peaks["build"], r["peak_mb"])

			r = _timed("inner", lambda: shape_tree.query_ball_tree(pos_tree, binning.r_min))
			ind_min = r["out"]
			totals["inner"] += r["t"]; peaks["inner"] = max(peaks["inner"], r["peak_mb"])

			r = _timed("outer", lambda: shape_tree.query_ball_tree(pos_tree, binning.r_max))
			ind_max = r["out"]
			totals["outer"] += r["t"]; peaks["outer"] = max(peaks["outer"], r["peak_mb"])

			r = _timed("setdiff", lambda: _setdiff2D(ind_max, ind_min))
			current = r["out"]
			totals["setdiff"] += r["t"]; peaks["setdiff"] = max(peaks["setdiff"], r["peak_mb"])

			r = _timed("setdiff_au", lambda: _setdiff2D(ind_max, ind_min, assume_unique=True))
			remedy_a = r["out"]
			totals["setdiff_au"] += r["t"]; peaks["setdiff_au"] = max(peaks["setdiff_au"], r["peak_mb"])

			# Remedy B still needs the candidate list as an ndarray: the
			# downstream fancy-indexing (positions[cand]) would otherwise have
			# to convert a Python list on every use.
			r = _timed("asarray", lambda: [np.asarray(x) for x in ind_max])
			remedy_b = r["out"]
			totals["asarray"] += r["t"]; peaks["asarray"] = max(peaks["asarray"], r["peak_mb"])

			n_inner += sum(len(x) for x in ind_min)
			n_outer += sum(len(x) for x in ind_max)
			n_kept += sum(len(x) for x in current)
			mismatch_A += sum(
				0 if np.array_equal(a, b) else 1 for a, b in zip(current, remedy_a))
			extra_B += sum(len(b) - len(a) for a, b in zip(current, remedy_b))

	scale = 1.0 / repeats
	return {
		"binning": binning_name,
		"t": {k: v * scale for k, v in totals.items()},
		"peak_mb": peaks,
		"n_inner": n_inner / repeats,
		"n_outer": n_outer / repeats,
		"n_kept": n_kept / repeats,
		"mismatch_A": mismatch_A,
		"extra_B": extra_B / repeats,
		"n_shape": len(shape),
	}


def report(res):
	t = res["t"]
	current = t["build"] + t["inner"] + t["outer"] + t["setdiff"]
	remedy_a = t["build"] + t["inner"] + t["outer"] + t["setdiff_au"]
	remedy_b = t["build"] + t["outer"] + t["asarray"]

	print(f"\n{'=' * 74}\n{res['binning']}   {res['n_shape']:,} shape galaxies\n{'=' * 74}")
	print("  candidates returned by the outer query : "
		  f"{res['n_outer']:>14,.0f}  ({res['n_outer'] / res['n_shape']:.1f}/galaxy)")
	print("  candidates returned by the inner query : "
		  f"{res['n_inner']:>14,.0f}  "
		  f"({100 * res['n_inner'] / max(res['n_outer'], 1):.4f}% of outer)")
	print("  candidates kept after the setdiff      : "
		  f"{res['n_kept']:>14,.0f}")

	print("\n  step timings (whole shape sample, one pass)")
	for k, label in (("build", "shape-tree build"), ("inner", "query @ r_min"),
					 ("outer", "query @ r_max"), ("setdiff", "setdiff2D (current)"),
					 ("setdiff_au", "setdiff2D assume_unique=True  [A]"),
					 ("asarray", "np.asarray only               [B]")):
		mem = (f"   peak alloc {res['peak_mb'][k]:6.2f} MB"
			   if res["peak_mb"][k] else "")
		print(f"    {label:<36} {t[k]:7.3f}s{mem}")

	print("\n  candidate selection, total per pass")
	print(f"    current                          {current:7.3f}s")
	print(f"    remedy A (assume_unique)         {remedy_a:7.3f}s"
		  f"   -> saves {current - remedy_a:6.3f}s "
		  f"({100 * (current - remedy_a) / current:5.1f}% of selection)")
	print(f"    remedy B (no inner query/setdiff){remedy_b:7.3f}s"
		  f"   -> saves {current - remedy_b:6.3f}s "
		  f"({100 * (current - remedy_b) / current:5.1f}% of selection)")

	print("\n  correctness")
	print(f"    A: candidate lists differing from current : {res['mismatch_A']}"
		  f"  {'(identical)' if res['mismatch_A'] == 0 else '(!! DIFFERS !!)'}")
	print(f"    B: extra candidates the mask must absorb  : {res['extra_B']:,.0f}"
		  f"  ({100 * res['extra_B'] / max(res['n_kept'], 1):.4f}% more rows through bin_pairs)")
	return current, remedy_a, remedy_b


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--n-shape", type=int, default=38_400)
	p.add_argument("--density-mode", default="fixed_density")
	p.add_argument("--repeats", type=int, default=1)
	p.add_argument("--memory", action="store_true",
				   help="also trace peak allocation per step (inflates the timings)")
	args = p.parse_args()

	global TRACE
	TRACE = args.memory

	mock = bench_lib.box_mock_for(args.n_shape, args.density_mode)
	scratch = os.path.join(_HERE, "_scratch")
	os.makedirs(scratch, exist_ok=True)
	out = os.path.join(scratch, "micro.hdf5")
	if os.path.exists(out):
		os.remove(out)

	import contextlib
	with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
		base = MeasureIABox(
			{k: mock[k] for k in
			 ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")},
			out, simulation=None, snapshot=None,
			separation_limits=[0.5, 20.0], num_bins_r=10, num_bins_pi=20,
			pi_max=20.0, boxsize=mock["boxsize"], num_nodes=1,
		)

	print(f"box {mock['boxsize']:.1f} Mpc/h, r in [{base.r_min}, {base.r_max}], "
		  f"{'tracemalloc ON (timings inflated)' if TRACE else 'timings only'}")
	for name in ("BoxRpPi", "BoxRMuR"):
		report(run(name, mock, base, args.repeats))
	if os.path.exists(out):
		os.remove(out)


if __name__ == "__main__":
	main()
