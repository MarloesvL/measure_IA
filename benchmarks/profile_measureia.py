"""Where measureia's time actually goes.

The cross-package sweep (run_sweep.py) answers "how fast"; this answers "why".
It profiles each of the four measurement paths on the same catalogue and prints
a ranked hot-spot table plus a grouped breakdown, so an optimisation effort has
somewhere to aim.

The grouping maps profiler entries onto the stages a reader can act on:

  tree_query   scipy KDTree construction and query_ball_tree
  setdiff      the annulus set-difference that removes the inner-radius ball
				 (numpy setdiff1d / unique / isin, called once per galaxy)
  binning      pair_kernel's bin_pairs — separations, windows, bin indices
  accumulate   the per-galaxy Python loop body in pair_kernel.accumulate
  add_at       np.add.at, the actual grid accumulation
  shapes       ellipticity projection (get_ellipticity, arccos)
  io           HDF5 reads and writes
  other            everything else

Run it directly:

	python benchmarks/profile_measureia.py
	python benchmarks/profile_measureia.py --n-shape 100000 --paths box_w

A note on interpreting the result: cProfile adds per-call overhead, so a stage
made of very many cheap calls is flattered downwards in wall-clock terms and
the absolute total exceeds the unprofiled runtime. Compare the *shares*, and
take the absolute numbers from run_sweep.py.
"""

import argparse
import cProfile
import io
import os
import pstats
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "validation"))

import numpy as np

import bench_lib
import bench_runner

# (substring of "file:lineno(function)", group). First match wins, so order
# matters: the more specific patterns come first.
GROUPS = [
	("_kdtree.py", "tree_query"),
	("ckdtree", "tree_query"),
	("setdiff1d", "setdiff"),
	("_arraysetops_impl.py", "setdiff"),
	("_unique_hash", "setdiff"),
	("bin_pairs", "binning"),
	("get_ellipticity", "shapes"),
	("arccos", "shapes"),
	("method 'at' of 'numpy.ufunc'", "add_at"),
	("pair_kernel.py", "accumulate"),
	("h5py", "io"),
	("write_data.py", "io"),
	("read_data.py", "io"),
]


def classify(entry):
	for pattern, group in GROUPS:
		if pattern in entry:
			return group
	return "other"


def profile_path(task, n_shape, density_mode, scratch_dir, top=12):
	"""Profile one measurement path; return (stats, grouped totals, wall time)."""
	cfg = dict(
		task=task, code="measureia", n_shape=n_shape, density_mode=density_mode,
		threads=1, scratch_dir=scratch_dir, scratch="local", variant="tree",
		boxsize=None, num_jk=0, bin_slop=None, repeats=1, warmup=0,
	)
	builder = bench_runner.BUILDERS[(task, "measureia")]
	with bench_runner._quiet():
		fn, _ = builder(cfg)
		fn()  # warm up outside the profiler

	pr = cProfile.Profile()
	with bench_runner._quiet():
		pr.enable()
		fn()
		pr.disable()

	stats = pstats.Stats(pr)
	grouped = {}
	for func, (_, _, tottime, _, _) in stats.stats.items():
		entry = f"{func[0]}:{func[1]}({func[2]})"
		grouped[classify(entry)] = grouped.get(classify(entry), 0.0) + tottime
	return stats, grouped, sum(grouped.values())


def print_report(task, stats, grouped, total, top):
	print(f"\n{'=' * 78}\n{task}   (profiled total {total:.2f}s)\n{'=' * 78}")
	print("\nstage breakdown")
	for name, t in sorted(grouped.items(), key=lambda kv: -kv[1]):
		bar = "#" * int(round(40 * t / total)) if total else ""
		print(f"  {name:<12} {t:7.3f}s  {100 * t / total:5.1f}%  {bar}")

	print(f"\ntop {top} by internal time")
	buf = io.StringIO()
	# `stats` is already a pstats.Stats; re-wrapping it raises, so point its
	# stream at the buffer instead of constructing a new one
	stats.stream = buf
	stats.sort_stats("tottime").print_stats(top)
	started = False
	for line in buf.getvalue().splitlines():
		if line.strip().startswith("ncalls"):
			started = True
		if started and line.strip():
			print("  " + line.strip())


def scaling_report(task, sizes, density_mode, scratch_dir, top):
	"""Profile one path at several sizes and give each stage its own slope.

	The question this answers: measureia's wall time scales as ~N^1.25-1.46 at
	fixed number density, where the pair count is linear in N and the ideal slope
	is 1.0 (benchmarks/FINDINGS.md F3). A single profile cannot say which part is
	responsible. Slope each stage separately and the culprit is whichever one
	exceeds 1.0 while carrying real weight -- a stage that is 2% of runtime
	cannot explain the total no matter how badly it scales.
	"""
	rows = {}
	for n in sizes:
		_, grouped, total = profile_path(task, n, density_mode, scratch_dir, top)
		rows[n] = (grouped, total)
		print(f"  profiled {task} at N={n:,}: {total:.2f}s")

	names = sorted({g for grouped, _ in rows.values() for g in grouped})
	n0, n1 = sizes[0], sizes[-1]

	print(f"\n{'=' * 82}\nper-stage scaling, {task}, {density_mode}\n{'=' * 82}")
	header = f"  {'stage':<13}" + "".join(f"{n:>12,}" for n in sizes) + f"{'slope':>9}{'share':>9}"
	print(header)
	print("  " + "-" * (len(header) - 2))
	slopes = {}
	for name in names:
		t0 = rows[n0][0].get(name, 0.0)
		t1 = rows[n1][0].get(name, 0.0)
		cells = "".join(f"{rows[n][0].get(name, 0.0):>11.3f}s" for n in sizes)
		if t0 > 0 and t1 > 0:
			slope = np.log(t1 / t0) / np.log(n1 / n0)
			slopes[name] = (slope, t1 / rows[n1][1])
			flag = "  <-- superlinear" if slope > 1.15 and t1 / rows[n1][1] > 0.10 else ""
			print(f"  {name:<13}{cells}{slope:>9.2f}{100 * t1 / rows[n1][1]:>8.0f}%{flag}")
		else:
			print(f"  {name:<13}{cells}{'-':>9}{'-':>9}")

	tot_slope = np.log(rows[n1][1] / rows[n0][1]) / np.log(n1 / n0)
	print(f"  {'TOTAL':<13}" + "".join(f"{rows[n][1]:>11.3f}s" for n in sizes)
		  + f"{tot_slope:>9.2f}")

	print("\n  Interpretation: a stage only explains the total if its slope exceeds 1.0")
	print("  *and* it carries a meaningful share of the runtime at the largest size.")
	worst = sorted(((sl, sh, nm) for nm, (sl, sh) in slopes.items()
					if sh > 0.10), reverse=True)
	if worst:
		sl, sh, nm = worst[0]
		print(f"  Dominant contributor: {nm} (slope {sl:.2f}, {100 * sh:.0f}% of runtime).")
	return rows


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--n-shape", type=int, default=38_400)
	p.add_argument("--density-mode", default="fixed_density",
				   choices=["fixed_density", "fixed_volume"])
	p.add_argument("--paths", nargs="+",
				   default=["box_w", "box_multipoles", "lc_w", "lc_multipoles"])
	p.add_argument("--scratch-dir", default=os.path.join(_HERE, "_scratch"))
	p.add_argument("--top", type=int, default=12)
	p.add_argument("--scaling", nargs="+", type=int, default=None,
				   metavar="N",
				   help="profile at these sizes and slope each stage separately, "
						"to find what makes the total superlinear")
	args = p.parse_args()

	os.makedirs(args.scratch_dir, exist_ok=True)
	print(f"catalogue: {args.n_shape:,} shape galaxies, {args.density_mode}")
	print(f"scratch  : {args.scratch_dir}")

	if args.scaling:
		for task in args.paths:
			scaling_report(task, sorted(args.scaling), args.density_mode,
						   args.scratch_dir, args.top)
		return

	summary = {}
	for task in args.paths:
		stats, grouped, total = profile_path(
			task, args.n_shape, args.density_mode, args.scratch_dir, args.top)
		print_report(task, stats, grouped, total, args.top)
		summary[task] = (grouped, total)

	print(f"\n{'=' * 78}\nshare of profiled time by stage\n{'=' * 78}")
	names = sorted({g for grouped, _ in summary.values() for g in grouped})
	header = "  " + "stage".ljust(12) + "".join(t.rjust(17) for t in args.paths)
	print(header)
	for name in names:
		row = "  " + name.ljust(12)
		for task in args.paths:
			grouped, total = summary[task]
			t = grouped.get(name, 0.0)
			row += f"{t:7.3f}s {100 * t / total:4.1f}%".rjust(17)
		print(row)


if __name__ == "__main__":
	main()
