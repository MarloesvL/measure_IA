"""Peak memory of a single measurement, for one configuration.

Runs one measurement in this process and reports both peak RSS
(``resource.getrusage``, i.e. what the OS saw, including the interpreter and
the catalogue arrays) and the peak Python allocation during the measurement
(``tracemalloc``, which excludes the baseline and so is far more sensitive to
the transient per-galaxy arrays).

Written for the F1 change (benchmarks/FINDINGS.md): dropping the inner KDTree
query means more candidates reach ``bin_pairs``, so the per-galaxy
``separation`` array is larger. How much larger depends strongly on
``separation_limits`` -- at [0.5, 20] the inner ball is a sliver, but at
[5, 20] it is a third of all candidates for the (r, mu_r) binning. This script
exists to measure the memory consequence at the stressing end rather than
assume it from the mild one.

One configuration per process, so RSS means something:

    python benchmarks/memcheck.py --task box_w --n-shape 100000 --sep-limits 5 20
"""

import argparse
import contextlib
import json
import os
import resource
import sys
import time
import tracemalloc

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_lib


def _peak_rss_mb():
	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return rss / (1024.0 ** 2) if sys.platform == "darwin" else rss / 1024.0


def run(task, n_shape, sep_limits, density_mode, scratch,
		trace=True, repeats=1):
	from measureia import MeasureIABox

	multipoles = task.endswith("multipoles")
	mock = bench_lib.box_mock_for(n_shape, density_mode)
	data = {k: mock[k] for k in
			("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")}
	os.makedirs(scratch, exist_ok=True)
	out = os.path.join(scratch, f"memcheck_{os.getpid()}.hdf5")
	if os.path.exists(out):
		os.remove(out)

	rss_before = _peak_rss_mb()
	with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
		ia = MeasureIABox(
			data, out, simulation=None, snapshot=None,
			separation_limits=list(sep_limits), num_bins_r=10, num_bins_pi=20,
			pi_max=20.0, boxsize=mock["boxsize"], num_nodes=1,
		)
		meth = ia.measure_xi_multipoles if multipoles else ia.measure_xi_w
		# warm up, and count candidates on that untimed call so the counting
		# never lands inside a timed region
		with bench_lib.CandidateCounter() as cc:
			meth("mem", "both", 0, temp_file_path=scratch + os.sep)

		# tracemalloc taxes allocation-heavy code, so it is off whenever the
		# timings are the point; with it on, treat the times as indicative only
		tm_peak = 0
		if trace:
			tracemalloc.start()
		times = []
		for _ in range(repeats):
			t0 = time.perf_counter()
			meth("mem", "both", 0, temp_file_path=scratch + os.sep)
			times.append(time.perf_counter() - t0)
		if trace:
			_, tm_peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()
		elapsed = float(np.min(times))

	if os.path.exists(out):
		os.remove(out)
	return {
		"task": task, "n_shape": n_shape, "sep_limits": list(sep_limits),
		"density_mode": density_mode,
		"n_shape_actual": int(len(mock["Position_shape_sample"])),
		"candidates": cc.candidates,
		"t": elapsed,
		"t_median": float(np.median(times)),
		"times": times,
		"traced": trace,
		"peak_rss_mb": _peak_rss_mb(),
		"rss_before_mb": rss_before,
		"tracemalloc_peak_mb": tm_peak / 1024.0 ** 2,
	}


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--task", default="box_w", choices=["box_w", "box_multipoles"])
	p.add_argument("--n-shape", type=int, default=38_400)
	p.add_argument("--sep-limits", type=float, nargs=2, default=[0.5, 20.0])
	p.add_argument("--density-mode", default="fixed_density")
	p.add_argument("--scratch", default=os.path.join(_HERE, "_scratch"))
	p.add_argument("--no-trace", action="store_true",
				   help="disable tracemalloc, so the timings are trustworthy")
	p.add_argument("--repeats", type=int, default=1)
	args = p.parse_args()
	print("@@MEM@@" + json.dumps(
		run(args.task, args.n_shape, args.sep_limits, args.density_mode,
			args.scratch, trace=not args.no_trace, repeats=args.repeats)))


if __name__ == "__main__":
	main()
