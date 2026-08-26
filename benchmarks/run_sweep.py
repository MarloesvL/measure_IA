"""Drive the measureia benchmark sweeps.

Builds the grid of benchmark points, runs each one in its own subprocess via
bench_runner.py, and appends the results to a per-machine JSONL file. The
sweep is resumable: a point whose key is already in the output file is
skipped, so an hours-long laptop run or a walltime-killed cluster job can be
restarted without repeating work.

Three sweeps, selectable with --sweeps:

  size        wall time vs catalogue size, measureia against halotools (box) and
			TreeCorr (lightcone), at one thread, in both scaling regimes.
  threads parallel scaling at one fixed size (measureia num_nodes against
			the reference codes' num_threads).
  internal    measureia's own w vs multipoles on an identical catalogue, with
			KDTree candidate counts recorded alongside the times.

Examples
--------
	python benchmarks/run_sweep.py --smoke --machine laptop
	python benchmarks/run_sweep.py --machine laptop --sweeps size internal
	python benchmarks/run_sweep.py --machine cluster --scratch nfs

Correctness is gated, not assumed: after the sweep, every measureia point is
compared against the reference point measured on the identical catalogue, at
the tolerances the cross-package validation established. Points that disagree
are reported and excluded from the plots by plot_results.py, so a timing is
never published for a configuration that computes the wrong answer.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_lib
import machines

RUNNER = os.path.join(_HERE, "bench_runner.py")

# Catalogue sizes (shape-sample galaxies). The smallest matches the validation
# mocks; the largest is around the COLIBRE-L400 scale quoted in CHANGELOG.md.
SIZES_FULL = [2_400, 9_600, 38_400, 100_000, 300_000]
SIZES_SMOKE = [2_400, 9_600]

# 'fixed_volume' packs every extra galaxy into the same 205 Mpc/h box, so the
# pair count per galaxy grows linearly with N and the total cost as N^2. The
# regime is worth measuring but only needs enough points to fix the slope;
# running it to 300k would take hours and tell us nothing the slope does not.
MAX_N_FIXED_VOLUME = 38_400

# Size at which the thread sweep is run.
THREAD_SWEEP_N = 38_400
THREAD_SWEEP_N_SMOKE = 9_600

DENSITY_MODES = ["fixed_density", "fixed_volume"]

# Jackknife realisation counts. The box requires a perfect cube (it splits into
# sub-boxes); the lightcone uses k-means sky patches and takes any count, and 9
# is what validation/run_lightcone_treecorr_cov.py uses.
NUM_JK_BOX = 27
NUM_JK_LC = 9
# Sizes for the jackknife sweep. Jackknife multiplies the work, so this stays
# well below the size sweep's ceiling.
SIZES_JK = [9_600, 38_400]
SIZES_JK_SMOKE = [2_400]

TASK_CODES = {
	"box_w": ["measureia", "halotools"],
	"lc_w": ["measureia", "treecorr"],
	"box_multipoles": ["measureia"],
	"lc_multipoles": ["measureia"],
}

# Gate tolerances, matching tests/test_validation_references.py.
GATES = {
	"halotools": dict(rtol=1e-10, atol=1e-8),
	"treecorr": dict(rtol=5e-3, atol=0.05),
}


def _sizes_for(sizes, density_mode):
	"""Sizes to run in a given scaling regime (see MAX_N_FIXED_VOLUME)."""
	if density_mode == "fixed_volume":
		return [n for n in sizes if n <= MAX_N_FIXED_VOLUME]
	return sizes


def build_grid(args, sizes, scratch_dir):
	"""All benchmark points requested, as a list of config dicts."""
	points = []
	thread_list = machines.MACHINES.get(args.machine, {}).get("threads", [1, 2, 4])
	if getattr(args, "max_cpus", None):
		thread_list = [t for t in thread_list if t <= args.max_cpus] or [1]

	def add(**kw):
		cfg = dict(
			scratch=args.scratch, scratch_dir=scratch_dir,
			repeats=args.repeats, warmup=args.warmup,
			boxsize=None, num_jk=0, bin_slop=None, variant="tree",
		)
		cfg.update(kw)
		points.append(cfg)

	def sizes_for(mode):
		"""Sizes to run in a given scaling regime (see MAX_N_FIXED_VOLUME)."""
		if mode == "fixed_volume":
			return [n for n in sizes if n <= MAX_N_FIXED_VOLUME]
		return sizes

	if "size" in args.sweeps:
		# Wall time vs catalogue size at one thread, measureia against the
		# reference code, in both scaling regimes. Every measureia point also
		# records its KDTree candidate count, so this doubles as the
		# w-vs-multipoles comparison at each size.
		for task in args.tasks:
			for code in TASK_CODES[task]:
				for mode in DENSITY_MODES:
					for n in sizes_for(mode):
						extra = {"bin_slop": 0} if code == "treecorr" else {}
						add(task=task, code=code, n_shape=n,
							density_mode=mode, threads=1, **extra)
		# TreeCorr at its own default bin_slop: how users actually run it, as
		# opposed to the accuracy-matched bin_slop=0 the validation uses.
		if "lc_w" in args.tasks:
			for n in sizes:
				add(task="lc_w", code="treecorr", n_shape=n,
					density_mode="fixed_density", threads=1, bin_slop=None)

	if "jackknife" in args.sweeps:
		# The full-sample w and multipoles paths turn out to cost about the
		# same; the jackknife and multiprocessing paths are where they can
		# diverge, so cross both axes rather than testing either alone.
		jk_sizes = [n for n in (SIZES_JK_SMOKE if args.smoke else SIZES_JK)
					if n <= max(sizes)]
		jk_threads = [t for t in thread_list if t in (1, 2, 4, 8)] or [1]
		for task in ("box_w", "box_multipoles", "lc_w", "lc_multipoles"):
			if task not in args.tasks:
				continue
			num_jk = NUM_JK_BOX if task.startswith("box") else NUM_JK_LC
			for n in jk_sizes:
				for jk in (0, num_jk):
					for t in jk_threads:
						add(task=task, code="measureia", n_shape=n,
							density_mode="fixed_density", threads=t, num_jk=jk)

	if "threads" in args.sweeps:
		n = THREAD_SWEEP_N_SMOKE if args.smoke else THREAD_SWEEP_N
		n = min(n, max(sizes))
		for task in args.tasks:
			for code in TASK_CODES[task]:
				for t in thread_list:
					extra = {"bin_slop": 0} if code == "treecorr" else {}
					add(task=task, code=code, n_shape=n,
						density_mode="fixed_density", threads=t, **extra)

	if "internal" in args.sweeps:
		# The w-vs-multipoles comparison at each size is already covered by the
		# size sweep, so the only points this adds are the boxsize series:
		# same sample, same everything, varying only the depth of the box.
		#
		# The box (rp, pi) KDTree is built on the 2D projection, so its query is
		# a cylinder through the full box depth and its candidate count should
		# grow with L. The (r, mu_r) query is a 3D ball, whose candidate count
		# does not depend on L at all.
		if "box_w" in args.tasks:
			n = sizes[min(1, len(sizes) - 1)]
			for L in (205.0, 400.0, 800.0):
				for task in ("box_w", "box_multipoles"):
					add(task=task, code="measureia", n_shape=n,
						density_mode="fixed_volume", threads=1, boxsize=L)

	# The sweeps overlap by design, so drop repeated points by key.
	unique, seen = [], set()
	for cfg in points:
		key = bench_lib.config_key(cfg)
		if key not in seen:
			seen.add(key)
			unique.append(cfg)
	return unique


def run_point(cfg, timeout):
	"""Run one point in a fresh subprocess with threads pinned in its environment."""
	env = dict(os.environ)
	nt = str(cfg["threads"])
	for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
				"NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
		env[var] = nt
	try:
		proc = subprocess.run(
			[sys.executable, RUNNER, json.dumps(cfg)],
			capture_output=True, text=True, timeout=timeout, env=env,
		)
	except subprocess.TimeoutExpired:
		rec = dict(cfg)
		rec["status"] = "timeout"
		rec["timeout_s"] = timeout
		return rec
	for line in proc.stdout.splitlines():
		if line.startswith("@@RESULT@@"):
			return json.loads(line[len("@@RESULT@@"):])
	rec = dict(cfg)
	rec["status"] = "error"
	rec["traceback"] = (proc.stderr or proc.stdout)[-4000:]
	return rec


def _label(cfg):
	bits = [cfg["task"], cfg["code"], f"N={cfg['n_shape']:,}", cfg["density_mode"],
			f"t={cfg['threads']}"]
	if cfg.get("boxsize"):
		bits.append(f"L={cfg['boxsize']:g}")
	if cfg.get("bin_slop") is not None:
		bits.append(f"slop={cfg['bin_slop']}")
	return "  ".join(bits)


def check_gates(path):
	"""Compare each measureia point against the reference on the same catalogue.

	Returns a list of (label, verdict, detail). A measureia record only gates
	when a reference record exists for the identical task, size, density mode
	and boxsize — the thread axis and bin_slop are irrelevant to the numbers.
	"""
	records = [r for r in bench_lib.load_records(path) if r.get("status") == "ok"]
	refs = {}
	for r in records:
		if r["code"] in GATES:
			refs.setdefault(
				(r["task"], r["n_shape"], r["density_mode"], r.get("boxsize")), r
			)
	out = []
	seen = set()
	for r in records:
		if r["code"] != "measureia":
			continue
		key = (r["task"], r["n_shape"], r["density_mode"], r.get("boxsize"))
		if key in seen or key not in refs:
			continue
		seen.add(key)
		ref = refs[key]
		tol = GATES[ref["code"]]
		# measureia divides S+ terms by the responsivity 2R; halotools does not.
		scale = 2.0 * r["responsivity_R"] if ref["code"] == "halotools" else 1.0
		verdict, detail = "ok", []
		for field, factor in (("result_g_plus", scale), ("result_gg", 1.0)):
			a = np.asarray(r[field], dtype=float) * factor
			b = np.asarray(ref[field], dtype=float)
			if a.shape != b.shape:
				verdict = "mismatch"
				detail.append(f"{field}: shape {a.shape} vs {b.shape}")
				continue
			if not np.allclose(a, b, **tol):
				verdict = "mismatch"
				with np.errstate(divide="ignore", invalid="ignore"):
					rel = np.abs(a - b) / np.abs(b)
				detail.append(f"{field}: max rel {np.nanmax(rel):.2e}")
		out.append((f"{r['task']} N={r['n_shape']:,} {r['density_mode']} "
					f"vs {ref['code']}", verdict, "; ".join(detail)))
	return out


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--machine", default="laptop", help="profile from machines.py")
	p.add_argument("--scratch", default="local", help="named scratch location in the profile")
	p.add_argument("--scratch-dir", default=None, help="explicit scratch dir, overrides the profile")
	p.add_argument("--out", default=None, help="output JSONL (default results/<machine>_<scratch>.jsonl)")
	p.add_argument("--sweeps", nargs="+", default=["size", "internal"],
				   choices=["size", "threads", "internal", "jackknife"])
	p.add_argument("--tasks", nargs="+", default=sorted(TASK_CODES),
				   choices=sorted(TASK_CODES))
	p.add_argument("--sizes", nargs="+", type=int, default=None)
	p.add_argument("--repeats", type=int, default=5)
	p.add_argument("--warmup", type=int, default=1)
	p.add_argument("--timeout", type=int, default=3600, help="per-point timeout in seconds")
	p.add_argument("--max-cpus", type=int, default=None,
				   help="cap the thread/num_nodes axis, so a run cannot exceed a CPU budget")
	p.add_argument("--smoke", action="store_true", help="smallest grid, for checking the harness")
	p.add_argument("--dry-run", action="store_true")
	p.add_argument("--check-only", action="store_true", help="only re-run the correctness gates")
	args = p.parse_args()

	scratch_dir = machines.resolve(args.machine, args.scratch, args.scratch_dir)
	out_path = args.out or os.path.join(
		_HERE, "results", f"{args.machine}_{args.scratch}.jsonl")

	if args.check_only:
		_report_gates(out_path)
		return

	if args.smoke:
		args.repeats = min(args.repeats, 3)
		args.warmup = 1
	sizes = args.sizes or (SIZES_SMOKE if args.smoke else SIZES_FULL)
	max_n = machines.MACHINES.get(args.machine, {}).get("max_n_shape")
	if max_n:
		sizes = [n for n in sizes if n <= max_n]

	points = build_grid(args, sizes, scratch_dir)
	done = bench_lib.completed_keys(out_path)
	todo = [c for c in points if bench_lib.config_key(c) not in done]

	print(f"scratch : {scratch_dir}")
	print(f"output  : {out_path}")
	print(f"points  : {len(points)} total, {len(points) - len(todo)} already done, "
		  f"{len(todo)} to run")
	if args.dry_run:
		for c in todo:
			print("  " + _label(c))
		return

	os.makedirs(scratch_dir, exist_ok=True)
	for i, cfg in enumerate(todo, 1):
		print(f"[{i}/{len(todo)}] {_label(cfg)} ... ", end="", flush=True)
		rec = run_point(cfg, args.timeout)
		bench_lib.append_record(out_path, rec)
		if rec["status"] == "ok":
			extra = ""
			if rec.get("candidates"):
				extra = f"  cand/gal {rec['candidates'] / max(rec.get('n_shape_actual') or rec['n_shape'], 1):.1f}"
			print(f"{rec['t_min']:.3f}s (median {rec['t_median']:.3f})"
				  f"  {rec['peak_rss_mb']:.0f} MB{extra}")
		else:
			print(rec["status"].upper())
			if rec.get("traceback"):
				print("    " + rec["traceback"].strip().splitlines()[-1])

	_report_gates(out_path)


def _report_gates(out_path):
	print("\n--- correctness gates (measureia vs the reference on the same catalogue) ---")
	gates = check_gates(out_path)
	if not gates:
		print("  no comparable pairs recorded yet")
		return
	for label, verdict, detail in gates:
		mark = "PASS" if verdict == "ok" else "FAIL"
		print(f"  {mark}  {label}" + (f"   [{detail}]" if detail else ""))


if __name__ == "__main__":
	main()
