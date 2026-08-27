"""Turn benchmark JSONL into figures and a markdown table.

Reads one or more result files written by run_sweep.py and produces:

  scaling_<mode>.png   wall time vs catalogue size, measureia against the
						 reference code, one panel per task
  internal_w_vs_mult.png measureia's own w vs multipoles, with the KDTree
						 candidate count per galaxy on a twin axis
  boxsize.png          the boxsize series at fixed N and fixed sample
  threads.png          parallel speedup (only if a thread sweep was run)
  memory.png           peak RSS vs catalogue size

plus a markdown summary table on stdout, ready to paste into paper/paper.md
or docs/.

Points whose correctness gate failed are excluded, not quietly plotted: a
timing for a configuration that computes the wrong answer is worse than no
timing. Records from different machines or scratch filesystems are never mixed
into one curve — pass one file at a time, or use --split-by.

Plotting style follows examples/example_read_and_plot.py: plain matplotlib, no
shared style module, LaTeX axis labels, savefig at dpi=150.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_lib
import run_sweep

TASK_LABELS = {
	"box_w": r"Box $w_{gg}$, $w_{g+}$",
	"box_multipoles": r"Box multipoles",
	"lc_w": r"Lightcone $w_{gg}$, $w_{g+}$",
	"lc_multipoles": r"Lightcone multipoles",
}
CODE_LABELS = {
	"measureia": "measureia",
	"halotools": "halotools",
	"treecorr": "TreeCorr",
}
MODE_LABELS = {
	"fixed_density": "fixed number density (box grows as $N^{1/3}$)",
	"fixed_volume": "fixed volume (density grows with $N$)",
}


def load(paths):
	"""Load records, drop failures and gate-failing measureia points."""
	records, failed = [], set()
	for path in paths:
		for label, verdict, _ in run_sweep.check_gates(path):
			if verdict != "ok":
				failed.add(label)
		for r in bench_lib.load_records(path):
			if r.get("status") == "ok":
				r["_source"] = os.path.basename(path)
				records.append(r)
	if failed:
		print(f"# excluded {len(failed)} gate-failing configuration(s): "
			  f"{sorted(failed)}", file=sys.stderr)
	return records


def _series(records, task, code, mode, boxsize=None, bin_slop="any", threads=1,
			num_jk=0):
	"""Records for one curve, as a size-ordered list.

	``num_jk`` defaults to 0 and must be filtered on: the jackknife sweep
	measures the same (task, N, density_mode, threads) points with jackknife
	enabled, and without this filter those land in the cross-package comparison
	against reference codes that compute no jackknife at all — inflating
	measureia's time by ~25% in whichever bins happen to be picked.
	"""
	sel = [r for r in records
		   if r["task"] == task and r["code"] == code
		   and r["density_mode"] == mode and r["threads"] == threads
		   and r.get("boxsize") == boxsize
		   and r.get("num_jk", 0) == num_jk
		   and (bin_slop == "any" or r.get("bin_slop") == bin_slop)]
	sel.sort(key=lambda r: r["n_shape"])
	return sel


def plot_scaling(records, mode, out_dir):
	tasks = [t for t in ("box_w", "lc_w", "box_multipoles", "lc_multipoles")
			 if any(r["task"] == t and r["density_mode"] == mode for r in records)]
	if not tasks:
		return None
	fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.0),
							 sharey=True, squeeze=False)
	for ax, task in zip(axes[0], tasks):
		for code in run_sweep.TASK_CODES[task]:
			slop = 0 if code == "treecorr" else "any"
			sel = _series(records, task, code, mode, bin_slop=slop)
			if not sel:
				continue
			n = [r.get("n_shape_actual", r["n_shape"]) for r in sel]
			t = [r["t_min"] for r in sel]
			ax.plot(n, t, marker="o", label=CODE_LABELS.get(code, code))
		# a linear-in-N guide anchored on the first measureia point
		mia = _series(records, task, "measureia", mode)
		if len(mia) > 1:
			n0, t0 = mia[0].get("n_shape_actual", mia[0]["n_shape"]), mia[0]["t_min"]
			nn = np.array([r.get("n_shape_actual", r["n_shape"]) for r in mia], dtype=float)
			ax.plot(nn, t0 * nn / n0, ls=":", color="grey", lw=1,
					label=r"$\propto N$")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xlabel(r"$N_{\rm shape}$")
		ax.set_title(TASK_LABELS.get(task, task), fontsize=10)
		ax.grid(alpha=0.3, which="both")
	axes[0][0].set_ylabel("wall time [s]")
	axes[0][-1].legend(fontsize=8)
	fig.suptitle(MODE_LABELS.get(mode, mode), fontsize=11)
	fig.tight_layout()
	path = os.path.join(out_dir, f"scaling_{mode}.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def plot_internal(records, out_dir):
	"""measureia's own w vs multipoles, with candidates per galaxy alongside."""
	pairs = [("box_w", "box_multipoles", "Box"), ("lc_w", "lc_multipoles", "Lightcone")]
	pairs = [p for p in pairs
			 if _series(records, p[0], "measureia", "fixed_density")
			 and _series(records, p[1], "measureia", "fixed_density")]
	if not pairs:
		return None
	fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.2), squeeze=False)
	for ax, (tw, tm, name) in zip(axes[0], pairs):
		w = _series(records, tw, "measureia", "fixed_density")
		m = _series(records, tm, "measureia", "fixed_density")
		common = sorted(set(r["n_shape"] for r in w) & set(r["n_shape"] for r in m))
		wd = {r["n_shape"]: r for r in w}
		md = {r["n_shape"]: r for r in m}
		n = [wd[k].get("n_shape_actual", wd[k]["n_shape"]) for k in common]
		ax.plot(n, [wd[k]["t_min"] / md[k]["t_min"] for k in common],
				marker="o", color="C0", label="time ratio  w / multipoles")
		ax.axhline(1.0, color="grey", lw=1, ls="--")
		ax.set_xscale("log")
		ax.set_xlabel(r"$N_{\rm shape}$")
		ax.set_ylabel("time ratio  $w$ / multipoles")
		ax.set_title(f"{name}: same catalogue, two estimators", fontsize=10)
		ax.grid(alpha=0.3)

		ax2 = ax.twinx()
		cw = [wd[k].get("candidates") for k in common]
		cm = [md[k].get("candidates") for k in common]
		if all(c for c in cw) and all(c for c in cm):
			ax2.plot(n, [a / b for a, b in zip(cw, cm)], marker="s", ls="--",
					 color="C3", label="KDTree candidate ratio")
			ax2.set_ylabel("candidate ratio  $w$ / multipoles", color="C3")
			ax2.tick_params(axis="y", labelcolor="C3")
		lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
		ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
	fig.tight_layout()
	path = os.path.join(out_dir, "internal_w_vs_mult.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def plot_boxsize(records, out_dir):
	sel = [r for r in records if r.get("boxsize") and not r.get("num_jk", 0)]
	if not sel:
		return None
	fig, ax = plt.subplots(figsize=(5.5, 4.2))
	ax2 = ax.twinx()
	for task, colour in (("box_w", "C0"), ("box_multipoles", "C1")):
		rows = sorted([r for r in sel if r["task"] == task], key=lambda r: r["boxsize"])
		if not rows:
			continue
		L = [r["boxsize"] for r in rows]
		ax.plot(L, [r["t_min"] for r in rows], marker="o", color=colour,
				label=f"{TASK_LABELS.get(task, task)} — time")
		if all(r.get("candidates") for r in rows):
			ax2.plot(L, [r["candidates"] / r.get("n_shape_actual", r["n_shape"]) for r in rows],
					 marker="s", ls="--", color=colour, alpha=0.6,
					 label=f"{TASK_LABELS.get(task, task)} — candidates/galaxy")
	ax.set_xlabel(r"box size $L$ [Mpc/$h$]")
	ax.set_ylabel("wall time [s]")
	ax2.set_ylabel("KDTree candidates per galaxy")
	ax.set_title("Fixed sample, varying box depth", fontsize=10)
	ax.grid(alpha=0.3)
	lines = ax.get_lines() + ax2.get_lines()
	ax.legend(lines, [l.get_label() for l in lines], fontsize=7)
	fig.tight_layout()
	path = os.path.join(out_dir, "boxsize.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def plot_threads(records, out_dir):
	by = defaultdict(list)
	for r in records:
		# key on num_jk and N as well: a speedup curve must hold everything but
		# the thread count fixed, or it compares unlike measurements
		by[(r["task"], r["code"], r.get("num_jk", 0), r["n_shape"],
			r["density_mode"])].append(r)
	by = {k: v for k, v in by.items() if len({r["threads"] for r in v}) > 1}
	if not by:
		return None
	fig, ax = plt.subplots(figsize=(5.5, 4.2))
	for (task, code, njk, n, _mode), rows in sorted(by.items()):
		rows.sort(key=lambda r: r["threads"])
		base = next((r["t_min"] for r in rows if r["threads"] == 1), None)
		if base is None:
			continue
		jk = f", jk={njk}" if njk else ""
		ax.plot([r["threads"] for r in rows], [base / r["t_min"] for r in rows],
				marker="o",
				label=f"{TASK_LABELS.get(task, task)}{jk}, N={n // 1000}k")
	t = sorted({r["threads"] for rows in by.values() for r in rows})
	ax.plot(t, t, ls=":", color="grey", lw=1, label="ideal")
	ax.set_xlabel("threads (measureia: num_nodes)")
	ax.set_ylabel("speedup over 1 thread")
	ax.grid(alpha=0.3)
	ax.legend(fontsize=7)
	fig.tight_layout()
	path = os.path.join(out_dir, "threads.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def plot_memory(records, out_dir):
	fig, ax = plt.subplots(figsize=(5.5, 4.2))
	drawn = False
	for task in ("box_w", "lc_w"):
		for code in run_sweep.TASK_CODES.get(task, []):
			sel = _series(records, task, code, "fixed_density",
						  bin_slop=0 if code == "treecorr" else "any")
			if len(sel) < 2:
				continue
			ax.plot([r.get("n_shape_actual", r["n_shape"]) for r in sel], [r["peak_rss_mb"] for r in sel],
					marker="o",
					label=f"{TASK_LABELS.get(task, task)} — {CODE_LABELS.get(code, code)}")
			drawn = True
	if not drawn:
		plt.close(fig)
		return None
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel(r"$N_{\rm shape}$")
	ax.set_ylabel("peak RSS [MB]")
	ax.grid(alpha=0.3, which="both")
	ax.legend(fontsize=7)
	fig.tight_layout()
	path = os.path.join(out_dir, "memory.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def markdown_table(records):
	"""Summary table for paper/paper.md and docs/.

	Reports peak memory alongside wall time. Memory is not a footnote here: it is
	the axis on which measureia does best, and for a user deciding whether a
	catalogue fits on their machine it matters at least as much as the runtime.
	"""
	lines = []
	env = records[0]["env"] if records else {}
	pk = env.get("packages", {})
	lines.append(f"Measured on {env.get('cpu', '?')} "
				 f"({env.get('cores_logical', '?')} logical cores), "
				 f"Python {env.get('python', '?')}, "
				 f"measureia {pk.get('measureia', '?')}, "
				 f"halotools {pk.get('halotools', '?')}, "
				 f"TreeCorr {pk.get('treecorr', '?')}. "
				 f"Single thread, fixed number density, best of "
				 f"{records[0].get('repeats', '?')} runs. Peak RSS includes the "
				 f"~160 MB interpreter and import baseline.\n")
	lines.append("| task | $N_{\\rm shape}$ | measureia | reference | time ratio "
				 "| measureia peak RSS | reference peak RSS | memory ratio |")
	lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
	for task in ("box_w", "lc_w"):
		codes = [c for c in run_sweep.TASK_CODES.get(task, []) if c != "measureia"]
		if not codes:
			continue
		ref_code = codes[0]
		mia = {r["n_shape"]: r for r in _series(records, task, "measureia", "fixed_density")}
		ref = {r["n_shape"]: r for r in
			   _series(records, task, ref_code, "fixed_density",
					   bin_slop=0 if ref_code == "treecorr" else "any")}
		for n in sorted(set(mia) & set(ref)):
			a, b = mia[n], ref[n]
			ma, mb = a["peak_rss_mb"], b["peak_rss_mb"]
			lines.append(
				f"| {TASK_LABELS.get(task, task)} "
				f"| {a.get('n_shape_actual', a['n_shape']):,} "
				f"| {a['t_min']:.3f} s | {b['t_min']:.3f} s "
				f"({CODE_LABELS.get(ref_code, ref_code)}) "
				f"| {a['t_min'] / b['t_min']:.1f}x "
				f"| {ma:,.0f} MB | {mb:,.0f} MB | {mb / ma:.1f}x |")

	# scaling slopes: the number that says whether a gap is a constant factor or
	# a scaling defect, which is a different thing to report and to fix
	lines.append("\nScaling at fixed number density, where the pair count is linear in N "
				 "so the ideal slope is 1.0:\n")
	lines.append("| code / task | $d\\log t / d\\log N$ |")
	lines.append("|---|---:|")
	for task in ("box_w", "box_multipoles", "lc_w", "lc_multipoles"):
		for code in run_sweep.TASK_CODES.get(task, []):
			sel = _series(records, task, code, "fixed_density",
						  bin_slop=0 if code == "treecorr" else "any")
			if len(sel) < 2:
				continue
			n0 = sel[0].get("n_shape_actual", sel[0]["n_shape"])
			n1 = sel[-1].get("n_shape_actual", sel[-1]["n_shape"])
			slope = (np.log(sel[-1]["t_min"] / sel[0]["t_min"]) / np.log(n1 / n0))
			lines.append(f"| {CODE_LABELS.get(code, code)} — "
						 f"{TASK_LABELS.get(task, task)} | {slope:.2f} |")
	return "\n".join(lines)


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("results", nargs="+", help="JSONL files from run_sweep.py")
	p.add_argument("--out-dir", default=os.path.join(_HERE, "results"))
	args = p.parse_args()

	records = load(args.results)
	if not records:
		raise SystemExit("no successful records found")
	os.makedirs(args.out_dir, exist_ok=True)

	made = []
	for mode in ("fixed_density", "fixed_volume"):
		made.append(plot_scaling(records, mode, args.out_dir))
	made.append(plot_internal(records, args.out_dir))
	made.append(plot_boxsize(records, args.out_dir))
	made.append(plot_threads(records, args.out_dir))
	made.append(plot_memory(records, args.out_dir))

	print(markdown_table(records))
	print("\nfigures written:", file=sys.stderr)
	for path in made:
		if path:
			print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
	main()
