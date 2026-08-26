"""Capture and compare every output array across a matrix of configurations.

Safety net for changes to `pair_kernel.accumulate` (see FINDINGS.md F1). Run it
on the unmodified code to capture a baseline, make the change, then run it again
in compare mode. It walks the **entire** output HDF5 for every configuration and
compares each dataset **bitwise**, so a change that perturbs any grid, any
jackknife realisation, any bin coordinate or any per-galaxy array is caught —
not just the final w(rp) or multipoles.

    python benchmarks/bitidentity_matrix.py capture --out baseline.npz
    # ... change src/measureia ...
    python benchmarks/bitidentity_matrix.py compare --ref baseline.npz

The matrix covers every code path the change touches: box and lightcone, w and
multipoles, brute / tree / multiprocessing backends, with and without jackknife.
The brute backend is included deliberately — it does *not* go through the
candidate-selection code being changed, so it acts as a control: if brute moves,
something far more serious is wrong than the intended edit.

Comparison is exact. Floats are compared with `np.array_equal` on the raw
values plus a `tobytes()` check, so +0.0/-0.0 and NaN payload differences are
not silently tolerated. NaNs are compared as equal-by-position, since NaN != NaN
would otherwise fail on legitimately undefined bins.
"""

import argparse
import contextlib
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_lib

SCRATCH = os.path.join(_HERE, "_scratch", "bitid")

# Small enough to run the whole matrix quickly, large enough that every branch
# has real pair counts. num_jk for the box must be a perfect cube.
N_SHAPE = 2_400
NUM_JK_BOX = 8
NUM_JK_LC = 4


def _quiet():
	devnull = open(os.devnull, "w")
	return contextlib.redirect_stdout(devnull)


def _walk(path):
	"""Every dataset in the file, as {hdf5 path: ndarray}."""
	out = {}

	def visit(name, obj):
		if isinstance(obj, h5py.Dataset):
			out[name] = obj[()]

	with h5py.File(path, "r") as f:
		f.visititems(visit)
	return out


def _fresh(name):
	os.makedirs(SCRATCH, exist_ok=True)
	p = os.path.join(SCRATCH, name)
	if os.path.exists(p):
		os.remove(p)
	return p


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------

def _box_masks(mock):
	"""A deliberately non-contiguous selection, so masking cannot accidentally
	coincide with a prefix (the failure mode a past P0 bug had)."""
	n_pos = len(mock["Position"])
	n_shape = len(mock["Position_shape_sample"])
	sel_pos = np.zeros(n_pos, dtype=bool)
	sel_pos[::3] = True
	sel_shape = np.zeros(n_shape, dtype=bool)
	sel_shape[::2] = True
	return {"Position": sel_pos, "Position_shape_sample": sel_shape,
			"Axis_Direction": sel_shape, "q": sel_shape}


def box_case(label, *, multipoles, backend, num_jk, num_nodes,
			 rp_cut=None, masks=False, sep_limits=(0.5, 20.0)):
	from measureia import MeasureIABox

	mock = bench_lib.box_mock_for(N_SHAPE, "fixed_volume")
	data = {k: mock[k] for k in
			("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")}
	out = _fresh(f"{label}.hdf5")
	temp = False if backend == "brute" else SCRATCH + os.sep
	with _quiet():
		ia = MeasureIABox(
			data, out, simulation=None, snapshot=None,
			separation_limits=list(sep_limits), num_bins_r=6, num_bins_pi=8,
			pi_max=20.0, boxsize=mock["boxsize"], num_nodes=num_nodes,
		)
		kwargs = {"temp_file_path": temp}
		if masks:
			kwargs["masks"] = _box_masks(mock)
		if multipoles:
			if rp_cut is not None:
				kwargs["rp_cut"] = rp_cut
			ia.measure_xi_multipoles("bitid", "both", num_jk, **kwargs)
		else:
			ia.measure_xi_w("bitid", "both", num_jk, **kwargs)
	arrays = _walk(out)
	os.remove(out)
	return arrays


def galaxy_contrib_case(label, *, num_jk, statistic, rp_cut=None):
	"""measure_galaxy_contributions -- this branch's new feature, which drives
	pair_kernel.accumulate through its per_galaxy branch."""
	from measureia import MeasureIABox

	mock = bench_lib.box_mock_for(N_SHAPE, "fixed_volume")
	data = {k: mock[k] for k in
			("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")}
	out = _fresh(f"{label}.hdf5")
	with _quiet():
		ia = MeasureIABox(
			data, out, simulation=None, snapshot=None,
			separation_limits=[0.5, 20.0], num_bins_r=6, num_bins_pi=8,
			pi_max=20.0, boxsize=mock["boxsize"], num_nodes=1,
		)
		res = ia.measure_galaxy_contributions(
			"bitid", num_jk=num_jk, statistic=statistic, rp_cut=rp_cut,
			temp_file_path=SCRATCH + os.sep, return_output=True)
	arrays = {k: np.asarray(v) for k, v in res.items()
			  if isinstance(v, (np.ndarray, list))}
	if os.path.exists(out):
		os.remove(out)
	return arrays


def lc_case(label, *, multipoles, backend, num_jk, num_nodes,
			estimator="galaxies", sep_limits=(0.5, 20.0)):
	import pyccl as ccl
	from measureia import MeasureIALightcone

	cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=0.7, sigma8=0.8, n_s=0.96)
	data, randoms, _ = bench_lib.lightcone_mock_for(N_SHAPE, "fixed_volume")
	for d in (data, randoms):
		d["Redshift"] = 1.0 / ccl.scale_factor_of_chi(cosmo, d.pop("r_com")) - 1.0
		d["Redshift_shape_sample"] = (
			1.0 / ccl.scale_factor_of_chi(cosmo, d.pop("r_com_shape_sample")) - 1.0)
	out = _fresh(f"{label}.hdf5")
	with _quiet():
		ia = MeasureIALightcone(
			data=data, randoms_data=randoms, output_file_name=out,
			separation_limits=list(sep_limits), num_bins_r=6, num_bins_pi=8,
			pi_max=20.0, num_nodes=num_nodes,
		)
		meth = ia.measure_xi_multipoles if multipoles else ia.measure_xi_w
		kwargs = dict(tree=(backend != "brute"), cosmology=cosmo, over_h=False,
					  temp_file_path=SCRATCH + os.sep, seed=99)
		if num_jk:
			kwargs["num_jk"] = num_jk
		meth(estimator, "bitid", "both", **kwargs)
	arrays = _walk(out)
	os.remove(out)
	return arrays


def build_matrix(skip_mp=False, only_mp=False):
	"""(label, callable) for every configuration to check.

	``skip_mp`` drops the multiprocessing cases, which are the only ones that
	use more than one CPU (they spawn ``num_nodes`` worker processes). The
	remaining matrix still covers both geometries, both estimators, the brute
	and tree backends and the jackknife, all in a single process.
	``only_mp`` keeps just those cases, for topping up a baseline captured
	earlier with ``--skip-mp``.
	"""
	cases = []
	if only_mp:
		skip_mp = False
	for multipoles in (False, True):
		kind = "mult" if multipoles else "w"
		# brute is the control: it bypasses the candidate-selection code entirely
		for backend in ("brute", "tree"):
			if only_mp:
				break
			for num_jk in (0, NUM_JK_BOX):
				label = f"box_{kind}_{backend}_jk{num_jk}"
				cases.append((label, lambda l=label, m=multipoles, b=backend, j=num_jk:
							  box_case(l, multipoles=m, backend=b, num_jk=j, num_nodes=1)))
		# multiprocessing: a different code path onto the same kernel
		if not skip_mp:
			for num_jk in (0, NUM_JK_BOX):
				label = f"box_{kind}_mp_jk{num_jk}"
				cases.append((label, lambda l=label, m=multipoles, j=num_jk:
							  box_case(l, multipoles=m, backend="tree", num_jk=j, num_nodes=2)))
		for backend in ("brute", "tree"):
			if only_mp:
				break
			for num_jk in (0, NUM_JK_LC):
				label = f"lc_{kind}_{backend}_jk{num_jk}"
				cases.append((label, lambda l=label, m=multipoles, b=backend, j=num_jk:
							  lc_case(l, multipoles=m, backend=b, num_jk=j, num_nodes=1)))
		if not skip_mp:
			for num_jk in (0, NUM_JK_LC):
				label = f"lc_{kind}_mp_jk{num_jk}"
				cases.append((label, lambda l=label, m=multipoles, j=num_jk:
							  lc_case(l, multipoles=m, backend="tree", num_jk=j, num_nodes=2)))

	if not only_mp:
		# ---- paths the plain matrix above does not reach -------------------
		# rp_cut lives *inside* the same bin_pairs mask the change now relies on
		for rp_cut in (1.0, 3.0):
			label = f"box_mult_rpcut{rp_cut:g}"
			cases.append((label, lambda l=label, c=rp_cut:
						  box_case(l, multipoles=True, backend="tree",
								   num_jk=0, num_nodes=1, rp_cut=c)))
		cases.append(("box_mult_rpcut_jk", lambda:
					  box_case("box_mult_rpcut_jk", multipoles=True, backend="tree",
							   num_jk=NUM_JK_BOX, num_nodes=1, rp_cut=2.0)))

		# masks: a non-contiguous selection, with and without jackknife
		for kind, mult in (("w", False), ("mult", True)):
			for num_jk in (0, NUM_JK_BOX):
				label = f"box_{kind}_masked_jk{num_jk}"
				cases.append((label, lambda l=label, m=mult, j=num_jk:
							  box_case(l, multipoles=m, backend="tree",
									   num_jk=j, num_nodes=1, masks=True)))

		# measure_galaxy_contributions -- the per_galaxy branch of accumulate
		for statistic in ("multipoles", "w"):
			for num_jk in (0, NUM_JK_BOX):
				label = f"box_contrib_{statistic}_jk{num_jk}"
				cases.append((label, lambda l=label, st=statistic, j=num_jk:
							  galaxy_contrib_case(l, num_jk=j, statistic=st)))

		# the lightcone 'clusters' estimator (the matrix above is all 'galaxies')
		for kind, mult in (("w", False), ("mult", True)):
			for num_jk in (0, NUM_JK_LC):
				label = f"lc_{kind}_clusters_jk{num_jk}"
				cases.append((label, lambda l=label, m=mult, j=num_jk:
							  lc_case(l, multipoles=m, backend="tree", num_jk=j,
									  num_nodes=1, estimator="clusters")))

		# The stressing case: r_min/r_max = 1/4 instead of 1/40, so the inner
		# ball the removed query used to discard is a large fraction of the
		# candidates rather than a sliver. If dropping that query is ever going
		# to change a number, it will show up here first.
		for kind, mult in (("w", False), ("mult", True)):
			for num_jk in (0, NUM_JK_BOX):
				label = f"box_{kind}_rmin5_jk{num_jk}"
				cases.append((label, lambda l=label, m=mult, j=num_jk:
							  box_case(l, multipoles=m, backend="tree", num_jk=j,
									   num_nodes=1, sep_limits=(5.0, 20.0))))
			label = f"lc_{kind}_rmin5"
			cases.append((label, lambda l=label, m=mult:
						  lc_case(l, multipoles=m, backend="tree", num_jk=0,
								  num_nodes=1, sep_limits=(5.0, 20.0))))
	return cases


# ---------------------------------------------------------------------------
# capture / compare
# ---------------------------------------------------------------------------

def capture(out_path, skip_mp=False, only_mp=False):
	store = {}
	matrix = build_matrix(skip_mp, only_mp)
	for label, fn in matrix:
		print(f"  {label} ... ", end="", flush=True)
		arrays = fn()
		for key, val in arrays.items():
			store[f"{label}||{key}"] = val
		print(f"{len(arrays)} datasets")
	np.savez(out_path, **store)
	print(f"\ncaptured {len(store)} arrays across {len(matrix)} configurations")
	print(f"-> {out_path}")


def _identical(a, b):
	"""Exact equality, tolerating NaN in the same positions."""
	a, b = np.asarray(a), np.asarray(b)
	if a.shape != b.shape or a.dtype != b.dtype:
		return False, "shape/dtype"
	if a.dtype.kind in "fc":
		na, nb = np.isnan(a), np.isnan(b)
		if not np.array_equal(na, nb):
			return False, "NaN pattern"
		if not np.array_equal(a[~na], b[~nb]):
			return False, "values"
		# catch +0.0 vs -0.0, which compares equal above
		if a[~na].tobytes() != b[~nb].tobytes():
			return False, "signed zero / bit pattern"
		return True, ""
	return (np.array_equal(a, b), "" if np.array_equal(a, b) else "values")


def compare(ref_path, skip_mp=False, only_mp=False):
	ref = np.load(ref_path, allow_pickle=False)
	ref_keys = set(ref.files)
	n_ok = n_bad = 0
	failures = []
	for label, fn in build_matrix(skip_mp, only_mp):
		arrays = fn()
		bad = []
		for key, val in arrays.items():
			full = f"{label}||{key}"
			if full not in ref_keys:
				bad.append((key, "missing from baseline"))
				continue
			ok, why = _identical(val, ref[full])
			if ok:
				n_ok += 1
			else:
				n_bad += 1
				bad.append((key, why))
		missing = {k for k in ref_keys if k.startswith(f"{label}||")} - {
			f"{label}||{k}" for k in arrays}
		for k in sorted(missing):
			bad.append((k.split("||", 1)[1], "absent after change"))
		mark = "OK  " if not bad else "FAIL"
		print(f"  {mark} {label}  ({len(arrays)} datasets)")
		for key, why in bad:
			print(f"         - {key}: {why}")
		failures.extend((label, k, w) for k, w in bad)

	print(f"\n{n_ok} arrays bit-identical, {n_bad} differing")
	if failures:
		print(f"\nFAILED — {len(failures)} dataset(s) changed:")
		for label, key, why in failures:
			print(f"  {label}  {key}  [{why}]")
		return 1
	print("\nPASS — every dataset in every configuration is bit-identical.")
	return 0


def main():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = p.add_subparsers(dest="mode", required=True)
	c = sub.add_parser("capture")
	c.add_argument("--out", default=os.path.join(SCRATCH, "baseline.npz"))
	c.add_argument("--skip-mp", action="store_true",
				   help="omit the multiprocessing cases (keeps the run on one CPU)")
	c.add_argument("--only-mp", action="store_true",
				   help="run ONLY the multiprocessing cases")
	d = sub.add_parser("compare")
	d.add_argument("--ref", default=os.path.join(SCRATCH, "baseline.npz"))
	d.add_argument("--skip-mp", action="store_true",
				   help="omit the multiprocessing cases (keeps the run on one CPU)")
	d.add_argument("--only-mp", action="store_true",
				   help="run ONLY the multiprocessing cases")
	args = p.parse_args()

	os.makedirs(SCRATCH, exist_ok=True)
	if args.mode == "capture":
		capture(args.out, args.skip_mp, args.only_mp)
		return 0
	return compare(args.ref, args.skip_mp, args.only_mp)


if __name__ == "__main__":
	sys.exit(main())
