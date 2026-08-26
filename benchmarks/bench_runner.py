"""Run exactly one benchmark point and emit one JSON result record.

Invoked as a fresh subprocess by run_sweep.py, once per (task, code, size,
thread count, ...) point. A subprocess per point is not an implementation
detail — it is required for the numbers to mean anything:

- Thread-count environment variables (OMP_NUM_THREADS and friends) only take
  effect if they are set before numpy / TreeCorr are imported. TreeCorr's
  num_threads otherwise defaults to every core on the machine.
- measureia's multiprocessing paths call
  ``mp.set_start_method("spawn", force=True)``, which is hostile to being
  driven repeatedly inside one long-lived process.
- Peak RSS (``resource.getrusage``) is only meaningful for a process that did
  one job.

Usage (normally via run_sweep.py):

	python bench_runner.py '<config json>'

The config is the benchmark point; the record printed to stdout after the
``@@RESULT@@`` marker is that config plus timings, peak memory, candidate
counts and a correctness verdict. Everything the measured codes print is
redirected to devnull while the clock runs, so console I/O is never timed.
"""

import contextlib
import io
import json
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "validation"))

import bench_lib

# --- configuration shared with the validation scripts -----------------------
# Imported rather than restated so a timed run uses the exact binning that the
# cross-package agreement was established on.
import run_box_halotools as V_BOX
import run_lightcone_treecorr as V_LC

# Correctness gate tolerances, matching tests/test_validation_references.py.
GATE_HALOTOOLS = dict(rtol=1e-10, atol=0.0)
GATE_TREECORR = dict(rtol=5e-3, atol=0.05)


@contextlib.contextmanager
def _quiet():
	"""Silence the measured code's own console output while timing."""
	with open(os.devnull, "w") as devnull:
		with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
			yield


def _scratch_file(scratch_dir, name):
	os.makedirs(scratch_dir, exist_ok=True)
	path = os.path.join(scratch_dir, name)
	if os.path.exists(path):
		os.remove(path)
	return path


# ----------------------------------------------------------------------------
# box: measureia vs halotools
# ----------------------------------------------------------------------------

def _box_data(cfg):
	mock = bench_lib.box_mock_for(
		cfg["n_shape"], cfg["density_mode"], boxsize=cfg.get("boxsize"),
		density=cfg.get("density"),
	)
	data = {k: mock[k] for k in
			("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")}
	return mock, data


def _make_box_measureia(cfg):
	"""Zero-argument callable running one MeasureIABox measurement."""
	from measureia import MeasureIABox

	mock, data = _box_data(cfg)
	scratch = cfg["scratch_dir"]
	multipoles = cfg["task"].endswith("multipoles")
	# temp_file_path=False selects the brute path with no temp storage (what the
	# validation script uses at num_nodes=1); the 'tree' variant keeps temp
	# storage on at every thread count so the thread axis compares like with like.
	temp = False if cfg.get("variant") == "brute" else scratch
	if cfg["threads"] > 1 and temp is False:
		raise ValueError("variant='brute' cannot use threads>1 (mp needs temp storage)")

	def run():
		out = _scratch_file(scratch, f"bench_box_{os.getpid()}.hdf5")
		ia = MeasureIABox(
			dict(data), out, simulation=None, snapshot=None,
			separation_limits=V_BOX.RP_LIMS,
			num_bins_r=V_BOX.NUM_BINS_RP,
			num_bins_pi=cfg.get("num_bins_pi", V_BOX.NUM_BINS_PI),
			pi_max=V_BOX.PI_MAX,
			boxsize=mock["boxsize"],
			num_nodes=cfg["threads"],
		)
		meth = ia.measure_xi_multipoles if multipoles else ia.measure_xi_w
		meth(V_BOX.DATASET, "both", cfg.get("num_jk", 0), temp_file_path=temp)
		import h5py
		with h5py.File(out, "r") as f:
			key = "multipoles_g_plus" if multipoles else "w_g_plus"
			gg_key = "multipoles_gg" if multipoles else "w_gg"
			gp = f[f"{key}/{V_BOX.DATASET}"][:]
			gg = f[f"{gg_key}/{V_BOX.DATASET}"][:]
		os.remove(out)
		return {"g_plus": gp, "gg": gg}

	return run, mock


def _make_box_halotools(cfg):
	from halotools.mock_observables import wp
	from halotools.mock_observables.ia_correlations import gi_plus_projected
	from measureia.mocks import halotools_inputs

	mock, _ = _box_data(cfg)
	shapes, orientations, e, density, period = halotools_inputs(mock)
	rp_bins = np.logspace(
		np.log10(V_BOX.RP_LIMS[0]), np.log10(V_BOX.RP_LIMS[1]), V_BOX.NUM_BINS_RP + 1
	)
	nt = cfg["threads"]

	def run():
		gp = gi_plus_projected(shapes, orientations, e, density, rp_bins,
							   V_BOX.PI_MAX, period=period, num_threads=nt)
		gg = wp(density, rp_bins, V_BOX.PI_MAX, sample2=shapes, period=period,
				num_threads=nt, do_auto=False, do_cross=True)
		return {"g_plus": gp, "gg": gg}

	return run, mock


# ----------------------------------------------------------------------------
# lightcone: measureia vs treecorr
# ----------------------------------------------------------------------------

def _lc_data(cfg):
	import pyccl as ccl

	data, randoms, info = bench_lib.lightcone_mock_for(
		cfg["n_shape"], cfg["density_mode"], density=cfg.get("density"))
	for d in (data, randoms):
		d["Redshift"] = 1.0 / ccl.scale_factor_of_chi(V_LC.COSMOLOGY, d.pop("r_com")) - 1.0
		d["Redshift_shape_sample"] = (
			1.0 / ccl.scale_factor_of_chi(V_LC.COSMOLOGY, d.pop("r_com_shape_sample")) - 1.0
		)
	dist = {
		"D": ccl.comoving_radial_distance(V_LC.COSMOLOGY, 1 / (1 + data["Redshift"])),
		"S": ccl.comoving_radial_distance(V_LC.COSMOLOGY, 1 / (1 + data["Redshift_shape_sample"])),
		"R_D": ccl.comoving_radial_distance(V_LC.COSMOLOGY, 1 / (1 + randoms["Redshift"])),
		"R_S": ccl.comoving_radial_distance(V_LC.COSMOLOGY, 1 / (1 + randoms["Redshift_shape_sample"])),
	}
	return data, randoms, info, dist


def _make_lc_measureia(cfg):
	from measureia import MeasureIALightcone

	data, randoms, info, _ = _lc_data(cfg)
	scratch = cfg["scratch_dir"]
	multipoles = cfg["task"].endswith("multipoles")
	tree = cfg.get("variant", "tree") != "brute"

	def run():
		out = _scratch_file(scratch, f"bench_lc_{os.getpid()}.hdf5")
		ia = MeasureIALightcone(
			data=dict(data), randoms_data=dict(randoms), output_file_name=out,
			separation_limits=V_LC.RP_LIMS,
			num_bins_r=V_LC.NUM_BINS_RP,
			num_bins_pi=cfg.get("num_bins_pi", V_LC.NUM_BINS_PI),
			pi_max=V_LC.PI_MAX,
			num_nodes=cfg["threads"],
		)
		meth = ia.measure_xi_multipoles if multipoles else ia.measure_xi_w
		kwargs = dict(tree=tree, cosmology=V_LC.COSMOLOGY, over_h=False,
					  temp_file_path=scratch if scratch.endswith(os.sep) else scratch + os.sep)
		if cfg.get("num_jk"):
			kwargs["num_jk"] = cfg["num_jk"]
		meth("galaxies", V_LC.DATASET, "both", **kwargs)
		import h5py
		with h5py.File(out, "r") as f:
			key = "multipoles_g_plus" if multipoles else "w_g_plus"
			gg_key = "multipoles_gg" if multipoles else "w_gg"
			gp = f[f"{key}/{V_LC.DATASET}"][:]
			gg = f[f"{gg_key}/{V_LC.DATASET}"][:]
		os.remove(out)
		return {"g_plus": gp, "gg": gg}

	return run, _lc_meta(data, randoms, info)


def _lc_meta(data, randoms, info):
	"""Sample sizes for a lightcone record.

	measureia.mocks' lightcone `info` dict carries only the mock parameters, not
	the catalogue sizes, so name them here — n_shape_actual is the x-axis of
	every scaling plot.
	"""
	return {
		"n_shape_actual": int(len(data["RA_shape_sample"])),
		"n_position": int(len(data["RA"])),
		"n_randoms_position": int(len(randoms["RA"])),
		"n_randoms_shape": int(len(randoms["RA_shape_sample"])),
		"seed": info.get("seed"),
	}


def _make_lc_treecorr(cfg):
	"""TreeCorr side: the full 6-runs-per-pi-slab estimator reconstruction.

	This is the honest unit of comparison — obtaining the same w_g+ and w_gg
	that one measureia call returns needs NG(D,S), NG(R_D,S), NN(D,S),
	NN(D,R_S), NN(R_D,S) and NN(R_D,R_S) for every signed pi slab, plus the
	user-side estimator assembly, all of which run_lightcone_treecorr already
	implements and is reused verbatim.
	"""
	import treecorr

	data, randoms, info, dist = _lc_data(cfg)
	r_bins = np.logspace(
		np.log10(V_LC.RP_LIMS[0]), np.log10(V_LC.RP_LIMS[1]), V_LC.NUM_BINS_RP + 1
	)
	n_pi = cfg.get("num_bins_pi", V_LC.NUM_BINS_PI)
	pi_bins = np.linspace(-V_LC.PI_MAX, V_LC.PI_MAX, n_pi + 1)
	bin_slop = cfg.get("bin_slop", 0)
	nt = cfg["threads"]

	def run():
		# reuse the validation implementation, overriding only bin_slop/threads
		cfg_kwargs = dict(nbins=V_LC.NUM_BINS_RP, min_sep=r_bins[0], max_sep=r_bins[-1],
						  metric="Rperp")
		if bin_slop is not None:
			cfg_kwargs["bin_slop"] = bin_slop

		pcat = treecorr.Catalog(ra=data["RA"], dec=data["DEC"], r=dist["D"],
								w=data["weight"], ra_units="deg", dec_units="deg")
		scat = treecorr.Catalog(ra=data["RA_shape_sample"], dec=data["DEC_shape_sample"],
								r=dist["S"], w=data["weight_shape_sample"],
								g1=-data["e1"], g2=-data["e2"],
								ra_units="deg", dec_units="deg")
		rdcat = treecorr.Catalog(ra=randoms["RA"], dec=randoms["DEC"], r=dist["R_D"],
								 w=randoms["weight"], ra_units="deg", dec_units="deg")
		rscat = treecorr.Catalog(ra=randoms["RA_shape_sample"], dec=randoms["DEC_shape_sample"],
								 r=dist["R_S"], w=randoms["weight_shape_sample"],
								 ra_units="deg", dec_units="deg")
		N_D, N_S, N_RD, N_RS = pcat.sumw, scat.sumw, rdcat.sumw, rscat.sumw

		xi_gp = np.zeros((V_LC.NUM_BINS_RP, n_pi))
		xi_gg = np.zeros((V_LC.NUM_BINS_RP, n_pi))
		for i in range(n_pi):
			slab = dict(min_rpar=pi_bins[i], max_rpar=pi_bins[i + 1])
			ng = treecorr.NGCorrelation(**cfg_kwargs, **slab)
			ng.process(pcat, scat, num_threads=nt)
			SpD = ng.xi * ng.weight
			ngr = treecorr.NGCorrelation(**cfg_kwargs, **slab)
			ngr.process(rdcat, scat, num_threads=nt)
			SpR = ngr.xi * ngr.weight

			def _nn(c1, c2):
				nn = treecorr.NNCorrelation(**cfg_kwargs, **slab)
				nn.process(c1, c2, num_threads=nt)
				return nn.weight

			DD = _nn(pcat, scat)
			RD = _nn(pcat, rscat)
			SR = _nn(rdcat, scat)
			RR = _nn(rdcat, rscat)
			norm_RR = RR / (N_RD * N_RS)
			with np.errstate(invalid="ignore", divide="ignore"):
				xi_gp[:, i] = (SpD / (N_S * N_D) - SpR / (N_S * N_RD)) / norm_RR
				xi_gg[:, i] = (DD / (N_D * N_S) - RD / (N_D * N_RS)
							   - SR / (N_S * N_RD)) / norm_RR + 1
		dpi = np.diff(pi_bins)
		return {"g_plus": np.sum(xi_gp * dpi, axis=1), "gg": np.sum(xi_gg * dpi, axis=1)}

	return run, _lc_meta(data, randoms, info)


BUILDERS = {
	("box_w", "measureia"): _make_box_measureia,
	("box_w", "halotools"): _make_box_halotools,
	("box_multipoles", "measureia"): _make_box_measureia,
	("lc_w", "measureia"): _make_lc_measureia,
	("lc_w", "treecorr"): _make_lc_treecorr,
	("lc_multipoles", "measureia"): _make_lc_measureia,
}


# ----------------------------------------------------------------------------
# I/O probe
# ----------------------------------------------------------------------------

def io_probe(scratch_dir, n_floats=1_000_000):
	"""Time an HDF5 write+read of a representative array in the scratch directory.

	measureia writes its output (and, on the tree/mp paths, a temp offload) to
	disk; halotools and TreeCorr write nothing. Rather than pretend that cost
	away or try to unpick it from inside the call, it is measured separately
	here, in the same directory, so the cost of an NFS scratch is visible as
	its own number.
	"""
	import h5py

	path = _scratch_file(scratch_dir, f"io_probe_{os.getpid()}.hdf5")
	arr = np.random.default_rng(0).random(n_floats)
	_, w = bench_lib.time_repeats(
		lambda: _h5_write(path, arr), repeats=3, warmup=1)
	_, r = bench_lib.time_repeats(lambda: _h5_read(path), repeats=3, warmup=1)
	os.remove(path)
	return {"n_floats": n_floats, "write_min": w["t_min"], "read_min": r["t_min"]}


def _h5_write(path, arr):
	import h5py
	with h5py.File(path, "w") as f:
		f["data"] = arr


def _h5_read(path):
	import h5py
	with h5py.File(path, "r") as f:
		return f["data"][:]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run_point(cfg):
	key = (cfg["task"], cfg["code"])
	if key not in BUILDERS:
		raise ValueError(f"bench_runner: no builder for {key}")
	builder = BUILDERS[key]

	record = dict(cfg)
	with _quiet():
		fn, meta = builder(cfg)

	# Candidate instrumentation (measureia only — it is the only code whose
	# candidate selection is visible from Python). Counting is folded into the
	# warmup call rather than run as an extra pass, so it costs nothing and
	# every measureia point carries a candidate count.
	candidates = None
	warmup = cfg.get("warmup", 1)
	if cfg["code"] == "measureia" and warmup > 0:
		with _quiet(), bench_lib.CandidateCounter() as cc:
			fn()
		candidates = cc.candidates
		warmup -= 1

	with _quiet():
		result, stats = bench_lib.time_repeats(
			fn, repeats=cfg.get("repeats", 5), warmup=warmup
		)

	record.update(stats)
	record["peak_rss_mb"] = bench_lib.peak_rss_mb()
	record["candidates"] = candidates
	record["status"] = "ok"
	record["result_g_plus"] = None if result["g_plus"] is None else result["g_plus"].tolist()
	record["result_gg"] = result["gg"].tolist()
	record["env"] = bench_lib.environment(cfg.get("scratch_dir"))
	if isinstance(meta, dict):
		if "n_shape_actual" in meta:
			record.update({k: v for k, v in meta.items() if k != "seed"})
		elif "Position_shape_sample" in meta:
			record["n_shape_actual"] = int(len(meta["Position_shape_sample"]))
		if "boxsize" in meta:
			record["boxsize_actual"] = float(meta["boxsize"])
			# The box responsivity R = 1 - <e^2>/2. measureia divides S+ terms by
			# 2R and halotools does not, so the correctness gate needs it.
			from measureia.mocks import responsivity
			record["responsivity_R"] = float(responsivity(meta))
	return record


def main():
	cfg = json.loads(sys.argv[1])
	try:
		record = run_point(cfg)
	except Exception:
		record = dict(cfg)
		record["status"] = "error"
		record["traceback"] = traceback.format_exc()
	print("@@RESULT@@" + json.dumps(record))


if __name__ == "__main__":
	main()
