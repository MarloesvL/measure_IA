"""Shared machinery for the measureia speed benchmarks.

Holds everything the benchmark scripts need but that says nothing about a
particular comparison: wall-clock timing with warmup and repeats, environment
capture, size-scaled mock generation, KDTree candidate-count instrumentation,
and the append-only JSONL result store.

Timing conventions used throughout (see benchmarks/README.md for the full
methodology and the fairness rules):

- One untimed warmup call, then ``repeats`` timed calls with
  ``time.perf_counter()``. Every raw time is recorded; ``t_min`` is the
  headline number (least contaminated by scheduler noise) and ``t_median``
  is reported alongside so an unstable point is visible rather than hidden.
- ``t_total`` covers arrays-in-memory to results-available, including
  measureia's HDF5 output write. ``t_compute`` excludes that write, since
  neither halotools nor TreeCorr performs one. Both are recorded; neither is
  quietly dropped.
- Peak RSS is read from ``resource.getrusage`` in the process that did the
  work, which is why every timing point runs in its own subprocess (see
  bench_runner.py).

Mock scaling (``box_mock_for`` / ``lightcone_mock_for``) drives the existing
``measureia.mocks`` generators through their own parameters; no new mock code
is introduced, so a benchmark catalogue at the reference size is byte-identical
to the one the validation scripts use.
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

# --- reference catalogue sizes (the validation mocks' own defaults) ---
REF_BOX_N_SHAPE = 2400      # radial_alignment_box_mock: 300 centrals x 8 satellites
REF_BOX_BOXSIZE = 205.0
REF_LC_N_SHAPE = 3200       # radial_alignment_lightcone_mock: 400 centrals x 8
REF_LC_RA_RANGE = (40.0, 50.0)
REF_LC_DEC_RANGE = (-5.0, 5.0)
N_SAT = 8


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------

def time_repeats(fn, repeats=5, warmup=1):
	"""Call ``fn`` ``warmup`` times untimed, then ``repeats`` times timed.

	Parameters
	----------
	fn : callable
		Zero-argument callable performing the work. Its return value from the
		final timed call is returned, so a caller can gate on correctness.
	repeats : int, optional
		Number of timed calls.
	warmup : int, optional
		Number of untimed calls first (page faults, lazy imports, file cache).

	Returns
	-------
	tuple
		``(result, stats)`` where ``stats`` is a dict with ``times`` (list of
		every raw timed duration in seconds), ``t_min``, ``t_median``,
		``t_max`` and ``repeats``.

	"""
	for _ in range(warmup):
		fn()
	times = []
	result = None
	for _ in range(repeats):
		t0 = time.perf_counter()
		result = fn()
		times.append(time.perf_counter() - t0)
	return result, {
		"times": times,
		"t_min": float(np.min(times)),
		"t_median": float(np.median(times)),
		"t_max": float(np.max(times)),
		"repeats": repeats,
	}


def peak_rss_mb():
	"""Peak resident set size of this process, in MB.

	``ru_maxrss`` is bytes on macOS and kilobytes on Linux; normalise both.
	"""
	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	if sys.platform == "darwin":
		return rss / (1024.0 ** 2)
	return rss / 1024.0


# ----------------------------------------------------------------------------
# environment capture
# ----------------------------------------------------------------------------

def _cpu_model():
	try:
		if sys.platform == "darwin":
			return subprocess.check_output(
				["sysctl", "-n", "machdep.cpu.brand_string"], text=True
			).strip()
		with open("/proc/cpuinfo") as f:
			for line in f:
				if line.startswith("model name"):
					return line.split(":", 1)[1].strip()
	except Exception:
		pass
	return platform.processor() or "unknown"


def _filesystem_of(path):
	"""Filesystem type and mount point of ``path``, best effort.

	Recorded with every result so an NFS scratch directory is never silently
	compared against a local one.
	"""
	if not path:
		return {"fstype": "none", "mount": None}
	try:
		out = subprocess.check_output(["df", "-P", path], text=True).strip().splitlines()
		mount = out[-1].split()[-1]
		source = out[-1].split()[0]
	except Exception:
		return {"fstype": "unknown", "mount": None}
	fstype = "unknown"
	try:
		if sys.platform == "darwin":
			mnt = subprocess.check_output(["mount"], text=True)
			for line in mnt.splitlines():
				if f" on {mount} " in line and "(" in line:
					fstype = line.split("(", 1)[1].split(",", 1)[0].strip(") ")
					break
		else:
			fstype = subprocess.check_output(
				["stat", "-f", "-c", "%T", path], text=True
			).strip()
	except Exception:
		pass
	return {"fstype": fstype, "mount": mount, "source": source}


def _package_versions():
	import importlib.metadata as md
	out = {}
	for pkg in ("measureia", "halotools", "treecorr", "numpy", "scipy", "h5py", "pyccl"):
		try:
			out[pkg] = md.version(pkg)
		except Exception:
			out[pkg] = None
	return out


def _git_sha():
	try:
		return subprocess.check_output(
			["git", "rev-parse", "--short", "HEAD"],
			cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
			text=True, stderr=subprocess.DEVNULL,
		).strip()
	except Exception:
		return None


def environment(scratch_dir=None):
	"""Full environment block stored on every result record."""
	return {
		"cpu": _cpu_model(),
		"cores_logical": os.cpu_count(),
		"platform": platform.platform(),
		"python": platform.python_version(),
		"packages": _package_versions(),
		"measureia_git": _git_sha(),
		"scratch_dir": scratch_dir,
		"scratch_fs": _filesystem_of(scratch_dir),
		"omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
	}


# ----------------------------------------------------------------------------
# size-scaled mocks
# ----------------------------------------------------------------------------

# Number density of the reference mock, and of a realistic simulation catalogue.
# The mock's own density is ~30x lower than a typical simulation, which matters a
# great deal for optimisation decisions: at the mock's density each galaxy has
# ~11-19 candidate partners within r_max, so NumPy per-call overhead dominates,
# while at simulation density it has ~300-400 and the actual pair arithmetic does.
# Anything measured only at REF_DENSITY should be treated as describing the mock.
REF_BOX_DENSITY = 2700 / 205.0 ** 3          # ~3.1e-4 per (Mpc/h)^3
SIM_BOX_DENSITY = 1.0e-2                     # TNG300-like, ~389 candidates/galaxy


def boxsize_for_density(n_total, density):
    """Box side that puts ``n_total`` objects at the given number density."""
    return float((n_total / density) ** (1.0 / 3.0))


def candidates_per_galaxy(density, r_max):
    """Expected neighbours within ``r_max`` at a given number density (3D ball).

    The single most useful number when reasoning about where time goes: it sets
    the length of the arrays every per-galaxy NumPy call operates on.
    """
    return density * (4.0 / 3.0) * np.pi * r_max ** 3


def box_mock_for(n_shape, density_mode, boxsize=None, seed=42, margin=0.0,
                 density=None):
	"""Radial-alignment box mock scaled to ``n_shape`` shape-sample galaxies.

	Parameters
	----------
	n_shape : int
		Target size of the shape sample. Rounded to a multiple of ``N_SAT``;
		the density sample is ``n_shape * (1 + 1/N_SAT)`` (centrals + satellites).
	density_mode : str
		``'fixed_density'`` scales the box as ``N^(1/3)`` from the reference
		205 Mpc/h at 2400 shape galaxies, so pairs-per-galaxy stays constant
		— what running a bigger simulation looks like. ``'fixed_volume'``
		holds the box at 205 Mpc/h so density and pairs-per-galaxy rise, which
		isolates the pair-count-limited regime.
	boxsize : float, optional
		Explicit override, used by the boxsize sweep that tests whether the
		box (rp, pi) candidate cost really grows with box depth.

	Returns
	-------
	dict
		A ``measureia.mocks.radial_alignment_box_mock`` output.

	"""
	from measureia.mocks import radial_alignment_box_mock

	n_centrals = max(1, int(round(n_shape / N_SAT)))
	n_shape_actual = n_centrals * N_SAT
	if boxsize is None and density is not None:
		# absolute number density, for measuring in the regime real catalogues
		# occupy rather than the one the reference mock happens to sit in
		boxsize = boxsize_for_density(n_shape_actual * (1.0 + 1.0 / N_SAT), density)
	if boxsize is None:
		if density_mode == "fixed_density":
			boxsize = REF_BOX_BOXSIZE * (n_shape_actual / REF_BOX_N_SHAPE) ** (1.0 / 3.0)
		elif density_mode == "fixed_volume":
			boxsize = REF_BOX_BOXSIZE
		else:
			raise ValueError(f"box_mock_for: unknown density_mode {density_mode!r}")
	return radial_alignment_box_mock(
		n_centrals=n_centrals, n_sat=N_SAT, boxsize=float(boxsize),
		seed=seed, margin=margin,
	)


def lightcone_mock_for(n_shape, density_mode, n_randoms_factor=5, seed=4242):
	"""Radial-alignment lightcone mock scaled to ``n_shape`` shape galaxies.

	``'fixed_density'`` scales the RA/DEC window area in proportion to N at a
	fixed comoving-distance shell, so surface density is constant;
	``'fixed_volume'`` keeps the reference 10x10 degree window. The randoms
	factor is held fixed so the randoms:data ratio never varies underneath the
	comparison — it matters, since the randoms dominate the pair count.
	"""
	from measureia.mocks import radial_alignment_lightcone_mock

	n_centrals = max(1, int(round(n_shape / N_SAT)))
	n_shape_actual = n_centrals * N_SAT
	if density_mode == "fixed_density":
		s = np.sqrt(n_shape_actual / REF_LC_N_SHAPE)
		ra0 = REF_LC_RA_RANGE[0]
		ra_range = (ra0, ra0 + (REF_LC_RA_RANGE[1] - REF_LC_RA_RANGE[0]) * s)
		dec_range = (REF_LC_DEC_RANGE[0] * s, REF_LC_DEC_RANGE[1] * s)
	elif density_mode == "fixed_volume":
		ra_range, dec_range = REF_LC_RA_RANGE, REF_LC_DEC_RANGE
	else:
		raise ValueError(f"lightcone_mock_for: unknown density_mode {density_mode!r}")
	return radial_alignment_lightcone_mock(
		n_centrals=n_centrals, n_sat=N_SAT,
		ra_range=ra_range, dec_range=dec_range,
		n_randoms_factor=n_randoms_factor, seed=seed,
	)


# ----------------------------------------------------------------------------
# KDTree candidate instrumentation
# ----------------------------------------------------------------------------

class CandidateCounter:
	"""Count KDTree candidates without touching src/measureia.

	measureia selects each galaxy's candidate partners with two
	``KDTree.query_ball_tree`` calls — one at the inner radius, one at the
	outer — and takes the set difference (``pair_kernel.accumulate``). The
	number of candidates actually processed is therefore
	``outer_total - inner_total``. Patching the scipy method records both
	without changing a line of the package, so the count is exactly what a
	real measurement pays.

	Use as a context manager; read ``.candidates`` afterwards. ``by_radius``
	keeps the per-radius totals in case a caller wants to check the split.
	"""

	def __init__(self):
		self.by_radius = {}
		self._orig = None

	def __enter__(self):
		from scipy.spatial import KDTree

		self._orig = KDTree.query_ball_tree
		by_radius = self.by_radius

		def counting_query_ball_tree(self_tree, other, r, *args, **kwargs):
			out = self._orig(self_tree, other, r, *args, **kwargs)
			total = sum(len(x) for x in out)
			key = round(float(r), 10)
			by_radius[key] = by_radius.get(key, 0) + total
			return out

		KDTree.query_ball_tree = counting_query_ball_tree
		return self

	def __exit__(self, *exc):
		from scipy.spatial import KDTree

		KDTree.query_ball_tree = self._orig
		return False

	@property
	def candidates(self):
		"""Candidates processed: outer-radius total minus inner-radius total.

		With a single query radius (r_min = 0 contributes nothing) this is just
		the outer total. Falls back to the plain sum if only one radius was seen.
		"""
		if not self.by_radius:
			return 0
		radii = sorted(self.by_radius)
		if len(radii) == 1:
			return self.by_radius[radii[0]]
		return self.by_radius[radii[-1]] - sum(self.by_radius[r] for r in radii[:-1])


# ----------------------------------------------------------------------------
# result store
# ----------------------------------------------------------------------------

# Fields that together identify a benchmark point, so a resumed sweep can skip
# what is already recorded. Timing and environment fields are deliberately not
# part of the key.
KEY_FIELDS = (
	"task", "code", "n_shape", "density_mode", "threads", "scratch",
	"boxsize", "num_jk", "bin_slop", "variant",
)


def config_key(record):
	"""Hashable identity of a benchmark point."""
	return tuple(record.get(f) for f in KEY_FIELDS)


def load_records(path):
	"""Read a JSONL result file; returns [] if it does not exist."""
	if not os.path.exists(path):
		return []
	out = []
	with open(path) as f:
		for line in f:
			line = line.strip()
			if line:
				out.append(json.loads(line))
	return out


def completed_keys(path):
	"""Keys of every point already recorded, for resumable sweeps."""
	return {config_key(r) for r in load_records(path)}


def append_record(path, record):
	"""Append one result record as a JSON line."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "a") as f:
		f.write(json.dumps(record) + "\n")
