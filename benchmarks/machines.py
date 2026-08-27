"""Named machine profiles for the benchmark sweeps.

Every path here is supplied explicitly. Nothing is auto-detected and no
temporary directory is inferred from the environment, because on a cluster the
difference between a shared NFS scratch and node-local disk is one of the
things being measured — guessing it would quietly destroy the measurement.

To benchmark on a new machine, add an entry:

	"mymachine": dict(
		scratch={"local": "/fast/local/path", "nfs": "/shared/nfs/path"},
		threads=[1, 2, 4, 8],
		max_n_shape=300_000,
	),

and run ``python benchmarks/run_sweep.py --machine mymachine --scratch local``.
Each named scratch location is measured separately and stored with the
filesystem type it resolved to, so an NFS curve is never plotted on top of a
local one.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MACHINES = {
	"laptop": dict(
		scratch={
			"local": os.path.join(_ROOT, "benchmarks", "_scratch"),
		},
		threads=[1, 2, 4, 8, 12],
		max_n_shape=300_000,
	),
	# ------------------------------------------------------------------
	# Cluster profile — FILL THESE IN before running there.
	#   "nfs":   a directory on the shared NFS filesystem
	#   "local": a directory on node-local disk
	# run_sweep.py refuses to start if the scratch location it was asked for
	# is still blank, rather than silently falling back to somewhere else.
	# ------------------------------------------------------------------
	"cluster": dict(
		scratch={
			"nfs": "",
			"local": "",
		},
		threads=[1, 2, 4, 8, 16, 32],
		max_n_shape=1_000_000,
	),
}


def resolve(machine, scratch_name, override=None):
	"""Return the scratch directory for ``machine``/``scratch_name``.

	``override`` (the --scratch-dir flag) wins over the profile, for one-off
	runs on a machine that has no entry here yet.
	"""
	if override:
		return override
	if machine not in MACHINES:
		raise SystemExit(
			f"Unknown machine {machine!r}. Known: {', '.join(sorted(MACHINES))}. "
			f"Add a profile to benchmarks/machines.py or pass --scratch-dir."
		)
	profile = MACHINES[machine]
	if scratch_name not in profile["scratch"]:
		raise SystemExit(
			f"Machine {machine!r} has no scratch location named {scratch_name!r}. "
			f"Known: {', '.join(sorted(profile['scratch']))}."
		)
	path = profile["scratch"][scratch_name]
	if not path:
		raise SystemExit(
			f"Scratch location {scratch_name!r} for machine {machine!r} is blank in "
			f"benchmarks/machines.py. Fill in the real path (this is deliberate — "
			f"the filesystem is a measured variable, so it must not be guessed), "
			f"or pass --scratch-dir explicitly."
		)
	return path
