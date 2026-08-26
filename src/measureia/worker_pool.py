"""One worker pool per measurement, instead of one per pair-count run.

Under the ``spawn`` start method — which the backends force, because ``fork`` is
unsafe alongside the threads that numpy's BLAS may hold — every worker process
re-imports numpy, scipy, h5py, pyccl and measureia before it can do anything.
Measured on an 8-worker pool that costs about **0.9 s**, and it is paid on pool
*creation*, independent of how much work the pool is then given.

Each backend used to create its own pool. That is invisible for a box ``w``
measurement, which performs a single pair-count run, but a lightcone
``corr_type="both"`` measurement performs six (S+D, S+R, DD, SR, RD, RR) and so
paid the startup six times: about 5.5 s of fixed overhead. That is why
multiprocessing measured *slower* than single-process below roughly 40,000
galaxies, and why the lightcone jackknife path was slower at every size tested
(``benchmarks/FINDINGS.md`` F4).

Usage. The measurement entry points open one pool for the whole measurement::

    with worker_pool.measurement_pool(self.num_nodes):
        ...  # every pair-count run inside here shares one pool

and the backends ask for whatever is open::

    with worker_pool.active_pool(num_nodes) as p:
        result = p.map(self._batch, indices)

``active_pool`` yields the shared pool when a measurement has one open and does
**not** close it; otherwise it creates a private pool for that call and closes
it on exit. A backend method called on its own therefore behaves exactly as it
did before.

Reusing a pool across ``map`` calls is safe here because the workers hold no
state between tasks: ``self`` is pickled afresh for every ``map``, carrying the
current shared-memory names with it, and the arrays themselves are passed
through ``multiprocessing.shared_memory`` by name rather than inherited.
"""

import functools
import multiprocessing as mp
from contextlib import contextmanager
from multiprocessing import Pool

# The pool opened by the innermost active `measurement_pool`, or None. Only ever
# touched from the parent process, single-threaded, so a plain module global is
# enough — there is no locking to get wrong.
_ACTIVE = None
_ACTIVE_NODES = None

_START_METHOD_SET = False


def _ensure_spawn():
	"""Force the ``spawn`` start method once per process.

	The backends each used to call this before every pool. Calling it repeatedly
	is harmless but pointless, and doing it once here keeps the reason for it in
	a single place: ``fork`` would inherit whatever threads numpy's BLAS holds,
	which is not safe.
	"""
	global _START_METHOD_SET
	if not _START_METHOD_SET:
		mp.set_start_method("spawn", force=True)
		_START_METHOD_SET = True


@contextmanager
def measurement_pool(num_nodes):
	"""Open one pool for the duration of a measurement.

	A no-op when ``num_nodes <= 1`` (no pool is created and none is advertised,
	so the backends take their single-process paths untouched), and a no-op when
	a pool for the same size is already open, so nesting is harmless.

	Parameters
	----------
	num_nodes : int
		Number of worker processes. ``<= 1`` disables pooling entirely.

	"""
	global _ACTIVE, _ACTIVE_NODES

	if num_nodes is None or num_nodes <= 1 or _ACTIVE is not None:
		yield _ACTIVE
		return

	_ensure_spawn()
	pool = Pool(num_nodes)
	_ACTIVE, _ACTIVE_NODES = pool, num_nodes
	try:
		yield pool
	finally:
		_ACTIVE, _ACTIVE_NODES = None, None
		pool.close()
		pool.join()


@contextmanager
def active_pool(num_nodes):
	"""Yield a pool of ``num_nodes`` workers for one ``map``.

	Reuses the pool opened by an enclosing `measurement_pool` when there is one
	of the right size — and leaves it open, since the measurement owns it.
	Otherwise creates a pool for this call alone and closes it on exit.
	"""
	if _ACTIVE is not None and _ACTIVE_NODES == num_nodes:
		yield _ACTIVE
		return

	_ensure_spawn()
	pool = Pool(num_nodes)
	try:
		yield pool
	finally:
		pool.close()
		pool.join()


def pooled(method):
	"""Open one `measurement_pool` for the whole of a measurement method.

	Applied to the public entry points (``measure_xi_w``,
	``measure_xi_multipoles``, ``measure_galaxy_contributions``) so that every
	pair-count run inside one measurement shares a single pool instead of
	creating its own. Uses ``self.num_nodes``, so it is inert when the object was
	built with ``num_nodes=1``.
	"""

	@functools.wraps(method)
	def wrapper(self, *args, **kwargs):
		with measurement_pool(getattr(self, "num_nodes", 1)):
			return method(self, *args, **kwargs)

	return wrapper
