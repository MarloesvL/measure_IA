"""K-means clustering on the sphere, used to build the lightcone jackknife patches.

Points on the sky are clustered by great-circle distance, which is what makes the
resulting patches compact regions rather than the elongated slivers a naive
clustering of (RA, DEC) as a flat plane would produce near the poles.

The implementation works on unit vectors rather than angles. For two unit vectors
the chord and arc distances are monotonically related,

    |x - c|^2 = 2 - 2 x.c        (both unit vectors)

so minimising the great-circle distance to a set of centres is the same as
maximising the dot product x.c. That also holds when a centre is not normalised,
since |x - c|^2 = 1 + |c|^2 - 2 x.c and |c| is the same for every point x: the
assignment step needs one matrix product and no trigonometry at all.

Cluster centres are updated with the spherical mean — the mean of the member unit
vectors, projected back onto the sphere — and the fit uses the usual two-pass
strategy for large catalogues: cluster a random subsample first to place the
centres, then run to convergence on the full sample starting from those centres.
"""

import numpy as np

_MAXITER = 100
_TOL = 1.0e-5


def radec_to_unit_vectors(ra_deg, dec_deg):
	"""Convert RA/DEC in degrees to unit vectors.

	Parameters
	----------
	ra_deg : ndarray
		Right ascension in degrees.
	dec_deg : ndarray
		Declination in degrees.

	Returns
	-------
	ndarray
		Array of shape (N, 3) of unit vectors.

	"""
	ra = np.radians(np.asarray(ra_deg, dtype=float))
	dec = np.radians(np.asarray(dec_deg, dtype=float))
	cos_dec = np.cos(dec)
	return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def _spherical_mean(vectors):
	"""Mean of unit vectors projected back onto the sphere.

	Returns None when the members average to (almost) the origin, i.e. they are
	spread symmetrically over the sphere and no mean direction is defined.
	"""
	mean = vectors.mean(axis=0)
	norm = np.sqrt(np.sum(mean ** 2))
	if norm < 1e-12:
		return None
	return mean / norm


class SphericalKMeans:
	"""A fitted set of k-means centres on the sphere.

	Attributes
	----------
	centers : ndarray
		Cluster centres as unit vectors, shape (num_centers, 3).
	labels : ndarray
		Index of the nearest centre for each point the model was fitted on.
	converged : bool
		True if the fit reached the tolerance before the iteration limit.

	"""

	def __init__(self, centers, labels=None, converged=False):
		self.centers = centers
		self.labels = labels
		self.converged = converged

	def find_nearest(self, X):
		"""Index of the nearest centre for each point.

		Parameters
		----------
		X : ndarray
			Array of shape (N, 2) with RA and DEC in degrees as columns.

		Returns
		-------
		ndarray
			Index of the closest centre for each point, shape (N,).

		"""
		X = np.asarray(X)
		vectors = radec_to_unit_vectors(X[:, 0], X[:, 1])
		return self._assign(vectors)

	def _assign(self, vectors):
		"""Nearest centre per unit vector: argmax of the dot product."""
		return np.argmax(vectors @ self.centers.T, axis=1)


def _lloyd(vectors, centers, maxiter=_MAXITER, tol=_TOL):
	"""Lloyd iterations on the sphere from a starting set of centres.

	Iterates assign-then-update until the mean angular distance to the assigned
	centre stops improving by more than `tol` (relative), or `maxiter` is reached.
	Centres that end up with no members keep their previous position.
	"""
	centers = centers.copy()
	labels = np.zeros(len(vectors), dtype=int)
	prev_distance = 0.0
	converged = False

	for _ in range(maxiter):
		dots = vectors @ centers.T
		labels = np.argmax(dots, axis=1)
		# mean angular distance to the assigned centre; clip guards arccos against
		# dot products a hair outside [-1, 1] from floating-point round-off
		best = np.clip(dots[np.arange(len(vectors)), labels], -1.0, 1.0)
		distance = np.arccos(best).mean()

		# same convergence test as the usual formulation: stop once the mean
		# distance stops decreasing by a relative tol
		if (1 - tol) * prev_distance <= distance <= prev_distance:
			converged = True
			break
		prev_distance = distance

		for index in range(len(centers)):
			members = vectors[labels == index]
			if len(members) == 0:
				continue
			mean = _spherical_mean(members)
			if mean is not None:
				centers[index] = mean

	# Label against the centres actually returned. On the converged path the break
	# happens before the centre update, so `labels` already matches; but when maxiter
	# is exhausted the loop exits *after* an update, leaving labels one step stale.
	# Callers mix `labels` with `find_nearest` (which uses `centers`) across samples,
	# so the two must describe the same partition.
	labels = np.argmax(vectors @ centers.T, axis=1)
	return centers, labels, converged


def kmeans_sample(X, num_centers, nsample=None, maxiter=_MAXITER, tol=_TOL, seed=None):
	"""Two-pass k-means on the sphere.

	Clusters a random subsample first to place the centres, then runs to
	convergence on the full sample. This is much faster than a single full-sample
	fit for large catalogues and gives the same kind of partition.

	Parameters
	----------
	X : ndarray
		Array of shape (N, 2) with RA and DEC in degrees as columns.
	num_centers : int
		Number of clusters to fit.
	nsample : int or NoneType, optional
		Size of the first-pass subsample. Defaults to max(2*sqrt(N), 10*num_centers),
		capped at N.
	maxiter : int, optional
		Maximum number of Lloyd iterations per pass. Default is 100.
	tol : float, optional
		Relative change in the mean distance to the centres that signals
		convergence. Default is 1e-5.
	seed : int or NoneType, optional
		Seed for choosing the subsample and the starting centres, making the fit
		reproducible. If None (default), the fit differs between runs. The global
		numpy and stdlib random states are never touched.

	Returns
	-------
	SphericalKMeans
		The fitted model, with `labels` giving the cluster index of each input point.

	Raises
	------
	ValueError
		If X is not an (N, 2) array, or there are fewer points than centres.

	"""
	X = np.asarray(X)
	if X.ndim != 2 or X.shape[1] != 2:
		raise ValueError(f"X must be an (N, 2) array of RA and DEC in degrees, got shape {X.shape}.")
	num_points = len(X)
	if num_points < num_centers:
		raise ValueError(
			f"Cannot fit {num_centers} clusters to {num_points} points: at least one point "
			f"per cluster is needed.")

	rng = np.random.default_rng(seed)
	vectors = radec_to_unit_vectors(X[:, 0], X[:, 1])

	if nsample is None:
		nsample = max(2 * np.sqrt(num_points), 10 * num_centers)
	nsample = int(min(nsample, num_points))

	# first pass: fit the centres on a subsample, starting from random points
	subsample = vectors[rng.choice(num_points, nsample, replace=False)]
	start = vectors[rng.choice(num_points, num_centers, replace=False)]
	centers, _, _ = _lloyd(subsample, start, maxiter=maxiter, tol=tol)

	# second pass: run to convergence on everything
	centers, labels, converged = _lloyd(vectors, centers, maxiter=maxiter, tol=tol)
	return SphericalKMeans(centers, labels, converged)
