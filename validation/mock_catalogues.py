"""Synthetic mock catalogues with a known, non-null intrinsic alignment signal.

The radial-alignment mock places centrals uniformly in a periodic box and
scatters satellites around them with a Gaussian profile. Satellite projected
major axes point at their own central (plus Gaussian angle noise), giving a
strong, deterministic w_g+ signal on scales up to a few times the satellite
profile scale, and the central-satellite clumps give a non-null w_gg.

All randomness is controlled by a single seed so that every validation script
and test sees byte-identical catalogues.
"""

import numpy as np


def radial_alignment_box_mock(
		n_centrals=300,
		n_sat=8,
		boxsize=205.0,
		sigma_sat=2.0,
		alignment_scatter=0.2,
		q_range=(0.3, 0.9),
		seed=42,
		margin=0.0,
):
	"""Generate a periodic-box mock with radially aligned satellites.

	Parameters
	----------
	n_centrals : int, optional
		Number of central galaxies (positions uniform in the box).
	n_sat : int, optional
		Number of satellites per central.
	boxsize : float, optional
		Periodic box size in Mpc/h.
	sigma_sat : float, optional
		Standard deviation (Mpc/h) of the Gaussian satellite offsets
		around their central.
	alignment_scatter : float, optional
		Standard deviation (radians) of the Gaussian noise angle added to
		the projected radial orientation. 0 gives perfect radial alignment.
	q_range : tuple of float, optional
		Uniform sampling range of the projected axis ratio q = b/a.
	seed : int, optional
		Seed for the random generator; fixes the full catalogue.
	margin : float, optional
		Keep centrals at least this far (Mpc/h) from every box face, so no
		periodically wrapped pair falls within the measured separation range.
		Needed when the same catalogue must also be measured without
		periodicity (e.g. the plane-parallel lightcone consistency check).
		Default 0 (fill the whole box).

	Returns
	-------
	dict
		Data dictionary in the MeasureIABox input format with the keys
		"Position" (centrals + satellites, the density sample),
		"Position_shape_sample" (satellites), "Axis_Direction" (projected
		2D unit vectors), "q", "LOS" (2, i.e. the z-axis), plus extras
		"boxsize" and "seed" for bookkeeping.

	"""
	rng = np.random.default_rng(seed)
	n_sats_total = n_centrals * n_sat

	centrals = rng.uniform(margin, boxsize - margin, (n_centrals, 3))
	offsets = rng.normal(0.0, sigma_sat, (n_sats_total, 3))
	satellites = (np.repeat(centrals, n_sat, axis=0) + offsets) % boxsize

	# projected (LOS = z) radial direction from central to satellite;
	# use the unwrapped offsets so periodic wrapping cannot flip directions
	radial = offsets[:, :2]
	norm = np.sqrt(np.sum(radial ** 2, axis=1))
	norm[norm < 1e-12] = 1e-12
	direction = radial / norm[:, None]

	# add alignment noise by rotating each direction by a Gaussian angle
	theta = rng.normal(0.0, alignment_scatter, n_sats_total)
	cos_t, sin_t = np.cos(theta), np.sin(theta)
	direction = np.column_stack([
		cos_t * direction[:, 0] - sin_t * direction[:, 1],
		sin_t * direction[:, 0] + cos_t * direction[:, 1],
	])

	q = rng.uniform(q_range[0], q_range[1], n_sats_total)
	positions = np.vstack([centrals, satellites])

	return {
		"Position": positions,
		"Position_shape_sample": satellites,
		"Axis_Direction": direction,
		"q": q,
		"LOS": 2,
		"boxsize": boxsize,
		"seed": seed,
	}


def halotools_inputs(mock, ellipticity="distortion"):
	"""Convert a radial-alignment mock to halotools ia_correlations inputs.

	Parameters
	----------
	mock : dict
		Output of radial_alignment_box_mock.
	ellipticity : str, optional
		'distortion' for e = (1-q^2)/(1+q^2) or 'ellipticity' for
		e = (1-q)/(1+q), matching the measureia option.

	Returns
	-------
	tuple
		(shape_positions, orientations, ellipticities, density_positions,
		period) ready for halotools gi_plus_projected.

	"""
	q = mock["q"]
	if ellipticity == "distortion":
		e = (1 - q ** 2) / (1 + q ** 2)
	elif ellipticity == "ellipticity":
		e = (1 - q) / (1 + q)
	else:
		raise ValueError(f"Unknown ellipticity definition: {ellipticity}")
	return (
		mock["Position_shape_sample"],
		mock["Axis_Direction"],
		e,
		mock["Position"],
		mock["boxsize"],
	)


def responsivity(mock, ellipticity="distortion"):
	"""Responsivity R = 1 - <e^2>/2 of the mock shape sample.

	measureia divides S+D by 2R; halotools does not, so
	w_g+^measureia * 2R = w_g+^halotools.
	"""
	q = mock["q"]
	if ellipticity == "distortion":
		e = (1 - q ** 2) / (1 + q ** 2)
	else:
		e = (1 - q) / (1 + q)
	return 1.0 - np.mean(e ** 2) / 2.0


def _uniform_cone(rng, n, ra_range, dec_range, r_range):
	"""Uniform comoving density in a cone section: RA uniform, sin(DEC)
	uniform, r^3 uniform. Returns (ra_deg, dec_deg, r)."""
	ra = rng.uniform(ra_range[0], ra_range[1], n)
	sin_dec = rng.uniform(np.sin(np.radians(dec_range[0])),
						  np.sin(np.radians(dec_range[1])), n)
	dec = np.degrees(np.arcsin(sin_dec))
	u = rng.uniform(r_range[0] ** 3, r_range[1] ** 3, n)
	r = u ** (1.0 / 3.0)
	return ra, dec, r


def _radec_to_cartesian(ra_deg, dec_deg, r):
	alpha, delta = np.radians(ra_deg), np.radians(dec_deg)
	return r[:, None] * np.column_stack([
		np.cos(delta) * np.cos(alpha),
		np.cos(delta) * np.sin(alpha),
		np.sin(delta),
	])


def _cartesian_to_radec(pos):
	r = np.sqrt(np.sum(pos ** 2, axis=1))
	dec = np.degrees(np.arcsin(pos[:, 2] / r))
	ra = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
	return ra, dec, r


def radial_alignment_lightcone_mock(
		n_centrals=400,
		n_sat=8,
		ra_range=(40.0, 50.0),
		dec_range=(-5.0, 5.0),
		r_range=(2450.0, 2650.0),
		sigma_sat=2.0,
		alignment_scatter=0.2,
		q_range=(0.3, 0.9),
		n_randoms_factor=5,
		seed=4242,
):
	"""Generate a lightcone mock with radially aligned satellites.

	Centrals are uniform in comoving volume inside a cone section
	(RA/DEC window x comoving-distance shell); satellites get Gaussian 3D
	comoving offsets around their central. Satellite ellipticities e1/e2
	follow the standard survey shear-catalogue convention that measureia
	(and TreeCorr) expect, with the major axis pointing at the satellite's
	own central plus Gaussian angle noise — i.e. radial alignment, giving
	e+ > 0 and w_g+ > 0 in the IA sign convention.

	The position (density) sample is the centrals; the shape sample is the
	satellites, so the two samples are disjoint (no self-pair corrections).
	Distances are comoving; the caller converts r <-> redshift with their
	cosmology (see r_com entries).

	Returns
	-------
	tuple of dict
		(data, randoms_data, info): MeasureIALightcone-format data and
		randoms dictionaries (with Redshift left as comoving distance in
		"r_com*" keys for the caller to convert), and an info dict with
		q, the generator parameters and seed.

	"""
	rng = np.random.default_rng(seed)
	n_sats_total = n_centrals * n_sat

	ra_c, dec_c, r_c = _uniform_cone(rng, n_centrals, ra_range, dec_range, r_range)
	pos_c = _radec_to_cartesian(ra_c, dec_c, r_c)

	offsets = rng.normal(0.0, sigma_sat, (n_sats_total, 3))
	pos_s = np.repeat(pos_c, n_sat, axis=0) + offsets
	ra_s, dec_s, r_s = _cartesian_to_radec(pos_s)

	# local tangent basis at each satellite
	alpha, delta = np.radians(ra_s), np.radians(dec_s)
	east = np.column_stack([-np.sin(alpha), np.cos(alpha), np.zeros(n_sats_total)])
	north = np.column_stack([-np.sin(delta) * np.cos(alpha),
							 -np.sin(delta) * np.sin(alpha),
							 np.cos(delta)])
	n_hat = pos_s / r_s[:, None]

	# projected direction from satellite to central in the tangent plane
	d = np.repeat(pos_c, n_sat, axis=0) - pos_s
	d_perp = d - np.sum(d * n_hat, axis=1, keepdims=True) * n_hat
	phi_axis = np.arctan2(np.sum(d_perp * north, axis=1),
						  np.sum(d_perp * east, axis=1))
	phi_axis += rng.normal(0.0, alignment_scatter, n_sats_total)

	q = rng.uniform(q_range[0], q_range[1], n_sats_total)
	e = (1 - q ** 2) / (1 + q ** 2)
	# survey-convention components: the internal (east, north) frame angle
	# phi_axis maps to the survey frame as -phi_axis (opposite handedness)
	e1 = e * np.cos(2 * phi_axis)
	e2 = -e * np.sin(2 * phi_axis)

	n_rand_d = n_randoms_factor * n_centrals
	n_rand_s = n_randoms_factor * n_sats_total
	ra_rd, dec_rd, r_rd = _uniform_cone(rng, n_rand_d, ra_range, dec_range, r_range)
	ra_rs, dec_rs, r_rs = _uniform_cone(rng, n_rand_s, ra_range, dec_range, r_range)

	data = {
		"RA": ra_c, "DEC": dec_c, "r_com": r_c,
		"RA_shape_sample": ra_s, "DEC_shape_sample": dec_s, "r_com_shape_sample": r_s,
		"e1": e1, "e2": e2,
		"weight": np.ones(n_centrals), "weight_shape_sample": np.ones(n_sats_total),
	}
	randoms_data = {
		"RA": ra_rd, "DEC": dec_rd, "r_com": r_rd,
		"RA_shape_sample": ra_rs, "DEC_shape_sample": dec_rs, "r_com_shape_sample": r_rs,
		"weight": np.ones(n_rand_d), "weight_shape_sample": np.ones(n_rand_s),
	}
	info = {"q": q, "seed": seed, "sigma_sat": sigma_sat,
			"alignment_scatter": alignment_scatter}
	return data, randoms_data, info


def embed_box_mock_on_lightcone(mock, distance=3000.0, alpha0=45.0, delta0=0.0,
								n_randoms_factor=10, seed=777):
	"""Embed a box mock at a large comoving distance on a lightcone.

	The box is placed with its centre at comoving distance `distance` in the
	direction (alpha0, delta0), with the box axes mapped as
	x -> east, y -> north, z -> line of sight (matching the box LOS = 2
	convention). Projected axis directions are converted to survey-convention
	e1/e2 exactly, using each galaxy's own local (east, north) tangent frame,
	so the only differences between MeasureIABox on `mock` and
	MeasureIALightcone on the embedding are the plane-parallel approximation
	itself and the randoms-based (rather than analytic) estimator.

	Randoms are drawn uniformly in the FULL box cube and embedded the same
	way, mirroring the analytic-RR assumption of the box estimator. Use a
	mock generated with a margin >= the maximum measured separation so that
	no periodically wrapped pair contributes to the box measurement.

	Returns
	-------
	tuple of dict
		(data, randoms_data) in MeasureIALightcone format, with comoving
		distances in "r_com*" keys for the caller to convert to redshift.

	"""
	rng = np.random.default_rng(seed)
	L = mock["boxsize"]

	a0, d0 = np.radians(alpha0), np.radians(delta0)
	n0 = np.array([np.cos(d0) * np.cos(a0), np.cos(d0) * np.sin(a0), np.sin(d0)])
	east0 = np.array([-np.sin(a0), np.cos(a0), 0.0])
	north0 = np.array([-np.sin(d0) * np.cos(a0), -np.sin(d0) * np.sin(a0), np.cos(d0)])

	def embed(pos):
		rel = pos - L / 2.0
		return (rel[:, 0, None] * east0 + rel[:, 1, None] * north0
				+ (distance + rel[:, 2])[:, None] * n0)

	pos_d = embed(mock["Position"])
	pos_s = embed(mock["Position_shape_sample"])
	ra_d, dec_d, r_d = _cartesian_to_radec(pos_d)
	ra_s, dec_s, r_s = _cartesian_to_radec(pos_s)

	# exact per-galaxy conversion of the projected axis to survey e1/e2
	axis3d = (mock["Axis_Direction"][:, 0, None] * east0
			  + mock["Axis_Direction"][:, 1, None] * north0)
	alpha, delta = np.radians(ra_s), np.radians(dec_s)
	east = np.column_stack([-np.sin(alpha), np.cos(alpha), np.zeros(len(ra_s))])
	north = np.column_stack([-np.sin(delta) * np.cos(alpha),
							 -np.sin(delta) * np.sin(alpha),
							 np.cos(delta)])
	n_hat = pos_s / r_s[:, None]
	axis_perp = axis3d - np.sum(axis3d * n_hat, axis=1, keepdims=True) * n_hat
	phi_axis = np.arctan2(np.sum(axis_perp * north, axis=1),
						  np.sum(axis_perp * east, axis=1))
	q = mock["q"]
	e = (1 - q ** 2) / (1 + q ** 2)
	e1 = e * np.cos(2 * phi_axis)
	e2 = -e * np.sin(2 * phi_axis)

	n_rand_d = n_randoms_factor * len(ra_d)
	n_rand_s = n_randoms_factor * len(ra_s)
	cube_rd = rng.uniform(0.0, L, (n_rand_d, 3))
	cube_rs = rng.uniform(0.0, L, (n_rand_s, 3))
	pos_rd = embed(cube_rd)
	pos_rs = embed(cube_rs)
	ra_rd, dec_rd, r_rd = _cartesian_to_radec(pos_rd)
	ra_rs, dec_rs, r_rs = _cartesian_to_radec(pos_rs)

	data = {
		"RA": ra_d, "DEC": dec_d, "r_com": r_d,
		"RA_shape_sample": ra_s, "DEC_shape_sample": dec_s, "r_com_shape_sample": r_s,
		"e1": e1, "e2": e2,
		"weight": np.ones(len(ra_d)), "weight_shape_sample": np.ones(len(ra_s)),
	}
	randoms_data = {
		"RA": ra_rd, "DEC": dec_rd, "r_com": r_rd,
		"RA_shape_sample": ra_rs, "DEC_shape_sample": dec_rs, "r_com_shape_sample": r_rs,
		"weight": np.ones(n_rand_d), "weight_shape_sample": np.ones(n_rand_s),
	}
	cube_coords = {"randoms_position": cube_rd, "randoms_shape": cube_rs}
	return data, randoms_data, cube_coords


def subbox_labels(positions, boxsize, L):
	"""Subbox index (0..L^3-1) per position — the same partition the box
	jackknife uses, for supplying identical patches to the lightcone."""
	idx = np.floor(np.asarray(positions) / (boxsize / L)).astype(int)
	idx = np.clip(idx, 0, L - 1)
	return idx[:, 0] * L ** 2 + idx[:, 1] * L + idx[:, 2]


if __name__ == "__main__":
	mock = radial_alignment_box_mock()
	print(f"density sample: {len(mock['Position'])}")
	print(f"shape sample:   {len(mock['Position_shape_sample'])}")
	print(f"responsivity R: {responsivity(mock):.6f}")
	lc_data, lc_rand, lc_info = radial_alignment_lightcone_mock()
	print(f"lightcone positions: {len(lc_data['RA'])}, shapes: {len(lc_data['RA_shape_sample'])}")
	print(f"lightcone randoms:   {len(lc_rand['RA'])}, {len(lc_rand['RA_shape_sample'])}")
