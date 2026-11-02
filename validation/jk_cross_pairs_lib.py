"""Helpers for the box-jackknife cross-pair diagnostic (run_box_jk_cross_pairs.py).

Everything here is deliberately independent of measureia's pair kernel, so that it
can act as a reference for it:

- ``patch_pair_counts`` counts (weighted) pairs and S+D sums into a *patch-pair matrix*
  ``M[a_pos, b_shape, f_pos, f_shape, i_r, i_pi]``, where ``a``/``b`` are the jackknife
  sub-box labels of the two members and ``f`` optional per-galaxy or per-pair flags
  (used to emulate the stochastic schemes).
- every jackknife scheme is then a *reduction* of that matrix: ``removed_counts``.
- ``rr_*`` give the analytic-RR variants and ``predicted_mean_ratio`` the
  zero-clustering expectation of DD_k/RR_k, from the sub-box self-overlap volume.

Conventions follow measureia's box (measureia_conventions): separation
s = shape - position with the minimum image, rp = |s_perp|, pi = s_LOS signed,
e+ = e cos(2 phi) with phi the angle between the projected major axis and s_perp,
and the bin index computed with the same floor formula as ``pair_kernel.BoxRpPi``.
"""

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- binning
def bin_index(x, edges, log):
	"""Bin index with measureia's floor formula (upper edge folded into last bin).

	Returns -1 for values outside [edges[0], edges[-1]).
	"""
	n = len(edges) - 1
	f = np.log10 if log else (lambda v: v)
	width = (f(edges[-1]) - f(edges[0])) / n
	out = np.full(len(x), -1, dtype=np.int64)
	ok = (x >= edges[0]) & (x < edges[-1])
	idx = np.floor(f(x[ok]) / width - f(edges[0]) / width).astype(np.int64)
	idx[idx >= n] = n - 1
	out[ok] = idx
	return out


def subbox_labels_measureia(positions, boxsize, L):
	"""Sub-box labels exactly as MeasureJackknife._get_jackknife_region_indices assigns
	them: strict bounds, so a coordinate exactly on an edge keeps the default label 0."""
	lab = np.zeros(len(positions), dtype=np.int64)
	edges = np.linspace(0.0, boxsize, L + 1)
	num = 0
	for i in range(L):
		for j in range(L):
			for k in range(L):
				m = ((positions[:, 0] > edges[i]) & (positions[:, 0] < edges[i + 1])
					 & (positions[:, 1] > edges[j]) & (positions[:, 1] < edges[j + 1])
					 & (positions[:, 2] > edges[k]) & (positions[:, 2] < edges[k + 1]))
				lab[m] = num
				num += 1
	return lab


# ------------------------------------------------------------------- pair counting
def patch_pair_counts(pos, shape_pos, *, boxsize, r_bins, pi_bins, lab_pos, lab_shape,
					  n_patch, w_pos=None, w_shape=None, axis=None, e=None, los=2,
					  log_bins=True, binning="rppi", flag_pos=None, flag_shape=None,
					  pair_flag_seed=None, chunk=2000):
	"""Patch-pair matrices of weighted pair counts (DD) and S+D sums.

	Parameters
	----------
	pos, shape_pos : (N, 3) arrays in [0, boxsize)
	r_bins : edges in rp (binning='rppi') or 3D r (binning='rmu')
	pi_bins : edges in signed pi (rppi) or mu in [-1, 1] (rmu)
	lab_pos, lab_shape : integer sub-box labels in [0, n_patch)
	axis, e : projected major-axis unit vectors (N_shape, 2) and ellipticity sizes;
		omit both for a DD-only count (S+D is then returned as None)
	flag_pos, flag_shape : optional per-galaxy integer flags in {0, 1} (corr_pc-style
		per-galaxy draws); the matrices then get a size-2 axis for that side
	pair_flag_seed : optional seed for independent per-pair flags (two Bernoulli(1/2)
		draws per pair, one per side). Mutually exclusive with flag_pos/flag_shape.

	Returns
	-------
	M_DD, M_SpD : arrays (n_patch, n_patch, F_pos, F_shape, n_r, n_pi)
	"""
	pos = np.asarray(pos, float)
	shape_pos = np.asarray(shape_pos, float)
	n_r, n_pi = len(r_bins) - 1, len(pi_bins) - 1
	w_pos = np.ones(len(pos)) if w_pos is None else np.asarray(w_pos, float)
	w_shape = np.ones(len(shape_pos)) if w_shape is None else np.asarray(w_shape, float)
	shapes = axis is not None
	if pair_flag_seed is not None and (flag_pos is not None or flag_shape is not None):
		raise ValueError("per-galaxy and per-pair flags are mutually exclusive")
	per_pair = pair_flag_seed is not None
	F_p = 2 if (flag_pos is not None or per_pair) else 1
	F_s = 2 if (flag_shape is not None or per_pair) else 1
	rng = np.random.default_rng(pair_flag_seed) if per_pair else None
	not_los = [i for i in range(3) if i != los]

	if binning == "rppi":
		q = float(np.sqrt(r_bins[-1] ** 2 + max(abs(pi_bins[0]), abs(pi_bins[-1])) ** 2))
	else:
		q = float(r_bins[-1])
	# several label sets (e.g. N_jk = 8, 27, 64) can be binned from ONE pair search
	multi = isinstance(lab_pos, (list, tuple))
	lab_pos_l = list(lab_pos) if multi else [lab_pos]
	lab_shape_l = list(lab_shape) if multi else [lab_shape]
	n_patch_l = list(n_patch) if multi else [n_patch]
	sizes = [n * n * F_p * F_s * n_r * n_pi for n in n_patch_l]
	M_DD = [np.zeros(s) for s in sizes]
	M_SpD = [np.zeros(s) for s in sizes] if shapes else None
	tree_p = cKDTree(pos, boxsize=boxsize)
	half = 0.5 * boxsize
	for s0 in range(0, len(shape_pos), chunk):
		s1 = min(len(shape_pos), s0 + chunk)
		tree_s = cKDTree(shape_pos[s0:s1], boxsize=boxsize)
		coo = tree_s.sparse_distance_matrix(tree_p, q, output_type="coo_matrix")
		i_s = coo.row.astype(np.int64) + s0
		i_p = coo.col.astype(np.int64)
		sep = shape_pos[i_s] - pos[i_p]
		sep[sep > half] -= boxsize
		sep[sep < -half] += boxsize
		perp = sep[:, not_los]
		rp = np.sqrt(np.sum(perp ** 2, axis=1))
		if binning == "rppi":
			ir = bin_index(rp, r_bins, log_bins)
			ipi = bin_index(sep[:, los], pi_bins, False)
		else:
			r3 = np.sqrt(np.sum(sep ** 2, axis=1))
			ir = bin_index(r3, r_bins, log_bins)
			with np.errstate(invalid="ignore", divide="ignore"):
				mu = sep[:, los] / r3
			ipi = bin_index(np.nan_to_num(mu, nan=-9.0), pi_bins, False)
		ok = (ir >= 0) & (ipi >= 0)
		i_s, i_p, ir, ipi = i_s[ok], i_p[ok], ir[ok], ipi[ok]
		if per_pair:
			fp = rng.integers(0, 2, len(i_s))
			fs = rng.integers(0, 2, len(i_s))
		else:
			fp = flag_pos[i_p] if flag_pos is not None else np.zeros(len(i_s), np.int64)
			fs = flag_shape[i_s] if flag_shape is not None else np.zeros(len(i_s), np.int64)
		ww = w_pos[i_p] * w_shape[i_s]
		if shapes:
			pp = perp[ok]
			rr = rp[ok]
			cos_phi = (pp[:, 0] * axis[i_s, 0] + pp[:, 1] * axis[i_s, 1]) / rr
			e_plus = e[i_s] * (2.0 * cos_phi * cos_phi - 1.0)
		tail = ((fp * F_s + fs) * n_r + ir) * n_pi + ipi
		for m, (lp, ls, n) in enumerate(zip(lab_pos_l, lab_shape_l, n_patch_l)):
			flat = (lp[i_p] * n + ls[i_s]) * (F_p * F_s * n_r * n_pi) + tail
			M_DD[m] += np.bincount(flat, weights=ww, minlength=sizes[m])
			if shapes:
				M_SpD[m] += np.bincount(flat, weights=ww * e_plus, minlength=sizes[m])
	out_DD = [M.reshape(n, n, F_p, F_s, n_r, n_pi) for M, n in zip(M_DD, n_patch_l)]
	out_SpD = ([M.reshape(n, n, F_p, F_s, n_r, n_pi) for M, n in zip(M_SpD, n_patch_l)]
			   if shapes else [None] * len(n_patch_l))
	if multi:
		return out_DD, out_SpD
	return out_DD[0], out_SpD[0]


# --------------------------------------------------------------------- reductions
def match_weight(n_patch):
	"""Mohammad & Percival (2022) eq. 27 cross-pair removal weight alpha (treecorr 'match')."""
	return n_patch / (2.0 + np.sqrt(2.0) * (n_patch - 1))


def removed_counts(M, scheme, c=None):
	"""Per-realisation *removed* grids, shape (n_patch, n_r, n_pi); retained = full - removed.

	scheme:
	  'weight'   deterministic: auto pairs in k weight 1, cross pairs touching k weight c
	             (c=1 union / treecorr 'simple', c=0.5 paper mean / treecorr 'mean',
	             c=match_weight(n) treecorr 'match')
	  'flagged'  stochastic, from the flag axes: auto pairs in k always removed; a cross
	             pair is removed from its position-side region iff f_pos == 1 and from its
	             shape-side region iff f_shape == 1 (corr_pc jk_check_PB / corel_jk rule
	             with per-galaxy flags, or the per-pair variant with per-pair flags)
	"""
	n = M.shape[0]
	out = np.zeros((n,) + M.shape[4:])
	if scheme == "weight":
		Mf = M.sum(axis=(2, 3))
		for k in range(n):
			auto = Mf[k, k]
			row = Mf[k].sum(axis=0) - auto  # position in k, shape elsewhere
			col = Mf[:, k].sum(axis=0) - auto  # shape in k, position elsewhere
			out[k] = auto + c * (row + col)
	elif scheme == "flagged":
		for k in range(n):
			auto = M[k, k].sum(axis=(0, 1))
			row = M[k, :, 1].sum(axis=(0, 1)) - M[k, k, 1].sum(axis=0)
			col = M[:, k, :, 1].sum(axis=(0, 1)) - M[k, k, :, 1].sum(axis=0)
			out[k] = auto + row + col
	else:
		raise ValueError(scheme)
	return out


def full_counts(M):
	return M.sum(axis=(0, 1, 2, 3))


# ---------------------------------------------------------------------- analytic RR
def shell_volumes(r_bins, pi_bins, binning="rppi"):
	"""Bin volumes: pi (rp2^2 - rp1^2) dpi for rppi, (2 pi/3)(r2^3 - r1^3) dmu for rmu."""
	r_bins, pi_bins = np.asarray(r_bins), np.asarray(pi_bins)
	if binning == "rppi":
		return np.pi * np.diff(r_bins ** 2)[:, None] * np.diff(pi_bins)[None, :]
	return (2.0 * np.pi / 3.0) * np.diff(r_bins ** 3)[:, None] * np.diff(pi_bins)[None, :]


def region_weight_sums(w, lab, n_patch):
	return np.bincount(lab, weights=w, minlength=n_patch)


def rr_current(Vshell, V, n_patch, Wp, Ws, Wp_k, Ws_k, ov=0.0, ov_k=None):
	"""measureia's current per-realisation analytic RR: retained weight sums over the
	retained volume V (N-1)/N, keeping the full-box bin shape."""
	ov_k = np.zeros(n_patch) if ov_k is None else ov_k
	amp = (Wp - Wp_k) * (Ws - Ws_k) - (ov - ov_k)
	return amp[:, None, None] * Vshell[None] / (V * (n_patch - 1) / n_patch)


def pair_amplitude(c, Wp, Ws, Wp_k, Ws_k, ov=0.0, ov_k=None):
	"""Weighted number of *possible* pairs retained in realisation k when cross pairs are
	removed with weight c: W_pW_s - [W_p,k W_s,k + c(W_p,k(W_s-W_s,k) + (W_p-W_p,k)W_s,k)],
	minus the retained self-pair overlap (self pairs are auto pairs, removed fully)."""
	ov_k = np.zeros(len(Wp_k)) if ov_k is None else ov_k
	removed = Wp_k * Ws_k + c * (Wp_k * (Ws - Ws_k) + (Wp - Wp_k) * Ws_k)
	return Wp * Ws - removed - (ov - ov_k)


def rr_pair_consistent(Vshell, V, c, Wp, Ws, Wp_k, Ws_k, ov=0.0, ov_k=None):
	"""Pair-consistent analytic RR: pair_amplitude over the full volume V (flat in r;
	exact in expectation only for c = 1/2)."""
	amp = pair_amplitude(c, Wp, Ws, Wp_k, Ws_k, ov, ov_k)
	return amp[:, None, None] * Vshell[None] / V


# ------------------------------------------------------------------ the prediction
def subbox_overlap_fraction(r_bins, pi_bins, L_sub, binning="rppi", n_mc=200000, seed=1):
	"""Bin-averaged V_ov(s)/V_k, with V_ov(s) = prod_i max(0, L_sub - |s_i|) the overlap of
	a cubic sub-box with itself shifted by s (valid for L_sub <= boxsize/2). Monte Carlo
	over separation vectors uniform in each bin's volume."""
	rng = np.random.default_rng(seed)
	n_r, n_pi = len(r_bins) - 1, len(pi_bins) - 1
	out = np.zeros((n_r, n_pi))
	for i in range(n_r):
		for j in range(n_pi):
			u = rng.random((3, n_mc))
			if binning == "rppi":
				rp = np.sqrt(r_bins[i] ** 2 + u[0] * (r_bins[i + 1] ** 2 - r_bins[i] ** 2))
				th = 2 * np.pi * u[1]
				s = np.array([rp * np.cos(th), rp * np.sin(th),
							  pi_bins[j] + u[2] * (pi_bins[j + 1] - pi_bins[j])])
			else:
				r = np.cbrt(r_bins[i] ** 3 + u[0] * (r_bins[i + 1] ** 3 - r_bins[i] ** 3))
				mu = pi_bins[j] + u[2] * (pi_bins[j + 1] - pi_bins[j])
				th = 2 * np.pi * u[1]
				st = np.sqrt(1 - mu ** 2)
				s = np.array([r * st * np.cos(th), r * st * np.sin(th), r * mu])
			ov = np.prod(np.clip(L_sub - np.abs(s), 0, None), axis=0)
			out[i, j] = ov.mean() / L_sub ** 3
	return out


def predicted_mean_ratio(c, n_patch, ov_frac):
	"""Zero-clustering expectation of retained DD_k / RR_k(current) for cross-pair weight
	c: (V - 2c V_k - (1 - 2c) V_ov) / (V - V_k). Union (c=1): (V - 2V_k + V_ov)/(V - V_k);
	c = 1/2: exactly 1."""
	vk = 1.0 / n_patch  # V_k / V
	return (1.0 - 2.0 * c * vk - (1.0 - 2.0 * c) * vk * ov_frac) / (1.0 - vk)


# ------------------------------------------- analytic DR / S+R (infinite randoms, option 2)
def arc_integrals(r, ax, bx, ay, by):
	"""For a circle of radius r about the origin and axis-aligned rectangles
	[ax, bx] x [ay, by] (arrays), the integrals over the arcs inside each rectangle of
	1, cos 2phi and sin 2phi d phi. Returns (I0, Ic, Is)."""
	cands = []
	for a in (ax, bx):
		t = np.arccos(np.clip(a / r, -1, 1))
		ok = np.abs(a) < r
		cands += [np.where(ok, t, 0.0), np.where(ok, -t, 0.0)]
	for b in (ay, by):
		t = np.arcsin(np.clip(b / r, -1, 1))
		ok = np.abs(b) < r
		cands += [np.where(ok, t, 0.0), np.where(ok, np.pi - t, 0.0)]
	G = len(ax)
	ang = np.mod(np.column_stack(cands), 2 * np.pi)
	ang = np.sort(np.column_stack([np.zeros(G), ang, np.full(G, 2 * np.pi)]), axis=1)
	lo, hi = ang[:, :-1], ang[:, 1:]
	mid = 0.5 * (lo + hi)
	xm, ym = r * np.cos(mid), r * np.sin(mid)
	inside = ((xm >= ax[:, None]) & (xm <= bx[:, None]) & (ym >= ay[:, None]) & (ym <= by[:, None]))
	I0 = np.sum((hi - lo) * inside, axis=1)
	Ic = np.sum(0.5 * (np.sin(2 * hi) - np.sin(2 * lo)) * inside, axis=1)
	Is = np.sum(0.5 * (np.cos(2 * lo) - np.cos(2 * hi)) * inside, axis=1)
	return I0, Ic, Is


def _image_intervals(lo_rel, width, boxsize, reach):
	"""Periodic images [lo + n L, lo + n L + width], n in {-1, 0, 1}, as a list of
	(lo, hi) arrays; intervals that cannot reach |t| <= reach are left in (zero overlap)."""
	return [(lo_rel + n * boxsize, lo_rel + n * boxsize + width) for n in (-1, 0, 1)]


def analytic_partial_shell(points, lab, n_patch, *, boxsize, r_bins, pi_bins, reflect_pi=False,
						   theta=None, e=None, n_gl=16):
	"""Per galaxy and per sub-box k != own label, the overlap of the galaxy's partner
	region {t in (rp, pi) bin : x + t in sub-box k} (periodic), integrated over:
	  - 1 (the overlap volume, for DR terms), and, when theta/e are given,
	  - e cos 2(phi_t - theta) (for S+R terms).
	Returns per-k SUMS over galaxies not in k: (sum_vol[k, r, pi], sum_ep[k, r, pi] or None).
	reflect_pi uses the bin with pi -> -pi (partner at x - s instead of x + s)."""
	L = round(n_patch ** (1 / 3))
	L_sub = boxsize / L
	n_r, n_pi = len(r_bins) - 1, len(pi_bins) - 1
	reach = np.sqrt(r_bins[-1] ** 2 + max(abs(pi_bins[0]), abs(pi_bins[-1])) ** 2)
	p_lo = -pi_bins[1:] if reflect_pi else pi_bins[:-1]
	p_hi = -pi_bins[:-1] if reflect_pi else pi_bins[1:]
	gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
	sum_vol = np.zeros((n_patch, n_r, n_pi))
	sum_ep = np.zeros((n_patch, n_r, n_pi)) if theta is not None else None
	c2t = np.cos(2 * theta) if theta is not None else None
	s2t = np.sin(2 * theta) if theta is not None else None
	for k in range(n_patch):
		ix, iy, iz = k // (L * L), (k // L) % L, k % L
		sel = np.where(lab != k)[0]
		x = points[sel]
		# z overlap per pi bin, summed over periodic images
		Lz = np.zeros((len(sel), n_pi))
		for (zlo, zhi) in _image_intervals(iz * L_sub - x[:, 2], L_sub, boxsize, reach):
			Lz += np.clip(np.minimum(zhi[:, None], p_hi[None]) - np.maximum(zlo[:, None], p_lo[None]), 0, None)
		has_z = Lz.sum(axis=1) > 0
		A = np.zeros((len(sel), n_r))
		Ac = np.zeros((len(sel), n_r))
		As = np.zeros((len(sel), n_r))
		for (xlo, xhi) in _image_intervals(ix * L_sub - x[:, 0], L_sub, boxsize, reach):
			for (ylo, yhi) in _image_intervals(iy * L_sub - x[:, 1], L_sub, boxsize, reach):
				# nearest distance from origin to the rectangle; skip if beyond rp_max
				dx = np.maximum(np.maximum(xlo, -xhi), 0)
				dy = np.maximum(np.maximum(ylo, -yhi), 0)
				near = np.where(has_z & (np.hypot(dx, dy) < r_bins[-1]))[0]
				if len(near) == 0:
					continue
				for i in range(n_r):
					r1, r2 = r_bins[i], r_bins[i + 1]
					for xg, wg in zip(gl_x, gl_w):
						r = 0.5 * (r2 - r1) * xg + 0.5 * (r1 + r2)
						wr = 0.5 * (r2 - r1) * wg * r
						I0, Ic, Is = arc_integrals(r, xlo[near], xhi[near], ylo[near], yhi[near])
						A[near, i] += wr * I0
						Ac[near, i] += wr * Ic
						As[near, i] += wr * Is
		sum_vol[k] = A.T @ Lz
		if theta is not None:
			ep = e[sel][:, None] * (Ac * c2t[sel][:, None] + As * s2t[sel][:, None])
			sum_ep[k] = ep.T @ Lz
	return sum_vol, sum_ep


def analytic_dr_terms(P, S, axis, e, lab_p, lab_s, n_patch, *, boxsize, r_bins, pi_bins,
					  w_p=None, w_s=None, n_gl=16):
	"""Normalised, retained, union-deletion D R_S, R_D S and S+R_D per realisation in the
	infinite-randoms limit (uniform randoms, unit weights):
	  D R_S,k  = [W_p,ret Vbin - sum_{i in D, not k} w_i Ov_ik] / [W_p,ret (V - V_k)]
	  R_D S,k  = same with shapes and the pi-reflected bin
	  S+R_D,k  = - sum_{j in S, not k} w_j e_j int_{Ov_jk} cos 2(phi - theta_j) / [W_s,ret (V - V_k)]
	(the full-shell e+ integral vanishes). Weights enter as point weights."""
	w_p = np.ones(len(P)) if w_p is None else w_p
	w_s = np.ones(len(S)) if w_s is None else w_s
	V = boxsize ** 3
	Vk = V / n_patch
	Vbin = shell_volumes(r_bins, pi_bins)
	theta = np.arctan2(axis[:, 1], axis[:, 0])
	sv_p, _ = analytic_partial_shell(P, lab_p, n_patch, boxsize=boxsize, r_bins=r_bins,
									 pi_bins=pi_bins, n_gl=n_gl)
	sv_s, sep_s = analytic_partial_shell(S, lab_s, n_patch, boxsize=boxsize, r_bins=r_bins,
										 pi_bins=pi_bins, reflect_pi=True, theta=theta,
										 e=e * w_s, n_gl=n_gl)
	# the shape weights multiply the volume sums too: redo with weights for the volume part
	Wp_ret = w_p.sum() - np.bincount(lab_p, weights=w_p, minlength=n_patch)
	Ws_ret = w_s.sum() - np.bincount(lab_s, weights=w_s, minlength=n_patch)
	if not np.allclose(w_p, 1) or not np.allclose(w_s, 1):
		raise NotImplementedError("prototype: unit weights only")
	DRs = (Wp_ret[:, None, None] * Vbin[None] - sv_p) / (Wp_ret[:, None, None] * (V - Vk))
	RdS = (Ws_ret[:, None, None] * Vbin[None] - sv_s) / (Ws_ret[:, None, None] * (V - Vk))
	SpRd = -sep_s / (Ws_ret[:, None, None] * (V - Vk))
	return DRs, RdS, SpRd


def _interval_overlap(lo1, hi1, lo2, hi2):
	return np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None)


def analytic_partial_shell_rmu(points, lab, n_patch, *, boxsize, r_bins, mu_bins, reflect_mu=False,
							   theta=None, e=None, n_gl=16, chunk=256):
	"""r-mu version of analytic_partial_shell. Bin b: 3D r in [r1, r2), mu = s_LOS / r in
	[mu1, mu2). In cylindrical coordinates (rho, h) about the galaxy, d^3s = rho drho dphi dh;
	at fixed rho the allowed h form exact intervals (the shell r1 <= sqrt(rho^2 + h^2) < r2,
	the mu bin h in [rho g(mu1), rho g(mu2)) with g(mu) = mu / sqrt(1 - mu^2), and the
	sub-box's z-range), so
	  O = int_0^{r2} rho drho I0(rho) Lh(rho),   Q likewise with cos2theta Ic + sin2theta Is,
	with the rho integral by Gauss-Legendre on segments split at the r_bins edges."""
	L = round(n_patch ** (1 / 3))
	L_sub = boxsize / L
	n_r, n_mu = len(r_bins) - 1, len(mu_bins) - 1
	mlo = -mu_bins[1:] if reflect_mu else mu_bins[:-1]
	mhi = -mu_bins[:-1] if reflect_mu else mu_bins[1:]
	with np.errstate(divide="ignore"):
		g_lo = np.where(np.abs(mlo) >= 1, np.sign(mlo) * np.inf, mlo / np.sqrt(np.clip(1 - mlo ** 2, 1e-300, None)))
		g_hi = np.where(np.abs(mhi) >= 1, np.sign(mhi) * np.inf, mhi / np.sqrt(np.clip(1 - mhi ** 2, 1e-300, None)))
	edges = np.concatenate([[0.0], r_bins])
	gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
	rho = np.concatenate([0.5 * (b - a) * gl_x + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
	wrho = np.concatenate([0.5 * (b - a) * gl_w for a, b in zip(edges[:-1], edges[1:])]) * rho
	# shell h-limits per (node, r bin): |h| in [a, b)
	a_sh = np.sqrt(np.clip(r_bins[:-1][None] ** 2 - rho[:, None] ** 2, 0, None))
	b_sh = np.sqrt(np.clip(r_bins[1:][None] ** 2 - rho[:, None] ** 2, 0, None))
	m_lo = rho[:, None] * g_lo[None]  # (node, mu)
	m_hi = rho[:, None] * g_hi[None]
	sum_vol = np.zeros((n_patch, n_r, n_mu))
	sum_ep = np.zeros((n_patch, n_r, n_mu)) if theta is not None else None
	for k in range(n_patch):
		ix, iy, iz = k // (L * L), (k // L) % L, k % L
		sel_all = np.where(lab != k)[0]
		for c0 in range(0, len(sel_all), chunk):
			sel = sel_all[c0:c0 + chunk]
			x = points[sel]
			G = len(sel)
			I0 = np.zeros((G, len(rho)))
			Ic = np.zeros((G, len(rho)))
			Is = np.zeros((G, len(rho)))
			for (xlo, xhi) in _image_intervals(ix * L_sub - x[:, 0], L_sub, boxsize, None):
				for (ylo, yhi) in _image_intervals(iy * L_sub - x[:, 1], L_sub, boxsize, None):
					dx = np.maximum(np.maximum(xlo, -xhi), 0)
					dy = np.maximum(np.maximum(ylo, -yhi), 0)
					near = np.where(np.hypot(dx, dy) < r_bins[-1])[0]
					if len(near) == 0:
						continue
					for q, r in enumerate(rho):
						a0, ac, as_ = arc_integrals(r, xlo[near], xhi[near], ylo[near], yhi[near])
						I0[near, q] += a0
						Ic[near, q] += ac
						Is[near, q] += as_
			if not np.any(I0):
				continue
			# Lh[g, node, r, mu]: summed over z images and the two signs of h
			Lh = np.zeros((G, len(rho), n_r, n_mu))
			for (zlo, zhi) in _image_intervals(iz * L_sub - x[:, 2], L_sub, boxsize, None):
				zl, zh = zlo[:, None, None, None], zhi[:, None, None, None]
				for sgn in (1, -1):
					lo_s = a_sh if sgn == 1 else -b_sh
					hi_s = b_sh if sgn == 1 else -a_sh
					lo = np.maximum(lo_s[None, :, :, None], m_lo[None, :, None, :])
					hi = np.minimum(hi_s[None, :, :, None], m_hi[None, :, None, :])
					Lh += _interval_overlap(lo, hi, zl, zh)
			sum_vol[k] += np.einsum("gq,q,gqrm->rm", I0, wrho, Lh)
			if theta is not None:
				ang = e[sel][:, None] * (Ic * np.cos(2 * theta[sel])[:, None] + Is * np.sin(2 * theta[sel])[:, None])
				sum_ep[k] += np.einsum("gq,q,gqrm->rm", ang, wrho, Lh)
	return sum_vol, sum_ep


def analytic_dr_terms_rmu(P, S, axis, e, lab_p, lab_s, n_patch, *, boxsize, r_bins, mu_bins, n_gl=16):
	"""r-mu counterpart of analytic_dr_terms (unit weights)."""
	V = boxsize ** 3
	Vk = V / n_patch
	Vbin = shell_volumes(r_bins, mu_bins, binning="rmu")
	theta = np.arctan2(axis[:, 1], axis[:, 0])
	sv_p, _ = analytic_partial_shell_rmu(P, lab_p, n_patch, boxsize=boxsize, r_bins=r_bins,
										 mu_bins=mu_bins, n_gl=n_gl)
	sv_s, sep_s = analytic_partial_shell_rmu(S, lab_s, n_patch, boxsize=boxsize, r_bins=r_bins,
											 mu_bins=mu_bins, reflect_mu=True, theta=theta, e=e,
											 n_gl=n_gl)
	Wp_ret = len(P) - np.bincount(lab_p, minlength=n_patch)
	Ws_ret = len(S) - np.bincount(lab_s, minlength=n_patch)
	DRs = (Wp_ret[:, None, None] * Vbin[None] - sv_p) / (Wp_ret[:, None, None] * (V - Vk))
	RdS = (Ws_ret[:, None, None] * Vbin[None] - sv_s) / (Ws_ret[:, None, None] * (V - Vk))
	SpRd = -sep_s / (Ws_ret[:, None, None] * (V - Vk))
	return DRs, RdS, SpRd


# ---------------------------------------------------------------- jackknife moments
def jk_cov(real):
	"""Delete-one covariance, (N-1)/N sum (x_k - mean)(x_k - mean)^T, as
	MeasureJackknife._combine_jackknife_information."""
	real = np.asarray(real)
	n = len(real)
	d = real - real.mean(axis=0)
	return (n - 1) / n * d.T @ d


def w_from_xi(xi, pi_bins):
	"""w = sum_pi xi * |dpi| (rppi grids, last axis is pi)."""
	return np.sum(xi * np.abs(np.diff(pi_bins)), axis=-1)


def responsivity_options(e, w, lab, n_patch, c):
	"""Per-realisation responsivity R_k for the candidate choices:
	  'retained'   shapes outside k only (measureia's compute_R_jk)
	  'full'       the full-sample R for every k
	  'pair'       shapes inside k with weight (1 - c)
	"""
	t = w * (1.0 - 0.5 * e ** 2)
	T, W = t.sum(), w.sum()
	T_k = np.bincount(lab, weights=t, minlength=n_patch)
	W_k = np.bincount(lab, weights=w, minlength=n_patch)
	return {
		"retained": (T - T_k) / (W - W_k),
		"full": np.full(n_patch, T / W),
		"pair": (T - c * T_k) / (W - c * W_k),
	}