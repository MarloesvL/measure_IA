r"""Analytic edge correction for the delete-one sub-box jackknife of a periodic box.

``MeasureIABox`` uses analytic randoms: RR is a formula, not a pair count. In a delete-one
jackknife realisation k, sub-box C_k is removed, and with it every pair that has a member in
C_k ("union" deletion). A random catalogue would supply per-realisation DR and S+R terms that
track which galaxies sit next to the hole; analytic randoms do not, which inflates the
large-scale jackknife errors (``validation/box_jk_analytic_randoms.md``). This module supplies
those terms in the limit of infinitely many randoms, i.e. as volumes.

For a galaxy at x and a separation bin b, the volume of the bin that lands in the hole is

    O_k(x; b) = int_b d^3s 1[x + s in C_k]                       (periodic),

and Oc_k, Os_k are the same integral weighted by cos 2 phi_s and sin 2 phi_s (phi_s the
azimuth of s in the projected plane). Per realisation, normalised per retained pair:

    DR_k  = sum_{i in D, i not in k} w_i [V_b - O_k(x_i; b)]          / [W_D,k (V - V_k)]
    RD_k  = sum_{j in S, j not in k} w_j [V_b - O_k(x_j; b~)]         / [W_S,k (V - V_k)]
    S+R_k = - sum_{j in S, j not in k} w_j e_j Q_k(x_j, theta_j; b~)  / [W_S,k (V - V_k)]

with Q = cos 2 theta Oc + sin 2 theta Os, b~ the bin with the line-of-sight coordinate (pi or
mu) reflected, and the exact union RR_k

    RR_k = (V_b / V) (1 - 2v + v f(b)) / (1 - v)^2,   v = 1 / N_jk,
    f(b) = < prod_a max(0, L_sub - |s_a|) >_b / L_sub^3.

Conventions: points are given with the line of sight as the LAST coordinate. Sub-box k has
indices (ix, iy, iz) = (k // L^2, (k // L) % L, k % L) along the three coordinates, L =
``L_subboxes``. The bins are (r_p, pi) for ``binning="rppi"`` and (r, mu) for
``binning="rmu"``. Every separation component must be at most half the box.

Accuracy: in rp-pi the sky-plane overlaps are exact (closed form for the area of a disc
inside a rectangle and its cos 2phi / sin 2phi moments). In r-mu the projected radius is
integrated with Gauss-Legendre quadrature on [0, r_i] and [r_i, r_{i+1}], separately for each
r bin (so a bin does not depend on the other bins): 16 nodes per segment at N_jk = 8 and 8 from
N_jk = 27. Per galaxy the error is up to ~5e-3 (16) or ~1.7e-2 (8) of V_b; against a 64-node
reference the jackknife errors change by <= 0.5% (16, N_jk = 8) and <= 0.6% (8, N_jk = 27, 64)
on the dense validation mock (``validation/run_box_jk_node_rule_check.py``; with 8 nodes at
N_jk = 8 it was 5.5%, hence the 16 there). The sub-box self-overlap f(b) is computed by
deterministic quadrature split at every kink.

Everything here is plain numpy, vectorised over galaxies in bounded-memory chunks. The
sky-plane part of an overlap depends only on the column (ix, iy) of the sub-box, so it is
computed once per column and combined with the line-of-sight part of every sub-box in that
column by one matrix product.
"""

import numpy as np

from . import worker_pool

__all__ = [
    "arc_integrals",
    "subbox_overlaps",
    "analytic_dr_terms",
    "self_overlap_fraction",
    "exact_union_rr",
    "bin_volumes",
    "check_half_box",
]

_N_GL_RMU = 8           # Gauss-Legendre nodes per rho segment (r-mu overlaps), N_jk >= 27
_N_GL_RMU_FEW_JK = 16   # the same at N_jk = 8 (see _n_gl_rmu)
_N_GL_F = 32         # nodes per segment for the self-overlap fraction f(b)
_CHUNK_ELEMENTS = 10_000_000  # rp-pi: target size (float64 elements, ~80 MB) of the largest per-chunk array
_CHUNK_BYTES_RMU = 200_000_000  # r-mu: target size of all of a chunk's temporaries (per worker)
_BLOCKS_PER_NODE = 4  # with num_nodes > 1: blocks of chunks per worker (galaxies near sub-box edges cost more)


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def check_half_box(boxsize, r_bins, bins2, binning):
    """Raises ValueError if a separation component can exceed half the box: r_max (and, in
    rp-pi, |pi|max) must be at most L/2. The periodic image sums, the sub-box self-overlap and
    minimum-image pair counting all assume this."""
    half = 0.5 * boxsize
    tol = 1e-12 * boxsize
    r_max = float(np.max(r_bins))
    if r_max > half + tol:
        raise ValueError(
            f"The maximum separation ({r_max:g}) exceeds half the box (L/2 = {half:g}). The analytic "
            f"jackknife edge correction needs every separation component to be at most L/2. Lower the "
            f"maximum separation, or pass jk_edge_correction=False.")
    if binning == "rppi" and float(np.max(np.abs(bins2))) > half + tol:
        raise ValueError(
            f"pi_max ({float(np.max(np.abs(bins2))):g}) exceeds half the box (L/2 = {half:g}). The analytic "
            f"jackknife edge correction needs every separation component to be at most L/2. Lower pi_max, "
            f"or pass jk_edge_correction=False.")


# ---------------------------------------------------------------------------
# Sky-plane geometry
# ---------------------------------------------------------------------------

def arc_integrals(r, ax, bx, ay, by):
    r"""Integrals of 1, cos 2phi and sin 2phi d phi over the arcs of the circle of radius r about
    the origin that lie inside the rectangle [ax, bx] x [ay, by]. All arguments broadcast.

    x in [ax, bx] means cos phi in [ax/r, bx/r]: phi in +-[alpha1, alpha2]. y in [ay, by] means
    sin phi in [ay/r, by/r]: phi in [beta1, beta2] or [pi - beta2, pi - beta1]. The arcs are
    the intersections of these angle intervals; the second y interval is also shifted by -2pi so
    that every piece inside (-pi, pi] is found.

    Every intersection endpoint is one of the interval endpoints, whose sin 2phi and cos 2phi
    follow from the ratios without further trigonometry (alpha = arccos c: cos 2alpha =
    2c^2 - 1, sin 2alpha = 2c sqrt(1 - c^2); beta = arcsin s: cos 2beta = 1 - 2s^2,
    sin 2beta = 2s sqrt(1 - s^2); -phi and pi - phi flip the sign of sin 2phi only).

    Returns
    -------
    I0, Ic, Is : ndarrays
    """
    r, ax, bx, ay, by = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (r, ax, bx, ay, by)))
    ok = r > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = [np.clip(v / np.where(ok, r, 1.0), -1.0, 1.0) for v in (bx, ax, ay, by)]
    cbx, cax, say, sby = ratios
    # (angle, sin 2phi, cos 2phi) of every endpoint
    def x_end(c, sign):
        return sign * np.arccos(c), sign * 2 * c * np.sqrt(1 - c * c), 2 * c * c - 1

    def y_end(s, shift):
        b = np.arcsin(s)
        s2, c2 = 2 * s * np.sqrt(1 - s * s), 1 - 2 * s * s
        if shift == 0:
            return b, s2, c2
        return shift - b, -s2, c2                      # pi - beta or -pi - beta

    x_int = [(x_end(cbx, 1), x_end(cax, 1)), (x_end(cax, -1), x_end(cbx, -1))]
    y_int = [(y_end(say, 0), y_end(sby, 0)), (y_end(sby, np.pi), y_end(say, np.pi)),
             (y_end(sby, -np.pi), y_end(say, -np.pi))]
    I0 = np.zeros(r.shape)
    Ic = np.zeros(r.shape)
    Is = np.zeros(r.shape)
    for xl, xh in x_int:
        for yl, yh in y_int:
            lo_x = xl[0] >= yl[0]
            hi_x = xh[0] <= yh[0]
            lo = np.where(lo_x, xl[0], yl[0])
            hi = np.where(hi_x, xh[0], yh[0])
            full = hi > lo                               # empty intersection contributes nothing
            I0 += np.where(full, hi - lo, 0.0)
            Ic += np.where(full, 0.5 * (np.where(hi_x, xh[1], yh[1]) - np.where(lo_x, xl[1], yl[1])), 0.0)
            Is += np.where(full, 0.5 * (np.where(lo_x, xl[2], yl[2]) - np.where(hi_x, xh[2], yh[2])), 0.0)
    return np.where(ok, I0, 0.0), np.where(ok, Ic, 0.0), np.where(ok, Is, 0.0)


def _quadrant_moments(a, b, r):
    r"""Integrals of (1, cos 2phi, sin 2phi) dA over the disc of radius r intersected with the
    rectangle [0, a] x [0, b] (a, b >= 0). Closed form: in polar coordinates the region is
    rho < min(r, a / cos phi, b / sin phi) for phi in [0, pi/2], so

      phi in [0, phi1]:     rho_max = a / cos phi   (the x = a wall)
      phi in [phi1, phi2]:  rho_max = r             (the circle)
      phi in [phi2, pi/2]:  rho_max = b / sin phi   (the y = b wall)

    with phi1 = min(arccos(a/r), phi_c), phi2 = max(arcsin(b/r), phi_c), phi_c = atan2(b, a)
    (when the disc covers the corner, phi1 = phi2 = phi_c). The phi integrals of
    rho_max^2 / 2 times (1, cos 2phi, sin 2phi) are elementary.
    """
    a, b, r = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, r)))
    ok = (a > 0) & (b > 0) & (r > 0)
    a_ = np.where(ok, a, 1.0)
    b_ = np.where(ok, b, 1.0)
    r_ = np.where(ok, r, 1.0)
    phic = np.arctan2(b_, a_)
    p1 = np.minimum(np.arccos(np.minimum(a_ / r_, 1.0)), phic)
    p2 = np.maximum(np.arcsin(np.minimum(b_ / r_, 1.0)), phic)
    t1 = np.tan(p1)
    c2 = 1.0 / np.tan(p2)
    a2, b2, r2 = a_ ** 2, b_ ** 2, r_ ** 2
    A = 0.5 * a2 * t1 + 0.5 * r2 * (p2 - p1) + 0.5 * b2 * c2
    C = (0.5 * a2 * (2 * p1 - t1) + 0.25 * r2 * (np.sin(2 * p2) - np.sin(2 * p1))
         + 0.5 * b2 * (c2 + 2 * p2 - np.pi))
    S = -a2 * np.log(np.cos(p1)) + 0.25 * r2 * (np.cos(2 * p1) - np.cos(2 * p2)) - b2 * np.log(np.sin(p2))
    return np.where(ok, A, 0.0), np.where(ok, C, 0.0), np.where(ok, S, 0.0)


def _disc_rect_moments(r, ax, bx, ay, by):
    r"""Integrals of (1, cos 2phi, sin 2phi) dA over the disc of radius r about the origin
    intersected with the rectangle [ax, bx] x [ay, by] (any position relative to the origin).

    With Phi(X, Y) the oriented integral over the rectangle spanned by the origin and (X, Y),
    the rectangle is Phi(bx, by) - Phi(ax, by) - Phi(bx, ay) + Phi(ax, ay). By the reflection
    symmetry of the integrands, Phi = sgn(X) sgn(Y) Q(|X|, |Y|) for 1 and cos 2phi, and
    Phi = Q(|X|, |Y|) for sin 2phi (odd in x and in y), with Q from ``_quadrant_moments``.
    Exactly zero where the disc does not reach the rectangle.
    """
    out = [0.0, 0.0, 0.0]
    for X, sx in ((bx, 1.0), (ax, -1.0)):
        for Y, sy in ((by, 1.0), (ay, -1.0)):
            A, C, S = _quadrant_moments(np.abs(X), np.abs(Y), r)
            sgn = np.sign(X) * np.sign(Y)
            out[0] = out[0] + sx * sy * sgn * A
            out[1] = out[1] + sx * sy * sgn * C
            out[2] = out[2] + sx * sy * S
    dx = np.maximum(np.maximum(ax, -bx), 0.0)
    dy = np.maximum(np.maximum(ay, -by), 0.0)
    reach = np.hypot(dx, dy) < r
    return tuple(np.where(reach, o, 0.0) for o in out)


# ---------------------------------------------------------------------------
# Bins, images, quadrature
# ---------------------------------------------------------------------------

def _second_bins(bins2, reflect):
    """Lower and upper edges of the second bin coordinate (pi or mu), reflected if asked."""
    bins2 = np.asarray(bins2, dtype=float)
    if reflect:
        return -bins2[1:], -bins2[:-1]
    return bins2[:-1], bins2[1:]


def _images(start, L_sub, boxsize):
    """Relative intervals [lo, lo + L_sub) of a sub-box's three nearest periodic images."""
    return [(start + n * boxsize, start + n * boxsize + L_sub) for n in (-1.0, 0.0, 1.0)]


def _n_gl_rmu(L_subboxes):
    """Gauss-Legendre nodes per rho segment in r-mu: _N_GL_RMU_FEW_JK at N_jk = 8, where each of
    the few realisations weighs heavily (8 nodes moved sigma_jk by up to 5.5% there), and
    _N_GL_RMU from N_jk = 27 (<= 0.6%; validation/run_box_jk_node_rule_check.py)."""
    return _N_GL_RMU_FEW_JK if L_subboxes <= 2 else _N_GL_RMU


def _gl_segments(segments, n_gl=None):
    """n_gl-point (default _N_GL_RMU) Gauss-Legendre nodes and weights (incl. the rho
    Jacobian) on the given segments; zero-length segments get zero weight."""
    x, w = np.polynomial.legendre.leggauss(_N_GL_RMU if n_gl is None else n_gl)
    rho = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in segments])
    wrho = np.concatenate([0.5 * (b - a) * w for a, b in segments]) * rho
    return rho, wrho


def _rmu_node_table(r_bins, rp_cut, n_gl=None):
    """Projected-radius nodes and weights for each r bin, shapes (n_r, 2 n_gl): bin i has its
    own n_gl-point Gauss-Legendre nodes on [rp_cut, r_i] and [r_i, r_{i+1}] (r_p >= rp_cut), so
    a bin's result does not depend on the other bins. Segments below rp_cut have zero length."""
    r_bins = np.asarray(r_bins, dtype=float)
    rows = [_gl_segments([(min(rp_cut, r_bins[i]), r_bins[i]),
                          (max(rp_cut, r_bins[i]), max(rp_cut, r_bins[i + 1]))], n_gl)
            for i in range(len(r_bins) - 1)]
    return np.array([r for r, _ in rows]), np.array([w for _, w in rows])


def _mu_slopes(mu):
    """h = rho * mu / sqrt(1 - mu^2): the height of the mu = const cone at projected radius rho,
    per unit rho (+-inf at mu = +-1)."""
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(mu) >= 1, np.sign(mu) * np.inf,
                        mu / np.sqrt(np.clip(1 - mu ** 2, 1e-300, None)))


def _shell_heights(rho_tab, r_bins):
    """Heights where bin i's spherical shell meets the cylinder of radius rho (bin i's
    nodes): a = sqrt(r_i^2 - rho^2), b = sqrt(r_{i+1}^2 - rho^2) (0 inside), shapes (n_q, n_r)."""
    r_bins = np.asarray(r_bins, dtype=float)
    a = np.sqrt(np.clip(r_bins[:-1][None] ** 2 - rho_tab.T ** 2, 0, None))
    b = np.sqrt(np.clip(r_bins[1:][None] ** 2 - rho_tab.T ** 2, 0, None))
    return a, b


def _rmu_height_overlap(zlo, zhi, a_sh, b_sh, m_lo, m_hi):
    """Length of the heights h that lie in the (r, mu) bin at projected radius rho AND in
    [zlo, zhi): the shell |h| in [a, b], the mu-bin h in [rho g_lo, rho g_hi].
    zlo, zhi: (G,); a_sh, b_sh: (n_r, 1, Q, 1); m_lo, m_hi: (n_r, 1, Q, n_mu)
    -> (n_r, G, Q, n_mu)."""
    zl, zh = zlo[None, :, None, None], zhi[None, :, None, None]
    out = 0.0
    for lo_s, hi_s in ((a_sh, b_sh), (-b_sh, -a_sh)):
        lo = np.maximum(lo_s, m_lo)
        hi = np.minimum(hi_s, m_hi)
        out = out + np.clip(np.minimum(hi, zh) - np.maximum(lo, zl), 0, None)
    return out


def _sky_rects(points, L_subboxes, L_sub, boxsize, reach):
    """For every galaxy and sub-box column c = ix * L + iy, the periodic images of the column's
    (x, y) rectangle relative to the galaxy that come within ``reach`` of it.
    Returns flat arrays (gal, col, ax, bx, ay, by).

    Vectorised over the 3L interval images per axis: the x and y intervals, and their
    distances from the galaxy, are (G, 3L) arrays, and one mask over (G, 3L, 3L) selects the
    rectangles in reach."""
    starts = (np.arange(L_subboxes)[:, None] * L_sub + np.array([-1.0, 0.0, 1.0])[None] * boxsize).ravel()
    idx = np.repeat(np.arange(L_subboxes), 3)                        # interval image -> column index
    lo = [starts[None, :] - points[:, a][:, None] for a in (0, 1)]   # (G, 3L)
    d = [np.maximum(np.maximum(l, -(l + L_sub)), 0.0) for l in lo]
    near_x, near_y = (di < reach for di in d)
    g, i, j = np.nonzero(near_x[:, :, None] & near_y[:, None, :]
                         & (np.hypot(d[0][:, :, None], d[1][:, None, :]) < reach))
    ax, ay = lo[0][g, i], lo[1][g, j]
    return g, idx[i] * L_subboxes + idx[j], ax, ax + L_sub, ay, ay + L_sub


def _scatter_sum(n_rows, rows, values):
    """out[r] = sum of values[i] over all i with rows[i] == r (rows may repeat). Equivalent to
    np.add.at on the first axis, but by sorting and np.add.reduceat, which is much faster."""
    out = np.zeros((n_rows,) + values.shape[1:])
    if len(rows) == 0:
        return out
    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    starts = np.flatnonzero(np.r_[True, rows_sorted[1:] != rows_sorted[:-1]])
    out[rows_sorted[starts]] = np.add.reduceat(values[order], starts, axis=0)
    return out


# ---------------------------------------------------------------------------
# Per-galaxy overlaps: sky-plane part per column, line-of-sight part per slab
# ---------------------------------------------------------------------------

def _sky_part(points, L_subboxes, *, boxsize, r_bins, rp_cut):
    """rp-pi sky-plane factors for every galaxy and sub-box column: the exact annulus moments
    (1, cos 2phi, sin 2phi), shape (3, G, L^2, n_r)."""
    L_sub = boxsize / L_subboxes
    r_bins = np.asarray(r_bins, dtype=float)
    G, ncol = len(points), L_subboxes ** 2
    gal, col, ax, bx, ay, by = _sky_rects(points, L_subboxes, L_sub, boxsize, r_bins[-1])
    radii = np.maximum(r_bins, rp_cut)          # r_p < rp_cut excluded: annulus from max(r1, rp_cut)
    F = _disc_rect_moments(radii[None, :], ax[:, None], bx[:, None], ay[:, None], by[:, None])
    vals = np.stack([np.diff(F[m], axis=1) for m in range(3)], axis=-1)          # (P, n_r, 3)
    out = _scatter_sum(G * ncol, gal * ncol + col, vals).reshape(G, ncol, len(r_bins) - 1, 3)
    return np.moveaxis(out, -1, 0)


def _los_part(points, L_subboxes, *, boxsize, bins2, reflect):
    """rp-pi line-of-sight factors: the overlap length of each pi bin with every sub-box slab
    iz (periodic), shape (G, L, n_pi)."""
    L_sub = boxsize / L_subboxes
    lo2, hi2 = _second_bins(bins2, reflect)
    z = points[:, 2]
    out = np.zeros((len(points), L_subboxes, len(lo2)))
    for iz in range(L_subboxes):
        for zl, zh in _images(iz * L_sub - z, L_sub, boxsize):
            out[:, iz] += np.clip(np.minimum(zh[:, None], hi2[None]) - np.maximum(zl[:, None], lo2[None]), 0, None)
    return out


def _sky_part_rmu(points, L_subboxes, moment_weights, *, boxsize, r_bins, rp_cut):
    """r-mu sky-plane factors, already weighted per galaxy. For every combination t of
    ``moment_weights`` (tuples (u_0, u_1, u_2) of per-galaxy arrays or None),

        A[r, t, c, g, q] = w_q(r) sum_m u_m[g] I_m(rho_q(r); g, c),

    with I_m the arc integrals of (1, cos 2phi, sin 2phi) over column c's periodic images about
    galaxy g, at bin r's node rho_q and with its quadrature weight w_q. Shape
    (n_r, T, L^2, G, n_q), so that A[r].reshape(T L^2, G n_q) is a view (no copy) for the
    matrix product with the line-of-sight factors.

    Arcs are evaluated only where the circle meets the rectangle (between its nearest point
    and its farthest corner), and accumulated with one bincount. That also adds up two images
    of the same column about one galaxy, which can both be in reach when r_max is close to L/2."""
    L_sub = boxsize / L_subboxes
    r_bins = np.asarray(r_bins, dtype=float)
    G, ncol, T = len(points), L_subboxes ** 2, len(moment_weights)
    rho_tab, w_tab = _rmu_node_table(r_bins, rp_cut, _n_gl_rmu(L_subboxes))
    n_r, Q = rho_tab.shape
    size = n_r * T * ncol * G * Q
    r_idx, q_idx = np.nonzero(w_tab != 0)          # zero-length segments (below rp_cut) add nothing
    nodes, node_w = rho_tab[r_idx, q_idx], w_tab[r_idx, q_idx]
    gal, col, ax, bx, ay, by = _sky_rects(points, L_subboxes, L_sub, boxsize, r_bins[-1])
    d_min = np.hypot(np.maximum(np.maximum(ax, -bx), 0.0), np.maximum(np.maximum(ay, -by), 0.0))
    d_max = np.hypot(np.maximum(np.abs(ax), np.abs(bx)), np.maximum(np.abs(ay), np.abs(by)))
    p, u = np.nonzero((nodes[None, :] > d_min[:, None]) & (nodes[None, :] < d_max[:, None]))
    if len(p) == 0:
        return np.zeros((n_r, T, ncol, G, Q))
    I = arc_integrals(nodes[u], ax[p], bx[p], ay[p], by[p])
    g = gal[p]
    flat = ((r_idx[u] * T * ncol + col[p]) * G + g) * Q + q_idx[u]          # index of (r, t = 0, c, g, q)
    vals = []
    for uw in moment_weights:
        v = 0.0
        for m, um in enumerate(uw):
            if um is not None:
                v = v + I[m] * um[g]
        vals.append(v * node_w[u])
    stride_t = ncol * G * Q
    out = np.bincount(np.concatenate([flat + t * stride_t for t in range(T)]),
                      weights=np.concatenate(vals), minlength=size)
    return out.reshape(n_r, T, ncol, G, Q)


def _los_part_rmu(points, L_subboxes, *, boxsize, r_bins, bins2, reflect, rp_cut):
    """r-mu line-of-sight factors: per (r, mu) bin, at each of bin r's rho nodes, the length of
    the bin's heights that fall in sub-box slab iz (periodic). Shape (n_r, G, n_q, L, n_mu), so
    that los[r].reshape(G n_q, L n_mu) is a view."""
    L_sub = boxsize / L_subboxes
    lo2, hi2 = _second_bins(bins2, reflect)
    z = points[:, 2]
    rho_tab, _ = _rmu_node_table(r_bins, rp_cut, _n_gl_rmu(L_subboxes))          # (n_r, Q)
    a_sh, b_sh = (h.T[:, None, :, None] for h in _shell_heights(rho_tab, r_bins))  # (n_r, 1, Q, 1)
    m_lo = (rho_tab[:, :, None] * _mu_slopes(lo2)[None, None])[:, None]            # (n_r, 1, Q, n_mu)
    m_hi = (rho_tab[:, :, None] * _mu_slopes(hi2)[None, None])[:, None]
    r_max = float(np.max(r_bins))
    out = np.zeros((rho_tab.shape[0], len(points), rho_tab.shape[1], L_subboxes, len(lo2)))
    for iz in range(L_subboxes):
        slab = out[:, :, :, iz]                                                    # view (n_r, G, Q, n_mu)
        for zl, zh in _images(iz * L_sub - z, L_sub, boxsize):
            near = np.nonzero((zh > -r_max) & (zl < r_max))[0]
            if len(near):
                slab[:, near] += _rmu_height_overlap(zl[near], zh[near], a_sh, b_sh, m_lo, m_hi)
    return out


def subbox_overlaps(points, k, L_subboxes, *, boxsize, r_bins, bins2, binning="rppi", reflect=False,
                    rp_cut=0.0):
    r"""Per galaxy, the volume of each bin (reflected along the line of sight if ``reflect``)
    that lies in sub-box k (periodic), and the same weighted by cos 2phi_s and sin 2phi_s.

    Parameters
    ----------
    points : ndarray (N, 3)
        Positions, line of sight last.
    k : int
        Sub-box index, k = ix * L^2 + iy * L + iz.
    L_subboxes : int
        Number of sub-boxes per side (N_jk = L_subboxes^3).
    boxsize : float
    r_bins, bins2 : ndarrays
        Bin edges: r_p and pi (``binning="rppi"``) or r and mu (``binning="rmu"``).
    binning : str
    reflect : bool
        Use the bin with the second coordinate reflected (pi -> -pi, mu -> -mu).
    rp_cut : float
        Exclude projected separations below rp_cut.

    Returns
    -------
    O, Oc, Os : ndarrays (N, n_r, n_2)
    """
    check_half_box(boxsize, r_bins, bins2, binning)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    L = L_subboxes
    c, iz = k // L, k % L                                            # column ix * L + iy, slab iz
    if binning == "rppi":
        sky = _sky_part(points, L, boxsize=boxsize, r_bins=r_bins, rp_cut=rp_cut)[:, :, c]   # (3, N, n_r)
        los = _los_part(points, L, boxsize=boxsize, bins2=bins2, reflect=reflect)[:, iz]      # (N, n_2)
        return tuple(sky[m][:, :, None] * los[:, None, :] for m in range(3))
    one = np.ones(len(points))
    moments = [(one, None, None), (None, one, None), (None, None, one)]
    sky = _sky_part_rmu(points, L, moments, boxsize=boxsize, r_bins=r_bins, rp_cut=rp_cut)[:, :, c]  # (n_r, 3, N, Q)
    los = _los_part_rmu(points, L, boxsize=boxsize, r_bins=r_bins, bins2=bins2, reflect=reflect,
                        rp_cut=rp_cut)[:, :, :, iz]                                                # (n_r, N, Q, n_2)
    return tuple(np.einsum("rgq,rgqm->grm", sky[:, m], los) for m in range(3))


# ---------------------------------------------------------------------------
# Per-realisation random terms
# ---------------------------------------------------------------------------

def _chunk_size(L_subboxes, r_bins, bins2, binning, boxsize):
    """Galaxies per chunk.
    rppi: the largest per-chunk array (galaxy x sub-box x bin) is ~_CHUNK_ELEMENTS.
    rmu: the temporaries of the whole chunk are ~_CHUNK_BYTES_RMU. They are dominated by the
    arc integrals, ~48 floats per (galaxy, column image within r_max, rho node); the sky and
    line-of-sight factors and their own-sub-box copies add 2 (n_r T L^2 Q + n_r Q L n_mu)."""
    n_r, n_2 = len(r_bins) - 1, len(bins2) - 1
    if binning == "rppi":
        return max(1, _CHUNK_ELEMENTS // (L_subboxes ** 3 * n_r * n_2))
    L_sub = boxsize / L_subboxes
    r_max = float(np.max(r_bins))
    images_per_axis = min(3 * L_subboxes, int(np.ceil(2 * r_max / L_sub)) + 1)
    Q = 2 * _n_gl_rmu(L_subboxes)
    per_galaxy = 48 * images_per_axis ** 2 * n_r * Q + 2 * (n_r * 3 * L_subboxes ** 2 * Q + n_r * Q * L_subboxes * n_2)
    return max(1, _CHUNK_BYTES_RMU // (8 * per_galaxy))


def _retained_overlap_sums(points, lab, L_subboxes, moment_weights, *, boxsize, r_bins, bins2, binning,
                           reflect, rp_cut, chunk_size, num_nodes=1):
    """For each sub-box k, sum over the galaxies NOT in k of sum_m u_m O^m_k, with O^m the
    overlap moments (1, cos 2phi, sin 2phi) and u_m = moment_weights[m] (per galaxy, or None).
    Returns a list of arrays (N_jk, n_r, n_2), one per requested combination.

    The sum over all galaxies is one matrix product per chunk: (galaxy x column) sky factors
    against (galaxy x slab) line-of-sight factors, contracted over galaxies (and rho nodes in
    r-mu). The galaxies' own sub-box is then subtracted.

    With num_nodes > 1 the chunks are grouped into contiguous blocks (about _BLOCKS_PER_NODE
    per worker, for load balance) that are summed in worker processes; each block's slice of
    the inputs is sent with its task. Chunk boundaries do not move, so only the order in which
    the partial sums are added changes (rounding)."""
    L = L_subboxes
    n_jk = L ** 3
    n_r, n_2 = len(r_bins) - 1, len(bins2) - 1
    T = len(moment_weights)
    kw = dict(boxsize=boxsize, r_bins=r_bins, bins2=bins2, binning=binning, reflect=reflect, rp_cut=rp_cut)
    n_chunks = -(-len(points) // chunk_size)
    n_blocks = min(n_chunks, _BLOCKS_PER_NODE * num_nodes) if num_nodes > 1 else min(n_chunks, 1)
    edges = [chunk_size * (n_chunks * b // n_blocks) for b in range(n_blocks)] + [len(points)]
    tasks = [(points[a:b], lab[a:b], L, [tuple(None if u is None else u[a:b] for u in uw) for uw in moment_weights],
              chunk_size, kw) for a, b in zip(edges[:-1], edges[1:])]
    if num_nodes > 1 and len(tasks) > 1:
        with worker_pool.active_pool(num_nodes) as pool:
            results = pool.map(_block_overlap_sums, tasks)
    else:
        results = [_block_overlap_sums(task) for task in tasks]
    totals = np.zeros((T, L * L, L, n_r, n_2))
    own = np.zeros((T, n_jk, n_r * n_2))
    for tot, own_b in results:
        totals += tot
        own += own_b
    return [(totals[t].reshape(n_jk, n_r, n_2) - own[t].reshape(n_jk, n_r, n_2)) for t in range(T)]


def _block_overlap_sums(task):
    """Worker (module level, so that it pickles): the summed ``_chunk_overlap_sums`` of one
    contiguous block of galaxies, chunk by chunk."""
    points, lab, L, moment_weights, chunk_size, kw = task
    totals, own = 0.0, 0.0
    for c0 in range(0, len(points), chunk_size):
        sl = slice(c0, c0 + chunk_size)
        uw_chunk = [tuple(None if u is None else u[sl] for u in uw) for uw in moment_weights]
        tot, own_c = _chunk_overlap_sums(points[sl], lab[sl], L, uw_chunk, **kw)
        totals = totals + tot
        own = own + own_c
    return totals, own


def _chunk_overlap_sums(x, lab_c, L, moment_weights, *, boxsize, r_bins, bins2, binning, reflect, rp_cut):
    """One chunk of galaxies' contribution to ``_retained_overlap_sums``: the sum over the chunk
    of sum_m u_m O^m for every (column, slab) sub-box, shape (T, L^2, L, n_r, n_2), and the
    same restricted to each galaxy's own sub-box, summed per sub-box, shape (T, N_jk, n_r n_2).
    ``moment_weights`` holds the chunk's slices."""
    G = len(x)
    n_jk = L ** 3
    n_r, n_2 = len(r_bins) - 1, len(bins2) - 1
    T = len(moment_weights)
    col_own, iz_own = lab_c // L, lab_c % L
    g = np.arange(G)
    tot = np.empty((T, L * L, L, n_r, n_2))
    own = np.zeros((T, n_jk, n_r * n_2))
    if binning == "rppi":
        sky = _sky_part(x, L, boxsize=boxsize, r_bins=r_bins, rp_cut=rp_cut)              # (3, G, L^2, n_r)
        los = _los_part(x, L, boxsize=boxsize, bins2=bins2, reflect=reflect)              # (G, L, n_2)
        for t, uw in enumerate(moment_weights):
            s = sum(sky[m] * u[:, None, None] for m, u in enumerate(uw) if u is not None)
            # (c, r) x (z, p) summed over galaxies
            tot[t] = np.einsum("gcr,gzp->czrp", s, los, optimize=True)
            own_vals = np.einsum("gr,gp->grp", s[g, col_own], los[g, iz_own])
            own[t] = _scatter_sum(n_jk, lab_c, own_vals.reshape(G, n_r * n_2))
        return tot, own
    A = _sky_part_rmu(x, L, moment_weights, boxsize=boxsize, r_bins=r_bins, rp_cut=rp_cut)  # (n_r, T, L^2, G, Q)
    los = _los_part_rmu(x, L, boxsize=boxsize, r_bins=r_bins, bins2=bins2, reflect=reflect,
                        rp_cut=rp_cut)                                                     # (n_r, G, Q, L, n_2)
    Q = A.shape[-1]
    # per r bin, all combinations at once: (combination x column, galaxy x node) @ (galaxy x node, slab x mu)
    for r in range(n_r):
        tot[:, :, :, r] = (A[r].reshape(T * L * L, G * Q) @ los[r].reshape(G * Q, L * n_2)).reshape(T, L * L, L, n_2)
    A_own = A[:, :, col_own, g]                                         # (n_r, T, G, Q)
    los_own = los[:, g, :, iz_own]                                      # (G, n_r, Q, n_2): advanced axes first
    own_vals = np.einsum("rtgq,grqm->tgrm", A_own, los_own)
    for t in range(T):
        own[t] = _scatter_sum(n_jk, lab_c, own_vals[t].reshape(G, n_r * n_2))
    return tot, own


def analytic_dr_terms(pos, shape, theta, e, lab_pos, lab_shape, L_subboxes, *, weight=None, weight_shape=None,
                      chunk_size=None, num_nodes=1, boxsize, r_bins, bins2, binning="rppi", rp_cut=0.0,
                      _single_pass=True):
    r"""Normalised per-realisation D.R_S, R_D.S and S+R_D (union deletion, infinite randoms).

    When the second-coordinate bins are symmetric about zero (always the case in measureia), both
    samples are handled in one geometry pass over their union; ``_single_pass=False`` forces one
    pass per sample (used by the tests to check that both give the same result).

    Parameters
    ----------
    pos, shape : ndarrays (N_D, 3), (N_S, 3)
        Position and shape samples, line of sight last.
    theta, e : ndarrays (N_S,)
        Major-axis angle in the projected plane and ellipticity of the shape sample.
    lab_pos, lab_shape : ndarrays of int
        Sub-box index of every galaxy (the deletion labels of the pair counts).
    L_subboxes : int
    weight, weight_shape : ndarrays or None
    chunk_size : int or None
        Galaxies per chunk (bounds peak memory; does not change the result beyond rounding).
    num_nodes : int
        Worker processes for the chunks (> 1 uses ``worker_pool.active_pool``, i.e. the
        measurement's pool when one is open). Does not change the result beyond rounding.
    boxsize, r_bins, bins2, binning, rp_cut
        As in ``subbox_overlaps``.

    Returns
    -------
    DR, RD, SpR : ndarrays (N_jk, n_r, n_2)
        Per retained pair: DR_k = sum_i w_i [V_b - O_ik] / (W_D,k (V - V_k)), RD_k likewise
        for the shape sample with the reflected bin, SpR_k = -sum_j w_j e_j Q_jk / (W_S,k (V - V_k)).
    """
    check_half_box(boxsize, r_bins, bins2, binning)
    pos = np.asarray(pos, dtype=float)
    shape = np.asarray(shape, dtype=float)
    lab_pos = np.asarray(lab_pos, dtype=int)
    lab_shape = np.asarray(lab_shape, dtype=int)
    w = np.ones(len(pos)) if weight is None else np.asarray(weight, dtype=float)
    ws = np.ones(len(shape)) if weight_shape is None else np.asarray(weight_shape, dtype=float)
    n_jk = L_subboxes ** 3
    V = boxsize ** 3
    Vk = V / n_jk
    Vb = bin_volumes(r_bins, bins2, binning, rp_cut)
    if chunk_size is None:
        chunk_size = _chunk_size(L_subboxes, r_bins, bins2, binning, boxsize)
    kw = dict(boxsize=boxsize, r_bins=r_bins, bins2=bins2, binning=binning, rp_cut=rp_cut, chunk_size=chunk_size,
              num_nodes=int(num_nodes or 1))
    ew = np.asarray(e, dtype=float) * ws
    theta = np.asarray(theta, dtype=float)
    shape_moments = (ew * np.cos(2 * theta), ew * np.sin(2 * theta))
    bins2 = np.asarray(bins2, dtype=float)
    union = _single_pass and np.allclose(bins2, -bins2[::-1], rtol=0, atol=1e-12 * np.abs(bins2).max())
    if union:
        # The reflected bin j is bin n-1-j, so the shape terms are the non-reflected overlaps in
        # reversed bin order, and both samples go through one geometry pass over their union.
        # Objects in both samples (identical coordinates) are computed once.
        points, inverse = np.unique(np.concatenate([pos, shape]), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        inv_p, inv_s = inverse[:len(pos)], inverse[len(pos):]
        lab = np.full(len(points), -1)
        lab[inv_p] = lab_pos
        union = np.array_equal(lab[inv_s][lab[inv_s] >= 0], lab_shape[lab[inv_s] >= 0])
    if union:
        lab[inv_s] = lab_shape
        to_union = lambda values, inv: np.bincount(inv, weights=values, minlength=len(points))
        sum_p, sum_s, sum_q = _retained_overlap_sums(
            points, lab, L_subboxes,
            [(to_union(w, inv_p), None, None), (to_union(ws, inv_s), None, None),
             (None, to_union(shape_moments[0], inv_s), to_union(shape_moments[1], inv_s))],
            reflect=False, **kw)
        sum_s, sum_q = sum_s[:, :, ::-1], sum_q[:, :, ::-1]
    else:
        (sum_p,) = _retained_overlap_sums(pos, lab_pos, L_subboxes, [(w, None, None)], reflect=False, **kw)
        sum_s, sum_q = _retained_overlap_sums(
            shape, lab_shape, L_subboxes, [(ws, None, None), (None,) + shape_moments], reflect=True, **kw)
    Wp = (w.sum() - np.bincount(lab_pos, weights=w, minlength=n_jk))[:, None, None]
    Wsk = (ws.sum() - np.bincount(lab_shape, weights=ws, minlength=n_jk))[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        DR = np.where(Wp > 0, (Wp * Vb[None] - sum_p) / (Wp * (V - Vk)), 0.0)
        RD = np.where(Wsk > 0, (Wsk * Vb[None] - sum_s) / (Wsk * (V - Vk)), 0.0)
        SpR = np.where(Wsk > 0, -sum_q / (Wsk * (V - Vk)), 0.0)
    return DR, RD, SpR


# ---------------------------------------------------------------------------
# Exact union RR
# ---------------------------------------------------------------------------

def _gl_cos(a, b, n):
    """Gauss-Legendre nodes/weights on [a, b] after rho = a + (b - a)(1 - cos(pi u))/2, which
    removes square-root behaviour at both segment ends."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (x + 1)
    rho = a + (b - a) * 0.5 * (1 - np.cos(np.pi * u))
    return rho, 0.5 * w * (b - a) * 0.5 * np.pi * np.sin(np.pi * u)


def _ring_self_overlap(rho, L):
    """g(rho) = int_0^{2pi} max(0, L - rho|cos phi|) max(0, L - rho|sin phi|) d phi (closed form)."""
    rho = np.asarray(rho, dtype=float)
    with np.errstate(divide="ignore"):
        q = np.where(rho > 0, np.minimum(L / np.where(rho > 0, rho, 1.0), 1.0), 1.0)
    pa, pb = np.arccos(q), np.arcsin(q)

    def P(p):
        return L ** 2 * p - L * rho * (np.sin(p) - np.cos(p)) + 0.5 * rho ** 2 * np.sin(p) ** 2

    return np.where(pb > pa, 4.0 * (P(pb) - P(pa)), 0.0)


def _height_weight_integral(lo, hi, L):
    """int_lo^hi max(0, L - |h|) dh for lo <= hi (elementwise)."""
    def H(z):
        m = np.minimum(np.abs(z), L)
        return np.sign(z) * (L * m - 0.5 * m ** 2)
    return H(hi) - H(lo)


def _bin_rho_integrals(r_bins, bins2, binning, rp_cut, L_sub):
    """Per bin, int rho drho 2pi J0(rho) (volume, restricted to r_p >= rp_cut) and
    int rho drho g(rho) J(rho) (self-overlap), J0 / J the length / (L_sub - |h|)-weighted
    length of the bin's heights at projected radius rho. Gauss-Legendre (cosine substitution)
    on segments split at every kink of the integrand."""
    r_bins, bins2 = np.asarray(r_bins, dtype=float), np.asarray(bins2, dtype=float)
    n_r, n_2 = len(r_bins) - 1, len(bins2) - 1
    vol = np.zeros((n_r, n_2))
    sov = np.zeros((n_r, n_2))
    kinks = [L_sub, np.sqrt(2.0) * L_sub, rp_cut]
    for i in range(n_r):
        r1, r2 = r_bins[i], r_bins[i + 1]
        for j in range(n_2):
            b1, b2 = bins2[j], bins2[j + 1]
            if binning == "rppi":
                lo = max(r1, rp_cut)
                brk = [lo, r2] + kinks
                g1, g2 = None, None
            else:
                lo = rp_cut
                g1, g2 = _mu_slopes([b1, b2])
                brk = [lo, r1, r2] + kinks
                for r in (r1, r2):
                    for mu in (b1, b2):
                        if abs(mu) < 1:
                            brk.append(r * np.sqrt(1 - mu ** 2))
                    if r > L_sub:
                        brk.append(np.sqrt(r ** 2 - L_sub ** 2))
                for g in (g1, g2):
                    if np.isfinite(g) and g != 0:
                        brk.append(L_sub / abs(g))
            brk = np.unique([p for p in brk if lo <= p <= r2])
            if len(brk) < 2:
                continue
            for a, b in zip(brk[:-1], brk[1:]):
                rho, wq = _gl_cos(a, b, _N_GL_F)
                if binning == "rppi":
                    J0 = np.full_like(rho, b2 - b1)
                    J = np.full_like(rho, _height_weight_integral(b1, b2, L_sub))
                else:
                    ah = np.sqrt(np.clip(r1 ** 2 - rho ** 2, 0, None))
                    bh = np.sqrt(np.clip(r2 ** 2 - rho ** 2, 0, None))
                    mlo = rho * g1 if np.isfinite(g1) else np.full_like(rho, -np.inf)
                    mhi = rho * g2 if np.isfinite(g2) else np.full_like(rho, np.inf)
                    J0 = np.zeros_like(rho)
                    J = np.zeros_like(rho)
                    for lo_s, hi_s in ((ah, bh), (-bh, -ah)):
                        hl = np.maximum(lo_s, mlo)
                        hh = np.maximum(np.minimum(hi_s, mhi), hl)
                        J0 += hh - hl
                        J += _height_weight_integral(hl, hh, L_sub)
                vol[i, j] += np.sum(wq * rho * 2 * np.pi * J0)
                sov[i, j] += np.sum(wq * rho * _ring_self_overlap(rho, L_sub) * J)
    return vol, sov


def bin_volumes(r_bins, bins2, binning="rppi", rp_cut=0.0):
    """Volume of every bin, restricted to projected separations >= rp_cut. Exact formulae
    without rp_cut (pi (r2^2 - r1^2) dpi, or (2 pi / 3)(r2^3 - r1^3) dmu); in r-mu with rp_cut
    by the same quadrature as f(b)."""
    r_bins, bins2 = np.asarray(r_bins, dtype=float), np.asarray(bins2, dtype=float)
    if binning == "rppi":
        r = np.maximum(r_bins, rp_cut)
        return np.pi * np.diff(r ** 2)[:, None] * np.diff(bins2)[None, :]
    if rp_cut <= 0:
        return (2.0 * np.pi / 3.0) * np.diff(r_bins ** 3)[:, None] * np.diff(bins2)[None, :]
    vol, _ = _bin_rho_integrals(r_bins, bins2, binning, rp_cut, L_sub=1.0)
    return vol


def self_overlap_fraction(L_subboxes, *, boxsize, r_bins, bins2, binning="rppi", rp_cut=0.0):
    r"""f(b) = (1 / V_b) int_b prod_a max(0, L_sub - |s_a|) / L_sub^3 d^3 s, the probability
    that both ends of a pair with separation uniform in bin b lie in the same sub-box, given
    that one does (restricted to r_p >= rp_cut). Deterministic quadrature: the angular
    integral is closed form, the projected radius is integrated by Gauss-Legendre on segments
    split at every kink."""
    check_half_box(boxsize, r_bins, bins2, binning)
    L_sub = boxsize / L_subboxes
    vol, sov = _bin_rho_integrals(r_bins, bins2, binning, rp_cut, L_sub)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, sov / (vol * L_sub ** 3), 0.0)


def exact_union_rr(L_subboxes, *, boxsize, r_bins, bins2, binning="rppi", rp_cut=0.0):
    r"""Normalised exact union RR_k (per retained pair, the same for every realisation):
    (V_b / V) (1 - 2v + v f(b)) / (1 - v)^2 with v = 1 / N_jk, V_b restricted to r_p >= rp_cut.
    Reduces to measureia's previous V_b / (V - V_k) where f -> 1 (separations << L_sub)."""
    v = 1.0 / L_subboxes ** 3
    f = self_overlap_fraction(L_subboxes, boxsize=boxsize, r_bins=r_bins, bins2=bins2, binning=binning,
                              rp_cut=rp_cut)
    Vb = bin_volumes(r_bins, bins2, binning, rp_cut)
    return Vb / boxsize ** 3 * (1 - 2 * v + v * f) / (1 - v) ** 2
