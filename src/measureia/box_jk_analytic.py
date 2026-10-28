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
integrated with 16-point Gauss-Legendre quadrature on [0, r_i] and [r_i, r_{i+1}], separately
for each r bin (so a bin does not depend on the other bins): up to ~6e-3 of V_b for a single
galaxy, which changes the jackknife errors by <= 0.1% (``validation/run_box_jk_quadrature_check.py``,
``validation/run_box_jk_node_layout.py``). The sub-box self-overlap f(b) is computed by
deterministic quadrature split at every kink.

Everything here is plain numpy, vectorised over galaxies in bounded-memory chunks. The
sky-plane part of an overlap depends only on the column (ix, iy) of the sub-box, so it is
computed once per column and combined with the line-of-sight part of every sub-box in that
column by one matrix product.
"""

import numpy as np

__all__ = [
    "arc_integrals",
    "subbox_overlaps",
    "analytic_dr_terms",
    "self_overlap_fraction",
    "exact_union_rr",
    "bin_volumes",
    "check_half_box",
]

_N_GL_RMU = 16       # Gauss-Legendre nodes per rho segment (r-mu overlaps)
_N_GL_F = 32         # nodes per segment for the self-overlap fraction f(b)
_CHUNK_ELEMENTS = 10_000_000  # target size (float64 elements, ~80 MB) of the largest per-chunk array


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


def _gl_segments(segments):
    """16-point Gauss-Legendre nodes and weights (incl. the rho Jacobian) on the given
    segments; zero-length segments get zero weight."""
    x, w = np.polynomial.legendre.leggauss(_N_GL_RMU)
    rho = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in segments])
    wrho = np.concatenate([0.5 * (b - a) * w for a, b in segments]) * rho
    return rho, wrho


def _rmu_node_table(r_bins, rp_cut):
    """Projected-radius nodes and weights for each r bin, shapes (n_r, 32): bin i has its own
    16-point Gauss-Legendre nodes on [rp_cut, r_i] and [r_i, r_{i+1}] (r_p >= rp_cut), so a
    bin's result does not depend on the other bins. Segments below rp_cut have zero length."""
    r_bins = np.asarray(r_bins, dtype=float)
    rows = [_gl_segments([(min(rp_cut, r_bins[i]), r_bins[i]),
                          (max(rp_cut, r_bins[i]), max(rp_cut, r_bins[i + 1]))])
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
    zlo, zhi: (G,); a_sh, b_sh: (Q, n_r); m_lo, m_hi: (Q, n_r, n_mu) -> (G, Q, n_r, n_mu)."""
    zl, zh = zlo[:, None, None, None], zhi[:, None, None, None]
    out = 0.0
    for lo_s, hi_s in ((a_sh, b_sh), (-b_sh, -a_sh)):
        lo = np.maximum(lo_s[None, :, :, None], m_lo[None])
        hi = np.minimum(hi_s[None, :, :, None], m_hi[None])
        out = out + np.clip(np.minimum(hi, zh) - np.maximum(lo, zl), 0, None)
    return out


def _sky_rects(points, L_subboxes, L_sub, boxsize, reach):
    """For every galaxy and sub-box column c = ix * L + iy, the periodic images of the column's
    (x, y) rectangle relative to the galaxy that come within ``reach`` of it.
    Returns flat arrays (gal, col, ax, bx, ay, by)."""
    gals, cols, rects = [], [], []
    for ix in range(L_subboxes):
        xs = _images(ix * L_sub - points[:, 0], L_sub, boxsize)
        for iy in range(L_subboxes):
            ys = _images(iy * L_sub - points[:, 1], L_sub, boxsize)
            for xl, xh in xs:
                dx = np.maximum(np.maximum(xl, -xh), 0.0)
                for yl, yh in ys:
                    dy = np.maximum(np.maximum(yl, -yh), 0.0)
                    near = np.nonzero(np.hypot(dx, dy) < reach)[0]
                    if len(near):
                        gals.append(near)
                        cols.append(np.full(len(near), ix * L_subboxes + iy))
                        rects.append(np.column_stack([xl[near], xh[near], yl[near], yh[near]]))
    if not gals:
        e = np.zeros(0)
        return e.astype(int), e.astype(int), e, e, e, e
    rect = np.concatenate(rects)
    return np.concatenate(gals), np.concatenate(cols), rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3]


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

def _sky_part(points, L_subboxes, *, boxsize, r_bins, binning, rp_cut):
    """Sky-plane factors for every galaxy and sub-box column, three moments (1, cos 2phi,
    sin 2phi).
    rppi: annulus moments, shape (3, G, L^2, n_r) — exact.
    rmu:  arc integrals at each bin's rho nodes times the node weights, shape
          (3, G, L^2, n_r, n_q); the arcs are evaluated once per distinct node."""
    L_sub = boxsize / L_subboxes
    r_bins = np.asarray(r_bins, dtype=float)
    G, ncol = len(points), L_subboxes ** 2
    gal, col, ax, bx, ay, by = _sky_rects(points, L_subboxes, L_sub, boxsize, r_bins[-1])
    if binning == "rppi":
        radii = np.maximum(r_bins, rp_cut)          # r_p < rp_cut excluded: annulus from max(r1, rp_cut)
        F = _disc_rect_moments(radii[None, :], ax[:, None], bx[:, None], ay[:, None], by[:, None])
        vals = np.stack([np.diff(F[m], axis=1) for m in range(3)], axis=-1)          # (P, n_r, 3)
        out = _scatter_sum(G * ncol, gal * ncol + col, vals).reshape(G, ncol, len(r_bins) - 1, 3)
        return np.moveaxis(out, -1, 0)
    rho_tab, w_tab = _rmu_node_table(r_bins, rp_cut)
    nodes, where = np.unique(rho_tab, return_inverse=True)
    where = where.reshape(rho_tab.shape)
    # the circle meets the rectangle only between its nearest point and its farthest corner;
    # elsewhere the arcs are exactly zero and are not evaluated
    d_min = np.hypot(np.maximum(np.maximum(ax, -bx), 0.0), np.maximum(np.maximum(ay, -by), 0.0))
    d_max = np.hypot(np.maximum(np.abs(ax), np.abs(bx)), np.maximum(np.abs(ay), np.abs(by)))
    p, u = np.nonzero((nodes[None, :] > d_min[:, None]) & (nodes[None, :] < d_max[:, None]))
    I = np.zeros((len(ax), len(nodes), 3))
    I[p, u] = np.stack(arc_integrals(nodes[u], ax[p], bx[p], ay[p], by[p]), axis=-1)
    vals = I[:, where] * w_tab[None, :, :, None]                                     # (P, n_r, n_q, 3)
    out = _scatter_sum(G * ncol, gal * ncol + col, vals).reshape((G, ncol) + rho_tab.shape + (3,))
    return np.moveaxis(out, -1, 0)


def _los_part(points, L_subboxes, *, boxsize, r_bins, bins2, binning, reflect, rp_cut):
    """Line-of-sight factors for every galaxy and sub-box slab iz.
    rppi: overlap length of each pi bin with the slab, shape (G, L, n_pi).
    rmu:  height overlap per (r, mu) bin at that bin's rho nodes, shape (G, L, n_q, n_r, n_mu)."""
    L_sub = boxsize / L_subboxes
    lo2, hi2 = _second_bins(bins2, reflect)
    z = points[:, 2]
    if binning == "rppi":
        out = np.zeros((len(points), L_subboxes, len(lo2)))
        for iz in range(L_subboxes):
            for zl, zh in _images(iz * L_sub - z, L_sub, boxsize):
                out[:, iz] += np.clip(np.minimum(zh[:, None], hi2[None]) - np.maximum(zl[:, None], lo2[None]), 0, None)
        return out
    rho_tab, _ = _rmu_node_table(r_bins, rp_cut)
    a_sh, b_sh = _shell_heights(rho_tab, r_bins)
    m_lo = rho_tab.T[:, :, None] * _mu_slopes(lo2)[None, None]
    m_hi = rho_tab.T[:, :, None] * _mu_slopes(hi2)[None, None]
    r_max = float(np.max(r_bins))
    out = np.zeros((len(points), L_subboxes, rho_tab.shape[1], len(r_bins) - 1, len(lo2)))
    for iz in range(L_subboxes):
        for zl, zh in _images(iz * L_sub - z, L_sub, boxsize):
            near = np.nonzero((zh > -r_max) & (zl < r_max))[0]
            if len(near):
                out[near, iz] += _rmu_height_overlap(zl[near], zh[near], a_sh, b_sh, m_lo, m_hi)
    return out


def _combine(sky, los, binning):
    """Overlap of every galaxy with every sub-box: (..., G, L^2, a) x (G, L, ...) -> per galaxy
    and sub-box k = c * L + iz."""
    if binning == "rppi":
        return np.einsum("...gcr,gzp->...gczrp", sky, los)
    return np.einsum("...gcrq,gzqrm->...gczrm", sky, los)


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
    ix, iy, iz = k // (L * L), (k // L) % L, k % L
    one = np.zeros(L * L, dtype=bool)
    one[ix * L + iy] = True
    kw = dict(boxsize=boxsize, r_bins=r_bins, binning=binning, rp_cut=rp_cut)
    sky = _sky_part(points, L, **kw)[:, :, one]                       # (3, N, 1, ...)
    los = _los_part(points, L, bins2=bins2, reflect=reflect, **kw)[:, iz:iz + 1]
    full = _combine(sky, los, binning)                               # (3, N, 1, 1, n_r, n_2)
    return tuple(full[m, :, 0, 0] for m in range(3))


# ---------------------------------------------------------------------------
# Per-realisation random terms
# ---------------------------------------------------------------------------

def _chunk_size(L_subboxes, r_bins, bins2, binning):
    if binning == "rppi":
        per = L_subboxes ** 3 * (len(r_bins) - 1) * (len(bins2) - 1)
    else:
        per = L_subboxes * _rmu_node_table(r_bins, 0.0)[0].shape[1] * (len(r_bins) - 1) * (len(bins2) - 1)
    return max(1, _CHUNK_ELEMENTS // per)


def _retained_overlap_sums(points, lab, L_subboxes, moment_weights, *, boxsize, r_bins, bins2, binning,
                           reflect, rp_cut, chunk_size):
    """For each sub-box k, sum over the galaxies NOT in k of sum_m u_m O^m_k, with O^m the
    overlap moments (1, cos 2phi, sin 2phi) and u_m = moment_weights[m] (per galaxy, or None).
    Returns a list of arrays (N_jk, n_r, n_2), one per requested combination.

    The sum over all galaxies is one matrix product per chunk: (galaxy x column) sky factors
    against (galaxy x slab) line-of-sight factors, contracted over galaxies (and rho nodes in
    r-mu). The galaxies' own sub-box is then subtracted."""
    L = L_subboxes
    n_jk = L ** 3
    n_r, n_2 = len(r_bins) - 1, len(bins2) - 1
    kw = dict(boxsize=boxsize, r_bins=r_bins, binning=binning, rp_cut=rp_cut)
    totals = [np.zeros((L * L, L, n_r, n_2)) for _ in moment_weights]
    own = [np.zeros((n_jk, n_r * n_2)) for _ in moment_weights]
    for c0 in range(0, len(points), chunk_size):
        sl = slice(c0, c0 + chunk_size)
        x = points[sl]
        G = len(x)
        sky = _sky_part(x, L, **kw)
        los = _los_part(x, L, bins2=bins2, reflect=reflect, **kw)
        lab_c = lab[sl]
        col_own, iz_own = lab_c // L, lab_c % L
        for t, uw in enumerate(moment_weights):
            s = sum(sky[m] * u[sl].reshape((G,) + (1,) * (sky.ndim - 2))
                    for m, u in enumerate(uw) if u is not None)
            if binning == "rppi":
                # (c, r) x (z, p) summed over galaxies
                tot = np.einsum("gcr,gzp->czrp", s, los, optimize=True)
                own_vals = np.einsum("gr,gp->grp", s[np.arange(G), col_own], los[np.arange(G), iz_own])
            else:
                # per r bin: (column, galaxy x node) @ (galaxy x node, slab x mu)
                Q = s.shape[-1]
                tot = np.empty((L * L, L, n_r, n_2))
                for r in range(n_r):
                    tot[:, :, r] = (s[:, :, r].transpose(1, 0, 2).reshape(L * L, G * Q)
                                    @ los[:, :, :, r].transpose(0, 2, 1, 3).reshape(G * Q, L * n_2)
                                    ).reshape(L * L, L, n_2)
                own_vals = np.einsum("grq,gqrm->grm", s[np.arange(G), col_own], los[np.arange(G), iz_own])
            totals[t] += tot
            own[t] += _scatter_sum(n_jk, lab_c, own_vals.reshape(G, n_r * n_2))
    return [tot.reshape(n_jk, n_r, n_2) - o.reshape(n_jk, n_r, n_2) for tot, o in zip(totals, own)]


def analytic_dr_terms(pos, shape, theta, e, lab_pos, lab_shape, L_subboxes, *, weight=None, weight_shape=None,
                      chunk_size=None, boxsize, r_bins, bins2, binning="rppi", rp_cut=0.0, _single_pass=True):
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
        chunk_size = _chunk_size(L_subboxes, r_bins, bins2, binning)
    kw = dict(boxsize=boxsize, r_bins=r_bins, bins2=bins2, binning=binning, rp_cut=rp_cut, chunk_size=chunk_size)
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
