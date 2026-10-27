"""
tests/box_jk_reference.py
=========================
Independent reference implementations ("oracles") for the analytic box-jackknife edge
correction in ``measureia.box_jk_analytic``. Used only by the tests.

- ``partial_shell_rppi`` / ``partial_shell_rmu`` are frozen copies of the loop-based
  prototype in ``validation/jk_cross_pairs_lib.py`` (``analytic_partial_shell`` and
  ``analytic_partial_shell_rmu``), which was validated against a 10x random catalogue on
  100 sparse, 100 strong-clustering and 40 dense mocks
  (``validation/box_jk_analytic_randoms.md``). The only change is an optional per-galaxy
  weight on the volume sums (the prototype was unit-weight only).
- ``mc_overlap`` is a brute-force Monte Carlo of the same overlaps, one galaxy at a time.
- ``precise_overlap_rppi`` is an adaptive-quadrature rp-pi overlap (~1e-12 of V_b), the
  reference for the module's exact closed form.
- ``self_overlap_fraction_grid`` integrates the sub-box self-overlap V_ov(s) over a bin on a
  dense deterministic grid (no shared code with the module under test).

Conventions (as in measureia): the line of sight is the last coordinate; a bin is
(r_p, pi) for ``binning="rppi"`` and (r, mu_r) for ``binning="rmu"``; sub-box labels follow
``k = ix * L^2 + iy * L + iz`` with ``i = floor(x / L_sub)``.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers (frozen prototype)
# ---------------------------------------------------------------------------

def shell_volumes(r_bins, bins2, binning="rppi"):
    """Bin volumes: pi (rp2^2 - rp1^2) dpi for rppi, (2 pi / 3)(r2^3 - r1^3) dmu for rmu."""
    r_bins, bins2 = np.asarray(r_bins, float), np.asarray(bins2, float)
    if binning == "rppi":
        return np.pi * np.diff(r_bins ** 2)[:, None] * np.diff(bins2)[None, :]
    return (2.0 * np.pi / 3.0) * np.diff(r_bins ** 3)[:, None] * np.diff(bins2)[None, :]


def arc_integrals(r, ax, bx, ay, by):
    """Integrals of 1, cos 2phi and sin 2phi d phi over the arcs of the circle of radius r
    (scalar) about the origin that lie inside the rectangles [ax, bx] x [ay, by] (arrays)."""
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


def _image_intervals(lo_rel, width, boxsize):
    return [(lo_rel + n * boxsize, lo_rel + n * boxsize + width) for n in (-1, 0, 1)]


def _interval_overlap(lo1, hi1, lo2, hi2):
    return np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None)


def partial_shell_rppi(points, lab, n_patch, *, boxsize, r_bins, pi_bins, reflect=False,
                       theta=None, e=None, w=None, n_gl=16):
    """Per sub-box k, sums over galaxies NOT in k of w_i O_ik(b) and (if theta is given)
    e_i Q_ik(b), with O the overlap volume of the galaxy's (rp, pi) bin with sub-box k and Q
    the same weighted by cos 2(phi - theta_i). Returns (sum_vol[k, r, pi], sum_ep or None)."""
    w = np.ones(len(points)) if w is None else np.asarray(w, float)
    L = round(n_patch ** (1 / 3))
    L_sub = boxsize / L
    n_r, n_pi = len(r_bins) - 1, len(pi_bins) - 1
    p_lo = -pi_bins[1:] if reflect else pi_bins[:-1]
    p_hi = -pi_bins[:-1] if reflect else pi_bins[1:]
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
    sum_vol = np.zeros((n_patch, n_r, n_pi))
    sum_ep = np.zeros((n_patch, n_r, n_pi)) if theta is not None else None
    c2t = np.cos(2 * theta) if theta is not None else None
    s2t = np.sin(2 * theta) if theta is not None else None
    for k in range(n_patch):
        ix, iy, iz = k // (L * L), (k // L) % L, k % L
        sel = np.where(lab != k)[0]
        x = points[sel]
        Lz = np.zeros((len(sel), n_pi))
        for (zlo, zhi) in _image_intervals(iz * L_sub - x[:, 2], L_sub, boxsize):
            Lz += np.clip(np.minimum(zhi[:, None], p_hi[None]) - np.maximum(zlo[:, None], p_lo[None]), 0, None)
        has_z = Lz.sum(axis=1) > 0
        A = np.zeros((len(sel), n_r))
        Ac = np.zeros((len(sel), n_r))
        As = np.zeros((len(sel), n_r))
        for (xlo, xhi) in _image_intervals(ix * L_sub - x[:, 0], L_sub, boxsize):
            for (ylo, yhi) in _image_intervals(iy * L_sub - x[:, 1], L_sub, boxsize):
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
        sum_vol[k] = (w[sel][:, None] * A).T @ Lz
        if theta is not None:
            ep = e[sel][:, None] * (Ac * c2t[sel][:, None] + As * s2t[sel][:, None])
            sum_ep[k] = ep.T @ Lz
    return sum_vol, sum_ep


def partial_shell_rmu(points, lab, n_patch, *, boxsize, r_bins, mu_bins, reflect=False,
                      theta=None, e=None, w=None, n_gl=16, chunk=256):
    """r-mu counterpart of ``partial_shell_rppi`` (cylindrical (rho, h) coordinates, exact
    h-intervals, Gauss-Legendre in rho on segments split at the r_bins edges)."""
    w = np.ones(len(points)) if w is None else np.asarray(w, float)
    L = round(n_patch ** (1 / 3))
    L_sub = boxsize / L
    n_r, n_mu = len(r_bins) - 1, len(mu_bins) - 1
    mlo = -mu_bins[1:] if reflect else mu_bins[:-1]
    mhi = -mu_bins[:-1] if reflect else mu_bins[1:]
    with np.errstate(divide="ignore", invalid="ignore"):  # sign(0) * inf in the unused np.where branch
        g_lo = np.where(np.abs(mlo) >= 1, np.sign(mlo) * np.inf, mlo / np.sqrt(np.clip(1 - mlo ** 2, 1e-300, None)))
        g_hi = np.where(np.abs(mhi) >= 1, np.sign(mhi) * np.inf, mhi / np.sqrt(np.clip(1 - mhi ** 2, 1e-300, None)))
    edges = np.concatenate([[0.0], r_bins])
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
    rho = np.concatenate([0.5 * (b - a) * gl_x + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    wrho = np.concatenate([0.5 * (b - a) * gl_w for a, b in zip(edges[:-1], edges[1:])]) * rho
    a_sh = np.sqrt(np.clip(r_bins[:-1][None] ** 2 - rho[:, None] ** 2, 0, None))
    b_sh = np.sqrt(np.clip(r_bins[1:][None] ** 2 - rho[:, None] ** 2, 0, None))
    m_lo = rho[:, None] * g_lo[None]
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
            for (xlo, xhi) in _image_intervals(ix * L_sub - x[:, 0], L_sub, boxsize):
                for (ylo, yhi) in _image_intervals(iy * L_sub - x[:, 1], L_sub, boxsize):
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
            Lh = np.zeros((G, len(rho), n_r, n_mu))
            for (zlo, zhi) in _image_intervals(iz * L_sub - x[:, 2], L_sub, boxsize):
                zl, zh = zlo[:, None, None, None], zhi[:, None, None, None]
                for sgn in (1, -1):
                    lo_s = a_sh if sgn == 1 else -b_sh
                    hi_s = b_sh if sgn == 1 else -a_sh
                    lo = np.maximum(lo_s[None, :, :, None], m_lo[None, :, None, :])
                    hi = np.minimum(hi_s[None, :, :, None], m_hi[None, :, None, :])
                    Lh += _interval_overlap(lo, hi, zl, zh)
            sum_vol[k] += np.einsum("gq,q,gqrm->rm", w[sel][:, None] * I0, wrho, Lh)
            if theta is not None:
                ang = e[sel][:, None] * (Ic * np.cos(2 * theta[sel])[:, None] + Is * np.sin(2 * theta[sel])[:, None])
                sum_ep[k] += np.einsum("gq,q,gqrm->rm", ang, wrho, Lh)
    return sum_vol, sum_ep


def dr_terms(pos, shape_pos, theta, e, lab_pos, lab_shape, n_patch, *, boxsize, r_bins, bins2,
             binning="rppi", w=None, w_shape=None):
    """Normalised union-deletion D.R_S, R_D.S and S+R_D per realisation in the
    infinite-randoms limit (the prototype's ``analytic_dr_terms``, with weights):
      DR_k  = sum_{i in D, not k} w_i [V_b - O_ik] / [W_D,k (V - V_k)]
      RD_k  = sum_{j in S, not k} w_j [V_b - O_jk(reflected b)] / [W_S,k (V - V_k)]
      S+R_k = - sum_{j in S, not k} w_j e_j Q_jk(reflected b) / [W_S,k (V - V_k)]"""
    w = np.ones(len(pos)) if w is None else np.asarray(w, float)
    w_shape = np.ones(len(shape_pos)) if w_shape is None else np.asarray(w_shape, float)
    V = boxsize ** 3
    Vk = V / n_patch
    Vbin = shell_volumes(r_bins, bins2, binning)
    fn = partial_shell_rppi if binning == "rppi" else partial_shell_rmu
    key = "pi_bins" if binning == "rppi" else "mu_bins"
    sv_p, _ = fn(pos, lab_pos, n_patch, boxsize=boxsize, r_bins=r_bins, **{key: bins2}, w=w)
    sv_s, sep_s = fn(shape_pos, lab_shape, n_patch, boxsize=boxsize, r_bins=r_bins, **{key: bins2},
                     reflect=True, theta=theta, e=e * w_shape, w=w_shape)
    Wp = w.sum() - np.bincount(lab_pos, weights=w, minlength=n_patch)
    Ws = w_shape.sum() - np.bincount(lab_shape, weights=w_shape, minlength=n_patch)
    DR = (Wp[:, None, None] * Vbin[None] - sv_p) / (Wp[:, None, None] * (V - Vk))
    RD = (Ws[:, None, None] * Vbin[None] - sv_s) / (Ws[:, None, None] * (V - Vk))
    SpR = -sep_s / (Ws[:, None, None] * (V - Vk))
    return DR, RD, SpR


# ---------------------------------------------------------------------------
# Brute-force Monte Carlo overlap (one galaxy, one sub-box)
# ---------------------------------------------------------------------------

def sample_bin(rng, n, r1, r2, b1, b2, binning="rppi"):
    """n separation vectors uniform in the volume of one bin (LOS = last axis)."""
    u = rng.random((3, n))
    phi = 2 * np.pi * u[1]
    if binning == "rppi":
        rp = np.sqrt(r1 ** 2 + u[0] * (r2 ** 2 - r1 ** 2))
        return np.column_stack([rp * np.cos(phi), rp * np.sin(phi), b1 + u[2] * (b2 - b1)])
    r = np.cbrt(r1 ** 3 + u[0] * (r2 ** 3 - r1 ** 3))
    mu = b1 + u[2] * (b2 - b1)
    st = np.sqrt(1 - mu ** 2)
    return np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu])


def mc_overlap(x, k, L_subboxes, *, boxsize, r1, r2, b1, b2, binning="rppi", reflect=False,
               theta=0.0, rp_cut=0.0, n=400_000, seed=0):
    """Monte Carlo estimate of (O, Q) for one galaxy at x, sub-box k, one bin:
    O = V_b * P(x + s in C_k), Q = V_b * E[1(x + s in C_k) cos 2(phi_s - theta)],
    with s uniform in the bin (reflected: s -> -s), restricted to r_p > rp_cut.
    Returns (O, Q, sigma_O, sigma_Q)."""
    rng = np.random.default_rng(seed)
    s = sample_bin(rng, n, r1, r2, b1, b2, binning)
    if reflect:
        s = -s
    Vb = shell_volumes([r1, r2], [b1, b2], binning)[0, 0]
    L_sub = boxsize / L_subboxes
    ix, iy, iz = k // (L_subboxes ** 2), (k // L_subboxes) % L_subboxes, k % L_subboxes
    y = np.mod(x[None, :] + s, boxsize)
    inside = ((np.floor(y[:, 0] / L_sub) == ix) & (np.floor(y[:, 1] / L_sub) == iy)
              & (np.floor(y[:, 2] / L_sub) == iz))
    inside &= np.hypot(s[:, 0], s[:, 1]) > rp_cut
    c = np.cos(2 * (np.arctan2(s[:, 1], s[:, 0]) - theta))
    fo, fq = inside.astype(float), inside * c
    return (Vb * fo.mean(), Vb * fq.mean(),
            Vb * fo.std() / np.sqrt(n), Vb * fq.std() / np.sqrt(n))


# ---------------------------------------------------------------------------
# High-precision rp-pi overlap (adaptive quadrature, one galaxy, one sub-box)
# ---------------------------------------------------------------------------

def precise_overlap_rppi(x, k, L_subboxes, *, boxsize, r1, r2, p1, p2, reflect=False):
    """(O, Oc, Os) for one galaxy at x, sub-box k and one (rp, pi) bin, to ~1e-12 of V_b
    (agrees with brute-force Monte Carlo within its noise):
    the sky-plane integral of r * (arc integrals) over [r1, r2] by scipy's adaptive quad,
    split at every radius where the integrand has a kink (distance to each rectangle edge and
    corner), times the exact line-of-sight overlap length."""
    from scipy.integrate import quad
    L_sub = boxsize / L_subboxes
    ix, iy, iz = k // (L_subboxes ** 2), (k // L_subboxes) % L_subboxes, k % L_subboxes
    lo, hi = (-p2, -p1) if reflect else (p1, p2)
    lz = sum(max(0.0, min(zh, hi) - max(zl, lo))
             for zl, zh in _image_intervals(iz * L_sub - x[2], L_sub, boxsize))
    rects = [(xl, xh, yl, yh) for xl, xh in _image_intervals(ix * L_sub - x[0], L_sub, boxsize)
             for yl, yh in _image_intervals(iy * L_sub - x[1], L_sub, boxsize)]
    a = np.array(rects, float)
    kinks = {r1, r2}
    for xl, xh, yl, yh in rects:
        kinks |= {abs(xl), abs(xh), abs(yl), abs(yh)}
        kinks |= {np.hypot(u, v) for u in (xl, xh) for v in (yl, yh)}
    pts = np.array(sorted(p for p in kinks if r1 <= p <= r2))

    def f(r, j):
        return r * arc_integrals(r, a[:, 0], a[:, 1], a[:, 2], a[:, 3])[j].sum()

    Vb = np.pi * (r2 ** 2 - r1 ** 2) * (p2 - p1)
    out = []
    for j in range(3):
        s = sum(quad(f, u, v, args=(j,), epsabs=1e-12 * Vb, epsrel=1e-12, limit=200)[0]
                for u, v in zip(pts[:-1], pts[1:]))
        out.append(lz * s)
    return tuple(out)


# ---------------------------------------------------------------------------
# Sub-box self-overlap: deterministic dense-grid integral
# ---------------------------------------------------------------------------

def self_overlap_fraction_grid(L_sub, r_bins, bins2, binning="rppi", rp_cut=0.0, n_r=64, n_b=64,
                               n_phi=256):
    """f(b) = (1 / V_b) int_b V_ov(s) / L_sub^3 d^3 s, V_ov(s) = prod_a max(0, L_sub - |s_a|),
    restricted to r_p > rp_cut, by a midpoint grid in (r, second coordinate, phi).
    The integrand is symmetric under phi -> -phi and phi -> pi - phi, so phi runs over
    [0, pi/2] with weight 4."""
    r_bins, bins2 = np.asarray(r_bins, float), np.asarray(bins2, float)
    out = np.zeros((len(r_bins) - 1, len(bins2) - 1))
    phi = (np.arange(n_phi) + 0.5) * (0.5 * np.pi / n_phi)
    cphi, sphi = np.cos(phi), np.sin(phi)
    for i in range(len(r_bins) - 1):
        r1, r2 = r_bins[i], r_bins[i + 1]
        rr = r1 + (np.arange(n_r) + 0.5) * (r2 - r1) / n_r
        for j in range(len(bins2) - 1):
            b1, b2 = bins2[j], bins2[j + 1]
            bb = b1 + (np.arange(n_b) + 0.5) * (b2 - b1) / n_b
            if binning == "rppi":
                rp, z = rr[:, None, None], bb[None, :, None]
                jac = rp  # d^3 s = rp drp dphi dpi
            else:
                r, mu = rr[:, None, None], bb[None, :, None]
                rp, z = r * np.sqrt(1 - mu ** 2), r * mu
                jac = r ** 2  # d^3 s = r^2 dr dmu dphi
            sx, sy = rp * cphi[None, None, :], rp * sphi[None, None, :]
            vov = (np.clip(L_sub - np.abs(sx), 0, None) * np.clip(L_sub - np.abs(sy), 0, None)
                   * np.clip(L_sub - np.abs(z), 0, None)) / L_sub ** 3
            vov = vov * (rp > rp_cut)
            cell = (r2 - r1) / n_r * (b2 - b1) / n_b * (0.5 * np.pi / n_phi) * 4
            integral = np.sum(vov * jac) * cell
            vol = np.sum(np.broadcast_to(jac * (rp > rp_cut), vov.shape)) * cell
            out[i, j] = integral / vol if vol > 0 else 0.0
    return out


def exact_union_rr_reference(L_subboxes, *, boxsize, r_bins, bins2, binning="rppi", rp_cut=0.0, **grid):
    """Normalised exact union RR_k (per retained pair):
    (V_b / V) (1 - 2v + v f(b)) / (1 - v)^2, v = 1 / N_jk, with V_b the bin volume restricted
    to r_p > rp_cut (grid estimate when rp_cut > 0)."""
    n_patch = L_subboxes ** 3
    v = 1.0 / n_patch
    V = boxsize ** 3
    f = self_overlap_fraction_grid(boxsize / L_subboxes, r_bins, bins2, binning, rp_cut, **grid)
    Vb = shell_volumes(r_bins, bins2, binning)
    if rp_cut > 0:
        Vb = Vb * cut_volume_fraction(r_bins, bins2, binning, rp_cut)
    return Vb / V * (1 - 2 * v + v * f) / (1 - v) ** 2


def cut_volume_fraction(r_bins, bins2, binning, rp_cut, n=4000):
    """Fraction of each bin's volume with r_p > rp_cut (midpoint grid; 1 for rppi bins with
    r_p >= rp_cut, since the cut is on the bin variable there)."""
    r_bins, bins2 = np.asarray(r_bins, float), np.asarray(bins2, float)
    out = np.ones((len(r_bins) - 1, len(bins2) - 1))
    if binning == "rppi":
        return out
    for i in range(len(r_bins) - 1):
        r1, r2 = r_bins[i], r_bins[i + 1]
        rr = r1 + (np.arange(n) + 0.5) * (r2 - r1) / n
        for j in range(len(bins2) - 1):
            b1, b2 = bins2[j], bins2[j + 1]
            mu = b1 + (np.arange(n) + 0.5) * (b2 - b1) / n
            rp = rr[:, None] * np.sqrt(1 - mu[None, :] ** 2)
            wgt = rr[:, None] ** 2 * np.ones_like(rp)
            out[i, j] = np.sum(wgt * (rp > rp_cut)) / np.sum(wgt)
    return out
