"""
tests/test_box_jk_analytic_geometry.py
======================================
Geometry of the analytic box-jackknife edge correction (``measureia.box_jk_analytic``).

In a delete-one sub-box jackknife, a galaxy next to the deleted sub-box C_k loses part of
its partner shell. With uniform randoms in a periodic box, the per-realisation DR and S+R
terms only need, per galaxy at x and per bin b,

    O_k(x; b)    = int_b d^3s 1[x + s in C_k]                      (overlap volume)
    Oc, Os       = the same weighted by cos 2 phi_s and sin 2 phi_s (for S+R),

plus the exact union RR_k, which needs the bin-averaged sub-box self-overlap
f(b) = <prod_a max(0, L_sub - |s_a|)> / L_sub^3.

Sections
--------
1. arc_integrals: closed-form arcs of a circle inside a rectangle
2. Partition identity: summed over all sub-boxes, the overlaps give the full bin
3. Galaxies far from / deep inside a sub-box
4. Agreement with brute-force Monte Carlo (incl. reflection and rp_cut)
5. Agreement with the validated loop prototype (incl. weights and chunking); in rp-pi the
   module's exact closed form also matches an adaptive high-precision reference
6. Exact union RR_k and the self-overlap fraction f(b)
7. Separations beyond half the box are refused

Accuracy: rp-pi overlaps are exact (closed form for the area of disc ∩ rectangle and its
cos 2phi / sin 2phi moments). r-mu overlaps use Gauss-Legendre quadrature in the projected
radius, 16 nodes on each of [0, r_i] and [r_i, r_{i+1}] for bin i: per galaxy up to ~6e-3 of
V_b, which changes the jackknife sigma by <= 0.1% (validation/run_box_jk_quadrature_check.py,
validation/run_box_jk_node_layout.py).
"""

import numpy as np
import pytest

from measureia import box_jk_analytic as bja
import box_jk_reference as ref


BOXSIZE = 150.0
R_BINS = np.logspace(0, np.log10(60.0), 6)       # reaches beyond L_sub = 37.5 at N_jk = 64
PI_BINS = np.linspace(-30.0, 30.0, 7)
MU_BINS = np.linspace(-1.0, 1.0, 7)
BINNINGS = [("rppi", PI_BINS), ("rmu", MU_BINS)]


def _kw(binning, bins2, **extra):
    return dict(boxsize=BOXSIZE, r_bins=R_BINS, bins2=bins2, binning=binning, **extra)


def _labels(points, L_subboxes, boxsize=BOXSIZE):
    i = np.floor(points / (boxsize / L_subboxes)).astype(int)
    return i[:, 0] * L_subboxes ** 2 + i[:, 1] * L_subboxes + i[:, 2]


# ---------------------------------------------------------------------------
# 1. arc_integrals
# ---------------------------------------------------------------------------

class TestArcIntegrals:

    def test_full_circle_inside_large_rectangle(self):
        big = np.array([-100.0])
        I0, Ic, Is = bja.arc_integrals(5.0, big, -big, big, -big)
        np.testing.assert_allclose([I0[0], Ic[0], Is[0]], [2 * np.pi, 0.0, 0.0], atol=1e-14)

    def test_first_quadrant(self):
        """Arc in [0, R]^2 for r < R: phi in [0, pi/2] -> (pi/2, 0, 1)."""
        I0, Ic, Is = bja.arc_integrals(3.0, np.array([0.0]), np.array([10.0]),
                                       np.array([0.0]), np.array([10.0]))
        np.testing.assert_allclose([I0[0], Ic[0], Is[0]], [np.pi / 2, 0.0, 1.0], atol=1e-14)

    def test_rectangle_outside_circle_gives_zero(self):
        I0, Ic, Is = bja.arc_integrals(2.0, np.array([5.0]), np.array([9.0]),
                                       np.array([-1.0]), np.array([1.0]))
        assert I0[0] == 0.0 and Ic[0] == 0.0 and Is[0] == 0.0

    def test_quadrants_partition_the_circle(self):
        r = 4.0
        rects = [(0, 10, 0, 10), (-10, 0, 0, 10), (-10, 0, -10, 0), (0, 10, -10, 0)]
        tot = np.zeros(3)
        for ax, bx, ay, by in rects:
            tot += np.array([v[0] for v in bja.arc_integrals(
                r, np.array([ax], float), np.array([bx], float),
                np.array([ay], float), np.array([by], float))])
        np.testing.assert_allclose(tot, [2 * np.pi, 0.0, 0.0], atol=1e-13)

    def test_matches_prototype_on_random_rectangles(self):
        rng = np.random.default_rng(1)
        n = 2000
        ax = rng.uniform(-30, 30, n)
        ay = rng.uniform(-30, 30, n)
        bx = ax + rng.uniform(0, 40, n)
        by = ay + rng.uniform(0, 40, n)
        for r in [0.5, 7.0, 21.0, 45.0]:
            got = bja.arc_integrals(r, ax, bx, ay, by)
            want = ref.arc_integrals(r, ax, bx, ay, by)
            for g, w in zip(got, want):
                np.testing.assert_allclose(g, w, atol=1e-12)

    def test_vectorised_over_radius(self):
        """r may be an array broadcasting against the rectangles: (n_r, 1) x (n_rect,)."""
        rng = np.random.default_rng(2)
        ax = rng.uniform(-30, 30, 50)
        ay = rng.uniform(-30, 30, 50)
        bx, by = ax + 20, ay + 20
        radii = np.array([1.0, 10.0, 30.0])
        got = bja.arc_integrals(radii[:, None], ax[None, :], bx[None, :], ay[None, :], by[None, :])
        for q, r in enumerate(radii):
            want = ref.arc_integrals(r, ax, bx, ay, by)
            for g, w in zip(got, want):
                np.testing.assert_allclose(g[q], w, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Partition identity
# ---------------------------------------------------------------------------

class TestPartitionIdentity:
    """The sub-boxes tile the (periodic) box, so for any galaxy the overlaps summed over
    ALL sub-boxes (including its own) cover the whole bin: sum_k O_k = V_b and
    sum_k Oc_k = sum_k Os_k = 0 (cos 2phi and sin 2phi integrate to zero over a full ring).
    In rp-pi this is exact (closed form). In r-mu the full bin is itself integrated by the
    Gauss-Legendre quadrature in rho, which for the full bin is accurate to 3.6e-3 of V_b (the
    same for the loop prototype: the error sits in the [r_i, r_{i+1}] segment both share)."""

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    @pytest.mark.parametrize("L_subboxes", [2, 3, 4])
    @pytest.mark.parametrize("reflect", [False, True])
    def test_sum_over_subboxes(self, binning, bins2, L_subboxes, reflect):
        rng = np.random.default_rng(10 + L_subboxes)
        pts = rng.uniform(0, BOXSIZE, (12, 3))
        Vb = ref.shell_volumes(R_BINS, bins2, binning)
        tot = np.zeros((3, len(pts)) + Vb.shape)
        for k in range(L_subboxes ** 3):
            O, Oc, Os = bja.subbox_overlaps(pts, k, L_subboxes, reflect=reflect, **_kw(binning, bins2))
            tot += np.array([O, Oc, Os])
        rtol = 1e-12 if binning == "rppi" else 5e-3
        np.testing.assert_allclose(tot[0], np.broadcast_to(Vb, tot[0].shape), rtol=rtol)
        np.testing.assert_allclose(tot[1], 0.0, atol=1e-11 * Vb.max())
        np.testing.assert_allclose(tot[2], 0.0, atol=1e-11 * Vb.max())

    def test_rp_cut_partition(self):
        """With rp_cut (r-mu only), the overlaps tile the bin restricted to r_p > rp_cut."""
        rp_cut = 6.0
        rng = np.random.default_rng(5)
        pts = rng.uniform(0, BOXSIZE, (6, 3))
        Vb = ref.shell_volumes(R_BINS, MU_BINS, "rmu") * ref.cut_volume_fraction(R_BINS, MU_BINS, "rmu", rp_cut)
        tot = 0.0
        for k in range(27):
            O, _, _ = bja.subbox_overlaps(pts, k, 3, rp_cut=rp_cut, **_kw("rmu", MU_BINS))
            tot = tot + O
        np.testing.assert_allclose(tot, np.broadcast_to(Vb, tot.shape), rtol=5e-3, atol=1e-6 * Vb.max())


# ---------------------------------------------------------------------------
# 3. Far from / deep inside a sub-box
# ---------------------------------------------------------------------------

class TestSimpleConfigurations:

    small_r = np.array([1.0, 5.0, 10.0])
    small_pi = np.array([-10.0, 0.0, 10.0])
    small_mu = np.array([-1.0, 0.0, 1.0])

    @pytest.mark.parametrize("binning", ["rppi", "rmu"])
    def test_distant_subbox_has_zero_overlap(self, binning):
        """Galaxy at the centre of sub-box 0 (L_sub = 50), bins up to 10: no reach into
        sub-box 13 (the box centre), exactly."""
        bins2 = self.small_pi if binning == "rppi" else self.small_mu
        x = np.array([[25.0, 25.0, 25.0]])
        O, Oc, Os = bja.subbox_overlaps(x, 13, 3, boxsize=BOXSIZE, r_bins=self.small_r, bins2=bins2,
                                        binning=binning)
        assert np.all(O == 0) and np.all(Oc == 0) and np.all(Os == 0)

    @pytest.mark.parametrize("binning", ["rppi", "rmu"])
    def test_own_subbox_contains_whole_bin(self, binning):
        bins2 = self.small_pi if binning == "rppi" else self.small_mu
        x = np.array([[25.0, 25.0, 25.0]])
        O, Oc, Os = bja.subbox_overlaps(x, 0, 3, boxsize=BOXSIZE, r_bins=self.small_r, bins2=bins2,
                                        binning=binning)
        Vb = ref.shell_volumes(self.small_r, bins2, binning)
        rtol = 1e-12 if binning == "rppi" else 5e-3   # r-mu: the quadrature accuracy (3.6e-3)
        np.testing.assert_allclose(O[0], Vb, rtol=rtol)
        np.testing.assert_allclose(Oc[0], 0.0, atol=1e-11 * Vb.max())
        np.testing.assert_allclose(Os[0], 0.0, atol=1e-11 * Vb.max())

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_periodic_wrap(self, binning, bins2):
        """A galaxy at x = 1 sits 1 Mpc/h from sub-box ix = 2 ([100, 150)) through the periodic
        boundary. Shifting everything by +100 in x gives a galaxy at x = 101 next to sub-box
        ix = 1 ([50, 100)) with no wrap involved; the overlaps must be identical."""
        k_wrap = 2 * 9 + 1 * 3 + 1      # ix = 2, iy = 1, iz = 1
        k_direct = 1 * 9 + 1 * 3 + 1    # ix = 1, iy = 1, iz = 1
        wrap = bja.subbox_overlaps(np.array([[1.0, 75.0, 75.0]]), k_wrap, 3, **_kw(binning, bins2))
        direct = bja.subbox_overlaps(np.array([[101.0, 75.0, 75.0]]), k_direct, 3, **_kw(binning, bins2))
        assert wrap[0].sum() > 0
        for a, b in zip(wrap, direct):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * direct[0].max())


# ---------------------------------------------------------------------------
# 4. Brute-force Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    @pytest.mark.parametrize("reflect", [False, True])
    def test_overlaps_match_mc(self, binning, bins2, reflect):
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, BOXSIZE, (4, 3))
        theta = rng.uniform(0, np.pi, 4)
        cases = [(4, 2), (3, 5), (2, 0), (4, 4)]
        zmax = 0.0
        for k in [0, 13, 26]:
            O, Oc, Os = bja.subbox_overlaps(pts, k, 3, reflect=reflect, **_kw(binning, bins2))
            for g in range(len(pts)):
                Q = np.cos(2 * theta[g]) * Oc[g] + np.sin(2 * theta[g]) * Os[g]
                for (i, j) in cases:
                    mo, mq, so, sq = ref.mc_overlap(
                        pts[g], k, 3, boxsize=BOXSIZE, r1=R_BINS[i], r2=R_BINS[i + 1], b1=bins2[j],
                        b2=bins2[j + 1], binning=binning, reflect=reflect, theta=theta[g],
                        n=100_000, seed=100 * k + g)
                    zmax = max(zmax, abs(O[g, i, j] - mo) / max(so, 1e-9),
                               abs(Q[i, j] - mq) / max(sq, 1e-9))
        assert zmax < 5.0, f"largest deviation from Monte Carlo: {zmax:.2f} sigma"

    def test_rp_cut_matches_mc(self):
        rp_cut = 6.0
        rng = np.random.default_rng(8)
        pts = rng.uniform(0, BOXSIZE, (3, 3))
        zmax = 0.0
        for k in [0, 13]:
            O, Oc, _ = bja.subbox_overlaps(pts, k, 3, rp_cut=rp_cut, **_kw("rmu", MU_BINS))
            for g in range(len(pts)):
                for (i, j) in [(2, 3), (3, 0), (4, 5)]:
                    mo, mq, so, sq = ref.mc_overlap(
                        pts[g], k, 3, boxsize=BOXSIZE, r1=R_BINS[i], r2=R_BINS[i + 1], b1=MU_BINS[j],
                        b2=MU_BINS[j + 1], binning="rmu", theta=0.0, rp_cut=rp_cut, n=100_000, seed=g + 10 * k)
                    zmax = max(zmax, abs(O[g, i, j] - mo) / max(so, 1e-9), abs(Oc[g, i, j] - mq) / max(sq, 1e-9))
        assert zmax < 5.0, f"largest deviation from Monte Carlo: {zmax:.2f} sigma"


# ---------------------------------------------------------------------------
# 5. The validated loop prototype
# ---------------------------------------------------------------------------

def _small_mock(seed, n_pos=120, n_shape=90):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, BOXSIZE, (n_pos, 3))
    shape = rng.uniform(0, BOXSIZE, (n_shape, 3))
    theta = rng.uniform(0, np.pi, n_shape)
    e = rng.uniform(0, 0.8, n_shape)
    w = rng.uniform(0.5, 2.0, n_pos)
    w_s = rng.uniform(0.5, 2.0, n_shape)
    return pos, shape, theta, e, w, w_s


class TestMatchesPrototype:

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    @pytest.mark.parametrize("L_subboxes", [2, 3])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_dr_terms(self, binning, bins2, L_subboxes, weighted):
        pos, shape, theta, e, w, w_s = _small_mock(20 + L_subboxes)
        if not weighted:
            w, w_s = None, None
        lab_p, lab_s = _labels(pos, L_subboxes), _labels(shape, L_subboxes)
        got = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, L_subboxes,
                                    weight=w, weight_shape=w_s, **_kw(binning, bins2))
        want = ref.dr_terms(pos, shape, theta, e, lab_p, lab_s, L_subboxes ** 3, boxsize=BOXSIZE,
                            r_bins=R_BINS, bins2=bins2, binning=binning, w=w, w_shape=w_s)
        V = BOXSIZE ** 3
        per_pair = ref.shell_volumes(R_BINS, bins2, binning) / (V - V / L_subboxes ** 3)
        # rppi: the exact closed form vs the prototype's 16-node quadrature, whose per-galaxy
        # error reaches ~5e-4 of V_b where a bin contains a tangent or corner radius (the exact
        # values are checked against adaptive quadrature in the next test).
        # rmu: per-bin rho nodes vs the prototype's shared nodes; per galaxy the two are within
        # 6.1e-3 and 9.6e-4 of V_b of the exact overlap (validation/run_box_jk_node_layout.py).
        tol = 1e-3 if binning == "rppi" else 1e-2
        for name, g, wt in zip(["DR", "RD", "S+R"], got, want):
            assert np.all(np.abs(g - wt) <= tol * per_pair[None]), name

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_chunk_size_does_not_change_the_result(self, binning, bins2):
        pos, shape, theta, e, w, w_s = _small_mock(30)
        lab_p, lab_s = _labels(pos, 2), _labels(shape, 2)
        a = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, 2, weight=w, weight_shape=w_s,
                                  chunk_size=7, **_kw(binning, bins2))
        b = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, 2, weight=w, weight_shape=w_s,
                                  chunk_size=10_000, **_kw(binning, bins2))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    @pytest.mark.parametrize("samples", ["same", "separate", "partial", "duplicates"])
    def test_single_pass_equals_two_passes(self, binning, bins2, samples):
        """With symmetric second-coordinate bins the reflected bin j is bin n-1-j, so both
        samples are handled in one geometry pass over their union (objects in both samples,
        matched by coordinates, computed once). Same result as one pass per sample."""
        pos, shape, theta, e, w, w_s = _small_mock(50, n_pos=100, n_shape=80)
        if samples == "same":
            shape = pos.copy()
            theta, e, w_s = (np.resize(v, len(pos)) for v in (theta, e, w_s))
        elif samples == "partial":
            shape[:40] = pos[10:50]
        elif samples == "duplicates":
            pos[5] = pos[6]
            shape[:30] = pos[:30]
        lab_p, lab_s = _labels(pos, 3), _labels(shape, 3)
        kw = dict(weight=w, weight_shape=w_s, **_kw(binning, bins2))
        one = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, 3, _single_pass=True, **kw)
        two = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, 3, _single_pass=False, **kw)
        for name, a, b in zip(["DR", "RD", "S+R"], one, two):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max(), err_msg=name)

    def test_asymmetric_bins_fall_back_to_two_passes(self):
        """Bins that are not symmetric about zero cannot be reflected by reversing their order;
        the result must still match the prototype."""
        pi_bins = np.array([-30.0, -10.0, 5.0, 30.0])
        pos, shape, theta, e, w, w_s = _small_mock(51)
        lab_p, lab_s = _labels(pos, 3), _labels(shape, 3)
        got = bja.analytic_dr_terms(pos, shape, theta, e, lab_p, lab_s, 3, weight=w, weight_shape=w_s,
                                    **_kw("rppi", pi_bins))
        want = ref.dr_terms(pos, shape, theta, e, lab_p, lab_s, 27, boxsize=BOXSIZE, r_bins=R_BINS,
                            bins2=pi_bins, binning="rppi", w=w, w_shape=w_s)
        per_pair = ref.shell_volumes(R_BINS, pi_bins) / (BOXSIZE ** 3 * (1 - 1 / 27))
        for name, g, wt in zip(["DR", "RD", "S+R"], got, want):
            assert np.all(np.abs(g - wt) <= 1e-3 * per_pair[None]), name

    @pytest.mark.parametrize("reflect", [False, True])
    def test_rppi_closed_form_matches_precise_reference(self, reflect):
        """rp-pi overlaps against adaptive quadrature split at every kink (tangent and corner
        radii), ~1e-12 of V_b. Galaxies are placed so that edge and corner distances to the
        sub-box fall inside the bins, including through the periodic boundary."""
        L_subboxes = 3
        cases = [(np.array([40.0, 140.0, 52.0]), 1 * 9 + 0 * 3 + 1),
                 (np.array([47.0, 53.0, 20.0]), 1 * 9 + 0 * 3 + 0),
                 (np.array([101.0, 2.0, 99.0]), 1 * 9 + 2 * 3 + 2),
                 (np.array([75.0, 75.0, 75.0]), 1 * 9 + 1 * 3 + 1)]
        Vb = ref.shell_volumes(R_BINS, PI_BINS)
        for x, k in cases:
            O, Oc, Os = bja.subbox_overlaps(x[None], k, L_subboxes, reflect=reflect, **_kw("rppi", PI_BINS))
            for i in range(len(R_BINS) - 1):
                for j in range(len(PI_BINS) - 1):
                    want = ref.precise_overlap_rppi(x, k, L_subboxes, boxsize=BOXSIZE, r1=R_BINS[i],
                                                    r2=R_BINS[i + 1], p1=PI_BINS[j], p2=PI_BINS[j + 1],
                                                    reflect=reflect)
                    got = (O[0, i, j], Oc[0, i, j], Os[0, i, j])
                    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10 * Vb[i, j],
                                               err_msg=f"x={x}, k={k}, bin=({i},{j})")

    def test_full_sample_limit(self):
        """With no galaxy anywhere near a deleted sub-box there is nothing to correct:
        DR = RD = V_b / (V - V_k) per pair and S+R = 0. Galaxies sit in a thin slab in the
        middle of sub-box column ix = 0, far (> r_max) from sub-box ix = 2."""
        rng = np.random.default_rng(31)
        r_bins = np.array([1.0, 5.0, 10.0])
        pi_bins = np.array([-10.0, 0.0, 10.0])
        n = 40
        pos = np.column_stack([rng.uniform(20, 30, n), rng.uniform(0, BOXSIZE, n), rng.uniform(0, BOXSIZE, n)])
        lab = _labels(pos, 3)
        DR, RD, SpR = bja.analytic_dr_terms(pos, pos, rng.uniform(0, np.pi, n), rng.uniform(0, 0.5, n),
                                            lab, lab, 3, boxsize=BOXSIZE, r_bins=r_bins, bins2=pi_bins,
                                            binning="rppi")
        V = BOXSIZE ** 3
        Vb = ref.shell_volumes(r_bins, pi_bins)
        k_far = [k for k in range(27) if k // 9 == 2]
        for k in k_far:
            np.testing.assert_allclose(DR[k], Vb / (V - V / 27), rtol=1e-12)
            np.testing.assert_allclose(RD[k], Vb / (V - V / 27), rtol=1e-12)
            np.testing.assert_allclose(SpR[k], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 6. Exact union RR_k
# ---------------------------------------------------------------------------

class TestExactUnionRR:

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    @pytest.mark.parametrize("L_subboxes", [3, 4])
    def test_self_overlap_fraction_matches_grid(self, binning, bins2, L_subboxes):
        f = bja.self_overlap_fraction(L_subboxes, **_kw(binning, bins2))
        f_ref = ref.self_overlap_fraction_grid(BOXSIZE / L_subboxes, R_BINS, bins2, binning)
        np.testing.assert_allclose(f, f_ref, atol=2e-4)

    def test_self_overlap_fraction_with_rp_cut(self):
        f = bja.self_overlap_fraction(3, rp_cut=6.0, **_kw("rmu", MU_BINS))
        f_ref = ref.self_overlap_fraction_grid(BOXSIZE / 3, R_BINS, MU_BINS, "rmu", rp_cut=6.0)
        np.testing.assert_allclose(f, f_ref, atol=5e-4)

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_self_overlap_equals_mean_own_subbox_overlap(self, binning, bins2):
        """f(b) V_b is also the mean, over points uniform in sub-box k, of their overlap with
        k itself: an independent route through subbox_overlaps."""
        rng = np.random.default_rng(40)
        L_subboxes = 3
        L_sub = BOXSIZE / L_subboxes
        pts = rng.uniform(0, L_sub, (1500, 3))  # sub-box 0
        O, _, _ = bja.subbox_overlaps(pts, 0, L_subboxes, **_kw(binning, bins2))
        Vb = ref.shell_volumes(R_BINS, bins2, binning)
        mean, err = O.mean(axis=0) / Vb, O.std(axis=0) / np.sqrt(len(pts)) / Vb
        f = bja.self_overlap_fraction(L_subboxes, **_kw(binning, bins2))
        assert np.all(np.abs(mean - f) < 5 * err + 2e-4)

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_exact_rr_formula(self, binning, bins2):
        """RR_k (per retained pair) = (V_b / V)(1 - 2v + v f(b)) / (1 - v)^2, v = 1 / N_jk."""
        for L_subboxes in [2, 3, 4]:
            v = 1.0 / L_subboxes ** 3
            f = bja.self_overlap_fraction(L_subboxes, **_kw(binning, bins2))
            Vb = ref.shell_volumes(R_BINS, bins2, binning)
            want = Vb / BOXSIZE ** 3 * (1 - 2 * v + v * f) / (1 - v) ** 2
            got = bja.exact_union_rr(L_subboxes, **_kw(binning, bins2))
            np.testing.assert_allclose(got, want, rtol=1e-13)

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_reduces_to_todays_rr_at_small_separation(self, binning, bins2):
        """For s << L_sub, f -> 1 and RR_k -> V_b / (V (1 - v)), measureia's current
        per-realisation RR (per retained pair)."""
        r_bins = np.array([0.01, 0.02, 0.05])
        b2 = bins2 if binning == "rmu" else np.array([-0.05, 0.0, 0.05])
        got = bja.exact_union_rr(3, boxsize=BOXSIZE, r_bins=r_bins, bins2=b2, binning=binning)
        Vb = ref.shell_volumes(r_bins, b2, binning)
        today = Vb / (BOXSIZE ** 3 * (1 - 1 / 27))
        np.testing.assert_allclose(got, today, rtol=2e-3)

    def test_rp_cut_matches_reference(self):
        got = bja.exact_union_rr(3, rp_cut=6.0, **_kw("rmu", MU_BINS))
        want = ref.exact_union_rr_reference(3, boxsize=BOXSIZE, r_bins=R_BINS, bins2=MU_BINS,
                                            binning="rmu", rp_cut=6.0)
        np.testing.assert_allclose(got, want, rtol=2e-3)


# ---------------------------------------------------------------------------
# 7. Separations beyond half the box
# ---------------------------------------------------------------------------

class TestHalfBoxLimit:
    """The periodic image sums and the self-overlap formula assume every separation component
    is at most L/2 (as does minimum-image pair counting). Beyond that the functions refuse."""

    @staticmethod
    def _calls(r_bins, bins2, binning):
        pts = np.array([[10.0, 20.0, 30.0]])
        kw = dict(boxsize=BOXSIZE, r_bins=r_bins, bins2=bins2, binning=binning)
        lab = _labels(pts, 2)
        return [
            lambda: bja.subbox_overlaps(pts, 0, 2, **kw),
            lambda: bja.analytic_dr_terms(pts, pts, np.zeros(1), np.zeros(1), lab, lab, 2, **kw),
            lambda: bja.exact_union_rr(2, **kw),
        ]

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_r_max_beyond_half_box_raises(self, binning, bins2):
        r_bins = np.array([1.0, 10.0, 0.5 * BOXSIZE + 1.0])
        for call in self._calls(r_bins, bins2, binning):
            with pytest.raises(ValueError, match="half the box"):
                call()

    def test_pi_max_beyond_half_box_raises(self):
        pi_bins = np.array([-0.5 * BOXSIZE - 1.0, 0.0, 0.5 * BOXSIZE + 1.0])
        for call in self._calls(R_BINS, pi_bins, "rppi"):
            with pytest.raises(ValueError, match="half the box"):
                call()

    @pytest.mark.parametrize("binning,bins2", BINNINGS)
    def test_exactly_half_box_is_allowed(self, binning, bins2):
        """pi_max defaults to L/2 in measureia, so equality must be accepted."""
        r_bins = np.array([1.0, 10.0, 0.5 * BOXSIZE])
        b2 = np.array([-0.5 * BOXSIZE, 0.0, 0.5 * BOXSIZE]) if binning == "rppi" else bins2
        for call in self._calls(r_bins, b2, binning):
            call()
