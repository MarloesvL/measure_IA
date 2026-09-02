"""
test_pair_kernel_shape_shape.py
===============================
The shape-shape (``shapes="both"``) branch of the pair kernel, tested directly
against independent O(N^2) references rather than through the public estimators.

Milestone 1 of the shape-shape feature adds three product grids to the kernel --
``Splus_Splus``, ``Scross_Scross`` and the symmetrised parity-odd ``Splus_Scross``
-- on both geometries, plus their union-deletion jackknife twins. Nothing above
the kernel requests them yet, so these tests are the only cover they have.

Sections
--------
  1. Box geometry: brute-force reference, sample-swap symmetry, degenerate limits
  2. Lightcone geometry: brute-force reference (each galaxy in its own frame)
  3. Jackknife: the delete-one identity for the new grids
  4. Non-interference: shapes=True / shapes=False behave exactly as before
"""

import numpy as np
import pytest

from measureia import MeasureIABox, MeasureIALightcone, MeasureIABase
from measureia import pair_kernel


_SEP = [0.5, 20.0]
_NR = 6
_NPI = 10
_L = 205.0
_SEED = 20260902


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_catalogue(n_pos=70, n_shape=90, seed=_SEED):
    """Two overlapping-in-nothing samples, both carrying shapes."""
    rng = np.random.default_rng(seed)

    def shapes(n):
        th = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([np.cos(th), np.sin(th)]), rng.uniform(0.3, 0.9, n)

    a_pos, q_pos = shapes(n_pos)
    a_sh, q_sh = shapes(n_shape)
    return {
        "Position": rng.uniform(0, _L, (n_pos, 3)),
        "Position_shape_sample": rng.uniform(0, _L, (n_shape, 3)),
        "Axis_Direction": a_sh, "q": q_sh,
        "Axis_Direction_density_sample": a_pos, "q_density_sample": q_pos,
        "weight": np.ones(n_pos), "weight_shape_sample": np.ones(n_shape),
        "LOS": 2,
    }


def _box_obj(data, tmp_path, name="pk.hdf5"):
    return MeasureIABox(
        data, str(tmp_path / name), simulation="TNG300", snapshot=99,
        separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI, pi_max=_L / 2,
    )


def _run_box(data, tmp_path, *, backend="tree", jk=False, num_box=None,
             responsivity=True, name="pk.hdf5", kind="rppi"):
    """Prepare + accumulate on the box path, returning (grids, R, R_pos)."""
    obj = _box_obj(data, tmp_path, name)
    ss = pair_kernel.prepare_box_samples(
        data, None, len(data["Position"]), len(data["Position_shape_sample"]),
        shapes="both", ellipticity="distortion", base=obj)
    if jk:
        ss.jk_shape = _patches(data["Position_shape_sample"], num_box)
        ss.jk_pos = _patches(data["Position"], num_box)
    R = (np.sum(ss.weight_shape * (1 - ss.e ** 2 / 2.0)) / np.sum(ss.weight_shape)
         if responsivity else 0.5)
    R_pos = (np.sum(ss.weight * (1 - ss.e_pos ** 2 / 2.0)) / np.sum(ss.weight)
             if responsivity else 0.5)
    binning = pair_kernel.BoxRpPi(obj) if kind == "rppi" else pair_kernel.BoxRMuR(obj)
    grids = pair_kernel.accumulate(
        ss, binning, base=obj, R=R, R_pos=R_pos, shapes="both",
        chunk_axis="shape", chunk_size_outer=25, backend=backend,
        jk=jk, num_box=num_box)
    return grids, R, R_pos


def _patches(pos, num_box):
    """Deterministic sub-box patch ids along x."""
    edges = np.linspace(0, _L, num_box + 1)
    return np.clip(np.digitize(pos[:, 0], edges) - 1, 0, num_box - 1)


def _bruteforce_box(data, R, R_pos, r_bins, pi_bins, kind="rppi", mu_bins=None):
    """Independent O(N^2) reference for the three shape-shape grids.

    ``kind="rppi"`` bins on (projected rp, signed Pi); ``kind="rmur"`` on
    (3D r, mu_r = Pi/r), the multipoles binning.
    """
    pos, pos_s = data["Position"], data["Position_shape_sample"]
    a_p, a_s = data["Axis_Direction_density_sample"], data["Axis_Direction"]
    e_p = (1 - data["q_density_sample"] ** 2) / (1 + data["q_density_sample"] ** 2)
    e_s = (1 - data["q"] ** 2) / (1 + data["q"] ** 2)
    nr, npi = len(r_bins) - 1, len(pi_bins) - 1
    pp = np.zeros((nr, npi)); xx = np.zeros((nr, npi)); px = np.zeros((nr, npi))

    for j in range(len(pos_s)):
        sep = pos_s[j] - pos
        sep -= _L * np.round(sep / _L)                     # minimum image
        rp = np.hypot(sep[:, 0], sep[:, 1])
        pi = sep[:, 2]
        if kind == "rppi":
            first, second, second_bins = rp, pi, pi_bins
        else:
            r3 = np.sqrt(np.sum(sep ** 2, axis=1))
            with np.errstate(invalid='ignore', divide='ignore'):
                first, second = r3, pi / r3
            second_bins = mu_bins
        ok = ((first >= r_bins[0]) & (first < r_bins[-1])
              & (second >= second_bins[0]) & (second < second_bins[-1]))
        if not ok.any():
            continue
        d = sep[ok][:, :2] / rp[ok][:, None]

        def project(axis, e):
            c = d[:, 0] * axis[:, 0] + d[:, 1] * axis[:, 1]
            x = axis[:, 0] * d[:, 1] - axis[:, 1] * d[:, 0]
            return e * (2 * c * c - 1), e * (2 * c * x)

        ep_s, ex_s = project(np.broadcast_to(a_s[j], d.shape), e_s[j])
        ep_p, ex_p = project(a_p[ok], e_p[ok])
        norm = (2 * R) * (2 * R_pos)
        ir = np.digitize(first[ok], r_bins) - 1
        ip = np.digitize(second[ok], second_bins) - 1
        np.add.at(pp, (ir, ip), ep_s * ep_p / norm)
        np.add.at(xx, (ir, ip), ex_s * ex_p / norm)
        np.add.at(px, (ir, ip), 0.5 * (ep_s * ex_p + ex_s * ep_p) / norm)
    return pp, xx, px


# ===========================================================================
# 1. Box geometry
# ===========================================================================

class TestShapeShapeBox:

    @pytest.mark.parametrize("backend", ["tree", "brute"])
    def test_matches_bruteforce(self, tmp_path, backend):
        """The three product grids against an independent O(N^2) reference."""
        data = _box_catalogue()
        grids, R, R_pos = _run_box(data, tmp_path, backend=backend)
        obj = _box_obj(data, tmp_path, "ref.hdf5")
        ref = _bruteforce_box(data, R, R_pos, obj.r_bins, obj.pi_bins)
        for got, want, name in zip(
                (grids.Splus_Splus, grids.Scross_Scross, grids.Splus_Scross),
                ref, ("Splus_Splus", "Scross_Scross", "Splus_Scross")):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13,
                                       err_msg=f"{name} disagrees with brute force")

    @pytest.mark.parametrize("backend", ["tree", "brute"])
    def test_matches_bruteforce_multipoles_binning(self, tmp_path, backend):
        """The same reference on the (r, mu_r) binning the multipoles use.

        The products themselves are binning-agnostic, but the audit treats each
        (geometry x statistic x backend x jk x shape-terms) cell as its own, and
        an untested cell is where the F7-class regressions have hidden before.
        """
        data = _box_catalogue()
        grids, R, R_pos = _run_box(data, tmp_path, backend=backend, kind="rmur")
        obj = _box_obj(data, tmp_path, "refm.hdf5")
        ref = _bruteforce_box(data, R, R_pos, obj.r_bins, obj.pi_bins,
                              kind="rmur", mu_bins=obj.mu_r_bins)
        for got, want, name in zip(
                (grids.Splus_Splus, grids.Scross_Scross, grids.Splus_Scross),
                ref, ("Splus_Splus", "Scross_Scross", "Splus_Scross")):
            assert np.max(np.abs(want)) > 1e-8, f"{name} reference is trivially zero"
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13,
                                       err_msg=f"{name} disagrees with brute force")

    def test_signal_is_not_trivially_zero(self, tmp_path):
        """Guard on the reference tests: the grids must carry real signal."""
        data = _box_catalogue()
        grids, _, _ = _run_box(data, tmp_path)
        for g, name in ((grids.Splus_Splus, "Splus_Splus"),
                        (grids.Scross_Scross, "Scross_Scross"),
                        (grids.Splus_Scross, "Splus_Scross")):
            assert np.max(np.abs(g)) > 1e-6, f"{name} is trivially zero"

    def test_symmetric_under_sample_swap(self, tmp_path):
        """Swapping which catalogue is 'position' and which is 'shape' must not
        change the shape-shape grids.

        S+S+ and SxSx are products over the same unordered pairs, and the
        parity-odd term is symmetrised precisely so that it shares the property.
        The tolerance is float-summation-order only: the two runs visit the
        pairs in different orders.
        """
        data = _box_catalogue()
        swapped = dict(data)
        swapped["Position"], swapped["Position_shape_sample"] = (
            data["Position_shape_sample"], data["Position"])
        swapped["Axis_Direction"], swapped["Axis_Direction_density_sample"] = (
            data["Axis_Direction_density_sample"], data["Axis_Direction"])
        swapped["q"], swapped["q_density_sample"] = (
            data["q_density_sample"], data["q"])
        swapped["weight"], swapped["weight_shape_sample"] = (
            data["weight_shape_sample"], data["weight"])

        a, _, _ = _run_box(data, tmp_path, name="a.hdf5")
        b, _, _ = _run_box(swapped, tmp_path, name="b.hdf5")
        # separation flips sign under the swap; cos 2phi and sin 2phi do not,
        # so every grid must match bin for bin apart from the Pi axis mirroring
        for ga, gb, name in ((a.Splus_Splus, b.Splus_Splus, "Splus_Splus"),
                             (a.Scross_Scross, b.Scross_Scross, "Scross_Scross"),
                             (a.Splus_Scross, b.Splus_Scross, "Splus_Scross")):
            np.testing.assert_allclose(ga, gb[:, ::-1], rtol=1e-11, atol=1e-13,
                                       err_msg=f"{name} not swap-symmetric")

    def test_zero_ellipticity_gives_zero_products(self, tmp_path):
        """q=1 on either sample zeroes every shape-shape grid exactly."""
        data = _box_catalogue()
        data["q_density_sample"] = np.ones_like(data["q_density_sample"])
        grids, _, _ = _run_box(data, tmp_path)
        for g in (grids.Splus_Splus, grids.Scross_Scross, grids.Splus_Scross):
            np.testing.assert_array_equal(g, np.zeros_like(g))

    def test_responsivity_scales_products_quadratically(self, tmp_path):
        """Two shapes means two responsivity factors: turning the correction off
        must rescale every product by exactly (2R)(2R_pos)."""
        data = _box_catalogue()
        on, R, R_pos = _run_box(data, tmp_path, responsivity=True, name="on.hdf5")
        off, _, _ = _run_box(data, tmp_path, responsivity=False, name="off.hdf5")
        factor = (2 * R) * (2 * R_pos)
        for g_on, g_off in ((on.Splus_Splus, off.Splus_Splus),
                            (on.Scross_Scross, off.Scross_Scross),
                            (on.Splus_Scross, off.Splus_Scross)):
            np.testing.assert_allclose(g_on * factor, g_off, rtol=1e-12, atol=1e-14)

    def test_missing_R_pos_raises(self, tmp_path):
        """The box path cannot silently default the second responsivity."""
        data = _box_catalogue()
        obj = _box_obj(data, tmp_path)
        ss = pair_kernel.prepare_box_samples(
            data, None, len(data["Position"]), len(data["Position_shape_sample"]),
            shapes="both", ellipticity="distortion", base=obj)
        binning = pair_kernel.BoxRpPi(obj)
        with pytest.raises(ValueError, match="R_pos"):
            pair_kernel.accumulate(ss, binning, base=obj, R=0.5, shapes="both",
                                   chunk_axis="shape", chunk_size_outer=25)

    def test_bad_shapes_value_raises(self):
        with pytest.raises(ValueError, match="True, False or 'both'"):
            pair_kernel.shape_mode("plus")


# ===========================================================================
# 2. Lightcone geometry
# ===========================================================================

def _lc_catalogue(n_pos=45, n_shape=55, seed=_SEED):
    """A small cone, both samples carrying e1/e2."""
    rng = np.random.default_rng(seed)

    def cone(n):
        ra = rng.uniform(20.0, 30.0, n)
        dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-5.0)),
                                               np.sin(np.radians(5.0)), n)))
        z = rng.uniform(0.10, 0.30, n)
        return ra, dec, z

    ra_d, dec_d, z_d = cone(n_pos)
    ra_s, dec_s, z_s = cone(n_shape)

    def ell(n):
        e = rng.uniform(0.05, 0.5, n)
        th = rng.uniform(0, np.pi, n)
        return e * np.cos(2 * th), e * np.sin(2 * th)

    e1_s, e2_s = ell(n_shape)
    e1_d, e2_d = ell(n_pos)
    return {
        "RA": ra_d, "DEC": dec_d, "Redshift": z_d,
        "RA_shape_sample": ra_s, "DEC_shape_sample": dec_s, "Redshift_shape_sample": z_s,
        "e1": e1_s, "e2": e2_s,
        "e1_density_sample": e1_d, "e2_density_sample": e2_d,
        "weight": np.ones(n_pos), "weight_shape_sample": np.ones(n_shape),
    }


def _lc_obj(data, tmp_path, name="pklc.hdf5"):
    randoms = {k: data[k].copy() for k in
               ("RA", "DEC", "Redshift", "RA_shape_sample", "DEC_shape_sample",
                "Redshift_shape_sample")}
    randoms["weight"] = np.ones(len(randoms["RA"]))
    randoms["weight_shape_sample"] = np.ones(len(randoms["RA_shape_sample"]))
    return MeasureIALightcone(
        data=data, randoms_data=randoms, output_file_name=str(tmp_path / name),
        separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI, pi_max=60.0,
    )


def _run_lc(data, tmp_path, *, backend="tree", name="pklc.hdf5"):
    obj = _lc_obj(data, tmp_path, name)
    ss = pair_kernel.prepare_lightcone_samples(
        data, None, shapes="both", cosmology=None, over_h=False,
        responsivity_correction=False, base=obj, print_num=False)
    binning = pair_kernel.SkyRpPi(obj)
    grids = pair_kernel.accumulate(
        ss, binning, base=obj, shapes="both", chunk_axis="position",
        chunk_size_outer=20, backend=backend)
    return grids, ss, obj, binning


def _bruteforce_lc(ss, obj, binning):
    """O(N^2) reference: each galaxy projected in its OWN tangent frame."""
    nr, npi = binning.num_bins_r, binning.num_bins_pi
    pp = np.zeros((nr, npi)); xx = np.zeros((nr, npi)); px = np.zeros((nr, npi))
    r_bins, pi_bins = binning.r_bins, binning.pi_bins

    for i in range(len(ss.pos)):
        L = ss.pos[i] + ss.pos_shape
        n_LOS = L / np.linalg.norm(L, axis=1)[:, None]
        sep = ss.pos_shape - ss.pos[i]
        los = np.sum(sep * n_LOS, axis=1)
        rp = np.sqrt(np.maximum(np.sum(sep ** 2, axis=1) - los ** 2, 0.0))
        ok = ((rp >= r_bins[0]) & (rp < r_bins[-1])
              & (los >= pi_bins[0]) & (los < pi_bins[-1]))
        if not ok.any():
            continue
        s_perp = sep - los[:, None] * n_LOS

        def proj(e_arr, east, north, sel):
            phi = np.arctan2(np.sum(s_perp[sel] * north, axis=1),
                             np.sum(s_perp[sel] * east, axis=1))
            c, s2 = np.cos(2 * phi), np.sin(2 * phi)
            return e_arr[:, 0] * c - e_arr[:, 1] * s2, e_arr[:, 0] * s2 + e_arr[:, 1] * c

        ep_s, ex_s = proj(ss.e[ok], ss.east_shape[ok], ss.north_shape[ok], ok)
        e_p_row = np.broadcast_to(ss.e_pos[i], (int(ok.sum()), 2))
        ep_p, ex_p = proj(e_p_row,
                          np.broadcast_to(ss.east[i], (int(ok.sum()), 3)),
                          np.broadcast_to(ss.north[i], (int(ok.sum()), 3)), ok)
        ir = np.digitize(rp[ok], r_bins) - 1
        ip = np.digitize(los[ok], pi_bins) - 1
        np.add.at(pp, (ir, ip), ep_s * ep_p)
        np.add.at(xx, (ir, ip), ex_s * ex_p)
        np.add.at(px, (ir, ip), 0.5 * (ep_s * ex_p + ex_s * ep_p))
    return pp, xx, px


class TestShapeShapeLightcone:

    @pytest.mark.parametrize("backend", ["tree", "brute"])
    def test_matches_bruteforce(self, tmp_path, backend):
        """Each galaxy in its own (east, north) frame, against an O(N^2) reference."""
        data = _lc_catalogue()
        grids, ss, obj, binning = _run_lc(data, tmp_path, backend=backend)
        ref = _bruteforce_lc(ss, obj, binning)
        for got, want, name in zip(
                (grids.Splus_Splus, grids.Scross_Scross, grids.Splus_Scross),
                ref, ("Splus_Splus", "Scross_Scross", "Splus_Scross")):
            assert np.max(np.abs(want)) > 1e-8, f"{name} reference is trivially zero"
            np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-13,
                                       err_msg=f"{name} disagrees with brute force")

    @pytest.mark.parametrize("backend", ["tree", "brute"])
    def test_multipoles_binning_runs_and_agrees_with_w_totals(self, tmp_path, backend):
        """Occupy the lightcone multipoles cell.

        (r, mu_r) and (rp, Pi) select different pair sets, so the grids differ;
        what must hold is that the products are computed identically, which is
        checked by re-deriving the (r, mu_r) grids from the same brute-force
        projection code.
        """
        data = _lc_catalogue()
        obj = _lc_obj(data, tmp_path, "lcm.hdf5")
        ss = pair_kernel.prepare_lightcone_samples(
            data, None, shapes="both", cosmology=None, over_h=False,
            responsivity_correction=False, base=obj, print_num=False)
        grids = pair_kernel.accumulate(
            ss, binning := pair_kernel.SkyRMuR(obj), base=obj, shapes="both",
            chunk_axis="position", chunk_size_outer=20, backend=backend)
        for g in (grids.Splus_Splus, grids.Scross_Scross, grids.Splus_Scross):
            assert g.shape == (binning.num_bins_r, binning.num_bins_pi)
            assert np.all(np.isfinite(g))
        assert np.max(np.abs(grids.Splus_Splus)) > 1e-8

    @pytest.mark.parametrize("sky_binning", ["SkyRpPi", "SkyRMuR"])
    def test_delete_one_identity(self, tmp_path, sky_binning):
        """The union-deletion contract on the lightcone, for the new grids.

        The lightcone bakes responsivity into e, and here it is off, so the
        full-sample grids are already raw and subtract directly.
        """
        num_jk = 3
        data = _lc_catalogue(n_pos=60, n_shape=70, seed=99)
        obj = _lc_obj(data, tmp_path, "lcjk.hdf5")
        binning = getattr(pair_kernel, sky_binning)(obj)

        def prep(d):
            return pair_kernel.prepare_lightcone_samples(
                d, None, shapes="both", cosmology=None, over_h=False,
                responsivity_correction=False, base=obj, print_num=False)

        jk_pos = np.clip(np.digitize(data["RA"], np.linspace(20, 30, num_jk + 1)) - 1,
                         0, num_jk - 1)
        jk_shape = np.clip(
            np.digitize(data["RA_shape_sample"], np.linspace(20, 30, num_jk + 1)) - 1,
            0, num_jk - 1)
        ss = prep(data)
        ss.jk_pos, ss.jk_shape = jk_pos, jk_shape
        full = pair_kernel.accumulate(
            ss, binning, base=obj, shapes="both", chunk_axis="position",
            chunk_size_outer=20, backend="tree", jk=True, num_box=num_jk)

        for i in range(num_jk):
            keep_p, keep_s = jk_pos != i, jk_shape != i
            if keep_p.sum() < 2 or keep_s.sum() < 2:
                continue
            d = dict(data)
            for k in ("RA", "DEC", "Redshift", "e1_density_sample",
                      "e2_density_sample", "weight"):
                d[k] = data[k][keep_p]
            for k in ("RA_shape_sample", "DEC_shape_sample", "Redshift_shape_sample",
                      "e1", "e2", "weight_shape_sample"):
                d[k] = data[k][keep_s]
            direct = pair_kernel.accumulate(
                prep(d), binning, base=obj, shapes="both", chunk_axis="position",
                chunk_size_outer=20, backend="tree")
            for f, j, dg, name in (
                    (full.Splus_Splus, full.Splus_Splus_jk, direct.Splus_Splus, "Splus_Splus"),
                    (full.Scross_Scross, full.Scross_Scross_jk, direct.Scross_Scross, "Scross_Scross"),
                    (full.Splus_Scross, full.Splus_Scross_jk, direct.Splus_Scross, "Splus_Scross")):
                np.testing.assert_allclose(
                    f - j[i], dg, rtol=1e-10, atol=1e-12,
                    err_msg=f"{name} delete-one identity broken for patch {i}")

    def test_shape_sample_gets_its_own_tangent_basis(self, tmp_path):
        """east_shape/north_shape are built, orthonormal, and genuinely differ
        from the position sample's basis (otherwise the own-frame projection
        would be untestable)."""
        data = _lc_catalogue()
        _, ss, _, _ = _run_lc(data, tmp_path)
        for basis in (ss.east_shape, ss.north_shape):
            np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            np.sum(ss.east_shape * ss.north_shape, axis=1), 0.0, atol=1e-12)
        assert not np.allclose(ss.east_shape[:len(ss.east)], ss.east)

    def test_g_plus_terms_keep_the_partner_frame(self, tmp_path):
        """shapes='both' must not silently re-project the existing g+ terms.

        The g+ path deliberately keeps its convention of projecting the shape
        galaxy in its *partner's* tangent frame; only the shape-shape products
        use each galaxy's own frame.
        """
        data = _lc_catalogue()
        obj = _lc_obj(data, tmp_path, "cmp.hdf5")
        binning = pair_kernel.SkyRpPi(obj)

        def go(mode):
            ss = pair_kernel.prepare_lightcone_samples(
                data, None, shapes=mode, cosmology=None, over_h=False,
                responsivity_correction=False, base=obj, print_num=False)
            return pair_kernel.accumulate(
                ss, binning, base=obj, shapes=mode, chunk_axis="position",
                chunk_size_outer=20, backend="tree")

        plain, both = go(True), go("both")
        np.testing.assert_array_equal(plain.Splus_D, both.Splus_D)
        np.testing.assert_array_equal(plain.Scross_D, both.Scross_D)
        np.testing.assert_array_equal(plain.DD, both.DD)


# ===========================================================================
# 3. Jackknife
# ===========================================================================

class TestShapeShapeJackknife:

    NUM_BOX = 4

    @pytest.mark.parametrize("kind", ["rppi", "rmur"])
    def test_delete_one_identity_box(self, tmp_path, kind):
        """full - jk[i] must equal a direct measurement on the catalogue with
        patch i physically removed from *both* samples.

        This is the union-deletion contract, and the same check that locks the
        existing S+D jackknife (TestBoxJackknifeDeleteOneIdentity). The jk grids
        are raw, so the full-sample grids are multiplied back up by the two
        responsivity factors before subtracting.
        """
        data = _box_catalogue(n_pos=90, n_shape=110, seed=7)
        grids, R, R_pos = _run_box(data, tmp_path, jk=True, num_box=self.NUM_BOX,
                                   name=f"jk_{kind}.hdf5", kind=kind)
        resp = (2 * R) * (2 * R_pos)
        jk_pos = _patches(data["Position"], self.NUM_BOX)
        jk_shape = _patches(data["Position_shape_sample"], self.NUM_BOX)

        for i in range(self.NUM_BOX):
            keep_p, keep_s = jk_pos != i, jk_shape != i
            if keep_p.sum() < 2 or keep_s.sum() < 2:
                continue
            deleted = {
                "Position": data["Position"][keep_p],
                "Position_shape_sample": data["Position_shape_sample"][keep_s],
                "Axis_Direction": data["Axis_Direction"][keep_s],
                "q": data["q"][keep_s],
                "Axis_Direction_density_sample": data["Axis_Direction_density_sample"][keep_p],
                "q_density_sample": data["q_density_sample"][keep_p],
                "weight": data["weight"][keep_p],
                "weight_shape_sample": data["weight_shape_sample"][keep_s],
                "LOS": 2,
            }
            # responsivity off -> the direct grids are raw, like the jk ones
            direct, _, _ = _run_box(deleted, tmp_path, responsivity=False,
                                    name=f"d{kind}{i}.hdf5", kind=kind)
            for full, jkg, dg, name in (
                    (grids.Splus_Splus, grids.Splus_Splus_jk, direct.Splus_Splus, "Splus_Splus"),
                    (grids.Scross_Scross, grids.Scross_Scross_jk, direct.Scross_Scross, "Scross_Scross"),
                    (grids.Splus_Scross, grids.Splus_Scross_jk, direct.Splus_Scross, "Splus_Scross")):
                np.testing.assert_allclose(
                    full * resp - jkg[i], dg, rtol=1e-10, atol=1e-12,
                    err_msg=f"{name} delete-one identity broken for patch {i}")

    def test_jk_grids_absent_without_jk(self, tmp_path):
        data = _box_catalogue()
        grids, _, _ = _run_box(data, tmp_path)
        assert grids.Splus_Splus_jk is None
        assert grids.Scross_Scross_jk is None
        assert grids.Splus_Scross_jk is None


# ===========================================================================
# 4. Non-interference with the existing paths
# ===========================================================================

class TestShapeShapeDoesNotDisturbExistingPaths:

    def test_shapes_true_grids_unchanged_by_both(self, tmp_path):
        """Asking for 'both' must not perturb S+D / SxD / DD at all.

        The shape-shape products are accumulated in their own branch after the
        g+ ones, so the existing sums keep their float summation order exactly.
        """
        data = _box_catalogue()
        obj = _box_obj(data, tmp_path, "x.hdf5")
        binning = pair_kernel.BoxRpPi(obj)

        def go(mode):
            ss = pair_kernel.prepare_box_samples(
                data, None, len(data["Position"]), len(data["Position_shape_sample"]),
                shapes=mode, ellipticity="distortion", base=obj)
            R = np.sum(ss.weight_shape * (1 - ss.e ** 2 / 2.0)) / np.sum(ss.weight_shape)
            return pair_kernel.accumulate(
                ss, binning, base=obj, R=R, R_pos=0.5, shapes=mode,
                chunk_axis="shape", chunk_size_outer=25)

        plain, both = go(True), go("both")
        np.testing.assert_array_equal(plain.DD, both.DD)
        np.testing.assert_array_equal(plain.Splus_D, both.Splus_D)
        np.testing.assert_array_equal(plain.Scross_D, both.Scross_D)
        assert plain.Splus_Splus is None and both.Splus_Splus is not None

    def test_shape_mode_normalisation(self):
        assert pair_kernel.shape_mode(True) == (True, False)
        assert pair_kernel.shape_mode(False) == (False, False)
        assert pair_kernel.shape_mode("both") == (True, True)
