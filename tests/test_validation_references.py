"""
Cross-package validation tests.

Compares measureia results on the synthetic radial-alignment mock against
committed reference outputs produced by external packages (halotools,
treecorr) via the scripts in validation/. The external packages are NOT
needed to run these tests — only the committed reference files are.

If a reference file is missing (e.g. freshly cloned before the validation
scripts were ever run with the external package installed), the
corresponding tests are skipped, not failed.
"""

import os
import sys

import h5py
import numpy as np
import pytest

_VALIDATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "validation")
sys.path.insert(0, _VALIDATION_DIR)

import run_box_halotools as box_halotools
import run_box_cov_bridge as box_cov_bridge
import run_box_multipoles_corrpc as box_multipoles
import run_lightcone_corrpc as lc_corrpc
import run_lightcone_multipoles_corrpc as lc_multipoles
import run_lightcone_treecorr as lc_treecorr
import run_plane_parallel as plane_parallel
from mock_catalogues import (radial_alignment_box_mock, responsivity,
                             radial_alignment_lightcone_mock)

_BOX_REF = box_halotools.REFERENCE_FILE
_LC_REF = lc_treecorr.REFERENCE_FILE

requires_box_reference = pytest.mark.skipif(
    not os.path.exists(_BOX_REF),
    reason="no committed halotools reference outputs; run "
           "validation/run_box_halotools.py with halotools installed first",
)


@pytest.fixture(scope="module")
def box_measureia_results(tmp_path_factory):
    """measureia w_g+ / w_gg on the validation mock (same config as the script)."""
    mock = radial_alignment_box_mock()
    out = str(tmp_path_factory.mktemp("validation") / "box_halotools_mock.hdf5")
    rp, w_g_plus, w_gg = box_halotools.run_measureia(mock, out)
    return {"rp": rp, "w_g_plus": w_g_plus, "w_gg": w_gg,
            "R": responsivity(mock)}


@requires_box_reference
class TestBoxAgainstHalotools:
    def test_reference_binning_matches(self, box_measureia_results):
        with h5py.File(_BOX_REF, "r") as f:
            rp_bins = f["rp_bins"][:]
            assert f.attrs["pi_max"] == box_halotools.PI_MAX
        expected = np.logspace(np.log10(box_halotools.RP_LIMS[0]),
                               np.log10(box_halotools.RP_LIMS[1]),
                               box_halotools.NUM_BINS_RP + 1)
        np.testing.assert_allclose(rp_bins, expected, rtol=1e-10)

    def test_w_g_plus_matches_halotools_up_to_2R(self, box_measureia_results):
        """measureia w_g+ times the responsivity 2R equals halotools w_g+."""
        with h5py.File(_BOX_REF, "r") as f:
            wgp_halotools = f["w_g_plus"][:]
        wgp_measureia = box_measureia_results["w_g_plus"]
        R = box_measureia_results["R"]
        np.testing.assert_allclose(wgp_measureia * 2 * R, wgp_halotools,
                                   rtol=1e-10, atol=1e-12)

    def test_w_gg_matches_halotools(self, box_measureia_results):
        with h5py.File(_BOX_REF, "r") as f:
            wgg_halotools = f["w_gg"][:]
        np.testing.assert_allclose(box_measureia_results["w_gg"], wgg_halotools,
                                   rtol=1e-10, atol=1e-12)


requires_lc_reference = pytest.mark.skipif(
    not os.path.exists(_LC_REF),
    reason="no committed treecorr reference outputs; run "
           "validation/run_lightcone_treecorr.py with treecorr installed first",
)


@pytest.fixture(scope="module")
def lc_measureia_results(tmp_path_factory):
    """measureia lightcone w_g+ / w_gg on the validation mock (same config
    as run_lightcone_treecorr.py)."""
    data, randoms, info, dist = lc_treecorr.build_catalogues()
    tmp = tmp_path_factory.mktemp("validation_lc")
    out = str(tmp / "lc_treecorr_mock.hdf5")
    ia, rp, w_g_plus, w_gg = lc_treecorr.run_measureia(data, randoms, out, str(tmp) + "/")
    return {"rp": rp, "w_g_plus": w_g_plus, "w_gg": w_gg,
            "r_bins": ia.r_bins, "pi_bins": ia.pi_bins}


@requires_lc_reference
class TestLightconeAgainstTreecorr:
    """Agreement is close but not machine precision: treecorr's Rperp
    separation definition and great-circle shear projection differ from
    measureia's midpoint-LOS / (east, north)-frame conventions by curvature
    terms, so a few pairs migrate bins and w_g+ picks up small frame
    differences. On this mock: w_g+ agrees to ~1e-5 (relative) in the
    high-signal bins, w_gg to <=0.5%; the atol term covers the near-zero
    outer bins."""

    def test_reference_binning_matches(self, lc_measureia_results):
        with h5py.File(_LC_REF, "r") as f:
            np.testing.assert_allclose(f["r_bins"][:], lc_measureia_results["r_bins"],
                                       rtol=1e-10)
            np.testing.assert_allclose(f["pi_bins"][:], lc_measureia_results["pi_bins"],
                                       rtol=1e-10)

    def test_w_g_plus_matches_treecorr(self, lc_measureia_results):
        with h5py.File(_LC_REF, "r") as f:
            wgp_treecorr = f["w_g_plus"][:]
        np.testing.assert_allclose(lc_measureia_results["w_g_plus"], wgp_treecorr,
                                   rtol=5e-3, atol=0.05)

    def test_w_gg_matches_treecorr(self, lc_measureia_results):
        with h5py.File(_LC_REF, "r") as f:
            wgg_treecorr = f["w_gg"][:]
        np.testing.assert_allclose(lc_measureia_results["w_gg"], wgg_treecorr,
                                   rtol=5e-3, atol=0.05)


@pytest.fixture(scope="module")
def plane_parallel_results(tmp_path_factory):
    """Box and lightcone measurements of the identical (margin) mock,
    plus the responsivity — see validation/run_plane_parallel.py."""
    mock = plane_parallel.build_mock()
    tmp = tmp_path_factory.mktemp("validation_pp")
    box = plane_parallel.run_box(mock, str(tmp / "pp_box.hdf5"))
    lc = plane_parallel.run_lightcone(mock, str(tmp / "pp_lc.hdf5"), str(tmp) + "/")
    return {"box": box, "lc": lc, "R": responsivity(mock)}


class TestPlaneParallelConsistency:
    """The box pipeline (validated against halotools at machine precision)
    and the lightcone pipeline (validated against treecorr) must agree on
    the identical catalogue in the plane-parallel limit.

    Known, intended differences (see validation/run_plane_parallel.py):
    the box divides S+ terms by the responsivity 2R, the lightcone does
    not; and the box's analytic RR assumes a periodic box while the
    lightcone's empirical randoms live in a bounded window. The tests
    below compare pair counts directly and the w statistics with the RR
    difference removed. The first and last separation bins are excluded
    where their pair counts are too small (single pair migrations between
    bins move them by several percent)."""

    def test_DD_pair_counts_agree(self, plane_parallel_results):
        box, lc = plane_parallel_results["box"], plane_parallel_results["lc"]
        ratio = lc["DD"].sum(axis=1) / box["DD"].sum(axis=1)
        np.testing.assert_allclose(ratio, 1.0, atol=0.025)

    def test_SplusD_agrees_up_to_2R(self, plane_parallel_results):
        """The raw S+D sums differ by exactly the responsivity factor 2R."""
        box, lc = plane_parallel_results["box"], plane_parallel_results["lc"]
        R = plane_parallel_results["R"]
        ratio = lc["SplusD"].sum(axis=1) / box["SplusD"].sum(axis=1) / (2 * R)
        np.testing.assert_allclose(ratio[1:7], 1.0, atol=0.03)

    def test_w_gg_agrees_with_matched_RR(self, plane_parallel_results):
        box, lc = plane_parallel_results["box"], plane_parallel_results["lc"]
        _, wgg_ana = plane_parallel.lightcone_w_with_analytic_RR(
            box, lc, plane_parallel_results["R"])
        np.testing.assert_allclose(wgg_ana / box["w_gg"], 1.0, atol=0.04)

    def test_w_g_plus_agrees_with_matched_RR(self, plane_parallel_results):
        box, lc = plane_parallel_results["box"], plane_parallel_results["lc"]
        wgp_ana, _ = plane_parallel.lightcone_w_with_analytic_RR(
            box, lc, plane_parallel_results["R"])
        np.testing.assert_allclose((wgp_ana / box["w_g_plus"])[1:6], 1.0, atol=0.05)


_COV_REF = os.path.join(_VALIDATION_DIR, "reference_outputs", "lightcone_treecorr_cov.hdf5")

requires_cov_reference = pytest.mark.skipif(
    not os.path.exists(_COV_REF),
    reason="no committed treecorr jackknife covariance reference; run "
           "validation/run_lightcone_treecorr_cov.py with treecorr installed first",
)


@requires_cov_reference
class TestJackknifeCovarianceAgainstTreecorr:
    """measureia's jackknife covariance vs an explicit delete-one-patch
    jackknife built from treecorr counts, using the identical (seeded)
    kmeans patch assignment — see validation/run_lightcone_treecorr_cov.py.
    Both compute the same deterministic statistic, so agreement is limited
    only by the estimator-level differences (~1e-5 on w_g+)."""

    @pytest.fixture(scope="class")
    def cov_results(self, tmp_path_factory):
        import run_lightcone_treecorr_cov as v
        data, randoms, info, dist = v.build_catalogues()
        tmp = tmp_path_factory.mktemp("validation_cov")
        out = str(tmp / "cov_mia.hdf5")
        ia = v.make_measureia(data, randoms, out)
        patches = ia.assign_jackknife_patches(data, randoms, v.NUM_JK,
                                              seed=v.PATCH_SEED)
        if "randoms_position" not in patches:
            patches["randoms_position"] = patches["randoms"]
            patches["randoms_shape"] = patches["randoms"]
        if os.path.exists(out):
            os.remove(out)
        ia, mia = v.run_measureia_jk(data, randoms, patches, out, str(tmp) + "/")
        with h5py.File(_COV_REF, "r") as f:
            assert f.attrs["patch_seed"] == v.PATCH_SEED
            return mia, f["cov_w_g_plus"][:], f["cov_w_gg"][:]

    @staticmethod
    def _corrmat(cov):
        s = np.sqrt(np.diag(cov))
        return cov / np.outer(s, s)

    def test_w_g_plus_jackknife_std(self, cov_results):
        mia, cov_gp_tc, _ = cov_results
        np.testing.assert_allclose(mia["std_gp"], np.sqrt(np.diag(cov_gp_tc)),
                                   rtol=0.03)

    def test_w_gg_jackknife_std(self, cov_results):
        mia, _, cov_gg_tc = cov_results
        np.testing.assert_allclose(mia["std_gg"], np.sqrt(np.diag(cov_gg_tc)),
                                   rtol=0.03)

    def test_correlation_matrix_structure(self, cov_results):
        mia, cov_gp_tc, cov_gg_tc = cov_results
        np.testing.assert_allclose(self._corrmat(mia["cov_gp"]),
                                   self._corrmat(cov_gp_tc), atol=0.05)
        np.testing.assert_allclose(self._corrmat(mia["cov_gg"]),
                                   self._corrmat(cov_gg_tc), atol=0.05)


class TestResponsivityOption:
    """The responsivity flag divides all S+ terms by 2R. Defaults: box True
    (shapes derive from raw axis ratios), lightcone False (e1/e2 are treated
    as calibrated shear estimates). Toggling the flag must rescale w_g+ by
    exactly 2R and leave w_gg untouched."""

    def test_box_flag_rescales_by_2R(self, tmp_path):
        import run_box_halotools as v
        mock = radial_alignment_box_mock()
        R = responsivity(mock)
        data = {k: mock[k] for k in
                ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
        from measureia import MeasureIABox
        results = {}
        for flag in (True, False):
            ia = MeasureIABox(
                data, str(tmp_path / f"box_resp_{flag}.hdf5"),
                simulation=None, snapshot=None,
                separation_limits=v.RP_LIMS, num_bins_r=v.NUM_BINS_RP,
                num_bins_pi=v.NUM_BINS_PI, pi_max=v.PI_MAX,
                boxsize=mock["boxsize"], num_nodes=1)
            ia.measure_xi_w(v.DATASET, "both", 0, temp_file_path=False,
                            responsivity=flag)
            with h5py.File(str(tmp_path / f"box_resp_{flag}.hdf5")) as f:
                results[flag] = (f[f"w_g_plus/{v.DATASET}"][:], f[f"w_gg/{v.DATASET}"][:])
        np.testing.assert_allclose(results[False][0], results[True][0] * 2 * R,
                                   rtol=1e-12)
        np.testing.assert_allclose(results[False][1], results[True][1], rtol=1e-12)

    def test_lightcone_flag_rescales_by_2R(self, tmp_path):
        import run_lightcone_treecorr as v
        data, randoms, info, dist = v.build_catalogues()
        e = np.sqrt(data["e1"] ** 2 + data["e2"] ** 2)
        w = data["weight_shape_sample"]
        R = np.sum(w * (1 - e ** 2 / 2.0)) / np.sum(w)
        from measureia import MeasureIALightcone
        results = {}
        for flag in (True, False):
            out = str(tmp_path / f"lc_resp_{flag}.hdf5")
            ia = MeasureIALightcone(
                data={k: v_ for k, v_ in data.items()},
                randoms_data={k: v_ for k, v_ in randoms.items()},
                output_file_name=out,
                separation_limits=v.RP_LIMS, num_bins_r=v.NUM_BINS_RP,
                num_bins_pi=v.NUM_BINS_PI, pi_max=v.PI_MAX, num_nodes=1)
            ia.measure_xi_w("galaxies", v.DATASET, "both", measure_cov=False,
                            tree=True, cosmology=v.COSMOLOGY, over_h=False,
                            temp_file_path=str(tmp_path) + "/", responsivity=flag)
            with h5py.File(out) as f:
                results[flag] = (f[f"w_g_plus/{v.DATASET}"][:], f[f"w_gg/{v.DATASET}"][:])
        np.testing.assert_allclose(results[False][0], results[True][0] * 2 * R,
                                   rtol=1e-10)
        np.testing.assert_allclose(results[False][1], results[True][1], rtol=1e-12)


_LC_CORRPC_REF = lc_corrpc.REFERENCE_FILE

requires_lc_corrpc_reference = pytest.mark.skipif(
    not os.path.exists(_LC_CORRPC_REF),
    reason="no committed corr_pc lightcone reference; build corr_pc (with the "
           "DRs patch) and run validation/run_lightcone_corrpc.py",
)


@requires_lc_corrpc_reference
class TestLightconeAgainstCorrPC:
    """measureia's lightcone 'galaxies' estimator vs corr_pc sky mode
    (coordinates=0), which implements the identical estimator with explicit
    randoms — a second, independent lightcone cross-check besides treecorr.
    Differences are limited by the rp/pi separation definitions (corr_pc:
    great-circle angle x mean comoving distance, plane-parallel radial pi;
    measureia: midpoint-LOS projections), the same class as treecorr's
    Rperp. On this mock: w_g+ agrees to <=0.15%, w_gg to <=0.4% (near-zero
    outer bins covered by the atol term). Reuses the treecorr-comparison
    mock and configuration, hence the shared fixture."""

    def test_w_g_plus_matches_corrpc(self, lc_measureia_results):
        with h5py.File(_LC_CORRPC_REF, "r") as f:
            wgp_pc = f["w_g_plus"][:]
        np.testing.assert_allclose(lc_measureia_results["w_g_plus"], wgp_pc,
                                   rtol=5e-3, atol=0.05)

    def test_w_gg_matches_corrpc(self, lc_measureia_results):
        with h5py.File(_LC_CORRPC_REF, "r") as f:
            wgg_pc = f["w_gg"][:]
        np.testing.assert_allclose(lc_measureia_results["w_gg"], wgg_pc,
                                   rtol=5e-3, atol=0.05)


_LC_MULTIPOLES_REF = lc_multipoles.REFERENCE_FILE

requires_lc_multipoles_reference = pytest.mark.skipif(
    not os.path.exists(_LC_MULTIPOLES_REF),
    reason="no committed corr_pc lightcone multipoles reference; build corr_pc "
           "(with the DRs patch) and run "
           "validation/run_lightcone_multipoles_corrpc.py",
)


@requires_lc_multipoles_reference
class TestLightconeMultipolesAgainstCorrPC:
    """measureia's lightcone multipoles vs corr_pc sky r-mu mode
    (coordinates=1), with measureia's Legendre integration applied to the
    corr_pc grid. The even-in-mu Legendre weights cancel corr_pc's internal
    signed-pi reordering, so the multipole level is the meaningful
    comparison. Differences are limited by the separation-definition
    curvature terms (as for w): <=0.4% in all signal bins on this mock,
    near-zero outer bins covered by atol."""

    @pytest.fixture(scope="class")
    def mp_results(self, tmp_path_factory):
        data, randoms, info = lc_multipoles.build_catalogues()
        tmp = tmp_path_factory.mktemp("validation_lc_mp")
        out = str(tmp / "lc_multipoles.hdf5")
        return lc_multipoles.run_measureia(data, randoms, out, str(tmp) + "/")

    def test_multipole_g_plus_matches_corrpc(self, mp_results):
        with h5py.File(_LC_MULTIPOLES_REF, "r") as f:
            mp_gp_pc = f["multipole_gp"][:]
        np.testing.assert_allclose(mp_results["multipole_gp"], mp_gp_pc,
                                   rtol=5e-3, atol=0.05)

    def test_multipole_gg_matches_corrpc(self, mp_results):
        with h5py.File(_LC_MULTIPOLES_REF, "r") as f:
            mp_gg_pc = f["multipole_gg"][:]
        np.testing.assert_allclose(mp_results["multipole_gg"], mp_gg_pc,
                                   rtol=5e-3, atol=0.05)

    def test_signal_is_non_null(self, mp_results):
        assert mp_results["multipole_gp"][0] > 10.0
        assert mp_results["multipole_gg"][0] > 100.0


_MULTIPOLES_REF = box_multipoles.REFERENCE_FILE

requires_multipoles_reference = pytest.mark.skipif(
    not os.path.exists(_MULTIPOLES_REF),
    reason="no committed corr_pc reference outputs; build corr_pc and run "
           "validation/run_box_multipoles_corrpc.py with CORR_PC_BIN set",
)


@pytest.fixture(scope="module")
def multipoles_measureia_results(tmp_path_factory):
    """measureia box multipoles on the validation mock (same config as
    run_box_multipoles_corrpc.py)."""
    mock = radial_alignment_box_mock()
    out = str(tmp_path_factory.mktemp("validation_mp") / "multipoles.hdf5")
    mia = box_multipoles.run_measureia(mock, out)
    n_pos = len(mock["Position"])
    return {"mia": mia, "R": responsivity(mock),
            "rr_norm": (n_pos - 1.0) / n_pos}


@requires_multipoles_reference
class TestBoxMultipolesAgainstCorrPC:
    """measureia's box multipoles vs corr_pc (github.com/sukhdeep2/corr_pc,
    Singh 2021) in periodic-box (r, mu) mode, with measureia's own Legendre
    integration applied to the corr_pc grid. Documented convention
    adjustments: responsivity 2R, the (N_pos-1)/N_pos analytic-RR
    normalisation, and corr_pc's opposite e2 chirality (handled when writing
    its inputs). Agreement is limited only by corr_pc's 6-significant-digit
    text output (~1e-6; a few 1e-5 in near-zero bins)."""

    def test_reference_binning_matches(self, multipoles_measureia_results):
        """Bin centres agree to corr_pc's 6-significant-digit text output."""
        with h5py.File(_MULTIPOLES_REF, "r") as f:
            np.testing.assert_allclose(f["r"][:], multipoles_measureia_results["mia"]["r"],
                                       rtol=1e-5)
            np.testing.assert_allclose(f["mu"][:], multipoles_measureia_results["mia"]["mu"],
                                       rtol=1e-5)

    def test_xi_grids_match_corrpc(self, multipoles_measureia_results):
        res = multipoles_measureia_results
        with h5py.File(_MULTIPOLES_REF, "r") as f:
            xi_gp_pc, xi_gg_pc = f["xi_gp"][:], f["xi_gg"][:]
        np.testing.assert_allclose(
            res["mia"]["xi_gp"] * 2 * res["R"] * res["rr_norm"], xi_gp_pc,
            rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            (res["mia"]["xi_gg"] + 1) * res["rr_norm"], xi_gg_pc + 1,
            rtol=1e-4, atol=1e-6)

    def test_multipoles_match_corrpc(self, multipoles_measureia_results):
        res = multipoles_measureia_results
        with h5py.File(_MULTIPOLES_REF, "r") as f:
            mp_gp_pc, mp_gg_pc = f["multipole_gp"][:], f["multipole_gg"][:]
        np.testing.assert_allclose(
            res["mia"]["multipole_gp"] * 2 * res["R"] * res["rr_norm"], mp_gp_pc,
            rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            (res["mia"]["multipole_gg"] + 1) * res["rr_norm"], mp_gg_pc + 1,
            rtol=1e-4, atol=1e-6)

    def test_signal_is_non_null(self, multipoles_measureia_results):
        mia = multipoles_measureia_results["mia"]
        assert np.max(np.abs(mia["multipole_gp"])) > 1.0
        assert mia["multipole_gg"][0] > 10.0


@pytest.fixture(scope="module")
def box_jk_delete_one(tmp_path_factory):
    """Box jackknife run plus independent direct measurements on each
    physically deleted subbox catalogue (see validation/run_box_cov_bridge.py)."""
    mock = plane_parallel.build_mock()
    tmp = tmp_path_factory.mktemp("validation_boxjk")
    box = box_cov_bridge.run_box_jk(mock, str(tmp / "box_jk.hdf5"), str(tmp) + "/")
    direct = box_cov_bridge.run_delete_one_direct(
        mock, str(tmp / "direct.hdf5"), str(tmp) + "/")
    return box, direct


class TestBoxJackknifeDeleteOneIdentity:
    """The box jackknife reconstructs each delete-one realisation by count
    subtraction (a pair is removed when either member is in the deleted
    subbox) with the analytic RR rescaled to the retained counts and volume.
    Each reconstruction must equal an independent direct measurement on the
    physically deleted catalogue at floating-point precision — no external
    package and no approximation is involved in this identity."""

    @pytest.fixture(scope="class")
    def identity(self, box_jk_delete_one):
        box, direct = box_jk_delete_one
        return box_cov_bridge.delete_one_identity(box, direct)

    def test_dd_count_grids(self, identity):
        assert identity["DD"] < 1e-10

    def test_splusd_grids_with_per_realisation_responsivity(self, identity):
        assert identity["SplusD"] < 1e-10

    def test_rr_volume_factor(self, identity):
        """RR_jk equals the direct-run analytic RR times exactly V/V_del."""
        assert identity["RR"] < 1e-12

    def test_w_realisations(self, identity):
        assert identity["w_g_plus"] < 1e-10
        assert identity["w_gg"] < 1e-10


_BRIDGE_REF = box_cov_bridge.REFERENCE_FILE

requires_bridge_reference = pytest.mark.skipif(
    not os.path.exists(_BRIDGE_REF),
    reason="no committed box covariance bridge reference; run "
           "validation/run_box_cov_bridge.py first",
)


@requires_bridge_reference
class TestBoxCovarianceBridge:
    """Box jackknife vs the treecorr-validated lightcone jackknife with the
    identical subbox partition on the plane-parallel embedding. Agreement is
    loose BY EXPECTATION (see validation/run_box_cov_bridge.py): with 8
    patches the delete-one deviations are only a few % of the mean, so the
    ~0.1-1% plane-parallel-vs-sky pair migrations plus the genuinely
    different estimator definitions move the stds by tens of percent. The
    sharp statement is that the box-style estimator rebuilt from the
    lightcone's own retained counts reproduces the lightcone stds — the
    residual lives in the counts/estimators, not the jackknife machinery
    (which TestBoxJackknifeDeleteOneIdentity locks at machine precision)."""

    def test_box_covariance_regression(self, box_jk_delete_one):
        """Fresh box jackknife covariance matches the committed snapshot."""
        box, _ = box_jk_delete_one
        with h5py.File(_BRIDGE_REF, "r") as f:
            np.testing.assert_allclose(box["std_gp"], f["box_std_gp"][:], rtol=1e-8)
            np.testing.assert_allclose(box["std_gg"], f["box_std_gg"][:], rtol=1e-8)
            np.testing.assert_allclose(box["cov_gp"], f["box_cov_gp"][:], rtol=1e-8,
                                       atol=1e-12)
            np.testing.assert_allclose(box["cov_gg"], f["box_cov_gg"][:], rtol=1e-8,
                                       atol=1e-12)

    def test_committed_identity_metrics(self):
        """The committed run's delete-one identity held at machine precision."""
        with h5py.File(_BRIDGE_REF, "r") as f:
            for key in ["DD", "SplusD", "RR", "w_g_plus", "w_gg"]:
                assert f.attrs[f"identity_{key}"] < 1e-9

    def test_bridge_agreement_level(self):
        """Cross-pipeline std ratios stay within the documented loose band."""
        with h5py.File(_BRIDGE_REF, "r") as f:
            for key in ["gp", "gg"]:
                ratio = f[f"box_std_{key}"][:] / f[f"lightcone_std_{key}"][:]
                assert np.all((ratio > 0.6) & (ratio < 1.15))

    def test_boxstyle_reconstruction_matches_lightcone(self):
        """Box-style estimator from lightcone counts reproduces lightcone stds."""
        with h5py.File(_BRIDGE_REF, "r") as f:
            for key in ["gp", "gg"]:
                ratio = f[f"boxstyle_std_{key}"][:] / f[f"lightcone_std_{key}"][:]
                assert np.all((ratio > 0.7) & (ratio < 1.15))

    def test_analytic_rr_shape_approximation_is_small(self):
        """The hole-boundary bin-shape the analytic RR misses stays ~2%."""
        with h5py.File(_BRIDGE_REF, "r") as f:
            assert f.attrs["rr_shape_error"] < 0.04


class TestMockCatalogue:
    """The mock itself must stay byte-identical across versions — the committed
    reference outputs are only valid for this exact catalogue."""

    def test_mock_is_deterministic(self):
        m1 = radial_alignment_box_mock()
        m2 = radial_alignment_box_mock()
        for key in ["Position", "Position_shape_sample", "Axis_Direction", "q"]:
            np.testing.assert_array_equal(m1[key], m2[key])

    def test_mock_fingerprint(self):
        """Guards the reference files against silent generator changes."""
        mock = radial_alignment_box_mock()
        fingerprint = np.array([
            mock["Position"].sum(),
            mock["Position_shape_sample"].sum(),
            mock["Axis_Direction"].sum(),
            mock["q"].sum(),
        ])
        np.testing.assert_allclose(
            fingerprint,
            [825592.8323787726, 734008.7125466282, -39.8313510979187, 1430.6731584753884],
            rtol=1e-12,
        )

    def test_signal_is_non_null(self, box_measureia_results):
        """w_g+ and w_gg must be clearly non-zero on small scales, otherwise
        the cross-package ratios are meaningless."""
        assert abs(box_measureia_results["w_g_plus"][0]) > 1.0
        assert box_measureia_results["w_gg"][0] > 10.0

    def test_lightcone_mock_fingerprint(self):
        """Guards the lightcone reference files against generator changes."""
        data, rand, info = radial_alignment_lightcone_mock()
        fingerprint = np.array([
            data["RA"].sum(),
            data["RA_shape_sample"].sum(),
            data["e1"].sum(),
            data["e2"].sum(),
            rand["RA"].sum() + rand["RA_shape_sample"].sum(),
        ])
        np.testing.assert_allclose(
            fingerprint,
            [18032.789003600938, 144263.78097673124, -30.911144187049196,
             12.974747765854334, 809976.0453178096],
            rtol=1e-12,
        )

    def test_lightcone_signal_is_non_null(self, lc_measureia_results):
        assert abs(lc_measureia_results["w_g_plus"][0]) > 100.0
        assert lc_measureia_results["w_gg"][0] > 100.0
