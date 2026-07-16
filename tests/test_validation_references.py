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
