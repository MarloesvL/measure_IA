"""
test_box_measure_xi_multipoles.py
==================================
Comprehensive tests for MeasureIABox.measure_xi_multipoles().

All tests use the synthetic catalogue fixtures from conftest.py — no TNG
simulation files are required.  Mirrors the 11-section structure of
test_box_measure_xi_w.py, adding RR and xi realisation checks (section 5)
that are specific to the multipole estimator.

Covers
------
  1.  corr_type variations: 'both', 'g+', 'gg', invalid
  2.  Ellipticity definitions: 'distortion' vs 'ellipticity'
  3.  num_jk input validation
  4.  Computation backends, no JK: brute == tree == multiproc
  5.  Computation backends, with JK: DD / SplusD / RR / xi per realisation
  6.  Output shape and r-bin consistency
  7.  Covariance matrix properties (symmetry, non-negative diagonal)
  8.  Masks: reduce pair count
  9.  Weights: 4× and 16× scaling laws
  10. Jackknife region assignment
  11. _combine_jackknife_information reproduces stored covariance
  12. Intermediate xi outputs: SplusD, RR, r/mu_r grids, ScrossD,
      xi_g_cross group, per-realisation bin grids
  13. Intermediate pair-count equality: brute == tree, tree == multiproc: SplusD, RR, r/mu_r grids, ScrossD,
      xi_g_cross group, per-realisation bin grids
"""

import numpy as np
import pytest
import h5py


NUM_JK = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(obj, group, key):
    with h5py.File(obj.output_file_name, "r") as f:
        return f[obj.snap_group + group][key][:]


# ---------------------------------------------------------------------------
# 1. corr_type variations
# ---------------------------------------------------------------------------

class TestCorrTypeM:

    def test_gp_matches_both(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mct_both", "both", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mct_gp",   "g+",   0, temp_file_path=False)

        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "mct_both"),
            _read(obj, "multipoles_g_plus", "mct_gp"))
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "mct_both_r"),
            _read(obj, "multipoles_g_plus", "mct_gp_r"))

    def test_gg_matches_both(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mct_both", "both", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mct_gg",   "gg",   0, temp_file_path=False)

        np.testing.assert_array_equal(
            _read(obj, "multipoles_gg", "mct_both"),
            _read(obj, "multipoles_gg", "mct_gg"))

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1):
        # corr_type is now validated up front (uniform ValueError), before any pair counting
        with pytest.raises(ValueError, match="corr_type"):
            IA_mock_TNG300_n1.measure_xi_multipoles(
                "bad", "gg+", 0, temp_file_path=False)

    def test_gg_count_pairs_matches_full_all_backends(self, IA_mock_TNG300_n1,
                                                      IA_mock_TNG300_n8, tmp_path):
        """corr_type='gg' dispatches to the DD-only count_pairs backends —
        DD grid and multipoles must match the full-loop ('both') result for
        brute, tree and multiprocessing."""
        tp = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mcp_full", "both", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mcp_brute", "gg", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mcp_tree", "gg", 0, temp_file_path=tp)
        ref_dd = _read(obj, "multipoles/xi_gg", "mcp_full_DD")
        ref_m = _read(obj, "multipoles_gg", "mcp_full")
        for name in ("mcp_brute", "mcp_tree"):
            np.testing.assert_array_equal(
                _read(obj, "multipoles/xi_gg", f"{name}_DD"), ref_dd)
            np.testing.assert_array_equal(
                _read(obj, "multipoles_gg", name), ref_m)
        obj8 = IA_mock_TNG300_n8
        obj8.measure_xi_multipoles("mcp_mp", "gg", 0, temp_file_path=tp,
                                   chunk_size=50)
        np.testing.assert_allclose(
            _read(obj8, "multipoles/xi_gg", "mcp_mp_DD"), ref_dd, rtol=1e-10)
        np.testing.assert_allclose(
            _read(obj8, "multipoles_gg", "mcp_mp"), ref_m, rtol=1e-10)

    def test_gg_count_pairs_jk_matches_full(self, IA_mock_TNG300_n1,
                                            IA_mock_TNG300_n8, tmp_path):
        """corr_type='gg' with num_jk>0 dispatches to the DD-only jk count
        backends — final multipoles, realisations and covariance must match
        the full-loop ('both') jk path for brute, tree and multiprocessing."""
        tp = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mjkcp_full", "both", NUM_JK,
                                  temp_file_path=tp)
        obj.measure_xi_multipoles("mjkcp_tree", "gg", NUM_JK,
                                  temp_file_path=tp)
        obj.measure_xi_multipoles("mjkcp_brute", "gg", NUM_JK,
                                  temp_file_path=False)
        ref_m = _read(obj, "multipoles_gg", "mjkcp_full")
        ref_cov = _read(obj, "multipoles_gg",
                        f"mjkcp_full_jackknife_cov_{NUM_JK}")
        for name, rt in (("mjkcp_tree", 1e-12), ("mjkcp_brute", 1e-10)):
            np.testing.assert_allclose(_read(obj, "multipoles_gg", name),
                                       ref_m, rtol=rt, atol=1e-13,
                                       err_msg=f"{name} multipoles mismatch")
            np.testing.assert_allclose(
                _read(obj, "multipoles_gg", f"{name}_jackknife_cov_{NUM_JK}"),
                ref_cov, rtol=1e-8, atol=1e-15,
                err_msg=f"{name} covariance mismatch")
        obj8 = IA_mock_TNG300_n8
        obj8.measure_xi_multipoles("mjkcp_mp", "gg", NUM_JK,
                                   temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(_read(obj8, "multipoles_gg", "mjkcp_mp"),
                                   ref_m, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(
            _read(obj8, "multipoles_gg", f"mjkcp_mp_jackknife_cov_{NUM_JK}"),
            ref_cov, rtol=1e-8, atol=1e-15)

    def test_rp_cut_is_forwarded_and_reduces_pairs(self, IA_mock_TNG300_n1):
        """Regression: rp_cut was accepted by measure_xi_multipoles but never
        forwarded to the backends (silently ignored)."""
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("rpcut_no", "gg", 0, temp_file_path=False)
        obj.measure_xi_multipoles("rpcut_yes", "gg", 0, temp_file_path=False,
                                  rp_cut=1.0)
        dd_no = _read(obj, "multipoles/xi_gg", "rpcut_no_DD")
        dd_yes = _read(obj, "multipoles/xi_gg", "rpcut_yes_DD")
        assert np.sum(dd_yes) < np.sum(dd_no)


# ---------------------------------------------------------------------------
# 2. Ellipticity definition
# ---------------------------------------------------------------------------

class TestEllipticityDefinitionM:

    def test_gg_multipoles_same_for_both_defs(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("md_dist", "gg", 0, temp_file_path=False,
                                  ellipticity="distortion")
        obj.measure_xi_multipoles("md_ell",  "gg", 0, temp_file_path=False,
                                  ellipticity="ellipticity")
        np.testing.assert_array_equal(
            _read(obj, "multipoles_gg", "md_dist"),
            _read(obj, "multipoles_gg", "md_ell"))

    def test_gp_multipoles_differ_by_definition(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mgp_dist", "g+", 0, temp_file_path=False,
                                  ellipticity="distortion")
        obj.measure_xi_multipoles("mgp_ell",  "g+", 0, temp_file_path=False,
                                  ellipticity="ellipticity")
        assert not np.allclose(
            _read(obj, "multipoles_g_plus", "mgp_dist"),
            _read(obj, "multipoles_g_plus", "mgp_ell")), \
            "g+ multipoles should differ between ellipticity definitions"

    def test_invalid_ellipticity_raises(self, IA_mock_TNG300_n1):
        with pytest.raises((KeyError, ValueError)):
            IA_mock_TNG300_n1.measure_xi_multipoles(
                "bad_ell", "both", 0, temp_file_path=False,
                ellipticity="wrong")


# ---------------------------------------------------------------------------
# 3. num_jk input validation
# ---------------------------------------------------------------------------

class TestNumJKValidationM:

    def test_non_cube_raises(self, IA_mock_TNG300_n1, tmp_path):
        with pytest.raises(ValueError):
            IA_mock_TNG300_n1.measure_xi_multipoles(
                "bad_jk", "both", 7,
                temp_file_path=str(tmp_path) + "/")

    def test_zero_jk_succeeds(self, IA_mock_TNG300_n1):
        IA_mock_TNG300_n1.measure_xi_multipoles(
            "zero_jk", "both", 0, temp_file_path=False)

    def test_jk_without_temp_path_raises(self, IA_mock_TNG300_n1):
        with pytest.raises(ValueError):
            IA_mock_TNG300_n1.measure_xi_multipoles(
                "no_path", "both", NUM_JK, temp_file_path=None)


# ---------------------------------------------------------------------------
# 4. Backends — no JK
# ---------------------------------------------------------------------------

class TestBackendsNoJKM:

    def test_brute_vs_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        tp  = str(tmp_path) + "/"
        obj.measure_xi_multipoles("mb_brute", "both", 0,
                                  temp_file_path=False)
        obj.measure_xi_multipoles("mb_tree",  "both", 0,
                                  temp_file_path=tp)

        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus", "mb_brute"),
            _read(obj, "multipoles_g_plus", "mb_tree"))
        np.testing.assert_allclose(
            _read(obj, "multipoles_gg", "mb_brute"),
            _read(obj, "multipoles_gg", "mb_tree"))
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "mb_brute_r"),
            _read(obj, "multipoles_g_plus", "mb_tree_r"))

    def test_tree_vs_multiproc(self, IA_mock_TNG300_n1, IA_mock_TNG300_n8,
                                tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_multipoles(
            "mmp_tree",  "both", 0, temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_multipoles(
            "mmp_multi", "both", 0, temp_file_path=tp, chunk_size=50)

        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "multipoles_g_plus", "mmp_tree"),
            _read(IA_mock_TNG300_n8, "multipoles_g_plus", "mmp_multi"))
        np.testing.assert_array_equal(
            _read(IA_mock_TNG300_n1, "multipoles_g_plus", "mmp_tree_r"),
            _read(IA_mock_TNG300_n8, "multipoles_g_plus", "mmp_multi_r"))


# ---------------------------------------------------------------------------
# 5. Backends — with JK (realisation-level checks)
# ---------------------------------------------------------------------------

class TestBackendsWithJKM:

    def _run_both(self, obj, tmp_path):
        tp = str(tmp_path) + "/"
        obj.measure_xi_multipoles("mjk_tree",  "both", NUM_JK,
                                  temp_file_path=tp)
        obj.measure_xi_multipoles("mjk_brute", "both", NUM_JK,
                                  temp_file_path=False)

    def test_dd_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_both(obj, tmp_path)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_gg/mjk_tree_jk{NUM_JK}",
                      f"mjk_tree_{i}_DD"),
                _read(obj, f"multipoles/xi_gg/mjk_brute_jk{NUM_JK}",
                      f"mjk_brute_{i}_DD"),
                rtol=1e-5, err_msg=f"DD mismatch at realisation {i}")

    def test_splusd_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_both(obj, tmp_path)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_g_plus/mjk_tree_jk{NUM_JK}",
                      f"mjk_tree_{i}_SplusD"),
                _read(obj, f"multipoles/xi_g_plus/mjk_brute_jk{NUM_JK}",
                      f"mjk_brute_{i}_SplusD"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"SplusD mismatch at realisation {i}")

    def test_rr_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_both(obj, tmp_path)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_g_plus/mjk_tree_jk{NUM_JK}",
                      f"mjk_tree_{i}_RR"),
                _read(obj, f"multipoles/xi_g_plus/mjk_brute_jk{NUM_JK}",
                      f"mjk_brute_{i}_RR"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"RR mismatch at realisation {i}")

    def test_xi_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_both(obj, tmp_path)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_g_plus/mjk_tree_jk{NUM_JK}",
                      f"mjk_tree_{i}"),
                _read(obj, f"multipoles/xi_g_plus/mjk_brute_jk{NUM_JK}",
                      f"mjk_brute_{i}"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"xi_g+ mismatch at realisation {i}")

    def test_multipoles_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_both(obj, tmp_path)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles_g_plus/mjk_tree_jk{NUM_JK}",
                      f"mjk_tree_{i}"),
                _read(obj, f"multipoles_g_plus/mjk_brute_jk{NUM_JK}",
                      f"mjk_brute_{i}"),
                rtol=1e-5)

    def test_multiproc_matches_tree_jk(self, IA_mock_TNG300_n1,
                                        IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_multipoles(
            "mjk_tree", "both", NUM_JK, temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_multipoles(
            "mjk_mp", "both", NUM_JK, temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "multipoles_g_plus", "mjk_tree"),
            _read(IA_mock_TNG300_n8, "multipoles_g_plus", "mjk_mp"),
            rtol=1e-5)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "multipoles_g_plus",
                  f"mjk_tree_jackknife_cov_{NUM_JK}"),
            _read(IA_mock_TNG300_n8, "multipoles_g_plus",
                  f"mjk_mp_jackknife_cov_{NUM_JK}"),
            rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Output shape / r-bin consistency
# ---------------------------------------------------------------------------

class TestOutputShapeM:

    def test_r_length_matches_num_bins_r(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mshp", "both", 0, temp_file_path=False)
        r = _read(obj, "multipoles_g_plus", "mshp_r")
        m = _read(obj, "multipoles_g_plus", "mshp")
        assert len(r) == obj.num_bins_r
        assert len(m) == obj.num_bins_r

    def test_r_bins_sorted_ascending(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mshp2", "both", 0, temp_file_path=False)
        r = _read(obj, "multipoles_g_plus", "mshp2_r")
        assert np.all(np.diff(r) > 0)

    def test_r_within_separation_limits(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mr_lim", "g+", 0, temp_file_path=False)
        r = _read(obj, "multipoles_g_plus", "mr_lim_r")
        assert r[0]  >= obj.r_bins[0]
        assert r[-1] <= obj.r_bins[-1]

    def test_r_consistent_across_corr_types(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mr_gp", "g+", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mr_gg", "gg", 0, temp_file_path=False)
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "mr_gp_r"),
            _read(obj, "multipoles_gg",     "mr_gg_r"))


# ---------------------------------------------------------------------------
# 7. Covariance properties
# ---------------------------------------------------------------------------

class TestCovariancePropertiesM:

    def _run_jk(self, obj, name, tmp_path):
        obj.measure_xi_multipoles(name, "both", NUM_JK,
                                  temp_file_path=str(tmp_path) + "/")

    def test_cov_gp_is_square(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "mcov_sq", tmp_path)
        cov = _read(obj, "multipoles_g_plus",
                    f"mcov_sq_jackknife_cov_{NUM_JK}")
        assert cov.ndim == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == obj.num_bins_r

    def test_cov_gp_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "mcov_sym", tmp_path)
        cov = _read(obj, "multipoles_g_plus",
                    f"mcov_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_cov_gp_diagonal_non_negative(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "mcov_diag", tmp_path)
        cov = _read(obj, "multipoles_g_plus",
                    f"mcov_diag_jackknife_cov_{NUM_JK}")
        assert np.all(np.diag(cov) >= 0)

    def test_cov_gg_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "mcov_gg_sym", tmp_path)
        cov = _read(obj, "multipoles_gg",
                    f"mcov_gg_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Masks
# ---------------------------------------------------------------------------

class TestMasksM:

    def test_mask_reduces_pair_count(self, IA_mock_TNG300_n1, box_mass):
        obj   = IA_mock_TNG300_n1
        masks = {k: box_mass > np.median(box_mass) for k in obj.data}
        obj.measure_xi_multipoles("mno_mask",   "gg", 0, temp_file_path=False)
        obj.measure_xi_multipoles("mwith_mask", "gg", 0, temp_file_path=False,
                                  masks=masks)
        dd_all  = _read(obj, "multipoles/xi_gg", "mno_mask_DD")
        dd_mask = _read(obj, "multipoles/xi_gg", "mwith_mask_DD")
        assert np.sum(dd_mask) < np.sum(dd_all)

    def test_masked_gp_nonzero(self, IA_mock_TNG300_n1, box_mass):
        obj   = IA_mock_TNG300_n1
        masks = {k: box_mass > np.median(box_mass) for k in obj.data}
        obj.measure_xi_multipoles("mmask_gp", "g+", 0, temp_file_path=False,
                                  masks=masks)
        assert np.any(_read(obj, "multipoles_g_plus", "mmask_gp") != 0)

    def test_shape_sample_only_mask(self, IA_mock_TNG300_n1, box_mass):
        """Mask on shape sample only — should run without error."""
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles(
            "mmask_ss_only", "g+", 0, temp_file_path=False,
            masks={"Position_shape_sample": box_mass > np.median(box_mass)})


# ---------------------------------------------------------------------------
# 9. Weights — scaling laws
# ---------------------------------------------------------------------------

class TestWeightsM:

    def _run_w(self, obj, name, w_val, tmp_path):
        N = len(obj.data["Position"])
        obj.data["weight"]              = np.full(N, w_val)
        obj.data["weight_shape_sample"] = np.full(N, w_val)
        obj.measure_xi_multipoles(name, "both", NUM_JK,
                                  temp_file_path=str(tmp_path) + "/")

    def test_weight_scaling_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "mwt_ones", 1.0, tmp_path)
        self._run_w(obj, "mwt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus", "mwt_ones"),
            4 * _read(obj, "multipoles_g_plus", "mwt_half"))

    def test_weight_scaling_dd(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "mwt_ones", 1.0, tmp_path)
        self._run_w(obj, "mwt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_gg", "mwt_ones_DD"),
            4 * _read(obj, "multipoles/xi_gg", "mwt_half_DD"))

    def test_weight_scaling_cov_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "mwt_ones", 1.0, tmp_path)
        self._run_w(obj, "mwt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus",
                  f"mwt_ones_jackknife_cov_{NUM_JK}"),
            16 * _read(obj, "multipoles_g_plus",
                       f"mwt_half_jackknife_cov_{NUM_JK}"))

    def test_weight_scaling_cov_gg(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "mwt_ones", 1.0, tmp_path)
        self._run_w(obj, "mwt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles_gg", f"mwt_ones_jackknife_cov_{NUM_JK}"),
            16 * _read(obj, "multipoles_gg",
                       f"mwt_half_jackknife_cov_{NUM_JK}"))

    def test_r_unchanged_by_weights(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "mwt_ones", 1.0, tmp_path)
        self._run_w(obj, "mwt_half", 0.5, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "mwt_ones_r"),
            _read(obj, "multipoles_g_plus", "mwt_half_r"))


# ---------------------------------------------------------------------------
# 10. Jackknife region assignment (same fixture as xi_w)
# ---------------------------------------------------------------------------

class TestJKRegionsM:

    def test_region_indices_consistent(self, jk_regions):
        jk_pos, jk_shape = jk_regions._get_jackknife_region_indices(None, 2)
        np.testing.assert_array_equal(jk_pos, jk_shape)

    def test_all_objects_assigned(self, jk_regions):
        jk_pos, _ = jk_regions._get_jackknife_region_indices(None, 2)
        assert len(jk_pos) == 4
        assert all(0 <= idx < 8 for idx in jk_pos)

    def test_expected_patch_values(self, jk_regions):
        jk_pos, _ = jk_regions._get_jackknife_region_indices(None, 2)
        assert jk_pos[0] == 0    # [1,1,1] → patch 0
        assert jk_pos[1] == 5    # [2,1,2] → patch 5
        assert jk_pos[2] == 7    # [2.5,2.5,1.51] → patch 7
        assert jk_pos[3] == 3    # [1,2,2] → patch 3


# ---------------------------------------------------------------------------
# 11. _combine_jackknife_information
# ---------------------------------------------------------------------------

class TestCombineJackknifeM:

    def test_combine_reproduces_covariance(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mjk_comb", "both", NUM_JK,
                                  temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "mjk_comb", f"mjk_comb_jk{NUM_JK}",
            ["multipoles_g_plus", "multipoles_gg"],
            NUM_JK, return_output=True)

        for cov, group in zip(covs, ["multipoles_g_plus", "multipoles_gg"]):
            stored = _read(obj, group, f"mjk_comb_jackknife_cov_{NUM_JK}")
            np.testing.assert_allclose(cov, stored, rtol=1e-5)

    def test_combined_cov_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj  = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mjk_comb", "both", NUM_JK,
                                  temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "mjk_comb", f"mjk_comb_jk{NUM_JK}",
            ["multipoles_g_plus", "multipoles_gg"],
            NUM_JK, return_output=True)
        for cov in covs:
            np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_combined_cov_diagonal_non_negative(self, IA_mock_TNG300_n1,
                                                 tmp_path):
        obj  = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mjk_comb", "both", NUM_JK,
                                  temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "mjk_comb", f"mjk_comb_jk{NUM_JK}",
            ["multipoles_g_plus", "multipoles_gg"],
            NUM_JK, return_output=True)
        for cov in covs:
            assert np.all(np.diag(cov) >= 0)


# ---------------------------------------------------------------------------
# 12. Intermediate xi outputs (full-sample and per-realisation)
# ---------------------------------------------------------------------------

class TestIntermediateOutputsM:
    """
    Verifies every dataset written to the multipoles/xi_g_plus/,
    multipoles/xi_g_cross/, and multipoles/xi_gg/ groups that the
    existing sections do not already cover.

    Key difference from TestIntermediateOutputsW: bin axes are r and mu_r
    (not rp and pi), so the grid datasets are named _r and _mu_r.

      Full-sample (no-JK):
        xi_g_plus/  : SplusD, RR_g_plus, r, mu_r, ScrossD
        xi_g_cross/ : xi, RR_g_cross, r, mu_r   (entire group)
        xi_gg/      : RR_gg, r, mu_r
      Per-realisation JK:
        _{i}_r and _{i}_mu_r match full-sample r/mu_r for every realisation
    """

    def _run_no_jk(self, obj):
        obj.measure_xi_multipoles("int_nojk", "both", 0,
                                   temp_file_path=False)

    def _run_jk(self, obj, tmp_path):
        obj.measure_xi_multipoles("int_jk", "both", NUM_JK,
                                   temp_file_path=str(tmp_path) + "/")

    # ------------------------------------------------------------------ full-sample SplusD / RR / r / mu_r

    def test_splusd_shape_and_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        splusd = _read(obj, "multipoles/xi_g_plus", "int_nojk_SplusD")
        assert splusd.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(np.isfinite(splusd))

    def test_rr_g_plus_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "multipoles/xi_g_plus", "int_nojk_RR_g_plus")
        assert rr.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(rr > 0)

    def test_r_grid_in_xi_group_sorted(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        r = _read(obj, "multipoles/xi_g_plus", "int_nojk_r")
        assert np.all(np.diff(r) > 0)

    def test_mu_r_grid_in_xi_group_sorted(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        mu_r = _read(obj, "multipoles/xi_g_plus", "int_nojk_mu_r")
        assert np.all(np.diff(mu_r) > 0)

    def test_r_grid_matches_r_bins(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        r   = _read(obj, "multipoles/xi_g_plus", "int_nojk_r")
        assert r[0]  >= obj.r_bins[0]
        assert r[-1] <= obj.r_bins[-1]

    def test_mu_r_grid_within_bounds(self, IA_mock_TNG300_n1):
        obj  = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        mu_r = _read(obj, "multipoles/xi_g_plus", "int_nojk_mu_r")
        assert mu_r[0]  >= obj.mu_r_bins[0]
        assert mu_r[-1] <= obj.mu_r_bins[-1]

    def test_scrossd_shape(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        sc = _read(obj, "multipoles/xi_g_cross", "int_nojk_ScrossD")
        assert sc.shape == (obj.num_bins_r, obj.num_bins_pi)

    # ------------------------------------------------------------------ xi_g_cross group

    def test_xi_g_cross_exists(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        xi_cross = _read(obj, "multipoles/xi_g_cross", "int_nojk")
        assert xi_cross.shape == (obj.num_bins_r, obj.num_bins_pi)

    def test_xi_g_cross_rr_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "multipoles/xi_g_cross", "int_nojk_RR_g_cross")
        assert np.all(rr > 0)

    def test_xi_g_cross_r_matches_xi_g_plus_r(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "int_nojk_r"),
            _read(obj, "multipoles/xi_g_plus",  "int_nojk_r"))

    def test_xi_g_cross_mu_r_matches_xi_g_plus_mu_r(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "int_nojk_mu_r"),
            _read(obj, "multipoles/xi_g_plus",  "int_nojk_mu_r"))

    # ------------------------------------------------------------------ xi_gg full-sample RR / r / mu_r

    def test_rr_gg_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "multipoles/xi_gg", "int_nojk_RR_gg")
        assert rr.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(rr > 0)

    def test_rr_gg_consistent_with_formula(self, IA_mock_TNG300_n1):
        """Sum of RR_gg over all bins should be close to the analytical
        prediction from get_random_pairs_r_mur over the full (r, mu_r) range.
        The per-bin analytic values telescope exactly to this total, so the
        agreement is at machine precision."""
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr  = _read(obj, "multipoles/xi_gg", "int_nojk_RR_gg")
        L3  = obj.boxsize ** 3
        N   = obj.Num_position
        total_rr_analytical = obj.get_random_pairs_r_mur(
            obj.r_bins[-1], obj.r_bins[0],
            obj.mu_r_bins[-1], obj.mu_r_bins[0],
            L3, "cross", N, N, obj.num_overlap)
        assert np.sum(rr) == pytest.approx(total_rr_analytical, rel=1e-10)

    def test_xi_gg_r_mu_r_grids_match_xi_g_plus(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg",     "int_nojk_r"),
            _read(obj, "multipoles/xi_g_plus", "int_nojk_r"))
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg",     "int_nojk_mu_r"),
            _read(obj, "multipoles/xi_g_plus", "int_nojk_mu_r"))

    # sigmasq was dropped from the multipoles jackknife path (kernel consolidation step 5b):
    # only the brute backend ever populated it, so the estimator is not trustworthy and is no
    # longer written. The former test_sigmasq_* checks were removed with it.

    # ------------------------------------------------------------------ per-realisation r / mu_r grids

    def test_per_jk_r_matches_fullsample_r(self, IA_mock_TNG300_n1, tmp_path):
        obj   = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        r_ref = _read(obj, "multipoles/xi_g_plus", "int_jk_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_g_plus/int_jk_jk{NUM_JK}",
                      f"int_jk_{i}_r"),
                r_ref,
                err_msg=f"r mismatch at JK realisation {i}")

    def test_per_jk_mu_r_matches_fullsample_mu_r(self, IA_mock_TNG300_n1,
                                                   tmp_path):
        obj      = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        mu_r_ref = _read(obj, "multipoles/xi_g_plus", "int_jk_mu_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_g_plus/int_jk_jk{NUM_JK}",
                      f"int_jk_{i}_mu_r"),
                mu_r_ref,
                err_msg=f"mu_r mismatch at JK realisation {i}")

    def test_per_jk_gg_r_matches_fullsample(self, IA_mock_TNG300_n1, tmp_path):
        obj   = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        r_ref = _read(obj, "multipoles/xi_gg", "int_jk_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_gg/int_jk_jk{NUM_JK}",
                      f"int_jk_{i}_r"),
                r_ref,
                err_msg=f"gg r mismatch at JK realisation {i}")


# ---------------------------------------------------------------------------
# 13. Intermediate pair-count equality: brute == tree
# ---------------------------------------------------------------------------

class TestIntermediatePairCountEqualityM:
    """
    Every intermediate pair-count array written to multipoles/xi_g_plus/,
    multipoles/xi_g_cross/, and multipoles/xi_gg/ must be identical between
    the brute and tree backends (given identical inputs).  Covers:
      xi_g_plus/ : SplusD, ScrossD, RR_g_plus, xi
      xi_g_cross/: xi, RR_g_cross
      xi_gg/     : xi, DD, RR_gg
    Bin-grid equality (r, mu_r) is already verified in section 12.
    Multiprocessing comparisons use allclose (machine-precision tolerance).
    """

    def _run_brute_and_tree(self, obj, tmp_path):
        obj.measure_xi_multipoles("mpce_brute", "both", 0,
                                   temp_file_path=False)
        obj.measure_xi_multipoles("mpce_tree",  "both", 0,
                                   temp_file_path=str(tmp_path) + "/")

    # xi_g_plus
    def test_splusd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_plus", "mpce_brute_SplusD"),
            _read(obj, "multipoles/xi_g_plus", "mpce_tree_SplusD"), rtol=1e-12, atol=1e-12)

    def test_scrossd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_cross", "mpce_brute_ScrossD"),
            _read(obj, "multipoles/xi_g_cross", "mpce_tree_ScrossD"), rtol=1e-12, atol=1e-15)

    def test_rr_g_plus_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_plus", "mpce_brute_RR_g_plus"),
            _read(obj, "multipoles/xi_g_plus", "mpce_tree_RR_g_plus"))

    def test_xi_g_plus_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_plus", "mpce_brute"),
            _read(obj, "multipoles/xi_g_plus", "mpce_tree"))

    # xi_g_cross
    def test_xi_g_cross_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_cross", "mpce_brute"),
            _read(obj, "multipoles/xi_g_cross", "mpce_tree"))

    def test_rr_g_cross_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "mpce_brute_RR_g_cross"),
            _read(obj, "multipoles/xi_g_cross", "mpce_tree_RR_g_cross"))

    # xi_gg
    def test_xi_gg_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_gg", "mpce_brute"),
            _read(obj, "multipoles/xi_gg", "mpce_tree"))

    def test_dd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg", "mpce_brute_DD"),
            _read(obj, "multipoles/xi_gg", "mpce_tree_DD"))

    def test_rr_gg_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg", "mpce_brute_RR_gg"),
            _read(obj, "multipoles/xi_gg", "mpce_tree_RR_gg"))

    # multiprocessing — allclose only
    def test_splusd_tree_allclose_multiproc(self, IA_mock_TNG300_n1,
                                             IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_multipoles("mpce_mp1", "both", 0,
                                                 temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_multipoles("mpce_mp8", "both", 0,
                                                 temp_file_path=tp,
                                                 chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "multipoles/xi_g_plus",
                  "mpce_mp1_SplusD"),
            _read(IA_mock_TNG300_n8, "multipoles/xi_g_plus",
                  "mpce_mp8_SplusD"),
            rtol=1e-10)

    def test_dd_tree_allclose_multiproc(self, IA_mock_TNG300_n1,
                                         IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_multipoles("mpce_mp1", "both", 0,
                                                 temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_multipoles("mpce_mp8", "both", 0,
                                                 temp_file_path=tp,
                                                 chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "multipoles/xi_gg", "mpce_mp1_DD"),
            _read(IA_mock_TNG300_n8, "multipoles/xi_gg", "mpce_mp8_DD"),
            rtol=1e-10)
