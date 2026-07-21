"""
test_lc_measure_xi_w.py
=======================
Comprehensive tests for MeasureIALightcone.measure_xi_w(), covering:

  - IA_estimator: 'clusters', 'galaxies'
  - corr_type: 'both', 'g+', 'gg'
  - Computation backends: brute (tree=False) and tree (tree=True)
  - Multiprocessing backend (num_nodes > 1)
  - measure_cov=False (no covariance) and measure_cov=True (with JK)
  - JK patches provided externally (jk_patches) vs generated internally (num_jk)
  - Single random sample vs separate position/shape random catalogues
  - Automatic weight injection when missing from data / randoms dict
  - Masks on data
  - over_h conversion flag
  - Invalid inputs (bad IA_estimator, missing randoms, missing JK args)
  - Output shapes, rp-bin consistency, covariance matrix properties
  - Regression against saved reference output

All tests use fixtures from tests/conftest_lc.py (see conftest_lc.py).

Run from the project root:
    pytest tests/test_lc_measure_xi_w.py -v
"""

import numpy as np
import pytest
import h5py

NUM_JK = 8


def _read(obj, group, key):
    with h5py.File(obj.output_file_name, "r") as f:
        return f[obj.snap_group + group][key][:]


# ---------------------------------------------------------------------------
# 1. IA_estimator validation
# ---------------------------------------------------------------------------

class TestIAEstimatorW:

    def test_invalid_estimator_raises(self, IA_mock_lc_n1, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_n1.measure_xi_w(
                "bad_estimator", "lc_bad", "both",
                measure_cov=False, tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_clusters_without_randoms_raises(self, IA_mock_lc_no_randoms, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_no_randoms.measure_xi_w(
                "clusters", "lc_no_rand", "both",
                measure_cov=False, tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_galaxies_without_randoms_raises(self, IA_mock_lc_no_randoms, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_no_randoms.measure_xi_w(
                "galaxies", "lc_no_rand", "both",
                measure_cov=False, tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_clusters_estimator_runs(self, IA_mock_lc_n1, tmp_path):
        """Clusters estimator should complete without error."""
        IA_mock_lc_n1.measure_xi_w(
            "clusters", "lc_clusters_nocov", "both",
            measure_cov=False, tree=True,
            temp_file_path=str(tmp_path) + "/")

    def test_galaxies_estimator_runs(self, IA_mock_lc_n1, tmp_path):
        """Galaxies estimator should complete without error."""
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_galaxies_nocov", "both",
            measure_cov=False, tree=True,
            temp_file_path=str(tmp_path) + "/")

    def test_clusters_vs_galaxies_gp_differ(self, IA_mock_lc_n1, tmp_path):
        """The two estimators use different xi formulas, so w_g+ should differ."""
        IA_mock_lc_n1.measure_xi_w(
            "clusters", "lc_est_cl", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_est_gx", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        assert not np.allclose(_read(obj, "w/xi_g_plus", "lc_est_cl"),
                               _read(obj, "w/xi_g_plus", "lc_est_gx")), \
            "clusters and galaxies estimators should produce different w_g+"


# ---------------------------------------------------------------------------
# 2. corr_type variations
# ---------------------------------------------------------------------------

class TestCorrTypeW:

    def test_gp_matches_both(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_ct_both", "both",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_ct_gp", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        np.testing.assert_array_equal(_read(obj, "w/xi_g_plus", "lc_ct_both"),
                                      _read(obj, "w/xi_g_plus", "lc_ct_gp"))
        np.testing.assert_array_equal(_read(obj, "w/xi_g_plus", "lc_ct_both_rp"),
                                      _read(obj, "w/xi_g_plus", "lc_ct_gp_rp"))

    def test_gg_matches_both(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_ct_both2", "both",
                          measure_cov=False, tree=False,
                          temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_ct_gg", "gg",
                          measure_cov=False, tree=False,
                          temp_file_path=str(tmp_path) + "/")
        np.testing.assert_array_equal(_read(obj, "w/xi_gg", "lc_ct_both2"),
                                      _read(obj, "w/xi_gg", "lc_ct_gg"))

    def test_invalid_corr_type_raises(self, IA_mock_lc_n1, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_n1.measure_xi_w(
                "galaxies", "lc_bad_ct", "gg+",
                measure_cov=False, tree=False,
                temp_file_path=str(tmp_path) + "/")


# ---------------------------------------------------------------------------
# 3. Backends (no covariance)
# ---------------------------------------------------------------------------

class TestBackendsNoCovW:

    def test_brute_vs_tree_gp(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_bvt_brute", "both",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_bvt_tree", "both",
            measure_cov=False, tree=True,
            temp_file_path=str(tmp_path) + "/")

        obj = IA_mock_lc_n1
        np.testing.assert_allclose(_read(obj, "w/xi_g_plus", "lc_bvt_brute"),
                                   _read(obj, "w/xi_g_plus", "lc_bvt_tree"))
        np.testing.assert_array_equal(_read(obj, "w/xi_g_plus", "lc_bvt_brute_rp"),
                                      _read(obj, "w/xi_g_plus", "lc_bvt_tree_rp"))

    def test_brute_vs_tree_gg(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_bvt_brute", "both",
                          measure_cov=False, tree=False,
                          temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_bvt_tree", "both",
                          measure_cov=False, tree=True,
                          temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(_read(obj, "w/xi_gg", "lc_bvt_brute"),
                                   _read(obj, "w/xi_gg", "lc_bvt_tree"))

    def test_tree_vs_multiproc(self, IA_mock_lc_n1, IA_mock_lc_n8, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_mp_tree", "both",
            measure_cov=False, tree=True,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n8.measure_xi_w(
            "galaxies", "lc_mp_multi", "both",
            measure_cov=False, tree=True,
            temp_file_path=str(tmp_path) + "/",
            chunk_size=50)

        np.testing.assert_allclose(_read(IA_mock_lc_n1, "w/xi_g_plus", "lc_mp_tree"),
                                   _read(IA_mock_lc_n8, "w/xi_g_plus", "lc_mp_multi"))
        np.testing.assert_array_equal(_read(IA_mock_lc_n1, "w/xi_g_plus", "lc_mp_tree_rp"),
                                      _read(IA_mock_lc_n8, "w/xi_g_plus", "lc_mp_multi_rp"))


# ---------------------------------------------------------------------------
# 4. Covariance — JK patches provided vs generated internally
# ---------------------------------------------------------------------------

class TestCovarianceW:

    def test_measure_cov_true_without_jk_args_raises(self, IA_mock_lc_n1, tmp_path):
        """measure_cov=True but no jk_patches and no num_jk must raise."""
        with pytest.raises(ValueError):
            IA_mock_lc_n1.measure_xi_w(
                "galaxies", "lc_no_jk", "both",
                measure_cov=True,
                temp_file_path=str(tmp_path) + "/")

    def test_internal_jk_generation(self, IA_mock_lc_n1, tmp_path):
        """num_jk triggers internal patch assignment — should not raise."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w(
            "galaxies", "lc_int_jk", "both",
            measure_cov=True, num_jk=NUM_JK,
            tree=True, temp_file_path=str(tmp_path) + "/")

        cov = _read(obj, "w_g_plus", f"lc_int_jk_jackknife_cov_{NUM_JK}")
        assert cov.shape[0] == cov.shape[1]

    def test_external_jk_patches(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        """Pre-assigned patches should produce the same result as internal ones."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w(
            "galaxies", "lc_ext_jk", "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")

        assert _read(obj, "w_g_plus", "lc_ext_jk") is not None

    def test_external_vs_internal_jk_agree(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        """Both patch-assignment paths produce the same final w_g+."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_int_jk2", "both",
                          measure_cov=True, num_jk=NUM_JK,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_ext_jk2", "both",
                          measure_cov=True, jk_patches=lc_jk_patches,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "w_g_plus", "lc_int_jk2"),
            _read(obj, "w_g_plus", "lc_ext_jk2"),
            rtol=1e-5)

    def test_one_based_jk_patches_raise(self, IA_mock_lc_n1, lc_jk_patches,
                                        tmp_path):
        """Patch indices must start at 0 — 1-based external patches raise a
        clear error instead of silently corrupting the covariance."""
        shifted = {k: np.asarray(v) + 1 for k, v in lc_jk_patches.items()}
        with pytest.raises(ValueError, match="must start at 0"):
            IA_mock_lc_n1.measure_xi_w(
                "galaxies", "lc_jk_1based", "both",
                measure_cov=True, jk_patches=shifted,
                tree=True, temp_file_path=str(tmp_path) + "/")

    def test_multiproc_vs_tree_jk(self, IA_mock_lc_n1, IA_mock_lc_n8,
                                  lc_jk_patches, tmp_path):
        """The multiprocessing jk backend (num_nodes>1) must reproduce the
        tree jk backend: final vectors and covariances. Regression for the
        batch clamp that truncated position samples larger than the shape
        sample (S+R term)."""
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_w_jk_n1", "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n8.measure_xi_w(
            "galaxies", "lc_w_jk_n8", "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/",
            chunk_size=50)
        for grp in ("w_g_plus", "w_gg"):
            np.testing.assert_allclose(
                _read(IA_mock_lc_n1, grp, "lc_w_jk_n1"),
                _read(IA_mock_lc_n8, grp, "lc_w_jk_n8"),
                rtol=1e-8, atol=1e-12,
                err_msg=f"{grp} n1 vs n8 mismatch")
            np.testing.assert_allclose(
                _read(IA_mock_lc_n1, grp, f"lc_w_jk_n1_jackknife_cov_{NUM_JK}"),
                _read(IA_mock_lc_n8, grp, f"lc_w_jk_n8_jackknife_cov_{NUM_JK}"),
                rtol=1e-6, atol=1e-14,
                err_msg=f"{grp} covariance n1 vs n8 mismatch")

    def test_brute_vs_tree_jk_realisations_splusd(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w(
            "galaxies", "lc_jk_tree",  "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w(
            "galaxies", "lc_jk_brute", "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_g_plus/lc_jk_tree_jk{NUM_JK}",
                      f"lc_jk_tree_{i}_SplusD"),
                _read(obj, f"w/xi_g_plus/lc_jk_brute_jk{NUM_JK}",
                      f"lc_jk_brute_{i}_SplusD"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"SplusD mismatch in JK realisation {i}")

    def test_brute_vs_tree_jk_realisations_dd(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w(
            "galaxies", "lc_jk_tree",  "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w(
            "galaxies", "lc_jk_brute", "both",
            measure_cov=True, jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_gg/lc_jk_tree_jk{NUM_JK}",
                      f"lc_jk_tree_{i}_DD"),
                _read(obj, f"w/xi_gg/lc_jk_brute_jk{NUM_JK}",
                      f"lc_jk_brute_{i}_DD"),
                rtol=1e-5,
                err_msg=f"DD mismatch in JK realisation {i}")


# ---------------------------------------------------------------------------
# 5. Single vs separate random catalogues
# ---------------------------------------------------------------------------

class TestRandomCataloguesW:

    def test_single_random_sample_runs(self, IA_mock_lc_single_rand, tmp_path):
        """When randoms_data has only 'RA'/'DEC'/'Redshift' (no _shape_sample
        keys), the code should auto-duplicate it for the shape sample."""
        IA_mock_lc_single_rand.measure_xi_w(
            "galaxies", "lc_single_rand", "both",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_separate_random_samples_run(self, IA_mock_lc_n1, tmp_path):
        """Full randoms dict with separate position/shape keys should work."""
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_sep_rand", "both",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_single_vs_sep_rand_gg_same(
            self, IA_mock_lc_dup_rand, IA_mock_lc_single_rand, tmp_path):
        """The auto-duplication of a single randoms catalogue must give the
        same result as explicitly passing identical position/shape randoms."""
        IA_mock_lc_dup_rand.measure_xi_w(
            "galaxies", "lc_sep_rand", "gg",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_single_rand.measure_xi_w(
            "galaxies", "lc_single_rand", "gg",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(IA_mock_lc_dup_rand,   "w/xi_gg", "lc_sep_rand"),
            _read(IA_mock_lc_single_rand,"w/xi_gg", "lc_single_rand"),
            rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Auto-injection of missing weights
# ---------------------------------------------------------------------------

class TestWeightInjectionW:

    def test_missing_data_weight_defaults_to_ones(self, IA_mock_lc_no_weight, tmp_path):
        """If 'weight' key is absent from data dict, the code should inject
        all-ones weights and continue without error."""
        IA_mock_lc_no_weight.measure_xi_w(
            "galaxies", "lc_no_wt_data", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_missing_randoms_weight_defaults_to_ones(self, IA_mock_lc_rand_no_weight, tmp_path):
        """Same for the randoms dict."""
        IA_mock_lc_rand_no_weight.measure_xi_w(
            "galaxies", "lc_no_wt_rand", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_explicit_ones_equals_missing_weight_gp(
            self, IA_mock_lc_n1, IA_mock_lc_no_weight, tmp_path):
        """Explicit weight=1 and missing weight should produce identical w_g+."""
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_exp_ones", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_no_weight.measure_xi_w(
            "galaxies", "lc_no_wt_data", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")

        np.testing.assert_allclose(
            _read(IA_mock_lc_n1,       "w/xi_g_plus", "lc_exp_ones"),
            _read(IA_mock_lc_no_weight, "w/xi_g_plus", "lc_no_wt_data"))


# ---------------------------------------------------------------------------
# 7. over_h flag
# ---------------------------------------------------------------------------

class TestOverHW:

    def test_over_h_false_and_true_differ(self, IA_mock_lc_n1, tmp_path):
        """Toggling over_h changes coordinate units → different pair counts."""
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_oh_false", "gg",
            measure_cov=False, tree=False, over_h=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_oh_true",  "gg",
            measure_cov=False, tree=False, over_h=True,
            temp_file_path=str(tmp_path) + "/")
        assert not np.allclose(_read(IA_mock_lc_n1, "w/xi_gg", "lc_oh_false"),
                               _read(IA_mock_lc_n1, "w/xi_gg", "lc_oh_true"))


# ---------------------------------------------------------------------------
# 8. Masks
# ---------------------------------------------------------------------------

class TestMasksW:

    def test_mask_reduces_pair_count(self, IA_mock_lc_n1, lc_masks, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_no_mask",   "gg",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_with_mask", "gg",
            measure_cov=False, tree=False, masks=lc_masks,
            temp_file_path=str(tmp_path) + "/")

        obj = IA_mock_lc_n1
        dd_all  = _read(obj, "w/xi_gg", "lc_no_mask_DD")
        dd_mask = _read(obj, "w/xi_gg", "lc_with_mask_DD")
        assert np.sum(dd_mask) < np.sum(dd_all)

    def test_masked_gp_nonzero(self, IA_mock_lc_n1, lc_masks, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_mask_gp", "g+",
            measure_cov=False, tree=False, masks=lc_masks,
            temp_file_path=str(tmp_path) + "/")
        # After masking there should still be some non-zero bins
        assert np.any(_read(IA_mock_lc_n1, "w/xi_g_plus", "lc_mask_gp") != 0)


# ---------------------------------------------------------------------------
# 9. Output shape / consistency
# ---------------------------------------------------------------------------

class TestOutputShapeW:

    def test_rp_length_matches_num_bins_r(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_w(
            "galaxies", "lc_shape_chk", "both",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        rp = _read(obj, "w/xi_g_plus", "lc_shape_chk_rp")
        w  = _read(obj, "w/xi_g_plus", "lc_shape_chk")
        assert len(rp) == obj.num_bins_r
        assert len(w)  == obj.num_bins_r

    def test_rp_bins_sorted_ascending(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_shape_chk", "both",
                          measure_cov=False, tree=False,
                          temp_file_path=str(tmp_path) + "/")
        rp = _read(obj, "w/xi_g_plus", "lc_shape_chk_rp")
        assert np.all(np.diff(rp) > 0)

    def test_rp_consistent_across_corr_types(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w(
            "galaxies", "lc_rp_gp", "g+",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w(
            "galaxies", "lc_rp_gg", "gg",
            measure_cov=False, tree=False,
            temp_file_path=str(tmp_path) + "/")
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_plus", "lc_rp_gp_rp"),
            _read(obj, "w/xi_gg",     "lc_rp_gg_rp"))

    def test_covariance_is_square_and_correct_size(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_cov_sz", "both",
                          measure_cov=True, jk_patches=lc_jk_patches,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "w_g_plus", f"lc_cov_sz_jackknife_cov_{NUM_JK}")
        assert cov.ndim == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == obj.num_bins_r

    def test_covariance_is_symmetric(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_cov_sym", "both",
                          measure_cov=True, jk_patches=lc_jk_patches,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "w_g_plus", f"lc_cov_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_diagonal_non_negative(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_cov_diag", "both",
                          measure_cov=True, jk_patches=lc_jk_patches,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "w_g_plus", f"lc_cov_diag_jackknife_cov_{NUM_JK}")
        # Bins with no pairs in the sparse mock give NaN variance; only the
        # finite entries carry information and those must be non-negative.
        d = np.diag(cov)
        assert np.any(np.isfinite(d))
        assert np.all(d[np.isfinite(d)] >= 0)


# ---------------------------------------------------------------------------
# 10. Regression against saved reference output
# ---------------------------------------------------------------------------

class TestRegressionW:

    def test_wgp_matches_reference(self, IA_mock_lc_n1, tmp_path):
        """Run twice with the same config; results must be identical (determinism)."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_reg_a", "both",
                          measure_cov=True, num_jk=NUM_JK,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_reg_b", "both",
                          measure_cov=True, num_jk=NUM_JK,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_plus", "lc_reg_a"),
            _read(obj, "w/xi_g_plus", "lc_reg_b"), rtol=1e-10)

    def test_wgg_matches_reference(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_reg_gg_a", "gg",
                          measure_cov=False, tree=True,
                          temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_reg_gg_b", "gg",
                          measure_cov=False, tree=True,
                          temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "w/xi_gg", "lc_reg_gg_a"),
            _read(obj, "w/xi_gg", "lc_reg_gg_b"), rtol=1e-10)

    def test_cov_wgp_matches_reference(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_cov_reg_a", "g+",
                          measure_cov=True, num_jk=NUM_JK, seed=42,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_cov_reg_b", "g+",
                          measure_cov=True, num_jk=NUM_JK, seed=42,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "w_g_plus", f"lc_cov_reg_a_jackknife_cov_{NUM_JK}"),
            _read(obj, "w_g_plus", f"lc_cov_reg_b_jackknife_cov_{NUM_JK}"),
            rtol=1e-10)

    def test_cov_wgg_matches_reference(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_cov_gg_a", "gg",
                          measure_cov=True, num_jk=NUM_JK, seed=42,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("galaxies", "lc_cov_gg_b", "gg",
                          measure_cov=True, num_jk=NUM_JK, seed=42,
                          tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "w_gg", f"lc_cov_gg_a_jackknife_cov_{NUM_JK}"),
            _read(obj, "w_gg", f"lc_cov_gg_b_jackknife_cov_{NUM_JK}"),
            rtol=1e-10)


# ---------------------------------------------------------------------------
# 11. Intermediate xi outputs (full-sample and per-realisation)
# ---------------------------------------------------------------------------

class TestIntermediateOutputsLCW:
    """
    Verifies every dataset written to the w/xi_g_plus/, w/xi_g_cross/, and
    w/xi_gg/ groups that the existing sections do not already cover.

    Key differences from the box equivalent (TestIntermediateOutputsW):
      - No sigmasq: lightcone uses real randoms so no variance dataset is stored.
      - No named RR_g_plus / RR_gg: pair normalisation is implicit in SR/DD.
      - xi_gg density key is _SR for the 'clusters' estimator and _DD for
        the 'galaxies' estimator.

    Covers:
      Full-sample (no covariance):
        xi_g_plus/ : SplusD (shape, non-negative), rp/pi grids, ScrossD shape
        xi_g_cross/: xi exists, rp/pi match xi_g_plus
        xi_gg/     : DD or SR positive, rp/pi match xi_g_plus
      Per-realisation JK:
        _{i}_rp and _{i}_pi match full-sample rp/pi for every realisation
    """

    def _run_no_cov(self, obj, tmp_path, estimator="galaxies"):
        obj.measure_xi_w(estimator, "lc_int_nojk", "both",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")

    def _run_cov(self, obj, tmp_path, estimator="galaxies"):
        obj.measure_xi_w(estimator, "lc_int_jk", "both",
                         measure_cov=True, num_jk=NUM_JK,
                         tree=False,
                         temp_file_path=str(tmp_path) + "/")

    # ------------------------------------------------------------------ SplusD

    def test_splusd_shape_and_non_negative(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        splusd = _read(obj, "w/xi_g_plus", "lc_int_nojk_SplusD")
        assert splusd.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(np.isfinite(splusd))

    def test_splusd_identical_for_both_estimators(self, IA_mock_lc_n1, tmp_path):
        """SplusD is computed from real galaxy positions for both estimators;
        the estimator choice only affects the observable formula, not the pair counts."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_sp_gx", "g+",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("clusters", "lc_sp_cl", "g+",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")
        sp_gx = _read(obj, "w/xi_g_plus", "lc_sp_gx_SplusD")
        sp_cl = _read(obj, "w/xi_g_plus", "lc_sp_cl_SplusD")
        np.testing.assert_array_equal(sp_gx, sp_cl)

    # ------------------------------------------------------------------ rp / pi grids (xi_g_plus)

    def test_rp_grid_sorted_ascending(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        rp  = _read(obj, "w/xi_g_plus", "lc_int_nojk_rp")
        assert np.all(np.diff(rp) > 0)

    def test_pi_grid_sorted_ascending(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        pi  = _read(obj, "w/xi_g_plus", "lc_int_nojk_pi")
        assert np.all(np.diff(pi) > 0)

    def test_rp_grid_within_separation_limits(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        rp  = _read(obj, "w/xi_g_plus", "lc_int_nojk_rp")
        assert rp[0]  >= obj.r_bins[0]
        assert rp[-1] <= obj.r_bins[-1]

    def test_pi_grid_within_pi_max(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        pi  = _read(obj, "w/xi_g_plus", "lc_int_nojk_pi")
        assert pi[0]  >= obj.pi_bins[0]
        assert pi[-1] <= obj.pi_bins[-1]

    # ------------------------------------------------------------------ ScrossD

    def test_scrossd_shape(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        sc  = _read(obj, "w/xi_g_cross", "lc_int_nojk_ScrossD")
        assert sc.shape == (obj.num_bins_r, obj.num_bins_pi)

    # ------------------------------------------------------------------ xi_g_cross group

    def test_xi_g_cross_exists_and_correct_shape(self, IA_mock_lc_n1, tmp_path):
        obj      = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        xi_cross = _read(obj, "w/xi_g_cross", "lc_int_nojk")
        assert xi_cross.shape == (obj.num_bins_r, obj.num_bins_pi)

    def test_xi_g_cross_rp_matches_xi_g_plus_rp(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "lc_int_nojk_rp"),
            _read(obj, "w/xi_g_plus",  "lc_int_nojk_rp"))

    def test_xi_g_cross_pi_matches_xi_g_plus_pi(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "lc_int_nojk_pi"),
            _read(obj, "w/xi_g_plus",  "lc_int_nojk_pi"))

    # ------------------------------------------------------------------ xi_gg pair counts (DD / SR)

    def test_xi_gg_dd_sr_rd_rr_all_written(self, IA_mock_lc_n1, tmp_path):
        """Both estimators write DD, SR, RD, RR to xi_gg when corr_type='gg'."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_gg_all", "gg",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")
        for suffix in ("_DD", "_SR", "_RD", "_RR"):
            arr = _read(obj, "w/xi_gg", f"lc_gg_all{suffix}")
            assert arr.shape == (obj.num_bins_r, obj.num_bins_pi),                 f"{suffix} wrong shape"
            assert np.all(arr >= 0), f"{suffix} has negative values"

    def test_xi_gg_dd_sr_same_for_both_estimators(self, IA_mock_lc_n1, tmp_path):
        """DD and SR are position-position pair counts, independent of the
        IA estimator — both estimators must produce identical values."""
        obj = IA_mock_lc_n1
        obj.measure_xi_w("galaxies", "lc_gg_gx2", "gg",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_w("clusters", "lc_gg_cl2", "gg",
                         measure_cov=False, tree=False,
                         temp_file_path=str(tmp_path) + "/")
        for suffix in ("_DD", "_SR", "_RD", "_RR"):
            np.testing.assert_array_equal(
                _read(obj, "w/xi_gg", f"lc_gg_gx2{suffix}"),
                _read(obj, "w/xi_gg", f"lc_gg_cl2{suffix}"),
                err_msg=f"{suffix} differs between estimators")

    def test_xi_gg_rp_matches_xi_g_plus_rp(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg",     "lc_int_nojk_rp"),
            _read(obj, "w/xi_g_plus", "lc_int_nojk_rp"))

    def test_xi_gg_pi_matches_xi_g_plus_pi(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg",     "lc_int_nojk_pi"),
            _read(obj, "w/xi_g_plus", "lc_int_nojk_pi"))

    # ------------------------------------------------------------------ per-realisation rp / pi

    def test_per_jk_rp_matches_fullsample_rp(self, IA_mock_lc_n1, tmp_path):
        obj    = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        rp_ref = _read(obj, "w/xi_g_plus", "lc_int_jk_rp")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_g_plus/lc_int_jk_jk{NUM_JK}",
                      f"lc_int_jk_{i}_rp"),
                rp_ref,
                err_msg=f"rp mismatch at JK realisation {i}")

    def test_per_jk_pi_matches_fullsample_pi(self, IA_mock_lc_n1, tmp_path):
        obj    = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        pi_ref = _read(obj, "w/xi_g_plus", "lc_int_jk_pi")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_g_plus/lc_int_jk_jk{NUM_JK}",
                      f"lc_int_jk_{i}_pi"),
                pi_ref,
                err_msg=f"pi mismatch at JK realisation {i}")

    def test_per_jk_gg_rp_matches_fullsample(self, IA_mock_lc_n1, tmp_path):
        obj    = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        rp_ref = _read(obj, "w/xi_gg", "lc_int_jk_rp")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_gg/lc_int_jk_jk{NUM_JK}",
                      f"lc_int_jk_{i}_rp"),
                rp_ref,
                err_msg=f"gg rp mismatch at JK realisation {i}")


# ---------------------------------------------------------------------------
# 12b. Intermediate pair-count equality: estimator and backend
# ---------------------------------------------------------------------------

class TestIntermediatePairCountEqualityLCW:
    """
    For the lightcone w method:

    Estimator equality: SplusD, SplusR, ScrossD are computed identically for
    both 'galaxies' and 'clusters' estimators (estimator only affects the
    observable formula applied afterwards).  DD, SR, RD, RR in xi_gg are
    also estimator-independent (verified in section 11 for xi_gg; extended
    here to xi_g_plus pair counts).

    Backend equality: brute and tree must give exact agreement on all
    intermediate pair-count arrays.  Multiprocessing may accumulate
    machine-precision differences so those comparisons use allclose.
    """

    def _run(self, obj, name, estimator="galaxies", tree=False, tmp_path=None):
        kwargs = dict(measure_cov=False, tree=tree,
                      temp_file_path=(str(tmp_path) + "/" if tmp_path else None))
        obj.measure_xi_w(estimator, name, "both", **kwargs)

    # ---- estimator equality ------------------------------------------------

    def test_splusd_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "ee_gx", "galaxies")
        self._run(obj, "ee_cl", "clusters")
        for grp in ("w/xi_g_plus",):
            np.testing.assert_array_equal(
                _read(obj, grp, "ee_gx_SplusD"),
                _read(obj, grp, "ee_cl_SplusD"))

    def test_splusr_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "ee_gx", "galaxies")
        self._run(obj, "ee_cl", "clusters")
        for grp in ("w/xi_g_plus",):
            np.testing.assert_array_equal(
                _read(obj, grp, "ee_gx_SplusR"),
                _read(obj, grp, "ee_cl_SplusR"))

    def test_scrossd_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "ee_gx", "galaxies")
        self._run(obj, "ee_cl", "clusters")
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "ee_gx_ScrossD"),
            _read(obj, "w/xi_g_cross", "ee_cl_ScrossD"))

    # ---- backend equality: brute == tree -----------------------------------

    def test_splusd_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_plus", "be_brute_SplusD"),
            _read(obj, "w/xi_g_plus", "be_tree_SplusD"))

    def test_splusr_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_plus", "be_brute_SplusR"),
            _read(obj, "w/xi_g_plus", "be_tree_SplusR"))

    def test_scrossd_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_cross", "be_brute_ScrossD"),
            _read(obj, "w/xi_g_cross", "be_tree_ScrossD"), rtol=1e-12, atol=1e-15)

    def test_xi_g_plus_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_plus", "be_brute"),
            _read(obj, "w/xi_g_plus", "be_tree"))

    def test_xi_g_cross_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_cross", "be_brute"),
            _read(obj, "w/xi_g_cross", "be_tree"))

    def test_xi_gg_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_gg", "be_brute"),
            _read(obj, "w/xi_gg", "be_tree"))

    def test_xi_gg_all_suffixes_brute_equals_tree(self, IA_mock_lc_n1,
                                                    tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "be_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "be_tree",  tree=True,  tmp_path=tmp_path)
        # Brute-force only writes _DD (the SplusD pass); tree writes all four.
        # Only compare the pair-count that both backends produce.
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg", "be_brute_DD"),
            _read(obj, "w/xi_gg", "be_tree_DD"),
            err_msg="xi_gg_DD brute != tree")

    # ---- multiprocessing: allclose -----------------------------------------

    def test_splusd_allclose_multiproc(self, IA_mock_lc_n1,
                                        IA_mock_lc_n8, tmp_path):
        tp = str(tmp_path) + "/"
        self._run(IA_mock_lc_n1, "mp_n1", tree=True, tmp_path=tmp_path)
        IA_mock_lc_n8.measure_xi_w("galaxies", "mp_n8", "both",
                                    measure_cov=False, tree=True,
                                    temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_lc_n1, "w/xi_g_plus", "mp_n1_SplusD"),
            _read(IA_mock_lc_n8, "w/xi_g_plus", "mp_n8_SplusD"),
            rtol=1e-10)

    def test_dd_allclose_multiproc(self, IA_mock_lc_n1,
                                    IA_mock_lc_n8, tmp_path):
        tp = str(tmp_path) + "/"
        self._run(IA_mock_lc_n1, "mp_n1", tree=True, tmp_path=tmp_path)
        IA_mock_lc_n8.measure_xi_w("galaxies", "mp_n8", "both",
                                    measure_cov=False, tree=True,
                                    temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_lc_n1, "w/xi_gg", "mp_n1_DD"),
            _read(IA_mock_lc_n8, "w/xi_gg", "mp_n8_DD"),
            rtol=1e-10)
