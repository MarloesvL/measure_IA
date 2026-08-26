"""
test_lc_measure_xi_multipoles.py
=================================
Comprehensive tests for MeasureIALightcone.measure_xi_multipoles(), covering:

  - IA_estimator: 'clusters', 'galaxies'
  - corr_type: 'both', 'g+', 'gg'
  - Computation backends: brute (tree=False) and tree (tree=True)
  - with and without jackknife covariance (num_jk / jk_patches)
  - JK patches provided externally (jk_patches) vs generated internally (num_jk)
  - Single random sample vs separate position/shape random catalogues
  - Auto weight injection
  - over_h flag
  - Masks on data
  - Invalid inputs
  - Output shapes (r-bins), covariance matrix properties
  - Realisation-level checks (SplusD, RR, xi per JK drop)
  - Determinism of repeated runs

All tests use fixtures from tests/conftest.py.

Run from the project root:
    pytest tests/test_lc_measure_xi_multipoles.py -v
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

class TestIAEstimatorM:

    def test_invalid_estimator_raises(self, IA_mock_lc_n1, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_n1.measure_xi_multipoles(
                "bad_estimator", "lc_m_bad", "both",
                tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_clusters_without_randoms_raises(self, IA_mock_lc_no_randoms, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_no_randoms.measure_xi_multipoles(
                "clusters", "lc_m_no_rand", "both",
                tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_galaxies_without_randoms_raises(self, IA_mock_lc_no_randoms, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_no_randoms.measure_xi_multipoles(
                "galaxies", "lc_m_no_rand", "both",
                tree=False,
                temp_file_path=str(tmp_path) + "/")

    def test_clusters_estimator_runs(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "clusters", "lc_m_clusters_nc", "both",
            tree=True,
            temp_file_path=str(tmp_path) + "/")

    def test_galaxies_estimator_runs(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_galaxies_nc", "both",
            tree=True,
            temp_file_path=str(tmp_path) + "/")

    def test_clusters_vs_galaxies_gp_differ(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "clusters", "lc_m_est_cl", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_est_gx", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        assert not np.allclose(_read(obj, "multipoles_g_plus", "lc_m_est_cl"),
                               _read(obj, "multipoles_g_plus", "lc_m_est_gx"))


# ---------------------------------------------------------------------------
# 2. corr_type variations
# ---------------------------------------------------------------------------

class TestCorrTypeM:

    def test_gp_matches_both(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_ct_both", "both",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_ct_gp", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        np.testing.assert_array_equal(_read(obj, "multipoles_g_plus", "lc_m_ct_both"),
                                      _read(obj, "multipoles_g_plus", "lc_m_ct_gp"))
        np.testing.assert_array_equal(_read(obj, "multipoles_g_plus", "lc_m_ct_both_r"),
                                      _read(obj, "multipoles_g_plus", "lc_m_ct_gp_r"))

    def test_gg_matches_both(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_ct_both2", "both",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_ct_gg", "gg",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        np.testing.assert_array_equal(_read(obj, "multipoles_gg", "lc_m_ct_both2"),
                                      _read(obj, "multipoles_gg", "lc_m_ct_gg"))

    def test_invalid_corr_type_raises(self, IA_mock_lc_n1, tmp_path):
        with pytest.raises(KeyError):
            IA_mock_lc_n1.measure_xi_multipoles(
                "galaxies", "lc_m_bad_ct", "gg+",
                tree=False,
                temp_file_path=str(tmp_path) + "/")


# ---------------------------------------------------------------------------
# 3. Backends (no covariance)
# ---------------------------------------------------------------------------

class TestBackendsNoCovM:

    def test_brute_vs_tree_gp(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_bvt_brute", "both",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_bvt_tree", "both",
            tree=True,
            temp_file_path=str(tmp_path) + "/")

        obj = IA_mock_lc_n1
        np.testing.assert_allclose(_read(obj, "multipoles_g_plus", "lc_m_bvt_brute"),
                                   _read(obj, "multipoles_g_plus", "lc_m_bvt_tree"))
        np.testing.assert_array_equal(_read(obj, "multipoles_g_plus", "lc_m_bvt_brute_r"),
                                      _read(obj, "multipoles_g_plus", "lc_m_bvt_tree_r"))

    def test_brute_vs_tree_gg(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_bvt_brute", "both",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_bvt_tree", "both",
                                   tree=True,
                                   temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(_read(obj, "multipoles_gg", "lc_m_bvt_brute"),
                                   _read(obj, "multipoles_gg", "lc_m_bvt_tree"))

    def test_tree_vs_multiproc(self, IA_mock_lc_n1, IA_mock_lc_n8, tmp_path):
        """The full-sample multiprocessing backend (num_nodes>1, no jackknife)
        must reproduce the tree backend.

        New in 0.5.0: before then the lightcone had no full-sample mp
        implementation at all, so num_nodes was accepted and silently ignored on
        this path and a test like this passed without exercising anything. The
        tolerance is allclose rather than exact because the mp path sums partial
        grids from separate position chunks, so the float summation order
        differs (measured: ~1e-14 relative).
        """
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_mp_tree", "both",
            tree=True,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n8.measure_xi_multipoles(
            "galaxies", "lc_m_mp_multi", "both",
            tree=True,
            temp_file_path=str(tmp_path) + "/",
            chunk_size=50)

        for grp in ("multipoles_g_plus", "multipoles_gg"):
            np.testing.assert_allclose(
                _read(IA_mock_lc_n1, grp, "lc_m_mp_tree"),
                _read(IA_mock_lc_n8, grp, "lc_m_mp_multi"),
                rtol=1e-8, atol=1e-12,
                err_msg=f"{grp}: full-sample mp does not reproduce tree")
        np.testing.assert_array_equal(
            _read(IA_mock_lc_n1, "multipoles_g_plus", "lc_m_mp_tree_r"),
            _read(IA_mock_lc_n8, "multipoles_g_plus", "lc_m_mp_multi_r"))

    def test_multiproc_honours_num_nodes(self, IA_mock_lc_n1, IA_mock_lc_n8, tmp_path):
        """num_nodes>1 must actually reach the multiprocessing backend.

        Guards the specific defect this path was added for: the dispatch used to
        fall through to the tree backend, so num_nodes was accepted and then had
        no effect (benchmarks/FINDINGS.md F4).

        The assertion is on a side effect rather than a spy, because under the
        'spawn' start method ``self`` is pickled to every worker -- so attaching
        a test double to the instance makes the run fail to pickle. Only the
        multiprocessing backends set ``shm_infos`` (the shared-memory block
        descriptors); the tree and brute backends never touch it.
        """
        for obj, name in ((IA_mock_lc_n1, "lc_m_disp_n1"), (IA_mock_lc_n8, "lc_m_disp_n8")):
            obj.measure_xi_multipoles("galaxies", name, "g+", tree=True,
                                      temp_file_path=str(tmp_path) + "/")

        assert not hasattr(IA_mock_lc_n1, "shm_infos"), \
            "num_nodes=1 unexpectedly used the multiprocessing backend"
        assert getattr(IA_mock_lc_n8, "shm_infos", None), \
            "num_nodes>1 did not reach the multiprocessing backend"

    def test_r_bins_same_across_backends(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_bvt_brute", "both",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_bvt_tree", "both",
                                   tree=True,
                                   temp_file_path=str(tmp_path) + "/")
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "lc_m_bvt_brute_r"),
            _read(obj, "multipoles_g_plus", "lc_m_bvt_tree_r"))


# ---------------------------------------------------------------------------
# 4. Covariance — JK patches provided vs generated internally
# ---------------------------------------------------------------------------

class TestCovarianceM:

    def test_no_jk_args_means_no_covariance(self, IA_mock_lc_n1, tmp_path):
        """Neither jk_patches nor num_jk means no covariance, matching the
        box's num_jk=0 default. It used to raise, when covariance was opt-out."""
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_no_jk", "both",
                                  temp_file_path=str(tmp_path) + "/")
        with h5py.File(obj.output_file_name, "r") as f:
            assert "lc_m_no_jk" in f["multipoles_g_plus"]
            assert not any(k.startswith("lc_m_no_jk_jackknife")
                           for k in f["multipoles_g_plus"])

    def test_internal_jk_generation(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_int_jk", "both",
            num_jk=NUM_JK,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        cov = _read(obj, "multipoles_g_plus", f"lc_m_int_jk_jackknife_cov_{NUM_JK}")
        assert cov.shape[0] == cov.shape[1]

    def test_external_jk_patches(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_ext_jk", "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        assert _read(obj, "multipoles_g_plus", "lc_m_ext_jk") is not None

    def test_internal_vs_external_agree(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_int_jk2", "both",
                                   num_jk=NUM_JK,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_ext_jk2", "both",
                                   jk_patches=lc_jk_patches,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus", "lc_m_int_jk2"),
            _read(obj, "multipoles_g_plus", "lc_m_ext_jk2"),
            rtol=1e-5)

    def test_multiproc_vs_tree_jk(self, IA_mock_lc_n1, IA_mock_lc_n8,
                                  lc_jk_patches, tmp_path):
        """The multiprocessing jk backend (num_nodes>1) must reproduce the
        tree jk backend: final vectors and covariances."""
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_jk_n1", "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n8.measure_xi_multipoles(
            "galaxies", "lc_m_jk_n8", "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/",
            chunk_size=50)
        for grp in ("multipoles_g_plus", "multipoles_gg"):
            np.testing.assert_allclose(
                _read(IA_mock_lc_n1, grp, "lc_m_jk_n1"),
                _read(IA_mock_lc_n8, grp, "lc_m_jk_n8"),
                rtol=1e-8, atol=1e-12,
                err_msg=f"{grp} n1 vs n8 mismatch")
            np.testing.assert_allclose(
                _read(IA_mock_lc_n1, grp, f"lc_m_jk_n1_jackknife_cov_{NUM_JK}"),
                _read(IA_mock_lc_n8, grp, f"lc_m_jk_n8_jackknife_cov_{NUM_JK}"),
                rtol=1e-6, atol=1e-14,
                err_msg=f"{grp} covariance n1 vs n8 mismatch")

    def test_brute_vs_tree_jk_realisations_splusd(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_tree",  "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_brute", "both",
            jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_g_plus/lc_m_jk_tree_jk{NUM_JK}",
                      f"lc_m_jk_tree_{i}_SplusD"),
                _read(obj, f"multipoles/xi_g_plus/lc_m_jk_brute_jk{NUM_JK}",
                      f"lc_m_jk_brute_{i}_SplusD"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"SplusD mismatch in JK realisation {i}")

    def test_brute_vs_tree_jk_realisations_rr(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_tree",  "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_brute", "both",
            jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_gg/lc_m_jk_tree_jk{NUM_JK}",
                      f"lc_m_jk_tree_{i}_RR"),
                _read(obj, f"multipoles/xi_gg/lc_m_jk_brute_jk{NUM_JK}",
                      f"lc_m_jk_brute_{i}_RR"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"RR mismatch in JK realisation {i}")

    def test_brute_vs_tree_jk_realisations_xi(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_tree",  "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_brute", "both",
            jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles/xi_g_plus/lc_m_jk_tree_jk{NUM_JK}",
                      f"lc_m_jk_tree_{i}"),
                _read(obj, f"multipoles/xi_g_plus/lc_m_jk_brute_jk{NUM_JK}",
                      f"lc_m_jk_brute_{i}"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"xi_g+ mismatch in JK realisation {i}")

    def test_brute_vs_tree_jk_realisations_multipoles(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_tree",  "both",
            jk_patches=lc_jk_patches,
            tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles(
            "galaxies", "lc_m_jk_brute", "both",
            jk_patches=lc_jk_patches,
            tree=False, temp_file_path=str(tmp_path) + "/")

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"multipoles_g_plus/lc_m_jk_tree_jk{NUM_JK}",
                      f"lc_m_jk_tree_{i}"),
                _read(obj, f"multipoles_g_plus/lc_m_jk_brute_jk{NUM_JK}",
                      f"lc_m_jk_brute_{i}"),
                rtol=1e-5)
            np.testing.assert_allclose(
                _read(obj, f"multipoles_gg/lc_m_jk_tree_jk{NUM_JK}",
                      f"lc_m_jk_tree_{i}"),
                _read(obj, f"multipoles_gg/lc_m_jk_brute_jk{NUM_JK}",
                      f"lc_m_jk_brute_{i}"),
                rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Single vs separate random catalogues
# ---------------------------------------------------------------------------

class TestRandomCataloguesM:

    def test_single_random_sample_runs(self, IA_mock_lc_single_rand, tmp_path):
        IA_mock_lc_single_rand.measure_xi_multipoles(
            "galaxies", "lc_m_single_rand", "both",
            tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_separate_random_samples_run(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_sep_rand", "both",
            tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_single_vs_sep_rand_gg_same(
            self, IA_mock_lc_dup_rand, IA_mock_lc_single_rand, tmp_path):
        """The auto-duplication of a single randoms catalogue must give the
        same result as explicitly passing identical position/shape randoms."""
        IA_mock_lc_dup_rand.measure_xi_multipoles(
            "galaxies", "lc_m_sep_rand", "gg",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_single_rand.measure_xi_multipoles(
            "galaxies", "lc_m_single_rand", "gg",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(IA_mock_lc_dup_rand,    "multipoles_gg", "lc_m_sep_rand"),
            _read(IA_mock_lc_single_rand, "multipoles_gg", "lc_m_single_rand"),
            rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Auto-injection of missing weights
# ---------------------------------------------------------------------------

class TestWeightInjectionM:

    def test_missing_data_weight_defaults_to_ones(self, IA_mock_lc_no_weight, tmp_path):
        IA_mock_lc_no_weight.measure_xi_multipoles(
            "galaxies", "lc_m_no_wt_data", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_missing_randoms_weight_defaults_to_ones(
            self, IA_mock_lc_rand_no_weight, tmp_path):
        IA_mock_lc_rand_no_weight.measure_xi_multipoles(
            "galaxies", "lc_m_no_wt_rand", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")

    def test_explicit_ones_equals_missing_weight(
            self, IA_mock_lc_n1, IA_mock_lc_no_weight, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_exp_ones", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_no_weight.measure_xi_multipoles(
            "galaxies", "lc_m_no_wt_data", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(IA_mock_lc_n1,       "multipoles_g_plus", "lc_m_exp_ones"),
            _read(IA_mock_lc_no_weight, "multipoles_g_plus", "lc_m_no_wt_data"))


# ---------------------------------------------------------------------------
# 7. over_h flag
# ---------------------------------------------------------------------------

class TestOverHM:

    def test_over_h_changes_result(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_oh_false", "gg",
            tree=False, over_h=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_oh_true",  "gg",
            tree=False, over_h=True,
            temp_file_path=str(tmp_path) + "/")
        assert not np.allclose(
            _read(IA_mock_lc_n1, "multipoles_gg", "lc_m_oh_false"),
            _read(IA_mock_lc_n1, "multipoles_gg", "lc_m_oh_true"))


# ---------------------------------------------------------------------------
# 8. Masks
# ---------------------------------------------------------------------------

class TestMasksM:

    def test_mask_reduces_pair_count(self, IA_mock_lc_n1, lc_masks, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_no_mask",   "gg",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_with_mask", "gg",
            tree=False, masks=lc_masks,
            temp_file_path=str(tmp_path) + "/")

        obj = IA_mock_lc_n1
        dd_all  = _read(obj, "multipoles/xi_gg", "lc_m_no_mask_DD")
        dd_mask = _read(obj, "multipoles/xi_gg", "lc_m_with_mask_DD")
        assert np.sum(dd_mask) < np.sum(dd_all)


# ---------------------------------------------------------------------------
# 9. Output shape / consistency
# ---------------------------------------------------------------------------

class TestOutputShapeM:

    def test_r_length_matches_num_bins_r(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_shape_chk", "both",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        r = _read(obj, "multipoles_g_plus", "lc_m_shape_chk_r")
        m = _read(obj, "multipoles_g_plus", "lc_m_shape_chk")
        assert len(r) == obj.num_bins_r
        assert len(m) == obj.num_bins_r

    def test_r_bins_sorted_ascending(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_shape_chk", "both",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        r = _read(obj, "multipoles_g_plus", "lc_m_shape_chk_r")
        assert np.all(np.diff(r) > 0)

    def test_r_bins_within_separation_limits(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_shape_chk", "both",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        r = _read(obj, "multipoles_g_plus", "lc_m_shape_chk_r")
        assert r[0]  >= obj.r_bins[0]
        assert r[-1] <= obj.r_bins[-1]

    def test_r_bins_consistent_across_corr_types(self, IA_mock_lc_n1, tmp_path):
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_r_gp", "g+",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        IA_mock_lc_n1.measure_xi_multipoles(
            "galaxies", "lc_m_r_gg", "gg",
            tree=False,
            temp_file_path=str(tmp_path) + "/")
        obj = IA_mock_lc_n1
        np.testing.assert_array_equal(
            _read(obj, "multipoles_g_plus", "lc_m_r_gp_r"),
            _read(obj, "multipoles_gg",     "lc_m_r_gg_r"))

    def test_covariance_is_square_and_correct_size(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_sz", "both",
                                   jk_patches=lc_jk_patches,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "multipoles_g_plus", f"lc_m_cov_sz_jackknife_cov_{NUM_JK}")
        assert cov.ndim == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == obj.num_bins_r

    def test_covariance_is_symmetric(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_sym", "both",
                                   jk_patches=lc_jk_patches,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "multipoles_g_plus", f"lc_m_cov_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_diagonal_non_negative(
            self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_diag", "both",
                                   jk_patches=lc_jk_patches,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "multipoles_g_plus", f"lc_m_cov_diag_jackknife_cov_{NUM_JK}")
        # Bins with no pairs in the sparse mock give NaN variance; only the
        # finite entries carry information and those must be non-negative.
        d = np.diag(cov)
        assert np.any(np.isfinite(d))
        assert np.all(d[np.isfinite(d)] >= 0)

    def test_gg_covariance_is_symmetric(self, IA_mock_lc_n1, lc_jk_patches, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_gg_sym", "both",
                                   jk_patches=lc_jk_patches,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        cov = _read(obj, "multipoles_gg", f"lc_m_cov_gg_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 10. Determinism: repeated runs of the same configuration agree
# ---------------------------------------------------------------------------

class TestDeterminismM:

    def test_gp_multipoles_is_deterministic(self, IA_mock_lc_n1, tmp_path):
        """Run twice; identical config must give identical result."""
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_reg_a", "both",
                                   num_jk=NUM_JK,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_reg_b", "both",
                                   num_jk=NUM_JK,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus", "lc_m_reg_a"),
            _read(obj, "multipoles_g_plus", "lc_m_reg_b"), rtol=1e-10)

    def test_gg_multipoles_is_deterministic(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_reg_gg_a", "gg",
                                   tree=True,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_reg_gg_b", "gg",
                                   tree=True,
                                   temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "multipoles_gg", "lc_m_reg_gg_a"),
            _read(obj, "multipoles_gg", "lc_m_reg_gg_b"), rtol=1e-10)

    def test_cov_gp_is_deterministic(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_reg_a", "g+",
                                   num_jk=NUM_JK, seed=42,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_reg_b", "g+",
                                   num_jk=NUM_JK, seed=42,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "multipoles_g_plus",
                  f"lc_m_cov_reg_a_jackknife_cov_{NUM_JK}"),
            _read(obj, "multipoles_g_plus",
                  f"lc_m_cov_reg_b_jackknife_cov_{NUM_JK}"),
            rtol=1e-10)

    def test_cov_gg_is_deterministic(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_gg_a", "gg",
                                   num_jk=NUM_JK, seed=42,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("galaxies", "lc_m_cov_gg_b", "gg",
                                   num_jk=NUM_JK, seed=42,
                                   tree=True, temp_file_path=str(tmp_path) + "/")
        np.testing.assert_allclose(
            _read(obj, "multipoles_gg",
                  f"lc_m_cov_gg_a_jackknife_cov_{NUM_JK}"),
            _read(obj, "multipoles_gg",
                  f"lc_m_cov_gg_b_jackknife_cov_{NUM_JK}"),
            rtol=1e-10)


# ---------------------------------------------------------------------------
# 11. Intermediate xi outputs (full-sample and per-realisation)
# ---------------------------------------------------------------------------

class TestIntermediateOutputsLCM:
    """
    Verifies every dataset written to the multipoles/xi_g_plus/,
    multipoles/xi_g_cross/, and multipoles/xi_gg/ groups that the
    existing sections do not already cover.

    Key differences from the box equivalent (TestIntermediateOutputsM):
      - No sigmasq: lightcone uses real randoms so no variance dataset is stored.
      - No named RR_g_plus / RR_gg stored analytically.
      - xi_gg density key is _SR for 'clusters' estimator, _DD for 'galaxies'.
      - Bin axes are r and mu_r (not rp and pi).

    Covers:
      Full-sample (no covariance):
        xi_g_plus/ : SplusD (shape, non-negative), r/mu_r grids, ScrossD shape
        xi_g_cross/: xi exists, r/mu_r match xi_g_plus
        xi_gg/     : DD or SR positive, r/mu_r match xi_g_plus
      Per-realisation JK:
        _{i}_r and _{i}_mu_r match full-sample r/mu_r for every realisation
    """

    def _run_no_cov(self, obj, estimator="galaxies", tmp_path=None):
        tp = str(tmp_path) + "/" if tmp_path else None
        obj.measure_xi_multipoles(estimator, "lc_m_int_nojk", "both",
                                   tree=False,
                                   temp_file_path=tp)

    def _run_cov(self, obj, tmp_path, estimator="galaxies"):
        obj.measure_xi_multipoles(estimator, "lc_m_int_jk", "both",
                                   num_jk=NUM_JK,
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")

    # ------------------------------------------------------------------ SplusD

    def test_splusd_shape_and_non_negative(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        splusd = _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_SplusD")
        assert splusd.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(np.isfinite(splusd))

    def test_splusd_identical_for_both_estimators(self, IA_mock_lc_n1, tmp_path):
        """SplusD is computed from real galaxy positions for both estimators;
        the estimator choice only affects the observable formula, not the pair counts."""
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_sp_gx", "g+",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("clusters", "lc_m_sp_cl", "g+",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        sp_gx = _read(obj, "multipoles/xi_g_plus", "lc_m_sp_gx_SplusD")
        sp_cl = _read(obj, "multipoles/xi_g_plus", "lc_m_sp_cl_SplusD")
        np.testing.assert_array_equal(sp_gx, sp_cl)

    # ------------------------------------------------------------------ r / mu_r grids (xi_g_plus)

    def test_r_grid_sorted_ascending(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        r   = _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_r")
        assert np.all(np.diff(r) > 0)

    def test_mu_r_grid_sorted_ascending(self, IA_mock_lc_n1):
        obj  = IA_mock_lc_n1
        self._run_no_cov(obj)
        mu_r = _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_mu_r")
        assert np.all(np.diff(mu_r) > 0)

    def test_r_grid_within_separation_limits(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        r   = _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_r")
        assert r[0]  >= obj.r_bins[0]
        assert r[-1] <= obj.r_bins[-1]

    def test_mu_r_grid_within_bounds(self, IA_mock_lc_n1):
        obj  = IA_mock_lc_n1
        self._run_no_cov(obj)
        mu_r = _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_mu_r")
        assert mu_r[0]  >= obj.mu_r_bins[0]
        assert mu_r[-1] <= obj.mu_r_bins[-1]

    # ------------------------------------------------------------------ ScrossD

    def test_scrossd_shape(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        sc  = _read(obj, "multipoles/xi_g_cross", "lc_m_int_nojk_ScrossD")
        assert sc.shape == (obj.num_bins_r, obj.num_bins_pi)

    # ------------------------------------------------------------------ xi_g_cross group

    def test_xi_g_cross_exists_and_correct_shape(self, IA_mock_lc_n1):
        obj      = IA_mock_lc_n1
        self._run_no_cov(obj)
        xi_cross = _read(obj, "multipoles/xi_g_cross", "lc_m_int_nojk")
        assert xi_cross.shape == (obj.num_bins_r, obj.num_bins_pi)

    def test_xi_g_cross_r_matches_xi_g_plus_r(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "lc_m_int_nojk_r"),
            _read(obj, "multipoles/xi_g_plus",  "lc_m_int_nojk_r"))

    def test_xi_g_cross_mu_r_matches_xi_g_plus_mu_r(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "lc_m_int_nojk_mu_r"),
            _read(obj, "multipoles/xi_g_plus",  "lc_m_int_nojk_mu_r"))

    # ------------------------------------------------------------------ xi_gg pair counts (DD / SR)

    def test_xi_gg_dd_sr_rd_rr_all_written(self, IA_mock_lc_n1, tmp_path):
        """Both estimators write DD, SR, RD, RR to xi_gg when corr_type='gg'."""
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_gg_all", "gg",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        for suffix in ("_DD", "_SR", "_RD", "_RR"):
            arr = _read(obj, "multipoles/xi_gg", f"lc_m_gg_all{suffix}")
            assert arr.shape == (obj.num_bins_r, obj.num_bins_pi),                 f"{suffix} wrong shape"
            assert np.all(arr >= 0), f"{suffix} has negative values"

    def test_xi_gg_dd_sr_same_for_both_estimators(self, IA_mock_lc_n1, tmp_path):
        """DD, SR, RD, RR are position-position pair counts, independent of the
        IA estimator — both estimators must produce identical values."""
        obj = IA_mock_lc_n1
        obj.measure_xi_multipoles("galaxies", "lc_m_gg_gx2", "gg",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        obj.measure_xi_multipoles("clusters", "lc_m_gg_cl2", "gg",
                                   tree=False,
                                   temp_file_path=str(tmp_path) + "/")
        for suffix in ("_DD", "_SR", "_RD", "_RR"):
            np.testing.assert_array_equal(
                _read(obj, "multipoles/xi_gg", f"lc_m_gg_gx2{suffix}"),
                _read(obj, "multipoles/xi_gg", f"lc_m_gg_cl2{suffix}"),
                err_msg=f"{suffix} differs between estimators")

    def test_xi_gg_r_matches_xi_g_plus_r(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg",     "lc_m_int_nojk_r"),
            _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_r"))

    def test_xi_gg_mu_r_matches_xi_g_plus_mu_r(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run_no_cov(obj)
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg",     "lc_m_int_nojk_mu_r"),
            _read(obj, "multipoles/xi_g_plus", "lc_m_int_nojk_mu_r"))

    # ------------------------------------------------------------------ per-realisation r / mu_r

    def test_per_jk_r_matches_fullsample_r(self, IA_mock_lc_n1, tmp_path):
        obj   = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        r_ref = _read(obj, "multipoles/xi_g_plus", "lc_m_int_jk_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_g_plus/lc_m_int_jk_jk{NUM_JK}",
                      f"lc_m_int_jk_{i}_r"),
                r_ref,
                err_msg=f"r mismatch at JK realisation {i}")

    def test_per_jk_mu_r_matches_fullsample_mu_r(self, IA_mock_lc_n1,
                                                   tmp_path):
        obj      = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        mu_r_ref = _read(obj, "multipoles/xi_g_plus", "lc_m_int_jk_mu_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_g_plus/lc_m_int_jk_jk{NUM_JK}",
                      f"lc_m_int_jk_{i}_mu_r"),
                mu_r_ref,
                err_msg=f"mu_r mismatch at JK realisation {i}")

    def test_per_jk_gg_r_matches_fullsample(self, IA_mock_lc_n1, tmp_path):
        obj   = IA_mock_lc_n1
        self._run_cov(obj, tmp_path)
        r_ref = _read(obj, "multipoles/xi_gg", "lc_m_int_jk_r")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"multipoles/xi_gg/lc_m_int_jk_jk{NUM_JK}",
                      f"lc_m_int_jk_{i}_r"),
                r_ref,
                err_msg=f"gg r mismatch at JK realisation {i}")


# ---------------------------------------------------------------------------
# 11b. Intermediate pair-count equality: estimator and backend
# ---------------------------------------------------------------------------

class TestIntermediatePairCountEqualityLCM:
    """
    For the lightcone multipoles method:

    Estimator equality: SplusD, SplusR, ScrossD are identical for both
    'galaxies' and 'clusters' estimators.  DD, SR, RD, RR in xi_gg are also
    estimator-independent.

    Backend equality: brute and tree give exact agreement on all intermediate
    pair-count arrays.  Multiprocessing comparisons use allclose.
    """

    def _run(self, obj, name, estimator="galaxies", tree=False, tmp_path=None):
        kwargs = dict(tree=tree,
                      temp_file_path=(str(tmp_path) + "/" if tmp_path else None))
        obj.measure_xi_multipoles(estimator, name, "both", **kwargs)

    # ---- estimator equality ------------------------------------------------

    def test_splusd_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "mee_gx", "galaxies")
        self._run(obj, "mee_cl", "clusters")
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_plus", "mee_gx_SplusD"),
            _read(obj, "multipoles/xi_g_plus", "mee_cl_SplusD"))

    def test_splusr_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "mee_gx", "galaxies")
        self._run(obj, "mee_cl", "clusters")
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_plus", "mee_gx_SplusR"),
            _read(obj, "multipoles/xi_g_plus", "mee_cl_SplusR"))

    def test_scrossd_estimator_equal(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        self._run(obj, "mee_gx", "galaxies")
        self._run(obj, "mee_cl", "clusters")
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_g_cross", "mee_gx_ScrossD"),
            _read(obj, "multipoles/xi_g_cross", "mee_cl_ScrossD"))

    # ---- backend equality: brute == tree -----------------------------------

    def test_splusd_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        # Float grids: brute and tree visit the sample in different orders since
        # the spatial chunk ordering (FINDINGS.md F5), so the sums differ in the
        # last few ulp. The pair *sets* are identical -- the integer DD/SR counts
        # still compare exactly -- and REFACTOR_PLAN.md section 4 specifies
        # allclose for brute-vs-tree anyway.
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_plus", "mbe_brute_SplusD"),
            _read(obj, "multipoles/xi_g_plus", "mbe_tree_SplusD"),
            rtol=1e-10, atol=1e-12)

    def test_splusr_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        # Float grids: brute and tree visit the sample in different orders since
        # the spatial chunk ordering (FINDINGS.md F5), so the sums differ in the
        # last few ulp. The pair *sets* are identical -- the integer DD/SR counts
        # still compare exactly -- and REFACTOR_PLAN.md section 4 specifies
        # allclose for brute-vs-tree anyway.
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_plus", "mbe_brute_SplusR"),
            _read(obj, "multipoles/xi_g_plus", "mbe_tree_SplusR"),
            rtol=1e-10, atol=1e-12)

    def test_scrossd_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_cross", "mbe_brute_ScrossD"),
            _read(obj, "multipoles/xi_g_cross", "mbe_tree_ScrossD"), rtol=1e-12, atol=1e-15)

    def test_xi_g_plus_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_plus", "mbe_brute"),
            _read(obj, "multipoles/xi_g_plus", "mbe_tree"))

    def test_xi_g_cross_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_g_cross", "mbe_brute"),
            _read(obj, "multipoles/xi_g_cross", "mbe_tree"))

    def test_xi_gg_brute_equals_tree(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        np.testing.assert_allclose(
            _read(obj, "multipoles/xi_gg", "mbe_brute"),
            _read(obj, "multipoles/xi_gg", "mbe_tree"))

    def test_xi_gg_all_suffixes_brute_equals_tree(self, IA_mock_lc_n1,
                                                    tmp_path):
        obj = IA_mock_lc_n1
        self._run(obj, "mbe_brute", tree=False, tmp_path=tmp_path)
        self._run(obj, "mbe_tree",  tree=True,  tmp_path=tmp_path)
        # Brute-force only writes _DD (the SplusD pass); tree writes all four.
        np.testing.assert_array_equal(
            _read(obj, "multipoles/xi_gg", "mbe_brute_DD"),
            _read(obj, "multipoles/xi_gg", "mbe_tree_DD"),
            err_msg="xi_gg_DD brute != tree")

    # ---- multiprocessing: allclose -----------------------------------------

    def test_splusd_allclose_multiproc(self, IA_mock_lc_n1,
                                        IA_mock_lc_n8, tmp_path):
        tp = str(tmp_path) + "/"
        self._run(IA_mock_lc_n1, "mmp_n1", tree=True, tmp_path=tmp_path)
        IA_mock_lc_n8.measure_xi_multipoles("galaxies", "mmp_n8", "both",
                                             tree=True,
                                             temp_file_path=tp,
                                             chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_lc_n1, "multipoles/xi_g_plus", "mmp_n1_SplusD"),
            _read(IA_mock_lc_n8, "multipoles/xi_g_plus", "mmp_n8_SplusD"),
            rtol=1e-10)

    def test_dd_allclose_multiproc(self, IA_mock_lc_n1,
                                    IA_mock_lc_n8, tmp_path):
        tp = str(tmp_path) + "/"
        self._run(IA_mock_lc_n1, "mmp_n1", tree=True, tmp_path=tmp_path)
        IA_mock_lc_n8.measure_xi_multipoles("galaxies", "mmp_n8", "both",
                                             tree=True,
                                             temp_file_path=tp,
                                             chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_lc_n1, "multipoles/xi_gg", "mmp_n1_DD"),
            _read(IA_mock_lc_n8, "multipoles/xi_gg", "mmp_n8_DD"),
            rtol=1e-10)
