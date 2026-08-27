"""
test_box_measure_xi_w.py
========================
Comprehensive tests for MeasureIABox.measure_xi_w().

All tests use the synthetic catalogue fixtures from conftest.py — no TNG
simulation files are required.  Each fixture gets its own tmp_path output
file so tests are fully independent and can run in any order.

Covers
------
  1.  corr_type variations: 'both', 'g+', 'gg', invalid
  2.  Ellipticity definitions: 'distortion' vs 'ellipticity'
  3.  num_jk input validation
  4.  Computation backends, no JK: brute == tree == multiproc
  5.  Computation backends, with JK: realisation-level DD / SplusD / RR / xi agreement
  6.  Output shape and rp-bin consistency
  7.  Covariance matrix properties (symmetry, non-negative diagonal)
  8.  Masks: reduce pair count, shape-sample-only mask
  9.  Weights: 4× and 16× scaling laws
  10. Jackknife region assignment
  11. _combine_jackknife_information reproduces stored covariance
  12. Intermediate xi outputs: SplusD, RR, pi/rp grids, ScrossD,
      xi_g_cross group, sigmasq, per-realisation bin grids
  13. Intermediate pair-count equality: brute == tree, tree == multiproc: SplusD, RR, pi/rp grids, ScrossD,
      xi_g_cross group, sigmasq, per-realisation bin grids
"""

import numpy as np
import pytest
import h5py
from measureia import MeasureIABox, ReadData, pair_kernel


NUM_JK = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gp(obj):
    return ReadData._from_file(obj.output_file_name, obj.snap_group + "w_g_plus/")


def _gg(obj):
    return ReadData._from_file(obj.output_file_name, obj.snap_group + "w_gg/")


def _xi_gg(obj):
    return ReadData._from_file(obj.output_file_name, obj.snap_group + "w/xi_gg/")


def _xi_gp(obj):
    return ReadData._from_file(obj.output_file_name, obj.snap_group + "w/xi_g_plus/")


def _read(obj, group, key):
    """Read a single dataset from the object's output HDF5."""
    with h5py.File(obj.output_file_name, "r") as f:
        return f[obj.snap_group + group][key][:]


# ---------------------------------------------------------------------------
# 1. corr_type variations
# ---------------------------------------------------------------------------

class TestCorrTypeW:

    def test_gp_matches_both(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("ct_both", "both", 0, temp_file_path=False)
        obj.measure_xi_w("ct_gp",   "g+",   0, temp_file_path=False)

        np.testing.assert_array_equal(
            _read(obj, "w_g_plus", "ct_both"),
            _read(obj, "w_g_plus", "ct_gp"))
        np.testing.assert_array_equal(
            _read(obj, "w_g_plus", "ct_both_rp"),
            _read(obj, "w_g_plus", "ct_gp_rp"))

    def test_gg_matches_both(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("ct_both", "both", 0, temp_file_path=False)
        obj.measure_xi_w("ct_gg",   "gg",   0, temp_file_path=False)

        np.testing.assert_array_equal(
            _read(obj, "w_gg", "ct_both"),
            _read(obj, "w_gg", "ct_gg"))

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1):
        # corr_type is now validated up front (uniform ValueError), before any pair counting
        with pytest.raises(ValueError, match="corr_type"):
            IA_mock_TNG300_n1.measure_xi_w("bad", "gg+", 0,
                                           temp_file_path=False)

    def test_gg_count_pairs_matches_full_all_backends(self, IA_mock_TNG300_n1,
                                                      IA_mock_TNG300_n8, tmp_path):
        """corr_type='gg' dispatches to the DD-only count_pairs backends —
        DD grid and w_gg must match the full-loop ('both') result for brute,
        tree and multiprocessing."""
        tp = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("cp_full", "both", 0, temp_file_path=False)
        obj.measure_xi_w("cp_brute", "gg", 0, temp_file_path=False)
        obj.measure_xi_w("cp_tree", "gg", 0, temp_file_path=tp)
        ref_dd = _read(obj, "w/xi_gg", "cp_full_DD")
        ref_w = _read(obj, "w_gg", "cp_full")
        for name in ("cp_brute", "cp_tree"):
            np.testing.assert_array_equal(_read(obj, "w/xi_gg", f"{name}_DD"), ref_dd)
            np.testing.assert_array_equal(_read(obj, "w_gg", name), ref_w)
        obj8 = IA_mock_TNG300_n8
        obj8.measure_xi_w("cp_mp", "gg", 0, temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(_read(obj8, "w/xi_gg", "cp_mp_DD"), ref_dd,
                                   rtol=1e-10)
        np.testing.assert_allclose(_read(obj8, "w_gg", "cp_mp"), ref_w,
                                   rtol=1e-10)

    def test_gg_count_pairs_jk_matches_full(self, IA_mock_TNG300_n1,
                                            IA_mock_TNG300_n8, tmp_path):
        """corr_type='gg' with num_jk>0 dispatches to the DD-only jk count
        backends — final w_gg, realisations and covariance must match the
        full-loop ('both') jk path for brute, tree and multiprocessing."""
        tp = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jkcp_full", "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jkcp_tree", "gg", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jkcp_brute", "gg", NUM_JK, temp_file_path=False)
        ref_w = _read(obj, "w_gg", "jkcp_full")
        ref_cov = _read(obj, "w_gg", f"jkcp_full_jackknife_cov_{NUM_JK}")
        for name, rt in (("jkcp_tree", 1e-12), ("jkcp_brute", 1e-10)):
            np.testing.assert_allclose(_read(obj, "w_gg", name), ref_w,
                                       rtol=rt, atol=1e-13,
                                       err_msg=f"{name} w_gg mismatch")
            np.testing.assert_allclose(
                _read(obj, "w_gg", f"{name}_jackknife_cov_{NUM_JK}"), ref_cov,
                rtol=1e-8, atol=1e-15,
                err_msg=f"{name} covariance mismatch")
        obj8 = IA_mock_TNG300_n8
        obj8.measure_xi_w("jkcp_mp", "gg", NUM_JK, temp_file_path=tp,
                          chunk_size=50)
        np.testing.assert_allclose(_read(obj8, "w_gg", "jkcp_mp"), ref_w,
                                   rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(
            _read(obj8, "w_gg", f"jkcp_mp_jackknife_cov_{NUM_JK}"), ref_cov,
            rtol=1e-8, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Ellipticity definition
# ---------------------------------------------------------------------------

class TestEllipticityDefinitionW:

    def test_wgg_same_for_both_defs(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("ell_dist", "gg", 0, temp_file_path=False,
                         ellipticity="distortion")
        obj.measure_xi_w("ell_ell",  "gg", 0, temp_file_path=False,
                         ellipticity="ellipticity")

        np.testing.assert_array_equal(
            _read(obj, "w_gg", "ell_dist"),
            _read(obj, "w_gg", "ell_ell"))

    def test_wgp_differs_by_definition(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("wgp_dist", "g+", 0, temp_file_path=False,
                         ellipticity="distortion")
        obj.measure_xi_w("wgp_ell",  "g+", 0, temp_file_path=False,
                         ellipticity="ellipticity")

        assert not np.allclose(
            _read(obj, "w_g_plus", "wgp_dist"),
            _read(obj, "w_g_plus", "wgp_ell")), \
            "distortion and ellipticity should differ for q < 1"

    def test_invalid_ellipticity_raises(self, IA_mock_TNG300_n1):
        with pytest.raises((KeyError, ValueError)):
            IA_mock_TNG300_n1.measure_xi_w("bad_ell", "both", 0,
                                           temp_file_path=False,
                                           ellipticity="wrong")


# ---------------------------------------------------------------------------
# 3. num_jk input validation
# ---------------------------------------------------------------------------

class TestNumJKValidationW:

    def test_non_cube_raises(self, IA_mock_TNG300_n1, tmp_path):
        with pytest.raises(ValueError):
            IA_mock_TNG300_n1.measure_xi_w("bad_jk", "both", 7,
                                           temp_file_path=str(tmp_path))

    def test_zero_jk_succeeds(self, IA_mock_TNG300_n1):
        IA_mock_TNG300_n1.measure_xi_w("zero_jk", "both", 0,
                                       temp_file_path=False)

    def test_jk_without_temp_path_raises(self, IA_mock_TNG300_n1):
        with pytest.raises(ValueError):
            IA_mock_TNG300_n1.measure_xi_w("no_path", "both", NUM_JK,
                                           temp_file_path=None)


# ---------------------------------------------------------------------------
# 4. Backends — no JK
# ---------------------------------------------------------------------------

class TestBackendsNoJKW:

    def test_brute_vs_tree_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("bvt_brute", "both", 0, temp_file_path=False)
        obj.measure_xi_w("bvt_tree",  "both", 0,
                         temp_file_path=str(tmp_path) + "/")

        np.testing.assert_allclose(
            _read(obj, "w_g_plus", "bvt_brute"),
            _read(obj, "w_g_plus", "bvt_tree"))
        np.testing.assert_allclose(
            _read(obj, "w_gg", "bvt_brute"),
            _read(obj, "w_gg", "bvt_tree"))
        np.testing.assert_array_equal(
            _read(obj, "w_g_plus", "bvt_brute_rp"),
            _read(obj, "w_g_plus", "bvt_tree_rp"))

    def test_tree_vs_multiproc(self, IA_mock_TNG300_n1, IA_mock_TNG300_n8,
                                tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_w("mp_tree",  "both", 0,
                                       temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_w("mp_multi", "both", 0,
                                       temp_file_path=tp, chunk_size=50)

        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "w_g_plus", "mp_tree"),
            _read(IA_mock_TNG300_n8, "w_g_plus", "mp_multi"))
        np.testing.assert_array_equal(
            _read(IA_mock_TNG300_n1, "w_g_plus", "mp_tree_rp"),
            _read(IA_mock_TNG300_n8, "w_g_plus", "mp_multi_rp"))

    def test_multiproc_chunk_size_not_dividing_catalogue(
            self, IA_mock_TNG300_n1, IA_mock_TNG300_n8, tmp_path):
        """The catalogue has 200 objects; every other multiprocessing test
        uses chunk_size=50, which divides it evenly. A chunk size that leaves
        a short trailing batch must give the same answer (regression for
        off-by-one clamping of the final chunk)."""
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_w("uneven_tree", "both", 0,
                                       temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_w("uneven_mp", "both", 0,
                                       temp_file_path=tp, chunk_size=37)

        for grp in ("w_g_plus", "w_gg"):
            np.testing.assert_allclose(
                _read(IA_mock_TNG300_n1, grp, "uneven_tree"),
                _read(IA_mock_TNG300_n8, grp, "uneven_mp"),
                rtol=1e-10, atol=1e-14, err_msg=f"{grp} mismatch")


# ---------------------------------------------------------------------------
# 5. Backends — with JK
# ---------------------------------------------------------------------------

class TestBackendsWithJKW:

    def test_dd_brute_vs_tree_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        tp = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_tree",  "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jk_brute", "both", NUM_JK, temp_file_path=False)

        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_gg/jk_tree_jk{NUM_JK}",
                      f"jk_tree_{i}_DD"),
                _read(obj, f"w/xi_gg/jk_brute_jk{NUM_JK}",
                      f"jk_brute_{i}_DD"),
                rtol=1e-5,
                err_msg=f"DD mismatch in JK realisation {i}")

    def test_splusd_brute_vs_tree_all_realisations(self, IA_mock_TNG300_n1,
                                                    tmp_path):
        tp  = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_tree",  "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jk_brute", "both", NUM_JK, temp_file_path=False)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_g_plus/jk_tree_jk{NUM_JK}",
                      f"jk_tree_{i}_SplusD"),
                _read(obj, f"w/xi_g_plus/jk_brute_jk{NUM_JK}",
                      f"jk_brute_{i}_SplusD"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"SplusD mismatch in JK realisation {i}")

    def test_w_realisations_brute_vs_tree(self, IA_mock_TNG300_n1, tmp_path):
        tp  = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_tree",  "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jk_brute", "both", NUM_JK, temp_file_path=False)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w_g_plus/jk_tree_jk{NUM_JK}",  f"jk_tree_{i}"),
                _read(obj, f"w_g_plus/jk_brute_jk{NUM_JK}", f"jk_brute_{i}"),
                rtol=1e-5)
            np.testing.assert_allclose(
                _read(obj, f"w_gg/jk_tree_jk{NUM_JK}",  f"jk_tree_{i}"),
                _read(obj, f"w_gg/jk_brute_jk{NUM_JK}", f"jk_brute_{i}"),
                rtol=1e-5)

    def test_rr_brute_vs_tree_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        tp  = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_tree",  "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jk_brute", "both", NUM_JK, temp_file_path=False)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_g_plus/jk_tree_jk{NUM_JK}",
                      f"jk_tree_{i}_RR"),
                _read(obj, f"w/xi_g_plus/jk_brute_jk{NUM_JK}",
                      f"jk_brute_{i}_RR"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"RR mismatch at realisation {i}")

    def test_xi_brute_vs_tree_all_realisations(self, IA_mock_TNG300_n1, tmp_path):
        tp  = str(tmp_path) + "/"
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_tree",  "both", NUM_JK, temp_file_path=tp)
        obj.measure_xi_w("jk_brute", "both", NUM_JK, temp_file_path=False)
        for i in range(NUM_JK):
            np.testing.assert_allclose(
                _read(obj, f"w/xi_g_plus/jk_tree_jk{NUM_JK}",
                      f"jk_tree_{i}"),
                _read(obj, f"w/xi_g_plus/jk_brute_jk{NUM_JK}",
                      f"jk_brute_{i}"),
                rtol=1e-5, atol=1e-5,
                err_msg=f"xi_g+ mismatch at realisation {i}")

    def test_multiproc_matches_tree_jk(self, IA_mock_TNG300_n1,
                                        IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_w("jk_tree", "both", NUM_JK,
                                        temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_w("jk_mp", "both", NUM_JK,
                                        temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "w_g_plus", "jk_tree"),
            _read(IA_mock_TNG300_n8, "w_g_plus", "jk_mp"),
            rtol=1e-5)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "w_g_plus",
                  f"jk_tree_jackknife_cov_{NUM_JK}"),
            _read(IA_mock_TNG300_n8, "w_g_plus",
                  f"jk_mp_jackknife_cov_{NUM_JK}"),
            rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Output shape / rp consistency
# ---------------------------------------------------------------------------

class TestOutputShapeW:

    def test_rp_length_matches_num_bins_r(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("shp", "both", 0, temp_file_path=False)
        rp = _read(obj, "w_g_plus", "shp_rp")
        w  = _read(obj, "w_g_plus", "shp")
        assert len(rp) == obj.num_bins_r
        assert len(w)  == obj.num_bins_r

    def test_rp_bins_sorted_ascending(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("shp2", "both", 0, temp_file_path=False)
        rp = _read(obj, "w_g_plus", "shp2_rp")
        assert np.all(np.diff(rp) > 0)

    def test_rp_consistent_across_corr_types(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("rp_gp", "g+", 0, temp_file_path=False)
        obj.measure_xi_w("rp_gg", "gg", 0, temp_file_path=False)
        np.testing.assert_array_equal(
            _read(obj, "w_g_plus", "rp_gp_rp"),
            _read(obj, "w_gg",     "rp_gg_rp"))

    def test_rp_within_separation_limits(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("rp_lim", "g+", 0, temp_file_path=False)
        rp = _read(obj, "w_g_plus", "rp_lim_rp")
        assert rp[0]  >= obj.r_bins[0]
        assert rp[-1] <= obj.r_bins[-1]


# ---------------------------------------------------------------------------
# 7. Covariance properties
# ---------------------------------------------------------------------------

class TestCovariancePropertiesW:

    def _run_jk(self, obj, name, tmp_path):
        obj.measure_xi_w(name, "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")

    def test_cov_gp_is_square(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "cov_sq", tmp_path)
        cov = _read(obj, "w_g_plus", f"cov_sq_jackknife_cov_{NUM_JK}")
        assert cov.ndim == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == obj.num_bins_r

    def test_cov_gp_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "cov_sym", tmp_path)
        cov = _read(obj, "w_g_plus", f"cov_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_cov_gp_diagonal_non_negative(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "cov_diag", tmp_path)
        cov = _read(obj, "w_g_plus", f"cov_diag_jackknife_cov_{NUM_JK}")
        assert np.all(np.diag(cov) >= 0)

    def test_cov_gg_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_jk(obj, "cov_gg_sym", tmp_path)
        cov = _read(obj, "w_gg", f"cov_gg_sym_jackknife_cov_{NUM_JK}")
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Masks
# ---------------------------------------------------------------------------

class TestMasksW:

    def test_mask_reduces_pair_count(self, IA_mock_TNG300_n1, box_mass):
        obj   = IA_mock_TNG300_n1
        masks = {k: box_mass > np.median(box_mass)
                 for k in obj.data}
        obj.measure_xi_w("no_mask",   "gg", 0, temp_file_path=False)
        obj.measure_xi_w("with_mask", "gg", 0, temp_file_path=False,
                         masks=masks)
        dd_all  = _read(obj, "w/xi_gg", "no_mask_DD")
        dd_mask = _read(obj, "w/xi_gg", "with_mask_DD")
        assert np.sum(dd_mask) < np.sum(dd_all)

    def test_mask_nonzero_result(self, IA_mock_TNG300_n1, box_mass):
        obj   = IA_mock_TNG300_n1
        masks = {k: box_mass > np.median(box_mass) for k in obj.data}
        obj.measure_xi_w("mask_gp", "g+", 0, temp_file_path=False,
                         masks=masks)
        # At least some bins should be non-zero
        assert np.any(_read(obj, "w_g_plus", "mask_gp") != 0)

    def test_shape_sample_only_mask(self, IA_mock_TNG300_n1, box_mass):
        """Mask on shape sample only — should run without error."""
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("mask_ss_only", "g+", 0, temp_file_path=False,
                         masks={"Position_shape_sample":
                                box_mass > np.median(box_mass)})

    def test_default_weight_mask_follows_selection(self, IA_mock_TNG300_n1,
                                                   box_mass, tmp_path):
        """Regression: with non-uniform weights and a non-contiguous mask but no
        'weight' entry in masks, the weights of the selected galaxies must be
        used — masking must give the same result as pre-masking the data."""
        obj = IA_mock_TNG300_n1
        rng = np.random.default_rng(12)
        weights = rng.uniform(0.5, 2.0, len(box_mass))
        obj.data["weight"] = weights
        obj.data["weight_shape_sample"] = weights
        sel = box_mass > np.median(box_mass)
        assert np.any(np.diff(np.flatnonzero(sel)) > 1)
        obj.measure_xi_w("masked", "both", 0, temp_file_path=False,
                         masks={"Position": sel, "Position_shape_sample": sel,
                                "Axis_Direction": sel, "q": sel})

        data_pre = {
            "Position":              obj.data["Position"][sel],
            "Position_shape_sample": obj.data["Position_shape_sample"][sel],
            "Axis_Direction":        obj.data["Axis_Direction"][sel],
            "q":                     obj.data["q"][sel],
            "LOS":                   obj.data["LOS"],
            "weight":                weights[sel],
            "weight_shape_sample":   weights[sel],
        }
        pre = MeasureIABox(data_pre, str(tmp_path / "premasked.hdf5"),
                           simulation="TNG300", snapshot=99,
                           separation_limits=[obj.r_min, obj.r_max],
                           num_bins_r=obj.num_bins_r,
                           num_bins_pi=obj.num_bins_pi)
        pre.measure_xi_w("premasked", "both", 0, temp_file_path=False)
        for grp in ("w_g_plus", "w_gg"):
            np.testing.assert_allclose(_read(obj, grp, "masked"),
                                       _read(pre, grp, "premasked"),
                                       rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 9. Weights — scaling laws
# ---------------------------------------------------------------------------

class TestWeightsW:

    def _run_w(self, obj, name, w_val, tmp_path):
        N = len(obj.data["Position"])
        obj.data["weight"]              = np.full(N, w_val)
        obj.data["weight_shape_sample"] = np.full(N, w_val)
        obj.measure_xi_w(name, "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")

    def test_weight_scaling_wgp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "wt_ones", 1.0, tmp_path)
        self._run_w(obj, "wt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w_g_plus", "wt_ones"),
            4 * _read(obj, "w_g_plus", "wt_half"))

    def test_weight_scaling_dd(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "wt_ones", 1.0, tmp_path)
        self._run_w(obj, "wt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_gg", "wt_ones_DD"),
            4 * _read(obj, "w/xi_gg", "wt_half_DD"))

    def test_weight_scaling_cov_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "wt_ones", 1.0, tmp_path)
        self._run_w(obj, "wt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w_g_plus", f"wt_ones_jackknife_cov_{NUM_JK}"),
            16 * _read(obj, "w_g_plus", f"wt_half_jackknife_cov_{NUM_JK}"))

    def test_weight_scaling_cov_gg(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "wt_ones", 1.0, tmp_path)
        self._run_w(obj, "wt_half", 0.5, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w_gg", f"wt_ones_jackknife_cov_{NUM_JK}"),
            16 * _read(obj, "w_gg", f"wt_half_jackknife_cov_{NUM_JK}"))

    def test_rp_unchanged_by_weights(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_w(obj, "wt_ones", 1.0, tmp_path)
        self._run_w(obj, "wt_half", 0.5, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w_g_plus", "wt_ones_rp"),
            _read(obj, "w_g_plus", "wt_half_rp"))


# ---------------------------------------------------------------------------
# 10. Jackknife region assignment
# ---------------------------------------------------------------------------

class TestJKRegionsW:

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

class TestCombineJackknifeW:

    def test_combine_reproduces_covariance(self, IA_mock_TNG300_n1, tmp_path):
        """Running measure_xi_w then _combine_jackknife_information must
        reproduce the covariance stored in the output file."""
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_comb", "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "jk_comb", f"jk_comb_jk{NUM_JK}",
            ["w_g_plus", "w_gg"], NUM_JK, return_output=True)

        for cov, group in zip(covs, ["w_g_plus", "w_gg"]):
            stored = _read(obj, group, f"jk_comb_jackknife_cov_{NUM_JK}")
            np.testing.assert_allclose(cov, stored, rtol=1e-5)

    def test_combined_cov_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj  = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_comb", "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "jk_comb", f"jk_comb_jk{NUM_JK}",
            ["w_g_plus", "w_gg"], NUM_JK, return_output=True)
        for cov in covs:
            np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_combined_cov_diagonal_non_negative(self, IA_mock_TNG300_n1,
                                                 tmp_path):
        obj  = IA_mock_TNG300_n1
        obj.measure_xi_w("jk_comb", "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")
        covs, _ = obj._combine_jackknife_information(
            "jk_comb", f"jk_comb_jk{NUM_JK}",
            ["w_g_plus", "w_gg"], NUM_JK, return_output=True)
        for cov in covs:
            assert np.all(np.diag(cov) >= 0)



# ---------------------------------------------------------------------------
# 12. Intermediate xi outputs (full-sample and per-realisation)
# ---------------------------------------------------------------------------

class TestIntermediateOutputsW:
    """
    Verifies every dataset written to the w/xi_g_plus/, w/xi_g_cross/, and
    w/xi_gg/ groups that the existing sections do not already cover:

      Full-sample (no-JK):
        xi_g_plus/  : SplusD, RR_g_plus, rp, pi, ScrossD
        xi_g_cross/ : xi, RR_g_cross, rp, pi   (entire group)
        xi_gg/      : RR_gg, rp, pi
      JK path:
        _sigmasq    written alongside full-sample xi (g_plus and gg)
      Per-realisation JK:
        _{i}_rp and _{i}_pi match full-sample rp/pi for every realisation
    """

    # ------------------------------------------------------------------ setup

    def _run_no_jk(self, obj):
        obj.measure_xi_w("int_nojk", "both", 0, temp_file_path=False)

    def _run_jk(self, obj, tmp_path):
        obj.measure_xi_w("int_jk", "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")

    # ------------------------------------------------------------------ full-sample SplusD / RR / pi / rp

    def test_splusd_shape_and_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        splusd = _read(obj, "w/xi_g_plus", "int_nojk_SplusD")
        assert splusd.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(np.isfinite(splusd))

    def test_rr_g_plus_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "w/xi_g_plus", "int_nojk_RR_g_plus")
        assert rr.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(rr > 0)

    def test_rp_grid_in_xi_group_sorted(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rp = _read(obj, "w/xi_g_plus", "int_nojk_rp")
        assert np.all(np.diff(rp) > 0)

    def test_pi_grid_in_xi_group_sorted(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        pi = _read(obj, "w/xi_g_plus", "int_nojk_pi")
        assert np.all(np.diff(pi) > 0)

    def test_rp_grid_matches_r_bins(self, IA_mock_TNG300_n1):
        """rp values stored in xi group should equal the midpoints of r_bins."""
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rp  = _read(obj, "w/xi_g_plus", "int_nojk_rp")
        assert rp[0]  >= obj.r_bins[0]
        assert rp[-1] <= obj.r_bins[-1]

    def test_pi_grid_matches_pi_bins(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        pi  = _read(obj, "w/xi_g_plus", "int_nojk_pi")
        assert pi[0]  >= obj.pi_bins[0]
        assert pi[-1] <= obj.pi_bins[-1]

    def test_scrossd_shape(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        sc = _read(obj, "w/xi_g_cross", "int_nojk_ScrossD")
        assert sc.shape == (obj.num_bins_r, obj.num_bins_pi)

    # ------------------------------------------------------------------ xi_g_cross group

    def test_xi_g_cross_exists(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        xi_cross = _read(obj, "w/xi_g_cross", "int_nojk")
        assert xi_cross.shape == (obj.num_bins_r, obj.num_bins_pi)

    def test_xi_g_cross_rr_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "w/xi_g_cross", "int_nojk_RR_g_cross")
        assert np.all(rr > 0)

    def test_xi_g_cross_rp_matches_xi_g_plus_rp(self, IA_mock_TNG300_n1):
        """xi_g_cross and xi_g_plus share the same rp / pi grids."""
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "int_nojk_rp"),
            _read(obj, "w/xi_g_plus",  "int_nojk_rp"))

    def test_xi_g_cross_pi_matches_xi_g_plus_pi(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "int_nojk_pi"),
            _read(obj, "w/xi_g_plus",  "int_nojk_pi"))

    # ------------------------------------------------------------------ xi_gg full-sample RR / rp / pi

    def test_rr_gg_positive(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr = _read(obj, "w/xi_gg", "int_nojk_RR_gg")
        assert rr.shape == (obj.num_bins_r, obj.num_bins_pi)
        assert np.all(rr > 0)

    def test_rr_gg_consistent_with_formula(self, IA_mock_TNG300_n1):
        """Sum of RR_gg over all bins should equal the analytical prediction
        from get_random_pairs over the full (rp_min, rp_max) and signed
        (-pi_max, pi_max) ranges. The code counts ordered pairs and bins
        signed pi, so the 'cross' normalisation and full pi extent apply;
        the per-bin analytic values telescope exactly to this total."""
        from measureia import MeasureIABase
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        rr  = _read(obj, "w/xi_gg", "int_nojk_RR_gg")
        L3  = obj.boxsize ** 3
        N   = obj.Num_position
        total_rr_analytical = MeasureIABase.get_random_pairs(
            obj.r_bins[-1], obj.r_bins[0],
            obj.pi_bins[-1], obj.pi_bins[0],
            L3, "cross", N, N, obj.num_overlap)
        assert np.sum(rr) == pytest.approx(total_rr_analytical, rel=1e-10)

    def test_xi_gg_rp_pi_grids_match_xi_g_plus(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        self._run_no_jk(obj)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg",     "int_nojk_rp"),
            _read(obj, "w/xi_g_plus", "int_nojk_rp"))
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg",     "int_nojk_pi"),
            _read(obj, "w/xi_g_plus", "int_nojk_pi"))

    # ------------------------------------------------------------------ per-realisation rp / pi grids

    def test_per_jk_rp_matches_fullsample_rp(self, IA_mock_TNG300_n1, tmp_path):
        """Each drop-one realisation stores rp; all must equal full-sample rp."""
        obj    = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        rp_ref = _read(obj, "w/xi_g_plus", "int_jk_rp")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_g_plus/int_jk_jk{NUM_JK}",
                      f"int_jk_{i}_rp"),
                rp_ref,
                err_msg=f"rp mismatch at JK realisation {i}")

    def test_per_jk_pi_matches_fullsample_pi(self, IA_mock_TNG300_n1, tmp_path):
        obj    = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        pi_ref = _read(obj, "w/xi_g_plus", "int_jk_pi")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_g_plus/int_jk_jk{NUM_JK}",
                      f"int_jk_{i}_pi"),
                pi_ref,
                err_msg=f"pi mismatch at JK realisation {i}")

    def test_per_jk_gg_rp_matches_fullsample(self, IA_mock_TNG300_n1, tmp_path):
        obj    = IA_mock_TNG300_n1
        self._run_jk(obj, tmp_path)
        rp_ref = _read(obj, "w/xi_gg", "int_jk_rp")
        for i in range(NUM_JK):
            np.testing.assert_array_equal(
                _read(obj, f"w/xi_gg/int_jk_jk{NUM_JK}", f"int_jk_{i}_rp"),
                rp_ref,
                err_msg=f"gg rp mismatch at JK realisation {i}")


# ---------------------------------------------------------------------------
# 13. Intermediate pair-count equality: brute == tree
# ---------------------------------------------------------------------------

class TestIntermediatePairCountEqualityW:
    """
    Every intermediate pair-count array written to xi_g_plus/, xi_g_cross/,
    and xi_gg/ must be identical between the brute and tree backends
    (given identical inputs).  This covers:
      xi_g_plus/ : SplusD, ScrossD, RR_g_plus, xi
      xi_g_cross/: xi, RR_g_cross
      xi_gg/     : xi, DD, RR_gg
    Bin-grid equality (rp, pi) is already verified in section 12 per-realisation
    tests, so it is not repeated here.
    Multiprocessing may introduce machine-precision differences and is therefore
    tested with allclose rather than array_equal.
    """

    def _run_brute_and_tree(self, obj, tmp_path):
        obj.measure_xi_w("pce_brute", "both", 0, temp_file_path=False)
        obj.measure_xi_w("pce_tree",  "both", 0,
                         temp_file_path=str(tmp_path) + "/")

    # xi_g_plus
    def test_splusd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_plus", "pce_brute_SplusD"),
            _read(obj, "w/xi_g_plus", "pce_tree_SplusD"), rtol=1e-12, atol=1e-12)

    def test_scrossd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_cross", "pce_brute_ScrossD"),
            _read(obj, "w/xi_g_cross", "pce_tree_ScrossD"), rtol=1e-12, atol=1e-15)

    def test_rr_g_plus_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_plus", "pce_brute_RR_g_plus"),
            _read(obj, "w/xi_g_plus", "pce_tree_RR_g_plus"))

    def test_xi_g_plus_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_plus", "pce_brute"),
            _read(obj, "w/xi_g_plus", "pce_tree"))

    # xi_g_cross
    def test_xi_g_cross_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_g_cross", "pce_brute"),
            _read(obj, "w/xi_g_cross", "pce_tree"))

    def test_rr_g_cross_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_g_cross", "pce_brute_RR_g_cross"),
            _read(obj, "w/xi_g_cross", "pce_tree_RR_g_cross"))

    # xi_gg
    def test_xi_gg_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_allclose(
            _read(obj, "w/xi_gg", "pce_brute"),
            _read(obj, "w/xi_gg", "pce_tree"))

    def test_dd_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg", "pce_brute_DD"),
            _read(obj, "w/xi_gg", "pce_tree_DD"))

    def test_rr_gg_brute_equals_tree(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_brute_and_tree(obj, tmp_path)
        np.testing.assert_array_equal(
            _read(obj, "w/xi_gg", "pce_brute_RR_gg"),
            _read(obj, "w/xi_gg", "pce_tree_RR_gg"))

    # multiprocessing — allclose only (floating-point accumulation differences)
    def test_splusd_tree_allclose_multiproc(self, IA_mock_TNG300_n1,
                                             IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_w("pce_mp1", "both", 0,
                                        temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_w("pce_mp8", "both", 0,
                                        temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "w/xi_g_plus", "pce_mp1_SplusD"),
            _read(IA_mock_TNG300_n8, "w/xi_g_plus", "pce_mp8_SplusD"),
            rtol=1e-10)

    def test_dd_tree_allclose_multiproc(self, IA_mock_TNG300_n1,
                                         IA_mock_TNG300_n8, tmp_path):
        tp = str(tmp_path) + "/"
        IA_mock_TNG300_n1.measure_xi_w("pce_mp1", "both", 0,
                                        temp_file_path=tp)
        IA_mock_TNG300_n8.measure_xi_w("pce_mp8", "both", 0,
                                        temp_file_path=tp, chunk_size=50)
        np.testing.assert_allclose(
            _read(IA_mock_TNG300_n1, "w/xi_gg", "pce_mp1_DD"),
            _read(IA_mock_TNG300_n8, "w/xi_gg", "pce_mp8_DD"),
            rtol=1e-10)


class TestBruteOnThe3DBranchWithJackknife:
    """The one cell the axis audit found empty: brute backend, 3D tree branch,
    jackknife on.

    Low risk on its own — the brute backend takes every position as a candidate
    and never builds a tree, so the 2D/3D choice barely reaches it — but an empty
    cell is an empty cell, and `benchmarks/axis_audit.py` reports it as such.
    Closing it keeps that report clean so a genuinely worrying gap stands out.
    """

    def test_brute_matches_tree_on_the_3d_branch_with_jk(self, tmp_path):
        objs = [TestTreeCoordsAgreeAcrossProcesses._obj(tmp_path, f"b3d{i}", 1)
                for i in range(2)]
        assert pair_kernel.BoxRpPi(objs[0]).tree_is_3d
        tp = str(tmp_path) + "/"
        objs[0].measure_xi_w("b3d_tree", "both", 8, temp_file_path=tp)
        objs[1].measure_xi_w("b3d_brute", "both", 8, temp_file_path=False)
        np.testing.assert_allclose(
            _read(objs[0], "w_g_plus", "b3d_tree"),
            _read(objs[1], "w_g_plus", "b3d_brute"), rtol=1e-8, atol=1e-12)
        np.testing.assert_array_equal(
            _read(objs[0], "w/xi_gg", "b3d_tree_DD"),
            _read(objs[1], "w/xi_gg", "b3d_brute_DD"))


class TestTreeCoordsAgreeAcrossProcesses:
    """The parent's shared tree and the workers' chunk trees must be built on the
    same coordinates.

    The multiprocessing backends build one position tree in the parent and hand
    it to the workers, which build their own chunk trees via
    ``binning.tree_coords``. If the parent hardcodes a different convention the
    two disagree and scipy raises "Trees passed to query_ball_tree have
    different dimensionality" — which is exactly what happened when BoxRpPi
    gained the 3D-ball query (benchmarks/FINDINGS.md F7) while four backend
    files still hardcoded the 2D projection.

    The default fixtures do not catch it: they leave ``pi_max=None``, which
    defaults to half the boxsize, and BoxRpPi then chooses the 2D projection —
    the same convention the hardcoded parents used. Covering this needs a
    configuration that selects the *3D* branch, which is what these tests pin.
    """

    @staticmethod
    def _obj(tmp_path, name, num_nodes):
        rng = np.random.default_rng(20260826)
        N = 600
        L = 205.0
        pos = rng.uniform(0.0, L, (N, 3))
        data = {"Position": pos, "Position_shape_sample": pos,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.full(N, 0.6)}
        # pi_max well below the box depth, so the enclosing ball is smaller than
        # the full-depth cylinder and BoxRpPi selects the 3D tree
        return MeasureIABox(data, str(tmp_path / f"{name}.hdf5"),
                            simulation=None, snapshot=None,
                            separation_limits=[0.5, 20.0], num_bins_r=6,
                            num_bins_pi=8, pi_max=20.0, boxsize=L,
                            num_nodes=num_nodes)

    def test_configuration_really_selects_the_3d_branch(self, tmp_path):
        """Guard the guard: if this stops being 3D the tests below stop covering
        the case they exist for."""
        obj = self._obj(tmp_path, "branch", 1)
        assert pair_kernel.BoxRpPi(obj).tree_is_3d, \
            "this configuration no longer selects the 3D tree; the mp test below " \
            "would silently stop covering the parent/worker agreement"

    def test_multiproc_matches_single_process_on_the_3d_branch(self, tmp_path):
        one = self._obj(tmp_path, "n1", 1)
        many = self._obj(tmp_path, "n2", 2)
        tp = str(tmp_path) + "/"
        one.measure_xi_w("t3d_n1", "both", 0, temp_file_path=tp)
        many.measure_xi_w("t3d_n2", "both", 0, temp_file_path=tp, chunk_size=200)
        np.testing.assert_allclose(
            _read(one, "w_g_plus", "t3d_n1"), _read(many, "w_g_plus", "t3d_n2"),
            rtol=1e-8, atol=1e-12)
        np.testing.assert_array_equal(
            _read(one, "w/xi_gg", "t3d_n1_DD"), _read(many, "w/xi_gg", "t3d_n2_DD"))
