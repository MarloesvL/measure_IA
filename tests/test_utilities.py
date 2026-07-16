"""
test_utilities.py
=================
Tests for utility classes and standalone functions that are shared across
MeasureIABox and MeasureIALightcone, i.e. they do not belong to a single
measurement method.

Covers
------
  1.  write_data: write_dataset_hdf5 / create_group_hdf5
  2.  ReadData: read_cat (full / cut / indices / bad-catalogue paths)
               read_MeasureIA_output (attribute population, num_jk=None,
               stale-value reset)
  3.  MeasureJackknife.measure_covariance_multiple_datasets
        - auto-covariance (1 dataset): shape, symmetry, non-negative diagonal,
          std == sqrt(diag(cov))
        - cross-covariance (2 datasets): shape, differs from auto
        - invalid corr_type raises ValueError
        - >2 datasets raises KeyError
  4.  MeasureJackknife.create_full_cov_matrix_projections
        - runs without error, returns 4 matrices of correct shape
        - pairwise sub-blocks match direct measure_covariance_multiple_datasets
        - invalid corr_type raises ValueError
"""

import numpy as np
import pytest
import h5py

from measureia import (
    ReadData,
    write_dataset_hdf5,
    create_group_hdf5,
)

NUM_JK = 8


# ===========================================================================
# 1. write_data utilities
# ===========================================================================

class TestWriteData:

    def test_write_dataset_creates_dataset(self, tmp_path):
        p = str(tmp_path / "wd_test.hdf5")
        with h5py.File(p, "w") as f:
            grp = f.create_group("grp")
            write_dataset_hdf5(grp, "arr", np.array([1.0, 2.0, 3.0]))
            np.testing.assert_array_equal(grp["arr"][:], [1.0, 2.0, 3.0])

    def test_write_dataset_overwrites_without_error(self, tmp_path):
        p = str(tmp_path / "wd_overwrite.hdf5")
        with h5py.File(p, "w") as f:
            grp = f.create_group("grp")
            write_dataset_hdf5(grp, "arr", np.array([1.0, 2.0]))
            write_dataset_hdf5(grp, "arr", np.array([9.0, 8.0]))
            np.testing.assert_array_equal(grp["arr"][:], [9.0, 8.0])

    def test_write_dataset_2d_array(self, tmp_path):
        p    = str(tmp_path / "wd_2d.hdf5")
        data = np.eye(4)
        with h5py.File(p, "w") as f:
            grp = f.create_group("grp")
            write_dataset_hdf5(grp, "mat", data)
            np.testing.assert_array_equal(grp["mat"][:], data)

    def test_create_group_hdf5_builds_nested_groups(self, tmp_path):
        p = str(tmp_path / "cg_test.hdf5")
        with h5py.File(p, "w") as f:
            grp = create_group_hdf5(f, "a/b/c")
            assert "a" in f
            assert "b" in f["a"]
            assert "c" in f["a/b"]
            assert grp == f["a/b/c"]

    def test_create_group_hdf5_returns_existing_group(self, tmp_path):
        p = str(tmp_path / "cg_exist.hdf5")
        with h5py.File(p, "w") as f:
            f.create_group("existing/path")
            grp = create_group_hdf5(f, "existing/path")
            assert grp == f["existing/path"]

    def test_create_group_hdf5_single_level(self, tmp_path):
        p = str(tmp_path / "cg_single.hdf5")
        with h5py.File(p, "w") as f:
            grp = create_group_hdf5(f, "solo")
            assert "solo" in f


# ===========================================================================
# 2. ReadData
# ===========================================================================

class TestReadData:

    def _make_reader(self, tmp_path, snap=99):
        """Write a minimal HDF5 and return a ReadData pointed at it."""
        p    = str(tmp_path / "rd_cat.hdf5")
        data = np.arange(20, dtype=float)
        with h5py.File(p, "w") as f:
            grp = create_group_hdf5(f, f"Snapshot_{snap}/")
            write_dataset_hdf5(grp, "MyData", data)
        return ReadData(
            simulation="TNG300",
            catalogue="rd_cat",
            snapshot=snap,
            sub_group="",
            data_path=str(tmp_path) + "/",
        ), data

    def test_read_cat_full(self, tmp_path):
        reader, data = self._make_reader(tmp_path)
        np.testing.assert_array_equal(reader.read_cat("MyData"), data)

    def test_read_cat_cut(self, tmp_path):
        reader, data = self._make_reader(tmp_path)
        np.testing.assert_array_equal(
            reader.read_cat("MyData", cut=[3, 8]), data[3:8])

    def test_read_cat_indices(self, tmp_path):
        reader, data = self._make_reader(tmp_path)
        idx = np.array([0, 5, 10, 15])
        np.testing.assert_array_equal(
            reader.read_cat("MyData", indices=idx), data[idx])

    def test_read_cat_subhalo_raises(self, tmp_path):
        r = ReadData("TNG300", "Subhalo", 99,
                     data_path=str(tmp_path) + "/")
        with pytest.raises(KeyError, match="read_subhalo"):
            r.read_cat("anything")

    def test_read_cat_snapshot_raises(self, tmp_path):
        r = ReadData("TNG300", "Snapshot", 99,
                     data_path=str(tmp_path) + "/")
        with pytest.raises(KeyError, match="read_snapshot"):
            r.read_cat("anything")

    # -----------------------------------------------------------------------
    # read_MeasureIA_output
    # -----------------------------------------------------------------------

    def _reader_for(self, obj, tmp_path):
        """Return a ReadData pointing at obj's output file."""
        import os
        out_dir  = os.path.dirname(obj.output_file_name) + "/"
        out_name = os.path.splitext(os.path.basename(obj.output_file_name))[0]
        return ReadData("TNG300", out_name, 99, data_path=out_dir)

    def test_read_measureia_output_w_populates_attributes(
            self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("rdm_w", "both", NUM_JK,
                         temp_file_path=str(tmp_path) + "/")
        reader = self._reader_for(obj, tmp_path)
        reader.read_MeasureIA_output("rdm_w", num_jk=NUM_JK)

        assert reader.w_gg      is not None
        assert reader.w_gp      is not None
        assert reader.rp        is not None
        assert len(reader.rp)   == obj.num_bins_r
        assert reader.cov_w_gg  is not None
        assert reader.cov_w_gp  is not None

    def test_read_measureia_output_multipoles_populates_attributes(
            self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("rdm_m", "both", NUM_JK,
                                   temp_file_path=str(tmp_path) + "/")
        reader = self._reader_for(obj, tmp_path)
        reader.read_MeasureIA_output("rdm_m", num_jk=NUM_JK)

        assert reader.multipoles_gg is not None
        assert reader.multipoles_gp is not None
        assert reader.r             is not None
        assert len(reader.r)        == obj.num_bins_r
        assert reader.cov_multipoles_gg is not None
        assert reader.cov_multipoles_gp is not None

    def test_read_measureia_output_num_jk_none_leaves_cov_none(
            self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("rdm_nocov", "both", 0, temp_file_path=False)
        reader = self._reader_for(obj, tmp_path)
        reader.read_MeasureIA_output("rdm_nocov", num_jk=None)

        assert reader.w_gg     is not None   # data present
        assert reader.cov_w_gg is None       # covariance not requested

    def test_read_measureia_output_resets_stale_values(
            self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("rdm_first",  "both", 0, temp_file_path=False)
        obj.measure_xi_multipoles("rdm_second", "both", 0,
                                   temp_file_path=False)
        reader = self._reader_for(obj, tmp_path)

        reader.read_MeasureIA_output("rdm_first", num_jk=None)
        assert reader.w_gg          is not None
        assert reader.multipoles_gg is None         # not written yet

        reader.read_MeasureIA_output("rdm_second", num_jk=None)
        assert reader.multipoles_gg is not None
        assert reader.w_gg          is None         # reset on second call


# ===========================================================================
# 3. MeasureJackknife.measure_covariance_multiple_datasets
# ===========================================================================

class TestMeasureCovarianceMultipleDatasets:

    def _run_two_jk(self, obj, tmp_path):
        """ds_A and ds_B are measured along different LOS axes so they are
        genuinely different datasets (identical inputs would make the cross
        covariance equal the auto covariance by construction)."""
        tp = str(tmp_path) + "/"
        obj.data["LOS"] = 2
        obj.measure_xi_w("ds_A", "g+", NUM_JK, temp_file_path=tp)
        obj.data["LOS"] = 0
        obj.measure_xi_w("ds_B", "g+", NUM_JK, temp_file_path=tp)
        obj.data["LOS"] = 2

    def test_auto_covariance_shape(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov, std = obj.measure_covariance_multiple_datasets(
            corr_types=["w_g_plus"],
            dataset_names=["ds_A"],
            num_box=NUM_JK,
            return_output=True,
        )
        assert cov.shape == (obj.num_bins_r, obj.num_bins_r)
        assert std.shape == (obj.num_bins_r,)

    def test_auto_covariance_is_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["ds_A"], NUM_JK, return_output=True)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_auto_covariance_diagonal_non_negative(self, IA_mock_TNG300_n1,
                                                    tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["ds_A"], NUM_JK, return_output=True)
        assert np.all(np.diag(cov) >= 0)

    def test_std_equals_sqrt_diag_cov(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov, std = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["ds_A"], NUM_JK, return_output=True)
        np.testing.assert_allclose(std, np.sqrt(np.diag(cov)), rtol=1e-10)

    def test_cross_covariance_shape(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov, std = obj.measure_covariance_multiple_datasets(
            corr_types=["w_g_plus", "w_g_plus"],
            dataset_names=["ds_A", "ds_B"],
            num_box=NUM_JK,
            return_output=True,
        )
        assert cov.shape == (obj.num_bins_r, obj.num_bins_r)
        assert std.shape == (obj.num_bins_r,)

    def test_cross_covariance_differs_from_auto(self, IA_mock_TNG300_n1,
                                                 tmp_path):
        obj = IA_mock_TNG300_n1
        self._run_two_jk(obj, tmp_path)
        cov_auto,  _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["ds_A"], NUM_JK, return_output=True)
        cov_cross, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["ds_A", "ds_B"],
            NUM_JK, return_output=True)
        assert not np.allclose(cov_auto, cov_cross)

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        with pytest.raises(ValueError, match="corr_type"):
            obj.measure_covariance_multiple_datasets(
                corr_types=["bad_type"],
                dataset_names=["ds_A"],
                num_box=NUM_JK,
                return_output=True,
            )

    def test_three_datasets_raises(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        with pytest.raises(KeyError):
            obj.measure_covariance_multiple_datasets(
                corr_types=["w_g_plus", "w_g_plus", "w_g_plus"],
                dataset_names=["ds_A", "ds_B", "ds_A"],
                num_box=NUM_JK,
                return_output=True,
            )

    def test_multipoles_auto_covariance(self, IA_mock_TNG300_n1, tmp_path):
        """Verify the method works with multipoles_g_plus corr_type too."""
        obj = IA_mock_TNG300_n1
        tp  = str(tmp_path) + "/"
        obj.measure_xi_multipoles("mds_A", "g+", NUM_JK, temp_file_path=tp)
        cov, std = obj.measure_covariance_multiple_datasets(
            ["multipoles_g_plus"], ["mds_A"], NUM_JK, return_output=True)
        assert cov.shape == (obj.num_bins_r, obj.num_bins_r)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ===========================================================================
# 4. MeasureJackknife.create_full_cov_matrix_projections
# ===========================================================================

class TestCreateFullCovMatrixProjections:

    def _setup_three_projections(self, obj, tmp_path):
        tp = str(tmp_path) + "/"
        for los, name in enumerate(("LOS_x", "LOS_y", "LOS_z")):
            obj.data["LOS"] = los
            obj.measure_xi_w(name, "g+", NUM_JK, temp_file_path=tp)
        obj.data["LOS"] = 2

    def test_runs_without_error(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        self._setup_three_projections(obj, tmp_path)
        obj.create_full_cov_matrix_projections(
            corr_type=["w_g_plus", "w_g_plus", "w_g_plus"],
            dataset_names=["LOS_x", "LOS_y", "LOS_z"],
            num_box=NUM_JK,
        )

    def test_return_output_gives_four_matrices(self, IA_mock_TNG300_n1,
                                                tmp_path):
        obj    = IA_mock_TNG300_n1
        self._setup_three_projections(obj, tmp_path)
        result = obj.create_full_cov_matrix_projections(
            corr_type=["w_g_plus", "w_g_plus", "w_g_plus"],
            dataset_names=["LOS_x", "LOS_y", "LOS_z"],
            num_box=NUM_JK,
            return_output=True,
        )
        assert len(result) == 4
        n = obj.num_bins_r
        cov3, cov2xy, cov2xz, cov2yz = result
        # cov3 stacks all three projections; the others stack two each
        assert cov3.shape == (3 * n, 3 * n)
        for cov2 in (cov2xy, cov2xz, cov2yz):
            assert cov2.shape == (2 * n, 2 * n)

    def test_pairwise_blocks_match_individual_calls(self, IA_mock_TNG300_n1,
                                                     tmp_path):
        obj = IA_mock_TNG300_n1
        self._setup_three_projections(obj, tmp_path)
        cov_xyz, cov_xy, cov_xz, cov_yz = \
            obj.create_full_cov_matrix_projections(
                corr_type=["w_g_plus", "w_g_plus", "w_g_plus"],
                dataset_names=["LOS_x", "LOS_y", "LOS_z"],
                num_box=NUM_JK,
                return_output=True,
            )
        ref_xy, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["LOS_x", "LOS_y"],
            NUM_JK, return_output=True)
        ref_xz, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["LOS_x", "LOS_z"],
            NUM_JK, return_output=True)
        ref_yz, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["LOS_y", "LOS_z"],
            NUM_JK, return_output=True)

        # The 2-projection matrices stack [[auto_a, cross], [cross.T, auto_b]];
        # the cross covariance sits in the upper-right block.
        n = obj.num_bins_r
        np.testing.assert_allclose(cov_xy[:n, n:], ref_xy, rtol=1e-10)
        np.testing.assert_allclose(cov_xz[:n, n:], ref_xz, rtol=1e-10)
        np.testing.assert_allclose(cov_yz[:n, n:], ref_yz, rtol=1e-10)

    def test_all_blocks_symmetric(self, IA_mock_TNG300_n1, tmp_path):
        obj    = IA_mock_TNG300_n1
        self._setup_three_projections(obj, tmp_path)
        result = obj.create_full_cov_matrix_projections(
            corr_type=["w_g_plus", "w_g_plus", "w_g_plus"],
            dataset_names=["LOS_x", "LOS_y", "LOS_z"],
            num_box=NUM_JK,
            return_output=True,
        )
        for cov in result:
            np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        with pytest.raises(ValueError, match="corr_type"):
            obj.create_full_cov_matrix_projections(
                corr_type=["bad_type", "bad_type", "bad_type"],
                dataset_names=["LOS_x", "LOS_y", "LOS_z"],
                num_box=NUM_JK,
            )
