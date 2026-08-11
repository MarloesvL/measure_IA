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
  2b. ReadData multi-file readers against synthetic TNG-layout file sets:
               read_subhalo / read_snapshot / read_snapshot_multiple /
               read_modelling_outputs
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
  5.  MeasureJackknife.assign_jackknife_patches
        - output structure, label range, every patch populated
        - seed determinism and global random-state restoration
        - invalid num_jk raises ValueError
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
# 2b. ReadData: multi-file simulation readers
# ===========================================================================

class TestReadDataSimulationFiles:
    """read_subhalo / read_snapshot / read_snapshot_multiple against synthetic
    multi-file HDF5 sets laid out like the TNG group-catalogue and snapshot
    files (``<folder>.<n>.hdf5``, one group per particle/subhalo type).

    The real ``N_files`` for TNG300 is 600; the fixtures write three files and
    override ``N_files`` so the readers walk exactly the files that exist.
    """

    SNAP = 99
    NFILES = 3

    def _write_file_set(self, tmp_path, folder, group, datasets, nfiles=NFILES,
                        skip_in=()):
        """Write ``nfiles`` HDF5 files, splitting each dataset row-wise.

        Parameters
        ----------
        datasets : dict
            name -> full array, split into ``nfiles`` equal chunks.
        skip_in : iterable of (file index, dataset name)
            Datasets deliberately omitted from a file, to exercise the
            "problem at file n" skip branch.

        Returns the dict of full arrays for comparison.
        """
        import os
        base = str(tmp_path) + "/" + folder            # folder starts with "/"
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for n in range(nfiles):
            with h5py.File(f"{base}.{n}.hdf5", "w") as f:
                grp = f.create_group(group)
                for name, full in datasets.items():
                    if (n, name) in skip_in:
                        continue
                    chunk = np.array_split(full, nfiles)[n]
                    grp.create_dataset(name, data=chunk)
        return datasets

    def _reader(self, tmp_path, catalogue, output_file_name=None):
        r = ReadData(
            simulation="TNG300",
            catalogue=catalogue,
            snapshot=self.SNAP,
            output_file_name=output_file_name,
            data_path=str(tmp_path),
        )
        r.N_files = self.NFILES
        return r

    # -----------------------------------------------------------------------
    # read_subhalo
    # -----------------------------------------------------------------------

    def test_read_subhalo_concatenates_1d(self, tmp_path):
        reader = self._reader(tmp_path, "Subhalo")
        full = np.arange(30, dtype=float)
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Mass": full})
        np.testing.assert_array_equal(reader.read_subhalo("Mass"), full)

    def test_read_subhalo_stacks_2d(self, tmp_path):
        reader = self._reader(tmp_path, "Subhalo")
        full = np.arange(30, dtype=float).reshape(10, 3)
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Pos": full})
        np.testing.assert_array_equal(reader.read_subhalo("Pos"), full)

    def test_read_subhalo_nfiles_argument_limits_files(self, tmp_path):
        """Nfiles=2 reads only the first two of the three written files."""
        reader = self._reader(tmp_path, "Subhalo")
        full = np.arange(30, dtype=float)
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Mass": full})
        expected = np.concatenate(np.array_split(full, self.NFILES)[:2])
        np.testing.assert_array_equal(reader.read_subhalo("Mass", Nfiles=2),
                                      expected)

    def test_read_subhalo_unknown_dataset_raises_with_options(self, tmp_path):
        reader = self._reader(tmp_path, "Subhalo")
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Mass": np.arange(30, dtype=float)})
        with pytest.raises(KeyError, match="Mass"):     # lists what is available
            reader.read_subhalo("NotThere")

    def test_read_subhalo_missing_file_raises_oserror(self, tmp_path):
        """Only file 0 exists but N_files says 3 -> clear OSError, not a raw
        h5py error."""
        reader = self._reader(tmp_path, "Subhalo")
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Mass": np.arange(30, dtype=float)}, nfiles=1)
        with pytest.raises(OSError, match="Could not open file 1"):
            reader.read_subhalo("Mass")

    def test_read_subhalo_skips_file_missing_the_dataset(self, tmp_path):
        """A file lacking the dataset is skipped, the rest still concatenate."""
        reader = self._reader(tmp_path, "Subhalo")
        full = np.arange(30, dtype=float)
        self._write_file_set(tmp_path, reader.fof_folder, "Subhalo",
                             {"Mass": full}, skip_in=[(1, "Mass")])
        chunks = np.array_split(full, self.NFILES)
        expected = np.concatenate([chunks[0], chunks[2]])
        np.testing.assert_array_equal(reader.read_subhalo("Mass"), expected)

    # -----------------------------------------------------------------------
    # read_snapshot
    # -----------------------------------------------------------------------

    def test_read_snapshot_concatenates_1d(self, tmp_path):
        reader = self._reader(tmp_path, "PartType4")
        full = np.arange(30, dtype=float)
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Mass": full})
        np.testing.assert_array_equal(reader.read_snapshot("Mass"), full)

    def test_read_snapshot_stacks_2d(self, tmp_path):
        reader = self._reader(tmp_path, "PartType4")
        full = np.arange(30, dtype=float).reshape(10, 3)
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Coordinates": full})
        np.testing.assert_array_equal(reader.read_snapshot("Coordinates"), full)

    def test_read_snapshot_writes_to_output_file(self, tmp_path):
        """With output_file_name set the data is streamed to the output file
        (resizable dataset) and nothing is returned."""
        out = str(tmp_path / "snap_out.hdf5")
        reader = self._reader(tmp_path, "PartType4", output_file_name=out)
        full = np.arange(30, dtype=float).reshape(10, 3)
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Coordinates": full})

        assert reader.read_snapshot("Coordinates") is None
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(
                f[f"Snapshot_{self.SNAP}/Coordinates"][:], full)

    def test_read_snapshot_output_file_overwrites_existing_dataset(self, tmp_path):
        """Re-reading into the same output file replaces, not appends."""
        out = str(tmp_path / "snap_out_twice.hdf5")
        reader = self._reader(tmp_path, "PartType4", output_file_name=out)
        full = np.arange(30, dtype=float)
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Mass": full})

        reader.read_snapshot("Mass")
        reader.read_snapshot("Mass")
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(
                f[f"Snapshot_{self.SNAP}/Mass"][:], full)

    def test_read_snapshot_unknown_dataset_raises_with_options(self, tmp_path):
        reader = self._reader(tmp_path, "PartType4")
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Mass": np.arange(30, dtype=float)})
        with pytest.raises(KeyError, match="Mass"):
            reader.read_snapshot("NotThere")

    # -----------------------------------------------------------------------
    # read_snapshot_multiple
    # -----------------------------------------------------------------------

    def test_read_snapshot_multiple_writes_all_datasets(self, tmp_path):
        out = str(tmp_path / "snap_multi.hdf5")
        reader = self._reader(tmp_path, "PartType4", output_file_name=out)
        mass = np.arange(30, dtype=float)
        pos = np.arange(90, dtype=float).reshape(30, 3)
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Mass": mass, "Coordinates": pos})

        assert reader.read_snapshot_multiple(["Mass", "Coordinates"]) is None
        with h5py.File(out, "r") as f:
            grp = f[f"Snapshot_{self.SNAP}"]
            np.testing.assert_array_equal(grp["Mass"][:], mass)
            np.testing.assert_array_equal(grp["Coordinates"][:], pos)

    def test_read_snapshot_multiple_unknown_dataset_raises_with_options(
            self, tmp_path):
        reader = self._reader(tmp_path, "PartType4")
        self._write_file_set(tmp_path, reader.snap_folder, "PartType4",
                             {"Mass": np.arange(30, dtype=float)})
        with pytest.raises(KeyError, match="NotThere"):
            reader.read_snapshot_multiple(["Mass", "NotThere"])

    # -----------------------------------------------------------------------
    # read_modelling_outputs
    # -----------------------------------------------------------------------

    def _write_modelling_file(self, tmp_path, name, groups, snap_group=None,
                              z=None):
        p = str(tmp_path / f"{name}.hdf5")
        with h5py.File(p, "w") as f:
            parent = f
            if snap_group is not None:
                parent = f.create_group(snap_group)
                parent.attrs["z"] = z
            for gname, attrs in groups.items():
                grp = parent.create_group(gname)
                for k, v in attrs.items():
                    grp.attrs[k] = v
        return p

    _FIT = {"A_IA": 1.5, "A_IA_err": 0.2, "b_g": 2.0, "b_g_err": 0.1}

    def test_read_modelling_outputs_populates_both_groups(self, tmp_path):
        reader = ReadData("TNG300", "unused", None, data_path=str(tmp_path))
        self._write_modelling_file(tmp_path, "fits",
                                   {"w": self._FIT, "multipoles": self._FIT})
        reader.read_modelling_outputs("fits")

        assert reader.w_A_IA == pytest.approx(1.5)
        assert reader.w_A_IA_err == pytest.approx(0.2)
        assert reader.w_b_g == pytest.approx(2.0)
        assert reader.w_b_g_err == pytest.approx(0.1)
        assert reader.multipoles_A_IA == pytest.approx(1.5)
        assert reader.multipoles_b_g_err == pytest.approx(0.1)

    def test_read_modelling_outputs_missing_group_is_skipped(self, tmp_path):
        reader = ReadData("TNG300", "unused", None, data_path=str(tmp_path))
        self._write_modelling_file(tmp_path, "fits_w_only", {"w": self._FIT})
        reader.read_modelling_outputs("fits_w_only")

        assert reader.w_A_IA == pytest.approx(1.5)
        assert not hasattr(reader, "multipoles_A_IA")

    def test_read_modelling_outputs_snapshot_group_reads_redshift(self, tmp_path):
        reader = ReadData("TNG300", "unused", 99, data_path=str(tmp_path))
        self._write_modelling_file(tmp_path, "fits_snap", {"w": self._FIT},
                                   snap_group="Snapshot_99", z=0.3)
        reader.read_modelling_outputs("fits_snap")

        assert reader.z == pytest.approx(0.3)
        assert reader.w_A_IA == pytest.approx(1.5)


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


# ===========================================================================
# 5. MeasureJackknife.assign_jackknife_patches
# ===========================================================================

class TestAssignJackknifePatches:
    """Unit tests for the spherical-k-means patch assignment: output
    structure, label ranges, seed determinism, and the geometric sanity of the
    assignment (nearby objects land in the same patch)."""

    NUM_JK = 6

    def _catalogs(self, N=200, NR=400, seed=7):
        rng = np.random.default_rng(seed)
        data = {
            "RA":  rng.uniform(150, 160, N),
            "DEC": rng.uniform(0, 8, N),
            "RA_shape_sample":  rng.uniform(150, 160, N),
            "DEC_shape_sample": rng.uniform(0, 8, N),
        }
        randoms = {
            "RA":  rng.uniform(150, 160, NR),
            "DEC": rng.uniform(0, 8, NR),
            "RA_shape_sample":  rng.uniform(150, 160, NR),
            "DEC_shape_sample": rng.uniform(0, 8, NR),
        }
        return data, randoms

    def _obj(self, tmp_path):
        from measureia import MeasureIALightcone
        data, randoms = self._catalogs()
        full = dict(data)
        full.update({
            "Redshift": np.full(len(data["RA"]), 0.2),
            "Redshift_shape_sample": np.full(len(data["RA"]), 0.2),
            "e1": np.zeros(len(data["RA"])),
            "e2": np.zeros(len(data["RA"])),
        })
        full_rand = dict(randoms)
        full_rand.update({
            "Redshift": np.full(len(randoms["RA"]), 0.2),
            "Redshift_shape_sample": np.full(len(randoms["RA"]), 0.2),
        })
        return MeasureIALightcone(full, full_rand,
                                  str(tmp_path / "jkp.hdf5"), pi_max=60.0)

    def test_returns_all_four_samples_with_matching_lengths(self, tmp_path):
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        p = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=1)

        assert set(p) == {"position", "shape",
                          "randoms_position", "randoms_shape"}
        assert len(p["position"])         == len(data["RA"])
        assert len(p["shape"])            == len(data["RA_shape_sample"])
        assert len(p["randoms_position"]) == len(randoms["RA"])
        assert len(p["randoms_shape"])    == len(randoms["RA_shape_sample"])

    def test_labels_are_zero_based_and_within_range(self, tmp_path):
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        p = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=1)
        for labels in p.values():
            assert labels.min() >= 0
            assert labels.max() < self.NUM_JK

    def test_every_patch_is_populated(self, tmp_path):
        """kmeans on a well-sampled random catalogue must fill all patches —
        an empty patch would produce a degenerate delete-one realisation."""
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        p = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=1)
        assert len(np.unique(p["randoms_position"])) == self.NUM_JK

    def test_seed_makes_assignment_reproducible(self, tmp_path):
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        a = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=123)
        b = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=123)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_different_seeds_give_different_assignment(self, tmp_path):
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        a = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=1)
        b = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=999)
        assert any(not np.array_equal(a[k], b[k]) for k in a)

    def test_global_random_state_is_untouched(self, tmp_path):
        """The fit draws from its own Generator, so neither the global numpy
        state nor the stdlib random state may move."""
        import random as _random
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()

        np.random.seed(5)
        _random.seed(5)
        np_before, py_before = np.random.random(), _random.random()

        np.random.seed(5)
        _random.seed(5)
        obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=42)
        assert np.random.random() == np_before
        assert _random.random() == py_before

    def test_identical_coordinates_share_a_patch(self, tmp_path):
        """Data objects placed exactly on random positions inherit those
        randoms' patch labels (find_nearest is a nearest-centre lookup)."""
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        data = dict(data)
        n = len(data["RA"])
        data["RA"], data["DEC"] = randoms["RA"][:n], randoms["DEC"][:n]

        p = obj.assign_jackknife_patches(data, randoms, self.NUM_JK, seed=1)
        np.testing.assert_array_equal(p["position"],
                                      p["randoms_position"][:n])

    @pytest.mark.parametrize("bad", [0, -3, 2.5, True, None])
    def test_invalid_num_jk_raises(self, tmp_path, bad):
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs()
        with pytest.raises(ValueError, match="num_jk must be an integer"):
            obj.assign_jackknife_patches(data, randoms, bad)

    def test_more_patches_than_randoms_raises(self, tmp_path):
        """The centres are fitted to the position randoms, so there must be at
        least one random per patch."""
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs(N=20, NR=50)
        with pytest.raises(ValueError,
                           match="cannot exceed the number of position randoms"):
            obj.assign_jackknife_patches(data, randoms, 51, seed=1)

    def test_few_randoms_per_patch_is_allowed(self, tmp_path):
        """Sparse randoms are fine: unlike the old kmeans_radec backend, which
        drew 10 points per patch without replacement, the fit only needs one
        random per patch."""
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs(N=20, NR=50)
        p = obj.assign_jackknife_patches(data, randoms, 6, seed=1)
        assert len(p["randoms_position"]) == 50
        assert p["randoms_position"].max() < 6

    def test_more_patches_than_data_objects_still_assigns(self, tmp_path):
        """num_jk may exceed the number of *data* objects — the patches are
        built from the randoms, and each datum simply falls in the nearest
        patch. Some patches then hold no data, which is a valid (if noisy)
        delete-one setup."""
        obj = self._obj(tmp_path)
        data, randoms = self._catalogs(N=5, NR=200)
        p = obj.assign_jackknife_patches(data, randoms, 10, seed=1)
        assert len(p["position"]) == 5
        assert len(np.unique(p["randoms_position"])) == 10


# ===========================================================================
# 6. Covariance-combining utilities on the lightcone
# ===========================================================================

class TestCovarianceUtilitiesLightcone:
    """measure_covariance_multiple_datasets and create_full_cov_matrix_projections
    are inherited from MeasureJackknife and were only covered for the box. The
    lightcone writes its jackknife realisations into the same group layout, so
    both must work there too."""

    LC_NUM_JK = 4

    def _run_two(self, obj, tmp_path):
        """Two genuinely different datasets: the same catalogue measured with
        the two IA estimators, which use different random terms."""
        tp = str(tmp_path) + "/"
        obj.measure_xi_w("galaxies", "lc_A", "g+", num_jk=self.LC_NUM_JK,
                         measure_cov=True, temp_file_path=tp)
        obj.measure_xi_w("clusters", "lc_B", "g+", num_jk=self.LC_NUM_JK,
                         measure_cov=True, temp_file_path=tp)

    def test_auto_covariance_shape_and_symmetry(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_two(obj, tmp_path)
        cov, std = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["lc_A"], self.LC_NUM_JK, return_output=True)

        n = obj.num_bins_r
        assert cov.shape == (n, n)
        assert std.shape == (n,)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        np.testing.assert_allclose(std, np.sqrt(np.diag(cov)), rtol=1e-12)

    def test_cross_covariance_differs_from_auto(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        self._run_two(obj, tmp_path)
        auto, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus"], ["lc_A"], self.LC_NUM_JK, return_output=True)
        cross, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["lc_A", "lc_B"], self.LC_NUM_JK,
            return_output=True)

        assert cross.shape == auto.shape
        assert not np.allclose(cross, auto, equal_nan=True)

    def test_invalid_corr_type_raises(self, IA_mock_lc_n1, tmp_path):
        obj = IA_mock_lc_n1
        with pytest.raises(ValueError, match="corr_type"):
            obj.measure_covariance_multiple_datasets(
                ["not_a_corr_type"], ["lc_A"], self.LC_NUM_JK)

    def test_full_cov_matrix_projections_blocks(self, IA_mock_lc_n1, tmp_path):
        """create_full_cov_matrix_projections always combines three datasets
        (it was written for the box LOS-x/y/z projections). The machinery is
        projection-agnostic, so on the lightcone it is exercised here with
        three distinct measurements; each returned block must be symmetric and
        the pairwise off-diagonal block must reproduce the direct two-dataset
        covariance."""
        obj = IA_mock_lc_n1
        self._run_two(obj, tmp_path)
        # third dataset: half the sample, so it differs from both of the above
        N = len(obj.data["RA"])
        half = np.arange(N) % 2 == 0
        obj.measure_xi_w("galaxies", "lc_C", "g+", num_jk=self.LC_NUM_JK,
                         measure_cov=True, masks={k: half for k in obj.data},
                         temp_file_path=str(tmp_path) + "/")

        ref, _ = obj.measure_covariance_multiple_datasets(
            ["w_g_plus", "w_g_plus"], ["lc_A", "lc_B"], self.LC_NUM_JK,
            return_output=True)
        result = obj.create_full_cov_matrix_projections(
            corr_type=["w_g_plus", "w_g_plus", "w_g_plus"],
            dataset_names=["lc_A", "lc_B", "lc_C"],
            num_box=self.LC_NUM_JK,
            return_output=True)

        n = obj.num_bins_r
        cov_ABC, cov_AB, cov_AC, cov_BC = result
        assert cov_ABC.shape == (3 * n, 3 * n)
        for cov in (cov_AB, cov_AC, cov_BC):
            assert cov.shape == (2 * n, 2 * n)
            np.testing.assert_allclose(cov, cov.T, atol=1e-12,
                                       equal_nan=True)
        # cov_AB stacks [[auto_A, cross_AB], [cross_AB.T, auto_B]]
        assert np.isfinite(ref).any(), "reference covariance is entirely NaN"
        np.testing.assert_allclose(cov_AB[:n, n:], ref, rtol=1e-10,
                                   equal_nan=True)
