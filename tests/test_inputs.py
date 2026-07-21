"""
test_inputs.py
==============
Tests for the CheckInput class and the custom input-key-name (remap-at-entry)
support in MeasureIABox and MeasureIALightcone.

Ported from the April limit_input_options branch (1639c90) onto the current
suite: the original try/except tests are rewritten with pytest.raises /
pytest.warns, and equivalence tests are added that verify custom key names
produce bit-identical results to the default names.
"""

import math
import numpy as np
import pytest
import h5py
import warnings

from measureia import MeasureIABox, MeasureIALightcone, CheckInput


BOXSIZE = 205.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_catalog(N=60, seed=11):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, N)
    return {
        "Position":              rng.uniform(0.0, BOXSIZE, (N, 3)),
        "Position_shape_sample": rng.uniform(0.0, BOXSIZE, (N, 3)),
        "Axis_Direction":        np.column_stack([np.cos(theta), np.sin(theta)]),
        "LOS":                   2,
        "q":                     rng.uniform(0.1, 1.0, N),
    }


_BOX_RENAME = {
    "Position": "COM",
    "Position_shape_sample": "COM_shape",
    "Axis_Direction": "e_dir",
    "q": "axis_ratio",
    "LOS": "los_index",
}

_BOX_NAME_KWARGS = {
    "positions_density_sample_name": "COM",
    "positions_shape_sample_name": "COM_shape",
    "axis_direction_name": "e_dir",
    "axis_ratio_name": "axis_ratio",
    "line_of_sight_index_name": "los_index",
}


def _lc_catalogs(N=80, NR=240, seed=13):
    rng = np.random.default_rng(seed)

    def sky(n):
        return {
            "RA":                    rng.uniform(150.0, 155.0, n),
            "DEC":                   rng.uniform(2.0, 6.0, n),
            "Redshift":              rng.uniform(0.1, 0.3, n),
            "RA_shape_sample":       rng.uniform(150.0, 155.0, n),
            "DEC_shape_sample":      rng.uniform(2.0, 6.0, n),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, n),
        }

    data = sky(N)
    data["e1"] = rng.uniform(-0.5, 0.5, N)
    data["e2"] = rng.uniform(-0.5, 0.5, N)
    randoms = sky(NR)
    return data, randoms


_LC_RENAME = {
    "RA": "ra_d", "RA_shape_sample": "ra_s",
    "DEC": "dec_d", "DEC_shape_sample": "dec_s",
    "Redshift": "z_d", "Redshift_shape_sample": "z_s",
    "e1": "eps1", "e2": "eps2",
}

_LC_NAME_KWARGS = {
    "RA_density_sample_name": "ra_d", "RA_shape_sample_name": "ra_s",
    "DEC_density_sample_name": "dec_d", "DEC_shape_sample_name": "dec_s",
    "redshift_density_sample_name": "z_d", "redshift_shape_sample_name": "z_s",
    "e1_name": "eps1", "e2_name": "eps2",
}


def _renamed(cat, mapping):
    return {mapping.get(k, k): v for k, v in cat.items()}


# ---------------------------------------------------------------------------
# 1. CheckInput unit tests
# ---------------------------------------------------------------------------

class TestCheckInput:
    def test_check_dict_raises_on_missing_key(self):
        with pytest.raises(KeyError, match="does not contain q"):
            CheckInput.check_dict({"Position": 1}, ["Position", "q"])

    def test_check_paths_missing_folder(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            CheckInput.check_paths(["/nonexistent_folder_xyz/out.hdf5"])

    def test_check_paths_bare_filename_ok(self):
        CheckInput.check_paths(["out.hdf5"])  # dirname == "" -> cwd, must not raise

    def test_check_units_coordinates(self):
        coords = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="do not agree with the boxsize"):
            CheckInput.check_units_coordinates(coords, 1.5)
        CheckInput.check_units_coordinates(coords, 2.5)  # in range, must not raise

    def test_check_type_input_data_rejects_bad_los(self):
        cat = _box_catalog(N=5)
        cat["LOS"] = 5
        with pytest.raises(ValueError, match="must be 0, 1 or 2"):
            CheckInput.check_type_input_data(
                cat, ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"))

    def test_check_type_input_data_accepts_numpy_integer_los(self):
        # np.integer LOS (e.g. np.int64(2)) used to fail the strict `type(...) == int` assert
        cat = _box_catalog(N=5)
        cat["LOS"] = np.int64(2)
        CheckInput.check_type_input_data(
            cat, ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"))

    def test_check_type_input_data_rejects_bad_shape(self):
        cat = _box_catalog(N=5)
        cat["Position"] = cat["Position"][:, :2]
        with pytest.raises(ValueError, match="must have shape"):
            CheckInput.check_type_input_data(
                cat, ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"))

    def test_check_type_input_data_rejects_length_mismatch(self):
        cat = _box_catalog(N=5)
        cat["q"] = cat["q"][:-1]  # shape-sample-aligned array with wrong length
        with pytest.raises(ValueError, match="must match"):
            CheckInput.check_type_input_data(
                cat, ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"))

    def test_check_type_input_data_rejects_non_finite(self):
        cat = _box_catalog(N=5)
        cat["Position"] = cat["Position"].copy()
        cat["Position"][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            CheckInput.check_type_input_data(
                cat, ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"))

    def test_jackknife_warning_max_separation(self):
        with pytest.warns(UserWarning, match="maximum separation exceeds"):
            CheckInput.check_jackknife_max_separation(64, 200, 60, 10)

    def test_jackknife_warning_num_r_bins(self):
        with pytest.warns(UserWarning, match="too many r"):
            CheckInput.check_jackknife_max_separation(64, 200, 40, 15)

    def test_jackknife_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CheckInput.check_jackknife_max_separation(8, 200, 20, 8)

    def test_rename_input_keys(self):
        renamed = CheckInput.rename_input_keys(
            {"COM": 1, "extra": 2}, {"COM": "Position"})
        assert renamed == {"Position": 1, "extra": 2}

    def test_rename_input_keys_none(self):
        assert CheckInput.rename_input_keys(None, {"COM": "Position"}) is None


# ---------------------------------------------------------------------------
# 2. MeasureIABox wiring
# ---------------------------------------------------------------------------

class TestBoxInputChecks:
    def test_missing_key_raises(self, tmp_path):
        cat = _box_catalog()
        del cat["Axis_Direction"]
        with pytest.raises(KeyError, match="does not contain Axis_Direction"):
            MeasureIABox(cat, str(tmp_path / "o.hdf5"), boxsize=BOXSIZE)

    def test_custom_name_missing_under_default_raises(self, tmp_path):
        cat = _renamed(_box_catalog(), _BOX_RENAME)
        with pytest.raises(KeyError, match="does not contain Position"):
            MeasureIABox(cat, str(tmp_path / "o.hdf5"), boxsize=BOXSIZE)

    def test_coordinates_outside_boxsize_raise(self, tmp_path):
        cat = _box_catalog()
        with pytest.raises(ValueError, match="do not agree with the boxsize"):
            MeasureIABox(cat, str(tmp_path / "o.hdf5"), boxsize=BOXSIZE / 10.0)

    def test_output_folder_must_exist(self, tmp_path):
        cat = _box_catalog()
        with pytest.raises(FileNotFoundError, match="does not exist"):
            MeasureIABox(cat, str(tmp_path / "missing_dir" / "o.hdf5"), boxsize=BOXSIZE)

    def test_jackknife_separation_warning(self, tmp_path):
        cat = _box_catalog()
        obj = MeasureIABox(cat, str(tmp_path / "o.hdf5"), boxsize=BOXSIZE,
                           separation_limits=[0.1, 60.0])
        with pytest.warns(UserWarning, match="maximum separation exceeds"):
            obj.measure_xi_w("warn_jk", "g+", num_jk=64, temp_file_path=False)

    def test_custom_names_bit_identical(self, tmp_path):
        cat = _box_catalog()
        obj_def = MeasureIABox(cat, str(tmp_path / "def.hdf5"), boxsize=BOXSIZE)
        obj_def.measure_xi_w("t", "both", temp_file_path=False)

        obj_cus = MeasureIABox(_renamed(cat, _BOX_RENAME), str(tmp_path / "cus.hdf5"),
                               boxsize=BOXSIZE, **_BOX_NAME_KWARGS)
        obj_cus.measure_xi_w("t", "both", temp_file_path=False)

        with h5py.File(tmp_path / "def.hdf5") as fd, h5py.File(tmp_path / "cus.hdf5") as fc:
            for ds in ("w_g_plus/t", "w_gg/t"):
                assert np.array_equal(fd[ds][:], fc[ds][:])

    def test_custom_named_masks_bit_identical(self, tmp_path):
        cat = _box_catalog()
        N = len(cat["q"])
        mask = np.zeros(N, dtype=bool)
        mask[: N // 2] = True
        masks_def = {"Position": mask, "Position_shape_sample": mask,
                     "Axis_Direction": mask, "q": mask}
        obj_def = MeasureIABox(cat, str(tmp_path / "def.hdf5"), boxsize=BOXSIZE)
        obj_def.measure_xi_w("t", "g+", temp_file_path=False, masks=masks_def)

        obj_cus = MeasureIABox(_renamed(cat, _BOX_RENAME), str(tmp_path / "cus.hdf5"),
                               boxsize=BOXSIZE, **_BOX_NAME_KWARGS)
        obj_cus.measure_xi_w("t", "g+", temp_file_path=False,
                             masks=_renamed(masks_def, _BOX_RENAME))

        with h5py.File(tmp_path / "def.hdf5") as fd, h5py.File(tmp_path / "cus.hdf5") as fc:
            assert np.array_equal(fd["w_g_plus/t"][:], fc["w_g_plus/t"][:])

    def test_user_dict_not_mutated(self, tmp_path):
        cat = _box_catalog()
        keys_before = set(cat.keys())
        MeasureIABox(cat, str(tmp_path / "o.hdf5"), boxsize=BOXSIZE)
        assert set(cat.keys()) == keys_before  # default weights go to the remapped copy


# ---------------------------------------------------------------------------
# 3. MeasureIALightcone wiring
# ---------------------------------------------------------------------------

class TestLightconeInputChecks:
    def test_missing_key_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        del data["e2"]
        with pytest.raises(KeyError, match="does not contain e2"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_randoms_missing_key_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        del randoms["RA"]
        with pytest.raises(KeyError, match="does not contain RA"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_bad_dec_range_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        data["DEC"] = data["DEC"].copy()
        data["DEC"][0] = 200.0
        with pytest.raises(ValueError, match=r"must be in \[-90, 90\]"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_bad_ra_range_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        data["RA"] = data["RA"].copy()
        data["RA"][0] = 400.0
        with pytest.raises(ValueError, match=r"must be in \[0, 360\]"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_non_finite_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        data["e1"] = data["e1"].copy()
        data["e1"][0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_shape_sample_length_mismatch_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        data["e2"] = data["e2"][:-1]  # shape-sample-aligned array, wrong length
        with pytest.raises(ValueError, match="must match"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60)

    def test_custom_name_missing_under_default_raises(self, tmp_path):
        data, randoms = _lc_catalogs()
        with pytest.raises(KeyError, match="does not contain ra_d"):
            MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"), pi_max=60,
                               **_LC_NAME_KWARGS)

    def test_custom_names_bit_identical(self, tmp_path):
        data, randoms = _lc_catalogs()
        kwargs = dict(separation_limits=[2.0, 20.0], num_bins_r=5, num_bins_pi=10,
                      pi_max=60)
        obj_def = MeasureIALightcone(dict(data), dict(randoms),
                                     str(tmp_path / "def.hdf5"), **kwargs)
        obj_def.measure_xi_w("galaxies", "t", "both", measure_cov=False)

        obj_cus = MeasureIALightcone(_renamed(data, _LC_RENAME),
                                     _renamed(randoms, _LC_RENAME),
                                     str(tmp_path / "cus.hdf5"), **kwargs,
                                     **_LC_NAME_KWARGS)
        obj_cus.measure_xi_w("galaxies", "t", "both", measure_cov=False)

        with h5py.File(tmp_path / "def.hdf5") as fd, h5py.File(tmp_path / "cus.hdf5") as fc:
            for ds in ("w_g_plus/t", "w_gg/t"):
                assert np.array_equal(fd[ds][:], fc[ds][:])

    def test_custom_names_jk_cov_bit_identical(self, tmp_path):
        data, randoms = _lc_catalogs(N=120, NR=360)
        kwargs = dict(separation_limits=[2.0, 20.0], num_bins_r=5, num_bins_pi=10,
                      pi_max=60)
        obj_def = MeasureIALightcone(dict(data), dict(randoms),
                                     str(tmp_path / "def.hdf5"), **kwargs)
        obj_def.measure_xi_w("galaxies", "t", "g+", num_jk=4, seed=3)

        obj_cus = MeasureIALightcone(_renamed(data, _LC_RENAME),
                                     _renamed(randoms, _LC_RENAME),
                                     str(tmp_path / "cus.hdf5"), **kwargs,
                                     **_LC_NAME_KWARGS)
        obj_cus.measure_xi_w("galaxies", "t", "g+", num_jk=4, seed=3)

        with h5py.File(tmp_path / "def.hdf5") as fd, h5py.File(tmp_path / "cus.hdf5") as fc:
            assert np.array_equal(fd["w_g_plus/t_jackknife_cov_4"][:],
                                  fc["w_g_plus/t_jackknife_cov_4"][:], equal_nan=True)

    def test_user_dicts_not_mutated(self, tmp_path):
        data, randoms = _lc_catalogs()
        data_keys = set(data.keys())
        randoms_keys = set(randoms.keys())
        obj = MeasureIALightcone(data, randoms, str(tmp_path / "o.hdf5"),
                                 separation_limits=[2.0, 20.0], num_bins_r=5,
                                 num_bins_pi=10, pi_max=60)
        obj.measure_xi_w("galaxies", "t", "gg", measure_cov=False)
        assert set(data.keys()) == data_keys  # weight defaults go to the remapped copies
        assert set(randoms.keys()) == randoms_keys
