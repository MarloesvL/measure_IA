"""
test_box_shape_shape.py
=======================
The public shape-shape ('++' / 'all') path on MeasureIABox: option validation,
the products that land in the output file, backend equality (including
multiprocessing), jackknife, and the analytic limits.

The kernel itself is covered against O(N^2) references in
test_pair_kernel_shape_shape.py; this module covers the chain above it --
dispatch, responsivity, RR, reduction and HDF5 layout.

Sections
--------
  1. corr_type validation and the density-sample-shapes requirement
  2. Output layout: which products appear for which corr_type
  3. Backend equality (brute / tree / multiprocessing)
  4. Jackknife
  5. Analytic limits
"""

import numpy as np
import pytest
import h5py

from measureia import MeasureIABox
from measureia.mocks import radial_alignment_box_mock


BOXSIZE = 205.0
_SEP = [0.5, 20.0]
_NR = 6
_NPI = 10
NUM_JK = 8


def _catalogue(seed=5, n_centrals=200, n_sat=6, density_shapes=True):
    m = radial_alignment_box_mock(n_centrals=n_centrals, n_sat=n_sat,
                                  boxsize=BOXSIZE, seed=seed)
    data = {
        "Position": m["Position"],
        "Position_shape_sample": m["Position_shape_sample"],
        "Axis_Direction": m["Axis_Direction"],
        "q": m["q"],
        "LOS": 2,
    }
    if density_shapes:
        rng = np.random.default_rng(seed + 1)
        n = len(m["Position"])
        th = rng.uniform(0, np.pi, n)
        data["Axis_Direction_density_sample"] = np.column_stack([np.cos(th), np.sin(th)])
        data["q_density_sample"] = rng.uniform(0.3, 0.9, n)
    return data


def _obj(data, tmp_path, name="ss.hdf5", num_nodes=1):
    return MeasureIABox(data, str(tmp_path / name), simulation="TNG300", snapshot=99,
                        separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI,
                        num_nodes=num_nodes)


def _products(obj, prefix):
    with h5py.File(obj.output_file_name, "r") as f:
        g = f[obj.snap_group]
        return {k: g[k]["t"][:] for k in g.keys()
                if k.startswith(prefix) and "t" in g[k]}


_SS_W = {"w_plus_plus", "w_cross_cross", "w_plus_cross"}
_GP_W = {"w_g_plus", "w_gg"}


# ===========================================================================
# 1. Option validation
# ===========================================================================

class TestCorrTypeValidation:

    def test_shape_shape_accepted(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "++", 0, temp_file_path=False)

    def test_all_accepted(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "all", 0, temp_file_path=False)

    def test_unknown_corr_type_lists_the_new_options(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        with pytest.raises(ValueError, match=r"\+\+"):
            obj.measure_xi_w("t", "gx", 0, temp_file_path=False)

    @pytest.mark.parametrize("corr_type", ["++", "all"])
    def test_refused_without_density_sample_shapes(self, tmp_path, corr_type):
        """Refused up front, not after a full pair count."""
        obj = _obj(_catalogue(density_shapes=False), tmp_path)
        with pytest.raises(ValueError, match="density sample needs shapes"):
            obj.measure_xi_w("t", corr_type, 0, temp_file_path=False)

    @pytest.mark.parametrize("corr_type", ["g+", "gg", "both"])
    def test_existing_corr_types_do_not_require_them(self, tmp_path, corr_type):
        obj = _obj(_catalogue(density_shapes=False), tmp_path)
        obj.measure_xi_w("t", corr_type, 0, temp_file_path=False)


# ===========================================================================
# 2. Output layout
# ===========================================================================

class TestOutputLayout:

    def test_shape_shape_products_written(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "++", 0, temp_file_path=False)
        assert _SS_W <= set(_products(obj, "w_"))

    def test_both_still_means_g_plus_and_gg(self, tmp_path):
        """Back-compat: 'both' must not start producing shape-shape output."""
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "both", 0, temp_file_path=False)
        got = set(_products(obj, "w_"))
        assert got == _GP_W, got

    def test_all_is_everything(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "all", 0, temp_file_path=False)
        assert set(_products(obj, "w_")) == _GP_W | _SS_W

    def test_multipoles_products(self, tmp_path):
        """Singh et al. give no convention for a cross-cross multipole, so only
        the plus-plus one is produced (see M_PRODUCTS)."""
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_multipoles("t", "all", 0, temp_file_path=False)
        got = set(_products(obj, "multipoles_"))
        assert got == {"multipoles_g_plus", "multipoles_gg", "multipoles_plus_plus"}

    def test_raw_grids_and_bins_are_written(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "++", 0, temp_file_path=False)
        with h5py.File(obj.output_file_name, "r") as f:
            g = f[obj.snap_group + "w/xi_plus_plus"]
            for suffix in ("", "_SplusSplus", "_RR", "_rp", "_pi"):
                assert f"t{suffix}" in g, suffix
            assert g["t"].shape == (_NR, _NPI)


# ===========================================================================
# 3. Backend equality
# ===========================================================================

class TestBackendEquality:

    @staticmethod
    def _run(tmp_path, name, *, temp_file_path, num_nodes, method="measure_xi_w", num_jk=0):
        obj = _obj(_catalogue(), tmp_path, name=name, num_nodes=num_nodes)
        getattr(obj, method)("t", "all", num_jk, temp_file_path=temp_file_path)
        return _products(obj, "w_" if method == "measure_xi_w" else "multipoles_")

    def test_brute_matches_tree(self, tmp_path):
        brute = self._run(tmp_path, "b.hdf5", temp_file_path=False, num_nodes=1)
        tree = self._run(tmp_path, "t.hdf5", temp_file_path=str(tmp_path), num_nodes=1)
        assert set(brute) == set(tree)
        for k in brute:
            np.testing.assert_allclose(brute[k], tree[k], rtol=1e-12, atol=1e-15,
                                       err_msg=k)

    @pytest.mark.parametrize("method,num_jk",
                             [("measure_xi_w", 0), ("measure_xi_w", NUM_JK),
                              ("measure_xi_multipoles", 0), ("measure_xi_multipoles", NUM_JK)])
    def test_multiprocessing_matches_single_process(self, tmp_path, method, num_jk):
        """The shape-shape grids must survive the trip through the workers.

        The batch methods build their SampleSet by hand from shared memory, so
        the density sample's shapes and the three product grids each needed
        their own plumbing; nothing else in the suite would notice if a worker
        silently returned zeros.
        """
        one = self._run(tmp_path, f"s1_{method}_{num_jk}.hdf5",
                        temp_file_path=str(tmp_path), num_nodes=1,
                        method=method, num_jk=num_jk)
        many = self._run(tmp_path, f"s4_{method}_{num_jk}.hdf5",
                         temp_file_path=str(tmp_path), num_nodes=4,
                         method=method, num_jk=num_jk)
        assert set(one) == set(many)
        for k in one:
            scale = max(np.nanmax(np.abs(one[k])), 1e-300)
            assert np.nanmax(np.abs(one[k] - many[k])) / scale < 1e-10, k

    def test_products_are_not_all_zero(self, tmp_path):
        """Guard: the equality tests above would pass on all-zero output."""
        got = self._run(tmp_path, "nz.hdf5", temp_file_path=False, num_nodes=1)
        for k in _SS_W:
            assert np.nanmax(np.abs(got[k])) > 1e-8, k


# ===========================================================================
# 4. Jackknife
# ===========================================================================

class TestJackknife:

    def test_covariance_written_for_every_product(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "++", NUM_JK, temp_file_path=False)
        with h5py.File(obj.output_file_name, "r") as f:
            g = f[obj.snap_group]
            for name in _SS_W:
                assert f"t_jackknife_cov_{NUM_JK}" in g[name], name
                cov = g[name][f"t_jackknife_cov_{NUM_JK}"][:]
                assert cov.shape == (_NR, _NR)
                assert np.all(np.diag(cov) >= 0)

    def test_realisations_written(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_w("t", "++", NUM_JK, temp_file_path=False)
        with h5py.File(obj.output_file_name, "r") as f:
            g = f[obj.snap_group + f"w/xi_plus_plus/t_jk{NUM_JK}"]
            for i in range(NUM_JK):
                assert f"t_{i}" in g
                assert g[f"t_{i}"].shape == (_NR, _NPI)

    def test_multipole_jackknife(self, tmp_path):
        obj = _obj(_catalogue(), tmp_path)
        obj.measure_xi_multipoles("t", "++", NUM_JK, temp_file_path=False)
        with h5py.File(obj.output_file_name, "r") as f:
            g = f[obj.snap_group + "multipoles_plus_plus"]
            assert f"t_jackknife_cov_{NUM_JK}" in g

    @pytest.mark.parametrize("method", ["measure_xi_w", "measure_xi_multipoles"])
    def test_errorbars_are_sqrt_diag_cov(self, tmp_path, method):
        """The written errorbars and covariance are accumulated separately, so
        check they agree for every product ('all' = g+, gg and shape-shape)."""
        obj = _obj(_catalogue(), tmp_path)
        getattr(obj, method)("t", "all", NUM_JK, temp_file_path=False)
        checked = 0
        with h5py.File(obj.output_file_name, "r") as f:
            g = f[obj.snap_group]
            for name in g.keys():
                if f"t_jackknife_cov_{NUM_JK}" not in g[name]:
                    continue
                cov = g[name][f"t_jackknife_cov_{NUM_JK}"][:]
                err = g[name][f"t_jackknife_{NUM_JK}"][:]
                np.testing.assert_allclose(err, np.sqrt(np.diag(cov)), rtol=1e-12,
                                           err_msg=name)
                checked += 1
        assert checked >= 3, f"only {checked} products carried a covariance"


# ===========================================================================
# 5. Analytic limits
# ===========================================================================

class TestAnalyticLimits:

    def test_zero_density_ellipticity_zeroes_every_product(self, tmp_path):
        """q=1 on the density sample kills all three products exactly, while
        leaving w_g+ (which only uses the shape sample) untouched."""
        data = _catalogue()
        data["q_density_sample"] = np.ones_like(data["q_density_sample"])
        obj = _obj(data, tmp_path)
        obj.measure_xi_w("t", "all", 0, temp_file_path=False)
        got = _products(obj, "w_")
        for k in _SS_W:
            np.testing.assert_array_equal(got[k], np.zeros_like(got[k]))
        assert np.nanmax(np.abs(got["w_g_plus"])) > 1e-8

    def test_responsivity_rescales_products_quadratically(self, tmp_path):
        """Two shapes means two responsivity factors, so switching the
        correction off must scale the products by exactly (2R)(2R_pos)."""
        data = _catalogue()
        on = _obj(data, tmp_path, "on.hdf5")
        on.measure_xi_w("t", "++", 0, temp_file_path=False, responsivity=True)
        off = _obj(data, tmp_path, "off.hdf5")
        off.measure_xi_w("t", "++", 0, temp_file_path=False, responsivity=False)

        e_s = (1 - data["q"] ** 2) / (1 + data["q"] ** 2)
        e_d = (1 - data["q_density_sample"] ** 2) / (1 + data["q_density_sample"] ** 2)
        R = np.mean(1 - e_s ** 2 / 2.0)
        R_pos = np.mean(1 - e_d ** 2 / 2.0)
        factor = (2 * R) * (2 * R_pos)

        a, b = _products(on, "w_"), _products(off, "w_")
        for k in _SS_W:
            np.testing.assert_allclose(a[k] * factor, b[k], rtol=1e-12, atol=1e-14,
                                       err_msg=k)

    def test_auto_case_runs(self, tmp_path):
        """The same catalogue in both slots -- the ordinary II auto-correlation.

        num_overlap detects the full overlap, and the ordered-pair convention of
        the loop and of available_pairs match, so no 'auto' RR factor is needed.
        """
        m = radial_alignment_box_mock(n_centrals=150, n_sat=6, boxsize=BOXSIZE, seed=3)
        pos = m["Position_shape_sample"]
        data = {
            "Position": pos, "Position_shape_sample": pos,
            "Axis_Direction": m["Axis_Direction"], "q": m["q"],
            "Axis_Direction_density_sample": m["Axis_Direction"],
            "q_density_sample": m["q"],
            "LOS": 2,
        }
        obj = _obj(data, tmp_path, "auto.hdf5")
        obj.measure_xi_w("t", "++", 0, temp_file_path=False)
        got = _products(obj, "w_")
        assert _SS_W <= set(got)
        assert np.nanmax(np.abs(got["w_plus_plus"])) > 1e-8
        assert np.all(np.isfinite(got["w_plus_plus"]))
