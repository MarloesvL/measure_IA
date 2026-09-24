"""
test_lc_shape_shape.py
======================
The public shape-shape ('++' / 'all') path on MeasureIALightcone: option
validation, the products that land in the output file, backend equality
(including multiprocessing), and jackknife.

The lightcone twin of test_box_shape_shape.py. The kernel itself is covered
against O(N^2) references in test_pair_kernel_shape_shape.py.

Sections
--------
  1. corr_type validation, the density-sample-shapes requirement, and the
     'clusters' refusal
  2. Output layout
  3. Backend equality (brute / tree / multiprocessing)
  4. Jackknife
"""

import numpy as np
import pytest
import h5py

from measureia import MeasureIALightcone


_SEP = [1.0, 20.0]
_NR = 5
_NPI = 8
NUM_JK = 4

_SS_W = {"w_plus_plus", "w_cross_cross", "w_plus_cross"}
_GP_W = {"w_g_plus", "w_gg"}


def _with_density_shapes(data, seed=31):
    """Give the density sample its own e1/e2, in the survey convention."""
    rng = np.random.default_rng(seed)
    n = len(data["RA"])
    th = rng.uniform(0, np.pi, n)
    e = rng.uniform(0.05, 0.5, n)
    data["e1_density_sample"] = e * np.cos(2 * th)
    data["e2_density_sample"] = -e * np.sin(2 * th)
    return data


def _build(data, randoms, tmp_path, name="lc_ss.hdf5", num_nodes=1):
    return MeasureIALightcone(
        data=data, randoms_data=randoms, output_file_name=str(tmp_path / name),
        separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI, pi_max=60.0,
        num_nodes=num_nodes)


@pytest.fixture()
def lc_ss(_lc_catalog, tmp_path):
    """A lightcone object whose density sample carries shapes."""
    data, rand = _lc_catalog
    data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    rand = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rand.items()}
    return _build(_with_density_shapes(data), rand, tmp_path)


@pytest.fixture()
def lc_plain(_lc_catalog, tmp_path):
    """The same, without density-sample shapes."""
    data, rand = _lc_catalog
    data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    rand = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rand.items()}
    return _build(data, rand, tmp_path, name="lc_plain.hdf5")


def _products(obj, prefix):
    with h5py.File(obj.output_file_name, "r") as f:
        g = f[obj.snap_group] if obj.snap_group else f
        return {k: g[k]["t"][:] for k in g.keys()
                if k.startswith(prefix) and "t" in g[k]}


# ===========================================================================
# 1. Option validation
# ===========================================================================

class TestCorrTypeValidation:

    @pytest.mark.parametrize("corr_type", ["++", "all"])
    def test_accepted(self, lc_ss, corr_type):
        lc_ss.measure_xi_w("galaxies", "t", corr_type, tree=True)

    @pytest.mark.parametrize("corr_type", ["++", "all"])
    def test_refused_without_density_sample_shapes(self, lc_plain, corr_type):
        with pytest.raises(ValueError, match="density sample needs shapes"):
            lc_plain.measure_xi_w("galaxies", "t", corr_type, tree=True)

    @pytest.mark.parametrize("corr_type", ["++", "all"])
    def test_clusters_estimator_refused(self, lc_ss, corr_type):
        """xi_++ = S+S+/RR has no published 'clusters' analogue, so rather than
        invent one the combination is refused."""
        with pytest.raises(ValueError, match="clusters"):
            lc_ss.measure_xi_w("clusters", "t", corr_type, tree=True)

    def test_clusters_still_fine_for_g_plus(self, lc_ss):
        lc_ss.measure_xi_w("clusters", "t", "g+", tree=True)


# ===========================================================================
# 2. Output layout
# ===========================================================================

class TestOutputLayout:

    def test_shape_shape_products_written(self, lc_ss):
        lc_ss.measure_xi_w("galaxies", "t", "++", tree=True)
        assert _SS_W <= set(_products(lc_ss, "w_"))

    def test_both_still_means_g_plus_and_gg(self, lc_ss):
        lc_ss.measure_xi_w("galaxies", "t", "both", tree=True)
        assert set(_products(lc_ss, "w_")) == _GP_W

    def test_all_is_everything(self, lc_ss):
        lc_ss.measure_xi_w("galaxies", "t", "all", tree=True)
        assert set(_products(lc_ss, "w_")) == _GP_W | _SS_W

    def test_multipoles(self, lc_ss):
        lc_ss.measure_xi_multipoles("galaxies", "t", "all", tree=True)
        assert set(_products(lc_ss, "multipoles_")) == {
            "multipoles_g_plus", "multipoles_gg", "multipoles_plus_plus"}

    def test_raw_sums_written(self, lc_ss):
        lc_ss.measure_xi_w("galaxies", "t", "++", tree=True)
        with h5py.File(lc_ss.output_file_name, "r") as f:
            g = f["w/xi_plus_plus"]
            for suffix in ("_SplusSplus", "_rp", "_pi"):
                assert f"t{suffix}" in g, suffix

    def test_no_splusr_analogue_is_written(self, lc_ss):
        """Randoms carry no shapes, so a pure '++' run does not even do the S+R
        pass; there is no shape-shape term to build from randoms."""
        lc_ss.measure_xi_w("galaxies", "t", "++", tree=True)
        with h5py.File(lc_ss.output_file_name, "r") as f:
            names = list(f["w/xi_plus_plus"].keys())
        assert not any(n.endswith("R") for n in names), names


# ===========================================================================
# 3. Backend equality
# ===========================================================================

class TestBackendEquality:

    def test_brute_matches_tree(self, _lc_catalog, tmp_path):
        def run(name, tree):
            data, rand = _lc_catalog
            data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            rand = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rand.items()}
            obj = _build(_with_density_shapes(data), rand, tmp_path, name=name)
            obj.measure_xi_w("galaxies", "t", "all", tree=tree)
            return _products(obj, "w_")
        a, b = run("b.hdf5", False), run("t.hdf5", True)
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-10, atol=1e-12, err_msg=k)

    @pytest.mark.parametrize("method,num_jk",
                             [("measure_xi_w", 0), ("measure_xi_w", NUM_JK),
                              ("measure_xi_multipoles", 0), ("measure_xi_multipoles", NUM_JK)])
    def test_multiprocessing_matches_single_process(self, _lc_catalog, tmp_path,
                                                    method, num_jk):
        """The shape-shape grids must survive the workers.

        On the lightcone the chunked axis is the *position* sample, so e_pos is
        sliced per batch while east_shape/north_shape are passed whole -- the
        opposite of the box. Nothing else in the suite would catch a worker
        silently returning zeros.
        """
        def run(name, nodes):
            data, rand = _lc_catalog
            data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            rand = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rand.items()}
            obj = _build(_with_density_shapes(data), rand, tmp_path, name=name,
                         num_nodes=nodes)
            kw = dict(dataset_name="t", corr_type="all")
            if num_jk:
                kw["num_jk"] = num_jk
            if nodes > 1:
                kw["temp_file_path"] = str(tmp_path)
            getattr(obj, method)("galaxies", **kw)
            return _products(obj, "w_" if method == "measure_xi_w" else "multipoles_")

        one = run(f"s1_{method}_{num_jk}.hdf5", 1)
        many = run(f"s4_{method}_{num_jk}.hdf5", 4)
        assert set(one) == set(many)
        nonzero = 0
        for k in one:
            m = np.isfinite(one[k]) & np.isfinite(many[k])
            assert m.any(), k
            scale = max(np.max(np.abs(one[k][m])), 1e-300)
            assert np.max(np.abs(one[k][m] - many[k][m])) / scale < 1e-10, k
            if np.max(np.abs(one[k][m])) > 1e-8:
                nonzero += 1
        assert nonzero > 0, "every product is zero; the equality check is vacuous"


# ===========================================================================
# 4. Jackknife
# ===========================================================================

class TestJackknife:

    def test_covariance_written(self, lc_ss):
        lc_ss.measure_xi_w("galaxies", "t", "++", num_jk=NUM_JK, tree=True)
        with h5py.File(lc_ss.output_file_name, "r") as f:
            for name in _SS_W:
                assert f"t_jackknife_cov_{NUM_JK}" in f[name], name
                assert f[name][f"t_jackknife_cov_{NUM_JK}"].shape == (_NR, _NR)

    def test_realisation_counts_are_written_and_distinct(self, lc_ss):
        """The jk group holds full - jk[i], the same convention as S+D.

        The delete-one *identity* is tested at kernel level in
        test_pair_kernel_shape_shape.py, where a realisation can be checked
        against a direct measurement on the physically deleted catalogue. What
        this covers is the plumbing above it: that every realisation is written,
        has the right shape, is finite, and actually differs from the full sample
        (an empty patch would silently give a realisation identical to it).

        Note there is no bound like |part| <= |full| to assert: these are signed
        sums, so removing a patch whose pairs contributed with the opposite sign
        makes the magnitude grow.
        """
        lc_ss.measure_xi_w("galaxies", "t", "++", num_jk=NUM_JK, tree=True)
        with h5py.File(lc_ss.output_file_name, "r") as f:
            full = f["w/xi_plus_plus/t_SplusSplus"][:]
            g = f[f"w/xi_plus_plus/t_jk{NUM_JK}"]
            assert np.max(np.abs(full)) > 1e-8, "full-sample sum is trivially zero"
            differing = 0
            for i in range(NUM_JK):
                part = g[f"t_{i}_SplusSplus"][:]
                assert part.shape == full.shape
                assert np.all(np.isfinite(part))
                if not np.array_equal(part, full):
                    differing += 1
            assert differing == NUM_JK, (
                f"only {differing}/{NUM_JK} realisations differ from the full sample")

    def test_multipole_jackknife(self, lc_ss):
        lc_ss.measure_xi_multipoles("galaxies", "t", "++", num_jk=NUM_JK, tree=True)
        with h5py.File(lc_ss.output_file_name, "r") as f:
            assert f"t_jackknife_cov_{NUM_JK}" in f["multipoles_plus_plus"]

    @pytest.mark.parametrize("method", ["measure_xi_w", "measure_xi_multipoles"])
    def test_errorbars_are_sqrt_diag_cov(self, lc_ss, method):
        """The written errorbars and covariance are accumulated separately, so
        check they agree for every product ('all' = g+, gg and shape-shape)."""
        getattr(lc_ss, method)("galaxies", "t", "all", num_jk=NUM_JK, tree=True)
        checked = 0
        with h5py.File(lc_ss.output_file_name, "r") as f:
            for name in f.keys():
                if not isinstance(f[name], h5py.Group) or f"t_jackknife_cov_{NUM_JK}" not in f[name]:
                    continue
                cov = f[name][f"t_jackknife_cov_{NUM_JK}"][:]
                err = f[name][f"t_jackknife_{NUM_JK}"][:]
                np.testing.assert_allclose(err, np.sqrt(np.diag(cov)), rtol=1e-12,
                                           err_msg=name)
                checked += 1
        assert checked >= 3, f"only {checked} products carried a covariance"
