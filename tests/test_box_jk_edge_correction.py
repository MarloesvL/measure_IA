"""
tests/test_box_jk_edge_correction.py
====================================
The analytic edge correction of the MeasureIABox delete-one jackknife
(``jk_edge_correction=True``, the default).

Each jackknife realisation k gets the per-realisation random terms of a uniform random
catalogue in the infinite-randoms limit (``measureia.box_jk_analytic``):

    xi_gg,k = (DD_k - RD_k - SR_k + RR_k) / RR_k
    xi_g+,k = (S+D_k - S+R_k) / RR_k            (both S+ terms already divided by 2 R_k)

with, in measureia's count units (N_k = the retained available pairs):
    _RD    = N_k * DR_k        positions x random shape-sample points  (lightcone naming)
    _SR    = N_k * RD_k        random position-sample points x shapes
    _SplusR= N_k * S+R_k / (2 R_k)
    _RR    = N_k * RR_k        exact union RR (incl. the sub-box self-overlap)

The full-sample measurement is unchanged. ``jk_edge_correction=False`` reproduces the
previous behaviour bit for bit.

Sections
--------
1. Realisation terms equal the independent reference (w and multipoles, weights, overlap)
2. Backends agree (brute, tree, multiprocessing) and the gg-only path matches
3. Full sample unchanged; option off reproduces the previous release (stored snapshot)
4. Masks, weights, empty sub-boxes, responsivity
5. Poisson null: the corrected realisations are unbiased (incl. rp_cut)
6. Line-of-sight invariance
7. Input validation (incl. separations beyond half the box) and shape-shape correlations
"""

import os
import numpy as np
import pytest
import h5py

from measureia import MeasureIABox
import box_jk_reference as ref


BOXSIZE = 100.0
SEP = [1.0, 20.0]
NUM_BINS_R = 5
NUM_BINS_PI = 6
PI_MAX = 20.0
NUM_JK = 8
L_SUB = 2
SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "data", "box_jk_before_edge_correction.npz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock(seed, n=260, n_shape=None, weights=False, boxsize=BOXSIZE):
    """Uniform box mock. n_shape=None: the shape sample is the position sample (the usual
    IA set-up, full overlap); otherwise an independent shape sample."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, boxsize, (n, 3))
    shape = pos.copy() if n_shape is None else rng.uniform(0, boxsize, (n_shape, 3))
    ns = len(shape)
    th = rng.uniform(0, np.pi, ns)
    data = {
        "Position": pos,
        "Position_shape_sample": shape,
        "Axis_Direction": np.column_stack([np.cos(th), np.sin(th)]),
        "LOS": 2,
        "q": rng.uniform(0.3, 1.0, ns),
    }
    if weights:
        data["weight"] = rng.uniform(0.5, 2.0, n)
        data["weight_shape_sample"] = rng.uniform(0.5, 2.0, ns)
    return data


def _box(tmp_path, data, name="out", num_nodes=1, **kw):
    return MeasureIABox(data, str(tmp_path / f"{name}.hdf5"), separation_limits=SEP,
                        num_bins_r=NUM_BINS_R, num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX,
                        boxsize=BOXSIZE, num_nodes=num_nodes, **kw)


def _run(obj, kind, name, corr_type="both", tmp=None, **kw):
    fn = obj.measure_xi_w if kind == "w" else obj.measure_xi_multipoles
    fn(name, corr_type, num_jk=kw.pop("num_jk", NUM_JK),
       temp_file_path=(str(tmp) + "/") if tmp is not None else False, **kw)


def _grp(kind):
    return "w" if kind == "w" else "multipoles"


def _jk(obj, kind, stat, name, suffix="", num_jk=NUM_JK):
    """Per-realisation datasets as an array [k, ...]."""
    g = f"{obj.snap_group}{_grp(kind)}/{stat}/{name}_jk{num_jk}"
    with h5py.File(obj.output_file_name, "r") as f:
        return np.array([f[g][f"{name}_{i}{suffix}"][:] for i in range(num_jk)])


def _has(obj, kind, stat, name, suffix, num_jk=NUM_JK):
    g = f"{obj.snap_group}{_grp(kind)}/{stat}/{name}_jk{num_jk}"
    with h5py.File(obj.output_file_name, "r") as f:
        return f"{name}_0{suffix}" in f[g]


def _all_datasets(obj):
    out = {}
    with h5py.File(obj.output_file_name, "r") as f:
        def visit(path, node):
            if isinstance(node, h5py.Dataset):
                out[path] = node[:]
        f.visititems(visit)
    return out


def _bins2(obj, kind):
    return obj.pi_bins if kind == "w" else obj.mu_r_bins


def _binning(kind):
    return "rppi" if kind == "w" else "rmu"


def _labels(points, L=L_SUB, boxsize=BOXSIZE):
    i = np.floor(points / (boxsize / L)).astype(int)
    return i[:, 0] * L ** 2 + i[:, 1] * L + i[:, 2]


def _oracle(data, obj, kind):
    """Reference DR, RD, S+R per realisation (normalised per retained pair) and R_k."""
    pos, shape = data["Position"], data["Position_shape_sample"]
    th = np.arctan2(data["Axis_Direction"][:, 1], data["Axis_Direction"][:, 0])
    q = data["q"]
    e = (1 - q ** 2) / (1 + q ** 2)
    w = data.get("weight")
    w_s = data.get("weight_shape_sample")
    lab_p, lab_s = _labels(pos), _labels(shape)
    DR, RD, SpR = ref.dr_terms(pos, shape, th, e, lab_p, lab_s, NUM_JK, boxsize=BOXSIZE,
                               r_bins=obj.r_bins, bins2=_bins2(obj, kind), binning=_binning(kind),
                               w=w, w_shape=w_s)
    ws = np.ones(len(shape)) if w_s is None else w_s
    R_k = np.array([np.sum((ws * (1 - e ** 2 / 2))[lab_s != k]) / np.sum(ws[lab_s != k])
                    for k in range(NUM_JK)])
    return DR, RD, SpR, R_k


def _n_pairs(obj_off, kind, name):
    """Retained available pairs N_k, from the previous release's realisation RR
    (= N_k V_b / (V (N_jk - 1) / N_jk), validated against an independent counter)."""
    RR_old = _jk(obj_off, kind, "xi_gg", name, "_RR")
    Vb = ref.shell_volumes(obj_off.r_bins, _bins2(obj_off, kind), _binning(kind))
    return (RR_old / Vb[None] * BOXSIZE ** 3 * (NUM_JK - 1) / NUM_JK)[:, 0, 0]


# ---------------------------------------------------------------------------
# 1. Realisation terms equal the independent reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["w", "multipoles"])
@pytest.mark.parametrize("mock_kw", [dict(), dict(n_shape=200, weights=True)],
                         ids=["same-sample", "separate-weighted"])
class TestRealisationTermsMatchReference:

    def _setup(self, tmp_path, kind, mock_kw):
        data = _mock(1, **mock_kw)
        on = _box(tmp_path, data, "on")
        _run(on, kind, "d")
        off = _box(tmp_path, data, "off")
        _run(off, kind, "d", jk_edge_correction=False)
        return data, on, off

    def test_random_terms(self, tmp_path, kind, mock_kw):
        """Against the validated loop prototype, to the accuracy of its quadrature: w uses the
        exact closed form (prototype per-galaxy error up to ~5e-4 of V_b); multipoles use
        per-bin rho nodes (per galaxy within 6.1e-3 of V_b, the prototype within 9.6e-4)."""
        data, on, off = self._setup(tmp_path, kind, mock_kw)
        N = _n_pairs(off, kind, "d")[:, None, None]
        DR, RD, SpR, R_k = _oracle(data, on, kind)
        got = [_jk(on, kind, "xi_gg", "d", "_RD"), _jk(on, kind, "xi_gg", "d", "_SR"),
               _jk(on, kind, "xi_g_plus", "d", "_SplusR")]
        want = [N * DR, N * RD, N * SpR / (2 * R_k[:, None, None])]
        Vb = ref.shell_volumes(on.r_bins, _bins2(on, kind), _binning(kind))
        per_pair = N * Vb[None] / (BOXSIZE ** 3 * (1 - 1 / NUM_JK))
        tol = 1e-3 if kind == "w" else 1e-2
        for name, g, w in zip(["RD", "SR", "SplusR"], got, want):
            assert np.all(np.abs(g - w) <= tol * per_pair), name

    def test_random_terms_equal_module(self, tmp_path, kind, mock_kw):
        """The pipeline writes exactly N_k times the module's normalised terms (S+R also
        divided by 2 R_k): checks the wiring (labels, weights, units) to rounding."""
        from measureia import box_jk_analytic as bja
        data, on, off = self._setup(tmp_path, kind, mock_kw)
        N = _n_pairs(off, kind, "d")[:, None, None]
        _, _, _, R_k = _oracle(data, on, kind)
        pos, shape = data["Position"], data["Position_shape_sample"]
        th = np.arctan2(data["Axis_Direction"][:, 1], data["Axis_Direction"][:, 0])
        e = (1 - data["q"] ** 2) / (1 + data["q"] ** 2)
        DR, RD, SpR = bja.analytic_dr_terms(
            pos, shape, th, e, _labels(pos), _labels(shape), L_SUB, weight=data.get("weight"),
            weight_shape=data.get("weight_shape_sample"), boxsize=BOXSIZE, r_bins=on.r_bins,
            bins2=_bins2(on, kind), binning=_binning(kind))
        np.testing.assert_allclose(_jk(on, kind, "xi_gg", "d", "_RD"), N * DR, rtol=1e-12)
        np.testing.assert_allclose(_jk(on, kind, "xi_gg", "d", "_SR"), N * RD, rtol=1e-12)
        want = N * SpR / (2 * R_k[:, None, None])
        np.testing.assert_allclose(_jk(on, kind, "xi_g_plus", "d", "_SplusR"), want,
                                   rtol=1e-12, atol=1e-12 * np.abs(want).max())

    def test_exact_union_rr(self, tmp_path, kind, mock_kw):
        data, on, off = self._setup(tmp_path, kind, mock_kw)
        N = _n_pairs(off, kind, "d")[:, None, None]
        rr = ref.exact_union_rr_reference(L_SUB, boxsize=BOXSIZE, r_bins=on.r_bins,
                                          bins2=_bins2(on, kind), binning=_binning(kind))
        for stat in ["xi_gg", "xi_g_plus"]:
            np.testing.assert_allclose(_jk(on, kind, stat, "d", "_RR"), N * rr[None], rtol=3e-4)

    def test_realisation_estimators(self, tmp_path, kind, mock_kw):
        """The written xi realisations are the corrected estimators of the written terms;
        DD and S+D are the same retained counts as before."""
        data, on, off = self._setup(tmp_path, kind, mock_kw)
        DD = _jk(on, kind, "xi_gg", "d", "_DD")
        RR = _jk(on, kind, "xi_gg", "d", "_RR")
        want_gg = (DD - _jk(on, kind, "xi_gg", "d", "_RD") - _jk(on, kind, "xi_gg", "d", "_SR") + RR) / RR
        np.testing.assert_allclose(_jk(on, kind, "xi_gg", "d"), want_gg, rtol=1e-12, atol=1e-14)
        SpD = _jk(on, kind, "xi_g_plus", "d", "_SplusD")
        want_gp = (SpD - _jk(on, kind, "xi_g_plus", "d", "_SplusR")) / _jk(on, kind, "xi_g_plus", "d", "_RR")
        np.testing.assert_allclose(_jk(on, kind, "xi_g_plus", "d"), want_gp, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(DD, _jk(off, kind, "xi_gg", "d", "_DD"))
        np.testing.assert_array_equal(SpD, _jk(off, kind, "xi_g_plus", "d", "_SplusD"))


# ---------------------------------------------------------------------------
# 2. Backends and the gg-only path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["w", "multipoles"])
class TestBackendsAgree:

    SUFFIXES = {"xi_gg": ["", "_DD", "_RR", "_RD", "_SR"], "xi_g_plus": ["", "_SplusD", "_SplusR", "_RR"]}

    def test_brute_tree_multiprocessing(self, tmp_path, kind):
        data = _mock(2, n_shape=180, weights=True)
        brute = _box(tmp_path, data, "brute")
        _run(brute, kind, "d")
        tree = _box(tmp_path, data, "tree")
        _run(tree, kind, "d", tmp=tmp_path)
        mp = _box(tmp_path, data, "mp", num_nodes=2)
        _run(mp, kind, "d", tmp=tmp_path)
        for stat, sufs in self.SUFFIXES.items():
            for s in sufs:
                b = _jk(brute, kind, stat, "d", s)
                for other in (tree, mp):
                    np.testing.assert_allclose(_jk(other, kind, stat, "d", s), b, rtol=1e-10,
                                               atol=1e-12 * max(np.abs(b).max(), 1), err_msg=f"{stat}{s}")

    @pytest.mark.parametrize("backend", ["brute", "tree", "mp"])
    def test_gg_only_path_matches_both(self, tmp_path, kind, backend):
        data = _mock(3, weights=True)
        nodes = 2 if backend == "mp" else 1
        tmp = None if backend == "brute" else tmp_path
        both = _box(tmp_path, data, "both", num_nodes=nodes)
        _run(both, kind, "d", "both", tmp=tmp)
        gg = _box(tmp_path, data, "gg", num_nodes=nodes)
        _run(gg, kind, "d", "gg", tmp=tmp)
        for s in self.SUFFIXES["xi_gg"]:
            np.testing.assert_allclose(_jk(gg, kind, "xi_gg", "d", s), _jk(both, kind, "xi_gg", "d", s),
                                       rtol=1e-12, atol=1e-14, err_msg=s)


# ---------------------------------------------------------------------------
# 3. Full sample unchanged; option off = previous release
# ---------------------------------------------------------------------------

def _is_jk_path(path):
    """Jackknife realisations and their summaries (mean, std, covariance)."""
    return f"_jk{NUM_JK}" in path or "_jackknife_" in path or "_mean_" in path


@pytest.mark.parametrize("kind", ["w", "multipoles"])
class TestBackwardCompatibility:

    def test_full_sample_bit_identical(self, tmp_path, kind):
        data = _mock(4, n_shape=150, weights=True)
        on = _box(tmp_path, data, "on")
        _run(on, kind, "d")
        off = _box(tmp_path, data, "off")
        _run(off, kind, "d", jk_edge_correction=False)
        a, b = _all_datasets(on), _all_datasets(off)
        full = [p for p in b if not _is_jk_path(p)]
        assert full, "no full-sample datasets found"
        for p in full:
            np.testing.assert_array_equal(a[p], b[p], err_msg=p)

    def test_full_sample_bit_identical_with_rp_cut(self, tmp_path, kind):
        """The realisation terms respect rp_cut, but the full-sample measurement (including
        its analytic RR, which ignores rp_cut in multipoles) stays exactly as before."""
        if kind == "w":
            pytest.skip("rp_cut only applies to multipoles")
        data = _mock(4, n_shape=150, weights=True)
        on = _box(tmp_path, data, "on")
        _run(on, kind, "d", rp_cut=4.0)
        off = _box(tmp_path, data, "off")
        _run(off, kind, "d", rp_cut=4.0, jk_edge_correction=False)
        a, b = _all_datasets(on), _all_datasets(off)
        full = [p for p in b if not _is_jk_path(p)]
        assert full, "no full-sample datasets found"
        for p in full:
            np.testing.assert_array_equal(a[p], b[p], err_msg=p)

    def test_off_writes_no_new_datasets(self, tmp_path, kind):
        off = _box(tmp_path, _mock(5), "off")
        _run(off, kind, "d", jk_edge_correction=False)
        for stat, s in [("xi_gg", "_RD"), ("xi_gg", "_SR"), ("xi_g_plus", "_SplusR")]:
            assert not _has(off, kind, stat, "d", s)

    @pytest.mark.parametrize("corr_type", ["both", "gg"])
    def test_off_reproduces_previous_release(self, tmp_path, kind, corr_type):
        """Snapshot of every output dataset written by measureia before this change
        (commit 636080d), for the mock and settings below; see
        tests/data/make_box_jk_snapshot.py."""
        snap = np.load(SNAPSHOT_FILE)
        off = _box(tmp_path, _mock(6, n_shape=150, weights=True), "off")
        _run(off, kind, "d", corr_type, jk_edge_correction=False)
        got = _all_datasets(off)
        prefix = f"{kind}/{corr_type}/"
        keys = [k[len(prefix):] for k in snap.files if k.startswith(prefix)]
        assert keys
        assert sorted(keys) == sorted(got), "set of datasets changed"
        for k in keys:
            np.testing.assert_allclose(got[k], snap[prefix + k], rtol=1e-12, atol=1e-300, err_msg=k)


# ---------------------------------------------------------------------------
# 4. Masks, weights, empty sub-boxes, responsivity
# ---------------------------------------------------------------------------

def _full_masks(data, keep_pos, keep_shape):
    m = {"Position": keep_pos, "Position_shape_sample": keep_shape,
         "Axis_Direction": keep_shape, "q": keep_shape}
    if "weight" in data:
        m["weight"] = keep_pos
        m["weight_shape_sample"] = keep_shape
    return m


def _filtered(data, keep_pos, keep_shape):
    out = dict(data)
    for k in ["Position", "weight"]:
        if k in data:
            out[k] = data[k][keep_pos]
    for k in ["Position_shape_sample", "Axis_Direction", "q", "weight_shape_sample"]:
        if k in data:
            out[k] = data[k][keep_shape]
    return out


@pytest.mark.parametrize("kind", ["w", "multipoles"])
class TestMasksWeightsResponsivity:

    def test_masked_equals_prefiltered(self, tmp_path, kind):
        data = _mock(7, n_shape=220, weights=True)
        rng = np.random.default_rng(70)
        kp = rng.random(len(data["Position"])) < 0.7
        ks = rng.random(len(data["Position_shape_sample"])) < 0.6
        masked = _box(tmp_path, data, "masked")
        _run(masked, kind, "d", masks=_full_masks(data, kp, ks))
        pre = _box(tmp_path, _filtered(data, kp, ks), "pre")
        _run(pre, kind, "d")
        for stat, sufs in TestBackendsAgree.SUFFIXES.items():
            for s in sufs:
                np.testing.assert_allclose(_jk(masked, kind, stat, "d", s), _jk(pre, kind, stat, "d", s),
                                           rtol=1e-12, atol=1e-14, err_msg=f"{stat}{s}")

    def test_covariance_invariant_under_weight_rescaling(self, tmp_path, kind):
        data = _mock(8, n_shape=200, weights=True)
        a = _box(tmp_path, data, "a")
        _run(a, kind, "d")
        scaled = dict(data, weight=3.0 * data["weight"], weight_shape_sample=0.5 * data["weight_shape_sample"])
        b = _box(tmp_path, scaled, "b")
        _run(b, kind, "d")
        for stat in ["xi_gg", "xi_g_plus"]:
            np.testing.assert_allclose(_jk(b, kind, stat, "d"), _jk(a, kind, stat, "d"), rtol=1e-10, atol=1e-13)

    def test_unit_weights_equal_no_weights(self, tmp_path, kind):
        data = _mock(9, n_shape=200)
        a = _box(tmp_path, data, "a")
        _run(a, kind, "d")
        ones = dict(data, weight=np.ones(len(data["Position"])),
                    weight_shape_sample=np.ones(len(data["Position_shape_sample"])))
        b = _box(tmp_path, ones, "b")
        _run(b, kind, "d")
        for stat in ["xi_gg", "xi_g_plus"]:
            np.testing.assert_allclose(_jk(b, kind, stat, "d"), _jk(a, kind, stat, "d"), rtol=1e-12, atol=1e-14)

    def test_empty_subbox(self, tmp_path, kind):
        """No galaxy in sub-box 0 ([0, 50)^3): runs, and every output is finite."""
        data = _mock(10, n=400)
        keep = ~np.all(data["Position"] < BOXSIZE / 2, axis=1)
        data = _filtered(data, keep, keep)
        obj = _box(tmp_path, data, "empty")
        _run(obj, kind, "d")
        for p, v in _all_datasets(obj).items():
            assert np.all(np.isfinite(v)), p

    def test_responsivity_unchanged(self, tmp_path, kind):
        """S+D realisations (divided by 2 R_k) are identical with the option on and off,
        so R_k is untouched."""
        data = _mock(11, n_shape=180, weights=True)
        on = _box(tmp_path, data, "on")
        _run(on, kind, "d")
        off = _box(tmp_path, data, "off")
        _run(off, kind, "d", jk_edge_correction=False)
        np.testing.assert_array_equal(_jk(on, kind, "xi_g_plus", "d", "_SplusD"),
                                      _jk(off, kind, "xi_g_plus", "d", "_SplusD"))


# ---------------------------------------------------------------------------
# 5. Poisson null
# ---------------------------------------------------------------------------

def _poisson_null(tmp_path, kind, edge, rp_cut=None, seed=12, n=4000):
    """Mean over realisations and over the second bin coordinate of the realisation xi_gg
    for an unclustered box, and its standard error (from the scatter over the second
    coordinate and realisations; conservative because realisations are correlated)."""
    data = _mock(seed, n=n)
    obj = MeasureIABox(data, str(tmp_path / f"null_{edge}.hdf5"), separation_limits=[2.0, 25.0],
                       num_bins_r=4, num_bins_pi=4, pi_max=20.0, boxsize=BOXSIZE)
    kw = dict(jk_edge_correction=edge)
    if rp_cut is not None:
        kw["rp_cut"] = rp_cut
    _run(obj, kind, "d", "gg", tmp=tmp_path, **kw)
    xi = _jk(obj, kind, "xi_gg", "d")            # [k, r, second]
    mean = xi.mean(axis=(0, 2))
    err = xi.std(axis=(0, 2), ddof=1)
    return mean, err


class TestPoissonNull:

    @pytest.mark.parametrize("kind", ["w", "multipoles"])
    def test_corrected_realisations_are_unbiased(self, tmp_path, kind):
        mean, err = _poisson_null(tmp_path, kind, True)
        assert np.all(np.abs(mean) < 4 * err + 1e-3), (mean, err)

    @pytest.mark.parametrize("kind", ["w", "multipoles"])
    def test_previous_realisations_are_biased_at_large_r(self, tmp_path, kind):
        """Sanity check that the null test has power: without the correction the
        realisations follow (V - 2 V_k + V_ov) / (V - V_k) - 1 < 0 at large r."""
        mean, err = _poisson_null(tmp_path, kind, False)
        assert mean[-1] < -5 * err[-1] and mean[-1] < -0.02

    def test_rp_cut_corrected_realisations_are_unbiased(self, tmp_path):
        mean, err = _poisson_null(tmp_path, "multipoles", True, rp_cut=4.0)
        assert np.all(np.abs(mean) < 4 * err + 1e-3), (mean, err)


# ---------------------------------------------------------------------------
# 6. Line-of-sight invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["w", "multipoles"])
def test_los_axis_invariance(tmp_path, kind):
    """LOS = 0 on (x, y, z) equals LOS = 2 on the permuted coordinates (y, z, x), whose
    projected plane is the same (y, z) plane. Sub-box labels are permuted, so compare the
    jackknife covariance, which does not depend on the realisation order."""
    data = _mock(13, n_shape=200, weights=True)
    d0 = dict(data, LOS=0)
    perm = [1, 2, 0]
    d2 = dict(data, LOS=2, Position=data["Position"][:, perm],
              Position_shape_sample=data["Position_shape_sample"][:, perm])
    a = _box(tmp_path, d0, "los0")
    _run(a, kind, "d")
    b = _box(tmp_path, d2, "los2")
    _run(b, kind, "d")
    stat_name = {"w": ["w_g_plus", "w_gg"], "multipoles": ["multipoles_g_plus", "multipoles_gg"]}[kind]
    with h5py.File(a.output_file_name, "r") as fa, h5py.File(b.output_file_name, "r") as fb:
        for s in stat_name:
            key = f"{a.snap_group}{s}/d_jackknife_cov_{NUM_JK}"
            np.testing.assert_allclose(fa[key][:], fb[key][:], rtol=1e-9, atol=1e-14, err_msg=s)


# ---------------------------------------------------------------------------
# 7. Input validation and shape-shape
# ---------------------------------------------------------------------------

class TestInputs:

    @pytest.mark.parametrize("bad", ["yes", 1, None])
    def test_non_bool_raises(self, tmp_path, bad):
        obj = _box(tmp_path, _mock(14), "bad")
        with pytest.raises(TypeError, match="jk_edge_correction must be True or False"):
            _run(obj, "w", "d", jk_edge_correction=bad)

    # r_max > L_sub also triggers measureia's existing "maximum separation exceeds the size of
    # your jackknife subboxes" warning; it is not what these tests are about.
    @pytest.mark.filterwarnings("ignore:WARNING. your maximum separation")
    @pytest.mark.parametrize("kind", ["w", "multipoles"])
    def test_r_max_beyond_half_box_raises(self, tmp_path, kind):
        data = _mock(17)
        obj = MeasureIABox(data, str(tmp_path / "far.hdf5"), separation_limits=[1.0, 0.5 * BOXSIZE + 5],
                           num_bins_r=NUM_BINS_R, num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX, boxsize=BOXSIZE)
        with pytest.raises(ValueError, match="half the box"):
            _run(obj, kind, "d")

    def test_pi_max_beyond_half_box_raises(self, tmp_path):
        data = _mock(18)
        obj = MeasureIABox(data, str(tmp_path / "far.hdf5"), separation_limits=SEP,
                           num_bins_r=NUM_BINS_R, num_bins_pi=NUM_BINS_PI, pi_max=0.5 * BOXSIZE + 5,
                           boxsize=BOXSIZE)
        with pytest.raises(ValueError, match="half the box"):
            _run(obj, "w", "d")

    @pytest.mark.filterwarnings("ignore:WARNING. your maximum separation")
    @pytest.mark.parametrize("kind", ["w", "multipoles"])
    def test_beyond_half_box_runs_without_correction(self, tmp_path, kind):
        """With jk_edge_correction=False the previous behaviour (no check) is kept."""
        data = _mock(17)
        obj = MeasureIABox(data, str(tmp_path / "far.hdf5"), separation_limits=[1.0, 0.5 * BOXSIZE + 5],
                           num_bins_r=NUM_BINS_R, num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX, boxsize=BOXSIZE)
        _run(obj, kind, "d", jk_edge_correction=False)

    def test_ignored_without_jackknife(self, tmp_path):
        a = _box(tmp_path, _mock(15), "a")
        a.measure_xi_w("d", "both", num_jk=0, temp_file_path=False, jk_edge_correction=True)
        b = _box(tmp_path, _mock(15), "b")
        b.measure_xi_w("d", "both", num_jk=0, temp_file_path=False, jk_edge_correction=False)
        da, db = _all_datasets(a), _all_datasets(b)
        assert sorted(da) == sorted(db)
        for p in da:
            np.testing.assert_array_equal(da[p], db[p])

    @pytest.mark.parametrize("kind", ["w", "multipoles"])
    def test_shape_shape_realisations_unchanged(self, tmp_path, kind):
        """Shape-shape (++, xx) realisations are out of scope: identical on and off."""
        data = _mock(16)
        data["Axis_Direction_density_sample"] = data["Axis_Direction"]
        data["q_density_sample"] = data["q"]
        on = _box(tmp_path, data, "on")
        _run(on, kind, "d", "++")
        off = _box(tmp_path, data, "off")
        _run(off, kind, "d", "++", jk_edge_correction=False)
        a, b = _all_datasets(on), _all_datasets(off)
        assert sorted(a) == sorted(b)
        for p in a:
            np.testing.assert_array_equal(a[p], b[p], err_msg=p)
