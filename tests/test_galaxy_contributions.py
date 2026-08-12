"""Per-galaxy alignment contributions (``per_galaxy`` in the kernel and
``MeasureIABox.measure_galaxy_contributions``).

The defining property throughout is that resolving the pair sums on the shape-galaxy axis
must change nothing: summing the per-galaxy arrays back over that axis has to reproduce
the aggregate grids, the multipole, and every jackknife realisation the package already
produces. These tests assert exactly that, and that the normal path is untouched when the
feature is off.
"""
from __future__ import annotations

import numpy as np
import pytest
import h5py

from measureia import MeasureIABox, pair_kernel
from measureia.mocks import radial_alignment_box_mock

BOXSIZE = 205.0
NUM_JK = 8
SEP_LIMS = [0.3, 8.0]
NUM_BINS_R = 6
NUM_BINS_PI = 10


@pytest.fixture(scope="module")
def mock_data():
    mock = radial_alignment_box_mock(n_centrals=300, n_sat=6, boxsize=BOXSIZE)
    return {
        "Position": mock["Position"],
        "Position_shape_sample": mock["Position_shape_sample"],
        "Axis_Direction": mock["Axis_Direction"],
        "LOS": 2,
        "q": mock["q"],
    }


def _box(mock_data, outfile, num_nodes=1):
    return MeasureIABox(mock_data, boxsize=BOXSIZE, separation_limits=SEP_LIMS,
                        num_bins_r=NUM_BINS_R, num_bins_pi=NUM_BINS_PI,
                        num_nodes=num_nodes, output_file_name=str(outfile))


def _sample_and_binning(mi, num_box=None, seed=0):
    ss = pair_kernel.prepare_box_samples(mi.data, None, mi.Num_position, mi.Num_shape,
                                         shapes=True, ellipticity="distortion", base=mi)
    R = sum(ss.weight_shape * (1 - ss.e ** 2 / 2.0)) / sum(ss.weight_shape)
    if num_box is not None:
        rng = np.random.default_rng(seed)
        ss.jk_pos = rng.integers(0, num_box, len(ss.pos))
        ss.jk_shape = rng.integers(0, num_box, len(ss.pos_shape))
    return ss, pair_kernel.BoxRMuR(mi, 0.0), R


# ---------------------------------------------------------------------------
# kernel level
# ---------------------------------------------------------------------------

def test_per_galaxy_sums_to_aggregate_grids(mock_data, tmp_path):
    """Unprojected per-galaxy grids sum back to DD and Splus_D."""
    mi = _box(mock_data, tmp_path / "o.hdf5")
    ss, binning, R = _sample_and_binning(mi)
    kw = dict(base=mi, R=R, shapes=True, chunk_axis="shape", chunk_size_outer=100,
              backend="tree")
    ref = pair_kernel.accumulate(ss, binning, **kw)
    gal = pair_kernel.accumulate(ss, binning, per_galaxy=True, **kw)

    assert gal.DD_gal.shape == (len(ss.pos_shape), NUM_BINS_R, NUM_BINS_PI)
    # unit weights make DD integer-valued, so this one is exact
    assert np.array_equal(gal.DD_gal.sum(axis=0), ref.DD)
    # S+D differs only by float summation order
    assert np.allclose(gal.Splus_D_gal.sum(axis=0), ref.Splus_D, rtol=1e-13, atol=0)


def test_per_galaxy_projection_contracts_the_mu_axis(mock_data, tmp_path):
    """With per_galaxy_proj=W, S+D is contracted with W and DD stays a plain count."""
    mi = _box(mock_data, tmp_path / "o.hdf5")
    ss, binning, R = _sample_and_binning(mi)
    kw = dict(base=mi, R=R, shapes=True, chunk_axis="shape", chunk_size_outer=100,
              backend="tree")
    ref = pair_kernel.accumulate(ss, binning, **kw)
    W = np.random.default_rng(1).normal(size=(NUM_BINS_R, NUM_BINS_PI))
    gal = pair_kernel.accumulate(ss, binning, per_galaxy=True, per_galaxy_proj=W, **kw)

    assert gal.Splus_D_gal.shape == (len(ss.pos_shape), NUM_BINS_R)
    assert np.allclose(gal.Splus_D_gal.sum(axis=0), (ref.Splus_D * W).sum(axis=1))
    assert np.allclose(gal.DD_gal.sum(axis=0), ref.DD.sum(axis=1))


def test_per_galaxy_jk_reproduces_union_deletion(mock_data, tmp_path):
    """The position-patch decomposition rebuilds the aggregate delete-one grids."""
    mi = _box(mock_data, tmp_path / "o.hdf5")
    ss, binning, R = _sample_and_binning(mi, num_box=NUM_JK)
    kw = dict(base=mi, R=R, shapes=True, chunk_axis="shape", chunk_size_outer=100,
              backend="tree", jk=True, num_box=NUM_JK)
    W = np.random.default_rng(2).normal(size=(NUM_BINS_R, NUM_BINS_PI))
    g = pair_kernel.accumulate(ss, binning, per_galaxy=True, per_galaxy_proj=W,
                               per_galaxy_jk=True, **kw)

    assert g.Splus_D_gal_jk.shape == (len(ss.pos_shape), NUM_JK, NUM_BINS_R)
    for n in range(NUM_JK):
        keep = ss.jk_shape != n
        rec_s = (g.Splus_D_gal_jk[keep].sum(axis=1) - g.Splus_D_gal_jk[keep, n]).sum(axis=0)
        rec_d = (g.DD_gal_jk[keep].sum(axis=1) - g.DD_gal_jk[keep, n]).sum(axis=0)
        # Splus_D_gal_jk is raw, as Splus_D_jk is; Splus_D has 2R divided out already
        exp_s = ((g.Splus_D * 2 * R - g.Splus_D_jk[n]) * W).sum(axis=1)
        exp_d = (g.DD - g.DD_jk[n]).sum(axis=1)
        assert np.allclose(rec_s, exp_s)
        assert np.allclose(rec_d, exp_d)


def test_per_galaxy_off_leaves_grids_bit_identical(mock_data, tmp_path):
    """The feature must not perturb the normal path in any way."""
    mi = _box(mock_data, tmp_path / "o.hdf5")
    ss, binning, R = _sample_and_binning(mi, num_box=NUM_JK)
    kw = dict(base=mi, R=R, shapes=True, chunk_axis="shape", chunk_size_outer=100,
              backend="tree", jk=True, num_box=NUM_JK)
    a = pair_kernel.accumulate(ss, binning, **kw)
    b = pair_kernel.accumulate(ss, binning, per_galaxy=True, **kw)
    for name in ("DD", "Splus_D", "Scross_D", "DD_jk", "Splus_D_jk"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.DD_gal is None and a.Splus_D_gal is None


@pytest.mark.parametrize("kwargs, match", [
    (dict(per_galaxy_jk=True), "requires per_galaxy=True"),
    (dict(per_galaxy=True, per_galaxy_jk=True), "requires per_galaxy_proj"),
    (dict(per_galaxy=True, per_galaxy_proj=np.zeros((2, 2))), "expected"),
])
def test_per_galaxy_input_errors(mock_data, tmp_path, kwargs, match):
    mi = _box(mock_data, tmp_path / "o.hdf5")
    ss, binning, R = _sample_and_binning(mi, num_box=NUM_JK)
    with pytest.raises(ValueError, match=match):
        pair_kernel.accumulate(ss, binning, base=mi, R=R, shapes=True,
                               chunk_axis="shape", backend="tree",
                               num_box=NUM_JK, **kwargs)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def test_contributions_rebuild_multipole_and_jackknife(mock_data, tmp_path):
    """measure_galaxy_contributions reproduces measure_xi_multipoles, realisation by
    realisation. This is the test the whole feature exists to pass."""
    outfile = tmp_path / "ref.hdf5"
    _box(mock_data, outfile).measure_xi_multipoles(
        "ref", "g+", num_jk=NUM_JK, temp_file_path=str(tmp_path))
    out = _box(mock_data, tmp_path / "gal.hdf5").measure_galaxy_contributions(
        "gal", num_jk=NUM_JK, return_output=True)

    with h5py.File(outfile, "r") as f:
        g = f["multipoles_g_plus"]
        ref = g["ref"][:]
        ref_jk = np.array([g[f"ref_jk{NUM_JK}"][f"ref_{i}"][:] for i in range(NUM_JK)])

    assert np.allclose(out["Y"].sum(axis=0), ref, rtol=1e-12, atol=0)

    Y_jk, jk_shape = out["Y_jk"], out["jk_shape"]
    for n in range(NUM_JK):
        keep = jk_shape != n
        rec = (Y_jk[keep].sum(axis=1) - Y_jk[keep, n]).sum(axis=0)
        rec /= out["rr_ratio"][n] * 2 * out["R_jk"][n]
        assert np.allclose(rec, ref_jk[n], rtol=1e-10, atol=0), f"realisation {n}"


def test_contributions_rebuild_wgplus_and_jackknife(mock_data, tmp_path):
    """Same identity for the projected statistic: statistic='w' reproduces
    measure_xi_w's w_g+ and each of its jackknife realisations."""
    outfile = tmp_path / "ref.hdf5"
    _box(mock_data, outfile).measure_xi_w(
        "ref", "g+", num_jk=NUM_JK, temp_file_path=str(tmp_path))
    out = _box(mock_data, tmp_path / "gal.hdf5").measure_galaxy_contributions(
        "gal", num_jk=NUM_JK, statistic="w", return_output=True)

    with h5py.File(outfile, "r") as f:
        g = f["w_g_plus"]
        ref = g["ref"][:]
        ref_jk = np.array([g[f"ref_jk{NUM_JK}"][f"ref_{i}"][:] for i in range(NUM_JK)])

    assert np.allclose(out["Y"].sum(axis=0), ref, rtol=1e-12, atol=0)

    Y_jk, jk_shape = out["Y_jk"], out["jk_shape"]
    for n in range(NUM_JK):
        keep = jk_shape != n
        rec = (Y_jk[keep].sum(axis=1) - Y_jk[keep, n]).sum(axis=0)
        rec /= out["rr_ratio"][n] * 2 * out["R_jk"][n]
        assert np.allclose(rec, ref_jk[n], rtol=1e-10, atol=0), f"realisation {n}"


@pytest.mark.parametrize("statistic", ["multipoles", "w"])
def test_multiprocessing_matches_single_process(mock_data, tmp_path, statistic):
    """The multiprocessing path must return exactly the single-process arrays: the galaxy
    axis is contiguous per slice and Pool.map preserves slice order, so concatenation
    reproduces the whole sample."""
    single = _box(mock_data, tmp_path / "s.hdf5").measure_galaxy_contributions(
        "gal", num_jk=NUM_JK, statistic=statistic, return_output=True)
    multi = _box(mock_data, tmp_path / "m.hdf5", num_nodes=2).measure_galaxy_contributions(
        "gal", num_jk=NUM_JK, statistic=statistic, temp_file_path=str(tmp_path),
        chunk_size=300, return_output=True)

    for key in ("Y", "P", "Y_jk", "P_jk", "jk_shape", "R_jk", "rr_ratio", "r"):
        assert np.array_equal(single[key], multi[key]), key


def test_multiprocessing_requires_temp_file_path(mock_data, tmp_path):
    mi = _box(mock_data, tmp_path / "o.hdf5", num_nodes=2)
    with pytest.raises(ValueError, match="requires temp_file_path"):
        mi.measure_galaxy_contributions("gal", num_jk=NUM_JK)


def test_unknown_statistic_rejected(mock_data, tmp_path):
    mi = _box(mock_data, tmp_path / "o.hdf5")
    with pytest.raises(ValueError, match="Unknown statistic"):
        mi.measure_galaxy_contributions("gal", statistic="xi")


def test_contributions_written_to_file(mock_data, tmp_path):
    outfile = tmp_path / "gal.hdf5"
    mi = _box(mock_data, outfile)
    assert mi.measure_galaxy_contributions("gal", num_jk=NUM_JK) is None
    with h5py.File(outfile, "r") as f:
        grp = f["galaxy_contributions/multipoles"]
        assert grp["gal_Y"].shape == (mi.Num_shape, NUM_BINS_R)
        assert grp["gal_P"].shape == (mi.Num_shape, NUM_BINS_R)
        assert grp["gal_r"].shape == (NUM_BINS_R,)
        jk = grp[f"gal_jk{NUM_JK}"]
        assert jk["Y_jk"].shape == (mi.Num_shape, NUM_JK, NUM_BINS_R)
        assert jk["jk_shape"].shape == (mi.Num_shape,)
        assert jk["rr_ratio"].shape == (NUM_JK,)


def _fit_bin(Y, P, x, b):
    """Per-bin pair-level OLS built from the per-galaxy moments alone.

    The pair response is scaled so that its mean over the pairs in the bin is the
    multipole, hence the ``P[:, b].sum()`` factor on the right-hand side.
    """
    X = np.column_stack([np.ones(len(Y)), x])
    XtX = np.einsum("jk,jl,j->kl", X, X, P[:, b])
    Xty = np.einsum("jk,j->k", X, Y[:, b] * P[:, b].sum())
    return np.linalg.solve(XtX, Xty)


def test_intercept_recovers_the_multipole_under_pair_weighted_centring(mock_data, tmp_path):
    """Science-level consistency check. The pair-level OLS intercept equals the ordinary
    multipole exactly when the regressors are centred with that bin's pair weights
    P_j(b) — a plain galaxy-average centring is *not* enough, because the design is
    weighted by how many neighbours each galaxy has."""
    out = _box(mock_data, tmp_path / "gal.hdf5").measure_galaxy_contributions(
        "gal", num_jk=0, return_output=True)
    Y, P = out["Y"], out["P"]

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(len(Y), 2))                    # two fake properties
    scale = raw.std(axis=0)                               # scale set once, globally

    for b in range(NUM_BINS_R):
        w = P[:, b]
        x_pair = (raw - np.average(raw, axis=0, weights=w)) / scale
        beta = _fit_bin(Y, P, x_pair, b)
        assert np.isclose(beta[0], Y[:, b].sum(), rtol=1e-10)


def test_slopes_are_invariant_to_the_centring_choice(mock_data, tmp_path):
    """Centring shifts the intercept only, so the coefficients that carry the physics are
    unaffected by the choice above."""
    out = _box(mock_data, tmp_path / "gal.hdf5").measure_galaxy_contributions(
        "gal", num_jk=0, return_output=True)
    Y, P = out["Y"], out["P"]

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(len(Y), 2))
    scale = raw.std(axis=0)

    for b in range(NUM_BINS_R):
        x_global = (raw - raw.mean(axis=0)) / scale
        x_pair = (raw - np.average(raw, axis=0, weights=P[:, b])) / scale
        assert np.allclose(_fit_bin(Y, P, x_global, b)[1:],
                           _fit_bin(Y, P, x_pair, b)[1:], rtol=1e-8)
