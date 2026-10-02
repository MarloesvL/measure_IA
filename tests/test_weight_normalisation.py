"""Weight normalisation of the estimators.

Pair counts are accumulated with the product of the two objects' weights, so they must be
normalised by the product of the samples' weight sums, not their sizes. The defining
property is invariance: multiplying any one sample's weights by a constant must leave every
correlation function (and its jackknife covariance) unchanged. For unit weights the weight
sums are the sample sizes, so the ordinary unweighted results are untouched.

Also covered here are two lightcone jackknife bugs in the same normalisation code: the
per-patch overlap count, and masks combined with jackknife patches, which must give exactly
what pre-filtering the catalogues gives.
"""
from __future__ import annotations

import numpy as np
import pytest
import h5py

from measureia import MeasureIABox, MeasureIALightcone
from measureia.measure_IA_base import weight_sum, overlap_weight, overlap_override_weight
from measureia.measure_galaxy_box import delete_one_estimator

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_weight_sum_is_the_size_for_unit_weights():
    assert weight_sum(np.ones(7)) == 7.0
    assert weight_sum(np.array([0.5, 2.0], dtype=np.float32)) == 2.5


def test_overlap_weight_sums_the_self_pair_weights():
    a = np.array([[0., 0.], [1., 1.], [2., 2.]])
    b = np.array([[2., 2.], [5., 5.], [0., 0.]])
    w_a = np.array([1., 2., 3.])
    w_b = np.array([10., 20., 30.])
    total, ind_a, ind_b = overlap_weight(a, b, w_a, w_b)
    # shared rows: (0,0) -> 1*30 and (2,2) -> 3*10
    assert total == 60.0
    assert sorted(zip(ind_a, ind_b)) == [(0, 2), (2, 0)]
    assert overlap_weight(a[:0], b, w_a[:0], w_b)[0] == 0.0


def test_overlap_override_scales_by_the_mean_weights():
    assert overlap_override_weight(0, np.full(4, 3.), np.full(5, 2.)) == 0.0
    assert overlap_override_weight(4, np.ones(4), np.ones(5)) == 4.0
    assert overlap_override_weight(4, np.full(4, 3.), np.full(5, 2.)) == 24.0


# ---------------------------------------------------------------------------
# lightcone
# ---------------------------------------------------------------------------

LC_SEP = [2.0, 40.0]
LC_BINS_R = 4
LC_BINS_PI = 10
LC_PI_MAX = 50.0
LC_NUM_JK = 4


def _sky(rng, n):
    return rng.uniform(150, 156, n), rng.uniform(2, 8, n), rng.uniform(0.2, 0.3, n)


@pytest.fixture(scope="module")
def lc_catalogue():
    """A shape sample drawn from the density sample (so D_S is non-trivial), with
    non-uniform weights and separate shape randoms."""
    rng = np.random.default_rng(7)
    n_d, n_s, n_r = 400, 250, 1200
    ra, dec, z = _sky(rng, n_d)
    shape = rng.choice(n_d, n_s, replace=False)
    ra_r, dec_r, z_r = _sky(rng, n_r)
    ra_rs, dec_rs, z_rs = _sky(rng, n_r)
    data = dict(
        RA=ra, DEC=dec, Redshift=z, weight=rng.uniform(0.2, 2.0, n_d),
        RA_shape_sample=ra[shape], DEC_shape_sample=dec[shape], Redshift_shape_sample=z[shape],
        weight_shape_sample=rng.uniform(0.5, 5.0, n_s),
        e1=rng.normal(0, 0.3, n_s), e2=rng.normal(0, 0.3, n_s))
    randoms = dict(
        RA=ra_r, DEC=dec_r, Redshift=z_r, weight=rng.uniform(0.1, 0.4, n_r),
        RA_shape_sample=ra_rs, DEC_shape_sample=dec_rs, Redshift_shape_sample=z_rs,
        weight_shape_sample=rng.uniform(0.5, 1.5, n_r))
    patches = dict(
        position=rng.integers(0, LC_NUM_JK, n_d), shape=rng.integers(0, LC_NUM_JK, n_s),
        randoms_position=rng.integers(0, LC_NUM_JK, n_r),
        randoms_shape=rng.integers(0, LC_NUM_JK, n_r))
    # a shared object must carry the same patch in both samples, as a real assignment would
    patches["shape"] = patches["position"][shape]
    return data, randoms, patches


def _copy(d):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def _lc(data, randoms, outfile):
    return MeasureIALightcone(data=_copy(data), randoms_data=_copy(randoms),
                              output_file_name=str(outfile), separation_limits=LC_SEP,
                              num_bins_r=LC_BINS_R, num_bins_pi=LC_BINS_PI, pi_max=LC_PI_MAX,
                              num_nodes=1)


def _run_lc(data, randoms, tmp_path, tag, statistic, estimator, patches=None, **kwargs):
    outfile = tmp_path / f"{tag}.hdf5"
    lc = _lc(data, randoms, outfile)
    measure = lc.measure_xi_w if statistic == "w" else lc.measure_xi_multipoles
    measure(IA_estimator=estimator, dataset_name="t", corr_type="both", over_h=True, tree=True,
            jk_patches=None if patches is None else _copy(patches),
            temp_file_path=str(tmp_path / f"tmp_{tag}"), **kwargs)
    prefix = "w" if statistic == "w" else "multipoles"
    out = {}
    with h5py.File(outfile, "r") as f:
        for name in (f"{prefix}_g_plus", f"{prefix}_gg"):
            out[name] = f[name]["t"][:]
            if patches is not None:
                out[name + "_cov"] = f[name][f"t_jackknife_cov_{LC_NUM_JK}"][:]
    return out


def _scaled(data, randoms, factors):
    data, randoms = _copy(data), _copy(randoms)
    data["weight"] *= factors[0]
    data["weight_shape_sample"] *= factors[1]
    randoms["weight"] *= factors[2]
    randoms["weight_shape_sample"] *= factors[3]
    return data, randoms


@pytest.mark.parametrize("statistic", ["w", "multipoles"])
@pytest.mark.parametrize("estimator", ["galaxies", "clusters"])
def test_lightcone_invariant_under_weight_rescaling(lc_catalogue, tmp_path, statistic, estimator):
    """The reported bug: rescaling one sample's weights changed xi_g+ (galaxies estimator)
    by that factor, and xi_gg by a non-constant amount. Every sample is rescaled by a
    different factor here, and the covariance must not move either."""
    data, randoms, patches = lc_catalogue
    ref = _run_lc(data, randoms, tmp_path, "ref", statistic, estimator, patches)
    sd, sr = _scaled(data, randoms, (0.3, 2.0, 5.0, 0.7))
    new = _run_lc(sd, sr, tmp_path, "scaled", statistic, estimator, patches)
    for name in ref:
        np.testing.assert_allclose(new[name], ref[name], rtol=1e-9, atol=1e-14, err_msg=name)


def test_lightcone_per_patch_overlap(lc_catalogue, tmp_path):
    """Each delete-one realisation keeps only the shared objects outside the removed patch.
    The old code subtracted len(np.where(...)) — always 1 — whatever the patch held."""
    data, randoms, patches = lc_catalogue
    lc = _lc(data, randoms, tmp_path / "o.hdf5")
    lc.data_dir, lc.randoms_data = _copy(data), _copy(randoms)
    full, per_patch = lc._sample_normalisation(None, None, jk_patches=patches, num_jk=LC_NUM_JK)

    shape_in_pos = {(r, d): i for i, (r, d) in enumerate(zip(data["RA"], data["DEC"]))}
    shared = [(shape_in_pos[(r, d)], j) for j, (r, d) in
              enumerate(zip(data["RA_shape_sample"], data["DEC_shape_sample"]))]
    w_pair = lambda i, j: data["weight"][i] * data["weight_shape_sample"][j]
    assert full["D_S"] == pytest.approx(sum(w_pair(i, j) for i, j in shared))
    assert full["D"] == pytest.approx(data["weight"].sum())
    assert full["R_S"] == pytest.approx(randoms["weight_shape_sample"].sum())
    for n, norm in enumerate(per_patch):
        expected = sum(w_pair(i, j) for i, j in shared
                       if patches["position"][i] != n and patches["shape"][j] != n)
        assert norm["D_S"] == pytest.approx(expected)
        assert norm["D"] == pytest.approx(data["weight"][patches["position"] != n].sum())
        assert norm["R_D"] == pytest.approx(randoms["weight"][patches["randoms_position"] != n].sum())


@pytest.mark.parametrize("statistic", ["w", "multipoles"])
def test_lightcone_masked_jackknife_equals_prefiltered(lc_catalogue, tmp_path, statistic):
    """Masking must be exactly equivalent to removing the objects beforehand, covariance
    included. The patch labels used to be passed unmasked into the masked pair counts."""
    data, randoms, patches = lc_catalogue
    m_d = data["RA"] < 154.0
    m_s = data["RA_shape_sample"] < 154.0
    m_r = randoms["RA"] < 154.0
    m_rs = randoms["RA_shape_sample"] < 154.0
    masks = {"RA": m_d, "RA_shape_sample": m_s}
    masks_randoms = {"RA": m_r, "RA_shape_sample": m_rs}
    masked = _run_lc(data, randoms, tmp_path, "masked", statistic, "galaxies", patches,
                     masks=masks, masks_randoms=masks_randoms)

    d_keys = ("RA", "DEC", "Redshift", "weight")
    s_keys = ("RA_shape_sample", "DEC_shape_sample", "Redshift_shape_sample",
              "weight_shape_sample", "e1", "e2")
    fd = {k: data[k][m_d] for k in d_keys} | {k: data[k][m_s] for k in s_keys}
    fr = ({k: randoms[k][m_r] for k in d_keys}
          | {k: randoms[k][m_rs] for k in s_keys if k in randoms})
    fp = dict(position=patches["position"][m_d], shape=patches["shape"][m_s],
              randoms_position=patches["randoms_position"][m_r],
              randoms_shape=patches["randoms_shape"][m_rs])
    filtered = _run_lc(fd, fr, tmp_path, "filtered", statistic, "galaxies", fp)
    for name in masked:
        np.testing.assert_allclose(masked[name], filtered[name], rtol=1e-12, atol=0, err_msg=name)


# ---------------------------------------------------------------------------
# box
# ---------------------------------------------------------------------------

BOXSIZE = 100.0
BOX_NUM_JK = 8


@pytest.fixture(scope="module")
def box_catalogue():
    """Shape sample drawn from the position sample, with non-uniform weights."""
    rng = np.random.default_rng(11)
    n, n_s = 600, 300
    pos = rng.uniform(0, BOXSIZE, (n, 3))
    shape = rng.choice(n, n_s, replace=False)
    axis = rng.normal(size=(n_s, 2))
    return dict(Position=pos, Position_shape_sample=pos[shape], Axis_Direction=axis,
                q=rng.uniform(0.3, 0.9, n_s), LOS=2,
                weight=rng.uniform(0.2, 2.0, n), weight_shape_sample=rng.uniform(0.5, 4.0, n_s))


def _box(data, outfile, **kwargs):
    return MeasureIABox(_copy(data), str(outfile), boxsize=BOXSIZE, separation_limits=[0.5, 15.0],
                        num_bins_r=5, num_bins_pi=8, num_nodes=1, **kwargs)


def _run_box(data, tmp_path, tag, statistic, **kwargs):
    outfile = tmp_path / f"{tag}.hdf5"
    box = _box(data, outfile, **kwargs)
    if statistic == "w":
        box.measure_xi_w("t", "both", BOX_NUM_JK, temp_file_path=str(tmp_path) + "/")
    else:
        box.measure_xi_multipoles("t", "both", BOX_NUM_JK, temp_file_path=str(tmp_path) + "/")
    out = {}
    with h5py.File(outfile, "r") as f:
        for name in (f"{statistic}_g_plus", f"{statistic}_gg"):
            out[name] = f[name]["t"][:]
            out[name + "_cov"] = f[name][f"t_jackknife_cov_{BOX_NUM_JK}"][:]
    return out


@pytest.mark.parametrize("statistic", ["w", "multipoles"])
def test_box_invariant_under_weight_rescaling(box_catalogue, tmp_path, statistic):
    ref = _run_box(box_catalogue, tmp_path, "ref", statistic)
    scaled = _copy(box_catalogue)
    scaled["weight"] *= 0.25
    scaled["weight_shape_sample"] *= 3.0
    new = _run_box(scaled, tmp_path, "scaled", statistic)
    for name in ref:
        np.testing.assert_allclose(new[name], ref[name], rtol=1e-9, atol=1e-14, err_msg=name)


def test_box_num_overlap_override_scales_with_the_weights(box_catalogue, tmp_path):
    """With constant weights c, the object-count override N_shape (every shape galaxy is
    also a position galaxy) is scaled to N_shape * c_pos * c_shape, exactly the weighted
    overlap measured automatically. Only the full-sample statistics are compared: an
    override is applied uniformly, with no per-region jackknife adjustment, by design."""
    data = _copy(box_catalogue)
    data["weight"] = np.full(len(data["Position"]), 3.0)
    data["weight_shape_sample"] = np.full(len(data["Position_shape_sample"]), 0.5)
    auto = _run_box(data, tmp_path, "auto", "w")
    override = _run_box(data, tmp_path, "override", "w",
                        num_overlap=len(data["Position_shape_sample"]))
    for name in ("w_g_plus", "w_gg"):
        np.testing.assert_allclose(override[name], auto[name], rtol=1e-12, atol=0, err_msg=name)


@pytest.mark.parametrize("statistic", ["multipoles", "w"])
def test_galaxy_contributions_rebuild_weighted_measurement(box_catalogue, tmp_path, statistic):
    """The per-galaxy decomposition shares the weighted RR, so with non-uniform weights it
    still reproduces the ordinary measurement and each jackknife realisation."""
    outfile = tmp_path / "ref.hdf5"
    box = _box(box_catalogue, outfile)
    measure = box.measure_xi_w if statistic == "w" else box.measure_xi_multipoles
    measure("ref", "g+", BOX_NUM_JK, temp_file_path=str(tmp_path) + "/")
    out = _box(box_catalogue, tmp_path / "gal.hdf5").measure_galaxy_contributions(
        "gal", num_jk=BOX_NUM_JK, statistic=statistic, return_output=True)
    with h5py.File(outfile, "r") as f:
        g = f[f"{statistic}_g_plus"]
        ref = g["ref"][:]
        ref_jk = [g[f"ref_jk{BOX_NUM_JK}"][f"ref_{i}"][:] for i in range(BOX_NUM_JK)]
    np.testing.assert_allclose(out["Y"].sum(axis=0), ref, rtol=1e-10, atol=1e-14)
    for n in range(BOX_NUM_JK):
        np.testing.assert_allclose(delete_one_estimator(out, n), ref_jk[n], rtol=1e-9, atol=1e-14,
                                   err_msg=f"realisation {n}")
