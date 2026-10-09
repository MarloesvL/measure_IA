"""Linearly spaced separation bins (``binning="linear"``).

Every quantity measureia writes per separation bin depends only on the pairs inside that
bin, so a run with N linear bins must reproduce, bin by bin, N separate runs that each have
a single bin spanning one linear bin's edges. A single bin has the same edges under either
scheme, so the reference runs use the default log binning and test the linear bin-index
computation independently of it.
"""
from __future__ import annotations

import numpy as np
import pytest
import h5py

from measureia import MeasureIABox, MeasureIALightcone


def _copy(d):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def _per_bin_datasets(path, num_bins):
    """Every dataset in the file whose first axis runs over the separation bins, keyed by its
    path. Jackknife covariances (num_bins x num_bins) are reduced to their diagonal."""
    out = {}

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset) or obj.ndim == 0 or obj.shape[0] != num_bins:
            return
        data = obj[:]
        if data.ndim == 2 and data.shape == (num_bins, num_bins) and "cov" in name:
            data = np.diag(data)
        out[name] = data

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def _assert_matches_single_bin_runs(run, edges, tmp_path):
    num_bins = len(edges) - 1
    full = _per_bin_datasets(run(tmp_path / "linear.hdf5", list(edges[[0, -1]]), num_bins, "linear"),
                             num_bins)
    assert full, "no per-bin datasets were written"
    for i in range(num_bins):
        single = _per_bin_datasets(run(tmp_path / f"bin_{i}.hdf5", [edges[i], edges[i + 1]], 1, "log"), 1)
        assert single.keys() == full.keys()
        for name in full:
            np.testing.assert_allclose(full[name][i], single[name][0], rtol=1e-10, atol=1e-14,
                                       err_msg=f"bin {i}: {name}")


# ---------------------------------------------------------------------------
# box
# ---------------------------------------------------------------------------

BOXSIZE = 100.0
BOX_NUM_JK = 8
BOX_EDGES = np.linspace(2.0, 14.0, 5)
BOX_BINS_PI = 6


@pytest.fixture(scope="module")
def box_catalogue():
    """Shape sample drawn from the position sample, with non-uniform weights."""
    rng = np.random.default_rng(23)
    n, n_s = 1500, 800
    pos = rng.uniform(0, BOXSIZE, (n, 3))
    shape = rng.choice(n, n_s, replace=False)
    return dict(Position=pos, Position_shape_sample=pos[shape], Axis_Direction=rng.normal(size=(n_s, 2)),
                q=rng.uniform(0.3, 0.9, n_s), LOS=2,
                weight=rng.uniform(0.2, 2.0, n), weight_shape_sample=rng.uniform(0.5, 4.0, n_s))


@pytest.mark.parametrize("statistic", ["w", "multipoles"])
def test_box_linear_bins_match_single_bin_runs(box_catalogue, tmp_path, statistic):
    def run(outfile, separation_limits, num_bins_r, binning):
        box = MeasureIABox(_copy(box_catalogue), str(outfile), boxsize=BOXSIZE,
                           separation_limits=separation_limits, num_bins_r=num_bins_r,
                           num_bins_pi=BOX_BINS_PI, num_nodes=1, binning=binning)
        measure = box.measure_xi_w if statistic == "w" else box.measure_xi_multipoles
        measure("t", "both", BOX_NUM_JK, temp_file_path=str(tmp_path) + "/")
        return outfile

    _assert_matches_single_bin_runs(run, BOX_EDGES, tmp_path)


# ---------------------------------------------------------------------------
# lightcone
# ---------------------------------------------------------------------------

LC_EDGES = np.linspace(5.0, 45.0, 5)
LC_BINS_PI = 6
LC_PI_MAX = 50.0
LC_NUM_JK = 4


def _sky(rng, n):
    return rng.uniform(150, 156, n), rng.uniform(2, 8, n), rng.uniform(0.2, 0.3, n)


@pytest.fixture(scope="module")
def lc_catalogue():
    rng = np.random.default_rng(29)
    n_d, n_s, n_r = 500, 300, 1500
    ra, dec, z = _sky(rng, n_d)
    shape = rng.choice(n_d, n_s, replace=False)
    ra_r, dec_r, z_r = _sky(rng, n_r)
    data = dict(
        RA=ra, DEC=dec, Redshift=z, weight=rng.uniform(0.2, 2.0, n_d),
        RA_shape_sample=ra[shape], DEC_shape_sample=dec[shape], Redshift_shape_sample=z[shape],
        weight_shape_sample=rng.uniform(0.5, 5.0, n_s),
        e1=rng.normal(0, 0.3, n_s), e2=rng.normal(0, 0.3, n_s))
    randoms = dict(RA=ra_r, DEC=dec_r, Redshift=z_r, weight=rng.uniform(0.1, 0.4, n_r))
    patches = dict(position=rng.integers(0, LC_NUM_JK, n_d), randoms=rng.integers(0, LC_NUM_JK, n_r))
    patches["shape"] = patches["position"][shape]
    return data, randoms, patches


@pytest.mark.parametrize("statistic", ["w", "multipoles"])
def test_lightcone_linear_bins_match_single_bin_runs(lc_catalogue, tmp_path, statistic):
    data, randoms, patches = lc_catalogue

    def run(outfile, separation_limits, num_bins_r, binning):
        lc = MeasureIALightcone(data=_copy(data), randoms_data=_copy(randoms), output_file_name=str(outfile),
                                separation_limits=separation_limits, num_bins_r=num_bins_r,
                                num_bins_pi=LC_BINS_PI, pi_max=LC_PI_MAX, num_nodes=1, binning=binning)
        measure = lc.measure_xi_w if statistic == "w" else lc.measure_xi_multipoles
        measure(IA_estimator="galaxies", dataset_name="t", corr_type="both", over_h=True, tree=True,
                jk_patches=_copy(patches), temp_file_path=str(tmp_path / f"tmp_{outfile.stem}"))
        return outfile

    _assert_matches_single_bin_runs(run, LC_EDGES, tmp_path)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_bin_edges_and_centres(tmp_path):
    box = MeasureIABox(None, None, boxsize=BOXSIZE, separation_limits=[60.0, 140.0], num_bins_r=8,
                       binning="linear")
    np.testing.assert_allclose(box.r_bins, np.linspace(60.0, 140.0, 9))
    default = MeasureIABox(None, None, boxsize=BOXSIZE, separation_limits=[0.1, 20.0], num_bins_r=8)
    assert default.binning == "log"
    np.testing.assert_allclose(default.r_bins, np.logspace(-1, np.log10(20.0), 9))


def test_unknown_binning_raises():
    with pytest.raises(ValueError, match="binning"):
        MeasureIABox(None, None, boxsize=BOXSIZE, binning="logarithmic")
