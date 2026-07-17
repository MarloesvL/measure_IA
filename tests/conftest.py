"""
tests/conftest.py
=================
Single combined conftest for the entire measure_IA test suite.

Sections
--------
A. Shared synthetic-data helpers  (pure NumPy, no HDF5 files needed)
B. MeasureIABase / pure-logic fixtures
C. MeasureIABox fixtures           (fully synthetic — no TNG data required)
D. MeasureIALightcone fixtures     (fully synthetic sky catalogue)

Run the whole suite with:
    pytest tests/ -v
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from measureia import MeasureIABase, MeasureIABox, MeasureIALightcone


# ===========================================================================
# A. SHARED SYNTHETIC-DATA HELPERS
# ===========================================================================

_BOX_SEED = 2024
_LC_SEED  = 42

_BOXSIZE   = 205.0          # matches TNG300
_N_BOX     = 200            # galaxies in the synthetic box catalogue
_N_LC_DATA = 150            # galaxies / clusters in the synthetic lightcone
_N_LC_RAND = 450            # randoms for the lightcone (3× data)

_SEP_LIMS   = [0.1, 20.0]
_NUM_BINS_R  = 8
_NUM_BINS_PI = 20
_NUM_JK      = 8

_RA_RANGE  = (150.0, 155.0)
_DEC_RANGE = (  2.0,   6.0)
_Z_RANGE   = (  0.1,   0.3)


# ---------------------------------------------------------------------------
# Box catalogue generator
# ---------------------------------------------------------------------------

def _make_box_catalog(N: int = _N_BOX,
                      boxsize: float = _BOXSIZE,
                      seed: int = _BOX_SEED) -> dict:
    """
    Generate a synthetic box catalogue compatible with MeasureIABox.

    Returns a plain dict (never touches disk) with keys:
        Position, Position_shape_sample, Axis_Direction, LOS, q, Mass
    """
    rng   = np.random.default_rng(seed)
    COM   = rng.uniform(0.0, boxsize, (N, 3))
    theta = rng.uniform(0.0, 2.0 * math.pi, N)
    e_dir = np.column_stack([np.cos(theta), np.sin(theta)])
    q     = rng.uniform(0.1, 1.0, N)
    mass  = rng.uniform(11.0, 12.0, N)
    return {
        "Position":              COM,
        "Position_shape_sample": COM,
        "Axis_Direction":        e_dir,
        "LOS":                   2,
        "q":                     q,
        "Mass":                  mass,
    }


# ---------------------------------------------------------------------------
# Lightcone catalogue generators
# ---------------------------------------------------------------------------

def _make_lc_data(N: int, rng: np.random.Generator) -> dict:
    return {
        "RA":                    rng.uniform(*_RA_RANGE,  N),
        "DEC":                   rng.uniform(*_DEC_RANGE, N),
        "Redshift":              rng.uniform(*_Z_RANGE,   N),
        "RA_shape_sample":       rng.uniform(*_RA_RANGE,  N),
        "DEC_shape_sample":      rng.uniform(*_DEC_RANGE, N),
        "Redshift_shape_sample": rng.uniform(*_Z_RANGE,   N),
        "e1":                    rng.uniform(-0.5, 0.5,   N),
        "e2":                    rng.uniform(-0.5, 0.5,   N),
        "weight":                np.ones(N),
        "weight_shape_sample":   np.ones(N),
    }


def _make_lc_randoms(N: int, rng: np.random.Generator) -> dict:
    return {
        "RA":                    rng.uniform(*_RA_RANGE,  N),
        "DEC":                   rng.uniform(*_DEC_RANGE, N),
        "Redshift":              rng.uniform(*_Z_RANGE,   N),
        "RA_shape_sample":       rng.uniform(*_RA_RANGE,  N),
        "DEC_shape_sample":      rng.uniform(*_DEC_RANGE, N),
        "Redshift_shape_sample": rng.uniform(*_Z_RANGE,   N),
        "weight":                np.ones(N),
        "weight_shape_sample":   np.ones(N),
    }


# ===========================================================================
# B. MeasureIABase / pure-logic fixtures
# ===========================================================================

@pytest.fixture()
def small_positions():
    """Four-galaxy position array used in geometric / JK-region tests."""
    return np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 2.0],
        [2.5, 2.5, 1.5],
        [1.0, 2.0, 2.0],
    ])


@pytest.fixture()
def unit_axis_directions():
    """Four unit-vector axis directions aligned with cardinal axes."""
    return np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])


@pytest.fixture()
def base_object_small(small_positions, unit_axis_directions):
    """
    MeasureIABase built from a tiny in-memory dataset.
    No HDF5 I/O ever occurs.
    """
    data = {
        "Position":              small_positions,
        "Position_shape_sample": small_positions,
        "Axis_Direction":        unit_axis_directions,
        "LOS":                   2,
        "q":                     np.array([0.5, 0.7, 0.3, 0.9]),
    }
    return MeasureIABase(
        data, "no_write.hdf5",
        simulation=None, snapshot=None,
        separation_limits=[0.1, 5.0],
        num_bins_r=4, num_bins_pi=4,
        pi_max=2.0, boxsize=3.0,
    )


# ===========================================================================
# C. MeasureIABox fixtures — 100 % synthetic, no TNG files required
# ===========================================================================

@pytest.fixture(scope="session")
def _box_catalog():
    """Session-scoped synthetic box catalogue dict (shared, never modified)."""
    return _make_box_catalog()


@pytest.fixture()
def IA_mock_TNG300_n1(_box_catalog, tmp_path):
    """
    1-node MeasureIABox built from the synthetic catalogue.
    Each test gets its own tmp output file so tests never share state.
    """
    out = str(tmp_path / "test_IA_mock_TNG300_n1.hdf5")
    # deep-copy mutable arrays so tests that mutate data don't affect others
    data = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in _box_catalog.items()}
    return MeasureIABox(
        data, out,
        simulation="TNG300", snapshot=99,
        separation_limits=_SEP_LIMS,
        num_bins_r=_NUM_BINS_R,
        num_bins_pi=_NUM_BINS_PI,
        pi_max=None,
        num_nodes=1,
    )


@pytest.fixture()
def IA_mock_TNG300_n8(_box_catalog, tmp_path):
    """8-node MeasureIABox for multiprocessing / chunking tests."""
    out = str(tmp_path / "test_IA_mock_TNG300_n8.hdf5")
    data = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in _box_catalog.items()}
    return MeasureIABox(
        data, out,
        simulation="TNG300", snapshot=99,
        separation_limits=_SEP_LIMS,
        num_bins_r=_NUM_BINS_R,
        num_bins_pi=_NUM_BINS_PI,
        pi_max=None,
        num_nodes=8,
    )


@pytest.fixture()
def IA_available_output(tmp_path):
    """
    MeasureIABox pointing at a pre-existing output file for
    _combine_jackknife_information tests.  The file is created
    by running IA_mock_TNG300_n1.measure_xi_w(...) in the test
    that needs it; this fixture just provides the reader object.
    """
    out = str(tmp_path / "mock_IA_TNG300.hdf5")
    return MeasureIABox(
        None, out,
        simulation="TNG300", snapshot=99,
        separation_limits=_SEP_LIMS,
        num_bins_r=_NUM_BINS_R,
        num_bins_pi=_NUM_BINS_PI,
    )


@pytest.fixture()
def jk_regions():
    """
    Tiny MeasureIABox (4 galaxies) for jackknife region-assignment tests.
    Positions are chosen so expected patch indices are deterministic.
    """
    COM = np.array([[1., 1., 1.],
                    [2., 1., 2.],
                    [2.5, 2.5, 1.51],
                    [1., 2., 2.]])
    data = {
        "Position":              COM,
        "Position_shape_sample": COM,
        "Axis_Direction":        np.zeros((4, 2)),
        "LOS":                   2,
        "q":                     np.ones(4),
    }
    return MeasureIABox(data, "no_write.hdf5",
                        simulation=None, snapshot=None, boxsize=3.0)


@pytest.fixture()
def box_mass(_box_catalog):
    """
    The synthetic Mass array from the box catalogue — used by mask tests
    that need a continuous property to cut on.
    """
    return _box_catalog["Mass"].copy()


# ===========================================================================
# D. MeasureIALightcone fixtures — 100 % synthetic sky catalogue
# ===========================================================================

@pytest.fixture(scope="session")
def _lc_catalog():
    """Session-scoped synthetic lightcone catalogue (shared, never modified)."""
    rng  = np.random.default_rng(_LC_SEED)
    data = _make_lc_data(_N_LC_DATA, rng)
    rand = _make_lc_randoms(_N_LC_RAND, rng)
    return data, rand


def _build_lc(data: dict, randoms, num_nodes: int, tmp_path) -> MeasureIALightcone:
    out = str(tmp_path / "test_IA_mock_lc.hdf5")
    return MeasureIALightcone(
        data=data,
        randoms_data=randoms,
        output_file_name=out,
        separation_limits=_SEP_LIMS,
        num_bins_r=_NUM_BINS_R,
        num_bins_pi=_NUM_BINS_PI,
        pi_max=60.0,
        num_nodes=num_nodes,
    )


def _copy_lc(cat: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in cat.items()}


@pytest.fixture()
def IA_mock_lc_n1(_lc_catalog, tmp_path):
    """Standard 1-node lightcone measurement object."""
    data, rand = _lc_catalog
    return _build_lc(_copy_lc(data), _copy_lc(rand), 1, tmp_path)


@pytest.fixture()
def IA_mock_lc_n8(_lc_catalog, tmp_path):
    """8-node lightcone measurement object for multiprocessing tests."""
    data, rand = _lc_catalog
    return _build_lc(_copy_lc(data), _copy_lc(rand), 8, tmp_path)


@pytest.fixture()
def IA_mock_lc_no_randoms(_lc_catalog, tmp_path):
    """Lightcone object with randoms_data=None — tests error paths."""
    data, _ = _lc_catalog
    return _build_lc(_copy_lc(data), None, 1, tmp_path)


@pytest.fixture()
def IA_mock_lc_no_weight(_lc_catalog, tmp_path):
    """Data dict missing 'weight' and 'weight_shape_sample' keys."""
    data, rand = _lc_catalog
    stripped = {k: v for k, v in data.items()
                if k not in ("weight", "weight_shape_sample")}
    return _build_lc(_copy_lc(stripped), _copy_lc(rand), 1, tmp_path)


@pytest.fixture()
def IA_mock_lc_rand_no_weight(_lc_catalog, tmp_path):
    """Randoms dict missing 'weight' and 'weight_shape_sample' keys."""
    data, rand = _lc_catalog
    stripped_rand = {k: v for k, v in rand.items()
                     if k not in ("weight", "weight_shape_sample")}
    return _build_lc(_copy_lc(data), _copy_lc(stripped_rand), 1, tmp_path)


@pytest.fixture()
def IA_mock_lc_single_rand(_lc_catalog, tmp_path):
    """
    Randoms dict with only one shared catalogue (no _shape_sample keys).
    The code should auto-duplicate the position randoms for the shape sample.
    """
    data, rand = _lc_catalog
    single_rand = {
        "RA":       rand["RA"].copy(),
        "DEC":      rand["DEC"].copy(),
        "Redshift": rand["Redshift"].copy(),
        "weight":   rand["weight"].copy(),
    }
    return _build_lc(_copy_lc(data), single_rand, 1, tmp_path)


@pytest.fixture()
def IA_mock_lc_dup_rand(_lc_catalog, tmp_path):
    """
    Randoms dict with explicit _shape_sample keys that are exact copies of the
    position randoms. Comparing against IA_mock_lc_single_rand isolates the
    auto-duplication code path: both objects use identical random samples, so
    their results must match exactly.
    """
    data, rand = _lc_catalog
    dup_rand = {
        "RA":                    rand["RA"].copy(),
        "DEC":                   rand["DEC"].copy(),
        "Redshift":              rand["Redshift"].copy(),
        "weight":                rand["weight"].copy(),
        "RA_shape_sample":       rand["RA"].copy(),
        "DEC_shape_sample":      rand["DEC"].copy(),
        "Redshift_shape_sample": rand["Redshift"].copy(),
        "weight_shape_sample":   rand["weight"].copy(),
    }
    return _build_lc(_copy_lc(data), dup_rand, 1, tmp_path)


@pytest.fixture()
def lc_jk_patches(_lc_catalog, tmp_path):
    """
    Pre-computed JK patch assignment dict for _NUM_JK=8 patches,
    produced by assign_jackknife_patches on the session catalogue.
    """
    data, rand = _lc_catalog
    obj     = _build_lc(_copy_lc(data), _copy_lc(rand), 1, tmp_path)
    patches = obj.assign_jackknife_patches(data, rand, _NUM_JK)
    if "randoms_position" not in patches:
        patches["randoms_position"] = patches["randoms"]
        patches["randoms_shape"]    = patches["randoms"]
    return patches


@pytest.fixture()
def lc_masks(_lc_catalog):
    """Boolean mask dict that keeps every other object (~half the sample)."""
    data, _ = _lc_catalog
    N    = len(data["RA"])
    mask = np.arange(N) % 2 == 0
    return {k: mask for k in data}
