"""
test_measure_ia_base_and_sim.py
================================
Pure-logic tests that require no HDF5 files or simulation data.

Covers
------
  MeasureIABase
    - Initialisation (bin shapes, log spacing, pi/mu_r symmetry, default weights,
      edge cases: missing pi_max, None data)
    - calculate_dot_product_arrays
    - get_ellipticity (1-D and 2-D modes)
    - get_random_pairs (both corr_types, volume scaling, formula verification)
    - get_random_pairs_r_mur
    - setdiff2D / setdiff_omit

  SimInfo / MeasureIABox initialisation
    - All known simulations (boxsize, h, L_0p5, snap_group)
    - Manual boxsize, no-simname, pi_max derivation
    - Snapshot string / group handling
    - Invalid simname / missing suffix raises KeyError
    - pi_max explicit vs derived

  MeasureIABase internal methods (direct unit tests)
    - _measure_w_g_i: return_output=True/False, corr_type routing,
      returned arrays match written values, invalid corr_type raises
    - _measure_multipoles: same coverage
"""

import math
import numpy as np
import pytest
from measureia import MeasureIABase, MeasureIABox, SimInfo


# ---------------------------------------------------------------------------
# Helpers / tiny fixtures
# ---------------------------------------------------------------------------

def make_base(boxsize=100.0, pi_max=None, num_bins_r=8, num_bins_pi=20,
              separation_limits=(0.1, 20.0)):
    """Return a MeasureIABase with minimal synthetic data (no file I/O)."""
    N = 20
    rng = np.random.default_rng(42)
    pos = rng.uniform(0, boxsize, (N, 3))
    e_dir = rng.uniform(-1, 1, (N, 2))
    # normalise so each row is a unit vector
    e_dir /= np.linalg.norm(e_dir, axis=1, keepdims=True)
    q = rng.uniform(0.1, 1.0, N)
    data = {
        "Position": pos,
        "Position_shape_sample": pos,
        "Axis_Direction": e_dir,
        "LOS": 2,
        "q": q,
    }
    return MeasureIABase(
        data=data,
        output_file_name="dummy_output.hdf5",  # never written in these tests
        simulation=None,
        snapshot=None,
        separation_limits=list(separation_limits),
        num_bins_r=num_bins_r,
        num_bins_pi=num_bins_pi,
        pi_max=pi_max if pi_max is not None else boxsize / 2.0,
        boxsize=boxsize,
    )


# ---------------------------------------------------------------------------
# SimInfo tests
# ---------------------------------------------------------------------------

class TestSimInfo:

    @pytest.mark.parametrize("simname, expected_box, expected_h", [
        ("TNG100",        75.0,             0.6774),
        ("TNG300",       205.0,             0.6774),
        ("EAGLE",        100.0 * 0.6777,    0.6777),
        ("HorizonAGN",   100.0,             0.704),
        ("FLAMINGO_L1",  1000.0 * 0.681,    0.681),
        ("FLAMINGO_L2p8",2800.0 * 0.681,    0.681),
        ("COLIBRE_L4",   400.0 * 0.681,     0.681),
        ("COLIBRE_L2",   200.0 * 0.681,     0.681),
    ])
    def test_known_simulation_specs(self, simname, expected_box, expected_h):
        info = SimInfo(simname, snapshot=None)
        assert info.boxsize == pytest.approx(expected_box)
        assert info.h       == pytest.approx(expected_h)
        assert info.L_0p5   == pytest.approx(expected_box / 2.0)

    def test_unknown_simname_raises(self):
        with pytest.raises(KeyError):
            SimInfo("UNKNOWN_SIM", snapshot=None)

    def test_snapshot_string_conversion(self):
        info = SimInfo("TNG100", snapshot=99)
        assert info.snapshot == "99"
        assert info.snap_group == "Snapshot_99/"

    def test_no_snapshot(self):
        info = SimInfo("TNG100", snapshot=None)
        assert info.snapshot is None
        assert info.snap_group == ""

    def test_manual_boxsize(self):
        info = SimInfo(None, snapshot=None, boxsize=300.0)
        assert info.boxsize == 300.0
        assert info.L_0p5   == 150.0
        assert info.simname is None

    def test_none_boxsize_gives_none_l0p5(self):
        info = SimInfo(None, snapshot=None, boxsize=None)
        assert info.L_0p5 is None

    def test_flamingo_missing_size_suffix_raises(self):
        with pytest.raises(KeyError):
            SimInfo("FLAMINGO", snapshot=None)

    def test_colibre_missing_size_suffix_raises(self):
        with pytest.raises(KeyError):
            SimInfo("COLIBRE", snapshot=None)


# ---------------------------------------------------------------------------
# MeasureIABase initialisation tests
# ---------------------------------------------------------------------------

class TestMeasureIABaseInit:

    def test_r_bins_shape(self):
        obj = make_base(num_bins_r=8)
        assert len(obj.r_bins) == 9           # n+1 edges

    def test_r_bins_log_spacing(self):
        obj = make_base(num_bins_r=8, separation_limits=(0.1, 20.0))
        log_bins = np.log10(obj.r_bins)
        diffs = np.diff(log_bins)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_r_min_max(self):
        obj = make_base(separation_limits=(0.5, 30.0))
        assert obj.r_bins[0]  == pytest.approx(0.5)
        assert obj.r_bins[-1] == pytest.approx(30.0)

    def test_pi_bins_shape(self):
        obj = make_base(num_bins_pi=20)
        assert len(obj.pi_bins) == 21

    def test_pi_bins_symmetric(self):
        obj = make_base(pi_max=50.0, num_bins_pi=10)
        assert obj.pi_bins[0]  == pytest.approx(-50.0)
        assert obj.pi_bins[-1] == pytest.approx( 50.0)

    def test_mu_r_bins_range(self):
        obj = make_base()
        assert obj.mu_r_bins[0]  == pytest.approx(-1.0)
        assert obj.mu_r_bins[-1] == pytest.approx( 1.0)

    def test_default_weights_ones(self):
        obj = make_base()
        np.testing.assert_array_equal(obj.data["weight"],
                                      np.ones(obj.Num_position))
        np.testing.assert_array_equal(obj.data["weight_shape_sample"],
                                      np.ones(obj.Num_shape))

    def test_num_position_and_shape(self):
        obj = make_base()
        assert obj.Num_position == 20
        assert obj.Num_shape    == 20

    def test_missing_pi_max_and_boxsize_raises(self):
        data = {"Position": np.zeros((5, 3)), "Position_shape_sample": np.zeros((5, 3)),
                "Axis_Direction": np.zeros((5, 2)), "LOS": 2, "q": np.ones(5)}
        with pytest.raises(ValueError, match="pi_max"):
            MeasureIABase(data, "dummy.hdf5", simulation=None, snapshot=None,
                          pi_max=None, boxsize=None)

    def test_none_data_gives_zero_counts(self):
        obj = MeasureIABase(None, "dummy.hdf5", simulation=None, snapshot=None,
                            pi_max=50.0, boxsize=100.0)
        assert obj.Num_position == 0
        assert obj.Num_shape    == 0


# ---------------------------------------------------------------------------
# calculate_dot_product_arrays
# ---------------------------------------------------------------------------

class TestDotProduct:

    def test_identical_unit_vectors(self):
        a = np.eye(3)
        result = MeasureIABase.calculate_dot_product_arrays(a, a)
        np.testing.assert_allclose(result, np.ones(3))

    def test_orthogonal_vectors(self):
        a1 = np.array([[1., 0., 0.], [0., 1., 0.]])
        a2 = np.array([[0., 1., 0.], [1., 0., 0.]])
        result = MeasureIABase.calculate_dot_product_arrays(a1, a2)
        np.testing.assert_allclose(result, np.zeros(2), atol=1e-15)

    def test_against_numpy_einsum(self):
        rng = np.random.default_rng(7)
        a1 = rng.standard_normal((50, 4))
        a2 = rng.standard_normal((50, 4))
        expected = np.einsum("ij,ij->i", a1, a2)
        result   = MeasureIABase.calculate_dot_product_arrays(a1, a2)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_single_row(self):
        a1 = np.array([[3., 4.]])
        a2 = np.array([[1., 2.]])
        assert MeasureIABase.calculate_dot_product_arrays(a1, a2)[0] == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# get_ellipticity
# ---------------------------------------------------------------------------

class TestGetEllipticity:

    def test_1d_zero_angle(self):
        """e_+ = e*cos(0) = e,  e_x = e*sin(0) = 0"""
        e   = np.array([0.5, 0.3, 0.7])
        phi = np.zeros(3)
        e_p, e_x = MeasureIABase.get_ellipticity(e, phi)
        np.testing.assert_allclose(e_p, e)
        np.testing.assert_allclose(e_x, np.zeros(3), atol=1e-15)

    def test_1d_pi_over_4(self):
        """At phi=pi/4: e_+ = e*cos(pi/2)=0, e_x = e*sin(pi/2)=e"""
        e   = np.array([1.0])
        phi = np.array([np.pi / 4])
        e_p, e_x = MeasureIABase.get_ellipticity(e, phi)
        np.testing.assert_allclose(e_p, [0.0], atol=1e-15)
        np.testing.assert_allclose(e_x, [1.0], atol=1e-15)

    def test_2d_components(self):
        """For 2D input (survey shear-catalogue convention, IA sign):
        e_+ = e1*cos2phi - e2*sin2phi, e_x = e1*sin2phi + e2*cos2phi"""
        e1   = np.array([1.0, 0.0])
        e2   = np.array([0.0, 1.0])
        phi  = np.zeros(2)
        e    = np.column_stack([e1, e2])
        e_p, e_x = MeasureIABase.get_ellipticity(e, phi)
        # phi=0 → cos2phi=1, sin2phi=0
        # e_+ = e1*1 - e2*0 = e1
        np.testing.assert_allclose(e_p, e1)
        # e_x = e1*0 + e2*1 = e2
        np.testing.assert_allclose(e_x, e2)

    def test_2d_radial_alignment_is_positive(self):
        """IA sign convention: a galaxy whose major axis points along the
        separation vector (radial alignment) must have e_+ > 0.

        A separation along the internal east axis (phi = 0) corresponds to
        a survey-frame position angle of 0, where a radially aligned galaxy
        has e1 = +e, e2 = 0."""
        e = np.array([[0.6, 0.0]])
        e_p, e_x = MeasureIABase.get_ellipticity(e, np.zeros(1))
        assert e_p[0] > 0
        np.testing.assert_allclose(e_p, [0.6])
        np.testing.assert_allclose(e_x, [0.0], atol=1e-15)

    def test_1d_output_shape(self):
        e   = np.ones(10)
        phi = np.linspace(0, np.pi, 10)
        e_p, e_x = MeasureIABase.get_ellipticity(e, phi)
        assert e_p.shape == (10,)
        assert e_x.shape == (10,)


# ---------------------------------------------------------------------------
# get_random_pairs (analytical RR)
# ---------------------------------------------------------------------------

class TestGetRandomPairs:

    @pytest.mark.parametrize("corrtype,factor", [
        ("cross", 1.0),
        ("auto",  0.5),   # auto is cross/2 (roughly, ignoring -1 term for large N)
    ])
    def test_cross_larger_than_auto(self, corrtype, factor):
        """Cross RR should be larger than auto RR (for Num_position > 1)."""
        Np, Ns = 100, 100
        rp_max, rp_min = 2.0, 1.0
        pi_max, pi_min = 1.0, 0.0
        L3 = 100.0 ** 3
        rr_cross = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min, L3,
                                                  "cross", Np, Ns)
        rr_auto  = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min, L3,
                                                  "auto", Np, Ns)
        assert rr_cross > rr_auto

    def test_rr_scales_with_volume(self):
        """Doubling the box volume halves RR."""
        Np, Ns = 50, 50
        rp_max, rp_min = 1.0, 0.5
        pi_max, pi_min = 0.5, 0.0
        rr1 = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min,
                                             100.0 ** 3, "cross", Np, Ns)
        rr2 = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min,
                                             200.0 ** 3, "cross", Np, Ns)
        assert rr2 == pytest.approx(rr1 * (100.0 / 200.0) ** 3)

    def test_unknown_corrtype_raises(self):
        with pytest.raises(ValueError, match="Unknown input"):
            MeasureIABase.get_random_pairs(2., 1., 1., 0., 1e6, "unknown", 10, 10)

    def test_formula_cross(self):
        """Manual formula check for cross RR."""
        Np, Ns = 10, 20
        rp_max, rp_min = 3.0, 1.0
        pi_max, pi_min = 2.0, 0.0
        L3 = 50.0 ** 3
        expected = Np * Ns * np.pi * (rp_max**2 - rp_min**2) * abs(pi_max - pi_min) / L3
        result   = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min,
                                                  L3, "cross", Np, Ns)
        assert result == pytest.approx(expected)

    def test_formula_auto(self):
        """Manual formula check for auto RR."""
        Np, Ns = 10, 20
        rp_max, rp_min = 3.0, 1.0
        pi_max, pi_min = 2.0, 0.0
        L3 = 50.0 ** 3
        geometry = np.pi * (rp_max**2 - rp_min**2) * abs(pi_max - pi_min) / L3
        # auto is the cross count halved; with no overlap declared the count is Np * Ns
        result = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min,
                                               L3, "auto", Np, Ns)
        assert result == pytest.approx(Np * Ns / 2.0 * geometry)
        # a true auto-correlation is one sample against itself: every object overlaps,
        # which recovers the familiar N (N - 1) / 2
        result = MeasureIABase.get_random_pairs(rp_max, rp_min, pi_max, pi_min,
                                               L3, "auto", Np, Np, Np)
        assert result == pytest.approx(Np * (Np - 1) / 2.0 * geometry)

    def test_overlap_term_interpolates_between_conventions(self):
        """num_overlap moves the count between the two limits it subsumes: disjoint
        samples (Np * Ns) and a shape sample drawn from the position sample
        (Ns * (Np - 1))."""
        Np, Ns = 10, 20
        args = (3.0, 1.0, 2.0, 0.0, 50.0 ** 3, "cross", Np, Ns)
        disjoint = MeasureIABase.get_random_pairs(*args, 0)
        subset = MeasureIABase.get_random_pairs(*args, Ns)
        assert subset == pytest.approx(disjoint * (Np * Ns - Ns) / (Np * Ns))
        assert subset < disjoint


# ---------------------------------------------------------------------------
# get_random_pairs_r_mur
# ---------------------------------------------------------------------------

class TestGetRandomPairsRMur:

    def test_positive_result(self):
        obj = make_base()
        rr = obj.get_random_pairs_r_mur(2.0, 1.0, 0.5, 0.0,
                                        100.0 ** 3, "cross", 50, 50)
        assert rr > 0.0

    def test_auto_smaller_than_cross(self):
        obj  = make_base()
        kwargs = dict(r_max=2.0, r_min=1.0, mur_max=1.0, mur_min=-1.0,
                      L3=100.0 ** 3, Num_position=100, Num_shape=100)
        rr_cross = obj.get_random_pairs_r_mur(**kwargs, corrtype="cross")
        rr_auto  = obj.get_random_pairs_r_mur(**kwargs, corrtype="auto")
        assert rr_cross > rr_auto

    def test_unknown_corrtype_raises(self):
        obj = make_base()
        with pytest.raises(ValueError, match="Unknown input"):
            obj.get_random_pairs_r_mur(2., 1., 1., 0., 1e6, "bad", 10, 10)

    def test_formula_cross(self):
        obj = make_base()
        Np, Ns = 10, 20
        r_max, r_min = 3.0, 1.0
        mur_max, mur_min = 0.5, 0.0
        L3 = 100.0 ** 3
        geometry = abs(2. * np.pi / 3. * (r_max**3 - r_min**3) * (mur_max - mur_min) / L3)
        # no overlap declared: the samples are treated as independent
        result = obj.get_random_pairs_r_mur(r_max, r_min, mur_max, mur_min,
                                            L3, "cross", Np, Ns)
        assert result == pytest.approx(Np * Ns * geometry)
        # shape sample drawn from the position sample: one self-pair per shape object
        result = obj.get_random_pairs_r_mur(r_max, r_min, mur_max, mur_min,
                                            L3, "cross", Np, Ns, Ns)
        assert result == pytest.approx(Ns * (Np - 1) * geometry)


# ---------------------------------------------------------------------------
# setdiff2D
# ---------------------------------------------------------------------------

class TestSetdiff2D:

    def test_basic_difference(self):
        a1 = [[1, 2, 3], [4, 5, 6]]
        a2 = [[2, 3],    [5, 6, 7]]
        diff = MeasureIABase.setdiff2D(a1, a2)
        assert list(diff[0]) == [1]
        assert list(diff[1]) == [4]

    def test_no_overlap(self):
        a1 = [[1, 2], [3, 4]]
        a2 = [[5, 6], [7, 8]]
        diff = MeasureIABase.setdiff2D(a1, a2)
        np.testing.assert_array_equal(diff[0], [1, 2])
        np.testing.assert_array_equal(diff[1], [3, 4])

    def test_complete_overlap(self):
        a1 = [[1, 2], [3, 4]]
        a2 = [[1, 2], [3, 4]]
        diff = MeasureIABase.setdiff2D(a1, a2)
        assert len(diff[0]) == 0
        assert len(diff[1]) == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            MeasureIABase.setdiff2D([[1, 2]], [[1], [2]])

    def test_returns_sorted(self):
        """np.setdiff1d returns sorted results."""
        a1 = [[3, 1, 2]]
        a2 = [[2]]
        diff = MeasureIABase.setdiff2D(a1, a2)
        assert list(diff[0]) == [1, 3]


# ---------------------------------------------------------------------------
# setdiff_omit
# ---------------------------------------------------------------------------

class TestSetdiffOmit:

    def test_basic(self):
        a1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a2 = [2, 5]
        incl_ind = [0, 1]
        diff = MeasureIABase.setdiff_omit(a1, a2, incl_ind)
        assert len(diff) == 2
        np.testing.assert_array_equal(diff[0], [1, 3])
        np.testing.assert_array_equal(diff[1], [4, 6])

    def test_no_included_indices(self):
        a1 = [[1, 2], [3, 4]]
        a2 = [1]
        diff = MeasureIABase.setdiff_omit(a1, a2, incl_ind=[])
        assert diff == []

    def test_all_indices_included(self):
        a1 = [[1, 2, 3], [4, 5, 6]]
        a2 = [1, 4]
        diff = MeasureIABase.setdiff_omit(a1, a2, incl_ind=[0, 1])
        np.testing.assert_array_equal(diff[0], [2, 3])
        np.testing.assert_array_equal(diff[1], [5, 6])
# Parameterised happy-path table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim, snap, expected_box, expected_snap_group", [
    ("TNG100",        99,   75.0,             "Snapshot_99/"),
    ("TNG100",        None, 75.0,             ""),
    ("TNG300",        99,   205.0,            "Snapshot_99/"),
    ("EAGLE",         28,   100.0 * 0.6777,   "Snapshot_28/"),
    ("HorizonAGN",    1,    100.0,            "Snapshot_1/"),
    ("FLAMINGO_L1",   0,    1000.0 * 0.681,   "Snapshot_0/"),
    ("FLAMINGO_L2p8", 0,    2800.0 * 0.681,   "Snapshot_0/"),
    ("COLIBRE_L4",    0,    400.0  * 0.681,   "Snapshot_0/"),
    ("COLIBRE_L2",    0,    200.0  * 0.681,   "Snapshot_0/"),
])
def test_known_sim_boxsize(sim, snap, expected_box, expected_snap_group):
    obj = MeasureIABox(None, None, sim, snap, pi_max=20.0)
    assert obj.boxsize == pytest.approx(expected_box, rel=1e-6)
    assert obj.snap_group == expected_snap_group


# ---------------------------------------------------------------------------
# Manual boxsize overrides simname
# ---------------------------------------------------------------------------

def test_manual_boxsize_ignores_simname():
    """When simulation AND boxsize are both given, simname boxsize wins."""
    obj = MeasureIABox(None, None, "TNG300", 99, boxsize=999.0, pi_max=20.0)
    # SimInfo gives priority to the hardcoded TNG300 value
    assert obj.boxsize == pytest.approx(205.0)


def test_manual_boxsize_no_simname():
    obj = MeasureIABox(None, None, None, 30, boxsize=300.0, pi_max=20.0)
    assert obj.boxsize  == pytest.approx(300.0)
    assert obj.L_0p5    == pytest.approx(150.0)
    assert obj.snapshot == "30"


def test_no_sim_no_boxsize_no_pimax_raises():
    """Must raise when neither boxsize nor pi_max can determine pi bins."""
    with pytest.raises(ValueError):
        MeasureIABox(None, None, None, None, boxsize=None, pi_max=None)


# ---------------------------------------------------------------------------
# Snapshot string handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snap_in, snap_str, group_str", [
    (99,    "99",   "Snapshot_99/"),
    (0,     "0",    "Snapshot_0/"),
    ("abc", "abc",  "Snapshot_abc/"),
    (None,  None,   ""),
])
def test_snapshot_attribute(snap_in, snap_str, group_str):
    obj = MeasureIABox(None, None, "TNG100", snap_in, pi_max=20.0)
    assert obj.snapshot   == snap_str
    assert obj.snap_group == group_str


# ---------------------------------------------------------------------------
# pi_max derived from boxsize when not given explicitly
# ---------------------------------------------------------------------------

def test_pi_max_derived_from_boxsize():
    """When pi_max=None, pi_bins should span ±L/2."""
    obj = MeasureIABox(None, None, "TNG100", None, pi_max=None)
    assert obj.pi_bins[0]  == pytest.approx(-37.5)
    assert obj.pi_bins[-1] == pytest.approx( 37.5)


def test_pi_max_explicit_overrides():
    obj = MeasureIABox(None, None, "TNG100", None, pi_max=10.0)
    assert obj.pi_bins[0]  == pytest.approx(-10.0)
    assert obj.pi_bins[-1] == pytest.approx( 10.0)


# ---------------------------------------------------------------------------
# Unknown / malformed simnames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sim", [
    "TNG50",
    "Millennium",
    "FLAMINGO",   # missing size suffix
    "COLIBRE",    # missing size suffix
    "",
])
def test_unknown_simname_raises(bad_sim):
    with pytest.raises(KeyError):
        MeasureIABox(None, None, bad_sim, None, pi_max=20.0)


# ---------------------------------------------------------------------------
# h-parameter sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim, h_expected", [
    ("TNG100",      0.6774),
    ("TNG300",      0.6774),
    ("EAGLE",       0.6777),
    ("HorizonAGN",  0.704),
    ("FLAMINGO_L1", 0.681),
    ("COLIBRE_L4",  0.681),
])
def test_h_parameter(sim, h_expected):
    obj = MeasureIABox(None, None, sim, None, pi_max=20.0)
    assert obj.h == pytest.approx(h_expected)


# ---------------------------------------------------------------------------
# L_0p5 consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim", ["TNG100", "TNG300", "HorizonAGN"])
def test_l0p5_is_half_boxsize(sim):
    obj = MeasureIABox(None, None, sim, None, pi_max=20.0)
    assert obj.L_0p5 == pytest.approx(obj.boxsize / 2.0)


# ===========================================================================
# _measure_w_g_i — return_output path and corr_type routing
# ===========================================================================

class TestMeasureWGI:
    """
    Direct tests for MeasureIABase._measure_w_g_i.
    We run measure_xi_w(..., return_output=False) first to populate the
    intermediate xi groups, then call _measure_w_g_i with return_output=True
    and verify the returned arrays match what was written to disk.
    """

    def test_return_output_true_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("wgi_gp", "g+", 0, temp_file_path=False)
        result = obj._measure_w_g_i("wgi_gp", corr_type="g+",
                                     return_output=True)
        # result shape: (num_bins_r, 2)  — columns are [rp, w_g+]
        assert result.shape == (obj.num_bins_r, 2)
        rp_ret = result[:, 0]
        w_ret  = result[:, 1]
        # rp must be sorted and positive
        assert np.all(np.diff(rp_ret) > 0)
        assert np.all(rp_ret > 0)
        # w values must be finite
        assert np.all(np.isfinite(w_ret))

    def test_return_output_true_gg(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("wgi_gg", "gg", 0, temp_file_path=False)
        result = obj._measure_w_g_i("wgi_gg", corr_type="gg",
                                     return_output=True)
        assert result.shape == (obj.num_bins_r, 2)

    def test_return_output_false_writes_to_file(self, IA_mock_TNG300_n1,
                                                 tmp_path):
        import h5py
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("wgi_write", "g+", 0, temp_file_path=False)
        obj._measure_w_g_i("wgi_write", corr_type="g+", return_output=False)
        with h5py.File(obj.output_file_name, "r") as f:
            assert "wgi_write" in f[f"{obj.snap_group}w_g_plus"]
            assert "wgi_write_rp" in f[f"{obj.snap_group}w_g_plus"]

    def test_return_output_matches_written_value(self, IA_mock_TNG300_n1,
                                                  tmp_path):
        import h5py
        obj = IA_mock_TNG300_n1
        obj.measure_xi_w("wgi_match", "g+", 0, temp_file_path=False)
        result = obj._measure_w_g_i("wgi_match", corr_type="g+",
                                     return_output=True)
        obj._measure_w_g_i("wgi_match", corr_type="g+", return_output=False)
        with h5py.File(obj.output_file_name, "r") as f:
            w_stored  = f[f"{obj.snap_group}w_g_plus/wgi_match"][:]
            rp_stored = f[f"{obj.snap_group}w_g_plus/wgi_match_rp"][:]
        np.testing.assert_allclose(result[:, 1], w_stored)
        np.testing.assert_allclose(result[:, 0], rp_stored)

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1):
        with pytest.raises(KeyError):
            IA_mock_TNG300_n1._measure_w_g_i("x", corr_type="bad",
                                              return_output=True)


# ===========================================================================
# _measure_multipoles — return_output path and corr_type routing
# ===========================================================================

class TestMeasureMultipoles:

    def test_return_output_true_gp(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mm_gp", "g+", 0, temp_file_path=False)
        result = obj._measure_multipoles("mm_gp", corr_type="g+",
                                          return_output=True)
        assert result.shape == (obj.num_bins_r, 2)
        r_ret = result[:, 0]
        assert np.all(np.diff(r_ret) > 0)
        assert np.all(r_ret > 0)
        assert np.all(np.isfinite(result[:, 1]))

    def test_return_output_true_gg(self, IA_mock_TNG300_n1, tmp_path):
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mm_gg", "gg", 0, temp_file_path=False)
        result = obj._measure_multipoles("mm_gg", corr_type="gg",
                                          return_output=True)
        assert result.shape == (obj.num_bins_r, 2)

    def test_return_output_false_writes_to_file(self, IA_mock_TNG300_n1,
                                                 tmp_path):
        import h5py
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mm_write", "g+", 0, temp_file_path=False)
        obj._measure_multipoles("mm_write", corr_type="g+",
                                 return_output=False)
        with h5py.File(obj.output_file_name, "r") as f:
            assert "mm_write"   in f[f"{obj.snap_group}multipoles_g_plus"]
            assert "mm_write_r" in f[f"{obj.snap_group}multipoles_g_plus"]

    def test_return_output_matches_written_value(self, IA_mock_TNG300_n1,
                                                  tmp_path):
        import h5py
        obj = IA_mock_TNG300_n1
        obj.measure_xi_multipoles("mm_match", "g+", 0, temp_file_path=False)
        result = obj._measure_multipoles("mm_match", corr_type="g+",
                                          return_output=True)
        obj._measure_multipoles("mm_match", corr_type="g+", return_output=False)
        with h5py.File(obj.output_file_name, "r") as f:
            m_stored = f[f"{obj.snap_group}multipoles_g_plus/mm_match"][:]
            r_stored = f[f"{obj.snap_group}multipoles_g_plus/mm_match_r"][:]
        np.testing.assert_allclose(result[:, 1], m_stored)
        np.testing.assert_allclose(result[:, 0], r_stored)

    def test_invalid_corr_type_raises(self, IA_mock_TNG300_n1):
        with pytest.raises(KeyError):
            IA_mock_TNG300_n1._measure_multipoles("x", corr_type="bad",
                                                   return_output=True)


# ---------------------------------------------------------------------------
# sample overlap: available_pairs / count_overlap / the num_overlap override
# ---------------------------------------------------------------------------

class TestSampleOverlap:
    """The analytic RR is normalised by the number of *available* pairs,
    Num_position * Num_shape - num_overlap, which subsumes both conventions in use."""

    def test_available_pairs_limits(self):
        from measureia.measure_IA_base import available_pairs
        Np, Ns = 10, 4
        assert available_pairs(Np, Ns, 0) == Np * Ns                  # independent samples
        assert available_pairs(Np, Ns, Ns) == Ns * (Np - 1)           # shapes drawn from positions
        assert available_pairs(Np, Np, Np, "auto") == Np * (Np - 1) / 2

    def test_available_pairs_rejects_unknown_corrtype(self):
        from measureia.measure_IA_base import available_pairs
        with pytest.raises(ValueError, match="Unknown input"):
            available_pairs(10, 10, 0, "bad")

    def test_count_overlap_matches_rows_exactly(self):
        from measureia.measure_IA_base import count_overlap, overlap_indices
        a = np.arange(30, dtype=float).reshape(10, 3)
        assert count_overlap(a, a) == 10
        assert count_overlap(a[:4], a[6:]) == 0
        assert count_overlap(a, a[3:7]) == 4
        np.testing.assert_array_equal(np.sort(overlap_indices(a[3:7], a)), [3, 4, 5, 6])

    def test_overlap_measured_from_the_catalogue(self, _box_catalog):
        """The usual IA setup: the shape sample is drawn from the position sample, so the
        overlap is found without the user declaring anything."""
        from measureia import MeasureIABox, pair_kernel
        obj = MeasureIABox(_box_catalog, "dummy_output.hdf5", boxsize=205.0)
        pair_kernel.prepare_box_samples(obj.data, None, obj.Num_position, obj.Num_shape,
                                        shapes=True, ellipticity="distortion", base=obj)
        assert obj.num_overlap == obj.Num_shape

    def test_num_overlap_override_is_respected(self, _box_catalog):
        from measureia import MeasureIABox, pair_kernel
        for override in (0, 3):
            obj = MeasureIABox(_box_catalog, "dummy_output.hdf5", boxsize=205.0,
                               num_overlap=override)
            pair_kernel.prepare_box_samples(obj.data, None, obj.Num_position, obj.Num_shape,
                                            shapes=True, ellipticity="distortion", base=obj)
            assert obj.num_overlap == override

    @pytest.mark.parametrize("bad", [-1, 1.5, True, "5"])
    def test_num_overlap_validated(self, _box_catalog, bad):
        from measureia import MeasureIABox
        with pytest.raises(ValueError, match="num_overlap"):
            MeasureIABox(_box_catalog, "dummy_output.hdf5", boxsize=205.0, num_overlap=bad)
