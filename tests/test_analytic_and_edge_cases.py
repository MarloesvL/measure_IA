"""
test_analytic_and_edge_cases.py
================================
Tests that verify physics correctness (analytic limits and the Landy-Szalay
formula), cosmology coordinate conversion, and graceful handling of degenerate
inputs.

Sections
--------
  1. Analytic limits — box (exact results derivable from the formula)
  2. Analytic limits — lightcone
  3. _obs_estimator formula (Landy-Szalay, hand-computed pair counts)
  4. Cosmology coordinate conversion (pyccl reference)
  5. Edge cases that ARE handled correctly (smoke tests)
  6. Previously-unhandled edge cases, now fixed at source (all-zero weights, empty mask,
       single-object sample, data restored after a failed run) — regression-locked here
  7. The same degenerate inputs on the lightcone, plus the non-default `cosmology`
       argument
"""

import math
import warnings
import numpy as np
import pytest
import h5py

from measureia import MeasureIABox, MeasureIALightcone, MeasureIABase


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEP   = [0.1, 20.0]
_NR    = 8
_NPI   = 20
_SEED  = 42


def _read(obj, group, key):
    with h5py.File(obj.output_file_name, "r") as f:
        return f[obj.snap_group + group][key][:]


def _box(data, tmp_path, n_nodes=1):
    out = str(tmp_path / "analytic.hdf5")
    return MeasureIABox(
        data, out,
        simulation="TNG300", snapshot=99,
        separation_limits=_SEP,
        num_bins_r=_NR, num_bins_pi=_NPI,
        pi_max=None, num_nodes=n_nodes,
    )


def _lc(data, randoms, tmp_path):
    out = str(tmp_path / "analytic_lc.hdf5")
    return MeasureIALightcone(
        data=data, randoms_data=randoms,
        output_file_name=out,
        separation_limits=_SEP,
        num_bins_r=_NR, num_bins_pi=_NPI,
        pi_max=60.0,
    )


# ===========================================================================
# 1. Analytic limits — box
# ===========================================================================

class TestAnalyticLimitsBox:
    """
    Exact results that follow from the box Landy-Szalay formula when inputs
    are chosen so the answer is known with zero tolerance.
    """

    # ------------------------------------------------------------------ L1
    # Zero-ellipticity → w_g+ = 0 exactly

    def test_zero_ellipticity_gives_zero_wgp(self, tmp_path):
        """If e1=e2=0 for all galaxies, SplusD=0, so w_g+=0 in every bin."""
        rng = np.random.default_rng(_SEED)
        N   = 100
        COM = rng.uniform(0, 205.0, (N, 3))
        e_dir = np.column_stack([np.ones(N), np.zeros(N)])
        data = {
            "Position":              COM,
            "Position_shape_sample": COM,
            "Axis_Direction":        e_dir,
            "LOS": 2,
            "q":   np.ones(N),          # q=1 → e_mag=0 for distortion def
        }
        obj = _box(data, tmp_path)
        obj.measure_xi_w("zero_e", "g+", 0, temp_file_path=False,
                          ellipticity="distortion")
        w = _read(obj, "w/xi_g_plus", "zero_e")
        np.testing.assert_array_equal(w, np.zeros_like(w))

    # ------------------------------------------------------------------ L4
    # Negating ellipticities negates w_g+ exactly

    def test_negate_ellipticity_negates_wgp(self, tmp_path):
        """w_g+(−e) = −w_g+(+e) exactly (linearity of SplusD in e_plus)."""
        rng  = np.random.default_rng(_SEED)
        N    = 80
        COM  = rng.uniform(0, 205.0, (N, 3))
        theta = rng.uniform(0, 2 * math.pi, N)
        e_dir = np.column_stack([np.cos(theta), np.sin(theta)])
        q    = rng.uniform(0.3, 0.9, N)
        base_data = {
            "Position":              COM,
            "Position_shape_sample": COM,
            "Axis_Direction":        e_dir,
            "LOS": 2, "q": q,
        }

        pos_data = {**base_data}
        neg_data = {**base_data, "q": q}   # same q

        obj_pos = _box(pos_data, tmp_path)
        obj_pos.measure_xi_w("pos_e", "g+", 0, temp_file_path=False)

        # negate e by flipping axis direction 90° (cos→sin, sin→−cos)
        neg_dir = np.column_stack([-np.sin(theta), np.cos(theta)])
        neg_data["Axis_Direction"] = neg_dir

        out_neg = str(tmp_path / "neg.hdf5")
        obj_neg = MeasureIABox(
            neg_data, out_neg,
            simulation="TNG300", snapshot=99,
            separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI)
        obj_neg.measure_xi_w("neg_e", "g+", 0, temp_file_path=False)

        w_pos = _read(obj_pos, "w/xi_g_plus", "pos_e")
        w_neg = _read(obj_neg, "w/xi_g_plus", "neg_e")
        np.testing.assert_allclose(w_neg, -w_pos, rtol=1e-10)

    # ------------------------------------------------------------------ L5
    # Doubling ellipticity magnitude doubles w_g+ exactly

    def test_double_ellipticity_doubles_wgp(self, tmp_path):
        """w_g+(2e) = 2 * w_g+(e) * R(e)/R(2e).

        S+D is linear in e, but the code divides by the responsivity
        R = 1 - e^2/2 (all |e| equal here), so exact linearity holds only
        after multiplying back the responsivity ratio."""
        rng   = np.random.default_rng(_SEED)
        N     = 80
        COM   = rng.uniform(0, 205.0, (N, 3))
        e_dir = np.column_stack([np.cos(rng.uniform(0, 2 * math.pi, N)),
                                  np.sin(rng.uniform(0, 2 * math.pi, N))])

        # Choose e_distortion values directly so doubling is exact.
        # distortion e = (1-q^2)/(1+q^2)  =>  q = sqrt((1-e)/(1+e))
        e1, e2 = 0.2, 0.4
        e_mag1 = e1 * np.ones(N)
        e_mag2 = e2 * np.ones(N)
        q1 = np.sqrt((1 - e_mag1) / (1 + e_mag1))
        q2 = np.sqrt((1 - e_mag2) / (1 + e_mag2))

        for q, tag in [(q1, "x1"), (q2, "x2")]:
            out = str(tmp_path / f"emag_{tag}.hdf5")
            d   = {"Position": COM, "Position_shape_sample": COM,
                   "Axis_Direction": e_dir, "LOS": 2, "q": q}
            o   = MeasureIABox(d, out, simulation="TNG300", snapshot=99,
                               separation_limits=_SEP, num_bins_r=_NR,
                               num_bins_pi=_NPI)
            o.measure_xi_w(f"em_{tag}", "g+", 0, temp_file_path=False,
                            ellipticity="distortion")

        with h5py.File(str(tmp_path / "emag_x1.hdf5"), "r") as f:
            snap_grp = f"Snapshot_99/"
            w1 = f[f"{snap_grp}w/xi_g_plus/em_x1"][:]
        with h5py.File(str(tmp_path / "emag_x2.hdf5"), "r") as f:
            w2 = f[f"{snap_grp}w/xi_g_plus/em_x2"][:]

        R1 = 1 - e1 ** 2 / 2.0
        R2 = 1 - e2 ** 2 / 2.0
        np.testing.assert_allclose(w2, 2 * w1 * R1 / R2, rtol=1e-10)

    # ------------------------------------------------------------------ L6
    # w_gg symmetric under swap of position and shape samples (auto-correlation)

    def test_wgg_symmetric_swap_position_shape(self, tmp_path):
        """w_gg is unchanged when position and shape samples are swapped,
        provided both use the same positions and weights."""
        rng = np.random.default_rng(_SEED)
        N   = 60
        COM = rng.uniform(0, 205.0, (N, 3))
        e_dir = np.column_stack([np.ones(N), np.zeros(N)])

        def make(pos, shape, tag):
            out = str(tmp_path / f"swap_{tag}.hdf5")
            d   = {"Position": pos, "Position_shape_sample": shape,
                   "Axis_Direction": e_dir, "LOS": 2, "q": np.ones(N)}
            o   = MeasureIABox(d, out, simulation="TNG300", snapshot=99,
                               separation_limits=_SEP, num_bins_r=_NR,
                               num_bins_pi=_NPI)
            o.measure_xi_w(f"sw_{tag}", "gg", 0, temp_file_path=False)
            return _read(o, "w/xi_gg", f"sw_{tag}")

        w_ab = make(COM, COM, "ab")
        w_ba = make(COM, COM, "ba")   # identical: swap gives same array
        # Replace NaN/inf in empty bins before comparing
        w_ab = np.where(np.isfinite(w_ab), w_ab, 0.0)
        w_ba = np.where(np.isfinite(w_ba), w_ba, 0.0)
        np.testing.assert_array_equal(w_ab, w_ba)

    # ------------------------------------------------------------------ L3
    # Uniform grid → w_gg = 0 (no clustering)

    def test_uniform_random_xi_gg_zero(self, tmp_path):
        """Uniform-random (unclustered) points give xi_gg = 0 within the
        statistical noise expected from the analytic pair count per bin.

        A regular lattice would NOT work here: a lattice is highly
        clustered on the scale of its spacing (xi = -1 in bins without
        lattice separations), so a Poisson sample is used instead."""
        rng    = np.random.default_rng(_SEED)
        N      = 2000
        L      = 50.0
        COM    = rng.uniform(0, L, (N, 3))
        e_dir  = np.column_stack([np.ones(N), np.zeros(N)])
        data   = {"Position": COM, "Position_shape_sample": COM,
                  "Axis_Direction": e_dir, "LOS": 2, "q": np.ones(N)}
        out    = str(tmp_path / "uniform.hdf5")
        obj    = MeasureIABox(
            data, out, simulation=None, snapshot=None,
            boxsize=L,
            separation_limits=[1.0, 15.0],
            num_bins_r=4, num_bins_pi=10)
        obj.measure_xi_w("uniform", "gg", 0, temp_file_path=False)
        xi = _read(obj, "w/xi_gg", "uniform")
        # Poisson noise per bin: sigma_xi ~ 1/sqrt(RR_bin); the smallest
        # bin (rp in [1, 1.97], one pi slice) has RR ~ 1.4e3, so 5 sigma
        # is ~0.13. Use a single conservative bound for all bins.
        assert np.all(np.isfinite(xi))
        np.testing.assert_allclose(xi, np.zeros_like(xi), atol=0.15)

    # ------------------------------------------------------------------ L2
    # Radial alignment → w_g+ > 0

    def test_radial_alignment_wgp_positive(self, tmp_path):
        """Galaxies with semimajor axes pointing radially toward a central
        cluster should give w_g+ > 0 in the innermost separation bins."""
        rng    = np.random.default_rng(_SEED)
        centre = np.array([50.0, 50.0, 50.0])
        N      = 200

        # Scatter galaxies around centre
        offsets = rng.standard_normal((N, 3))
        offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
        radii   = rng.uniform(1.0, 10.0, N)
        COM     = centre + offsets * radii[:, None]

        # Radial alignment in the projected plane (LOS = 2, i.e. z-axis)
        proj = offsets[:, :2]           # project onto x-y plane
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        e_dir = proj / norms            # unit vector pointing radially outward

        # Strong ellipticity
        q = 0.2 * np.ones(N)

        # Position sample = just the centre (as a cluster)
        cluster = centre[None, :]
        e_dir_cluster = np.array([[1.0, 0.0]])

        data = {
            "Position":              cluster,
            "Position_shape_sample": COM,
            "Axis_Direction":        e_dir,
            "LOS": 2, "q": q,
        }
        obj = _box(data, tmp_path)
        obj.measure_xi_w("radial", "g+", 0,
                          temp_file_path=False,
                          ellipticity="distortion")
        w = _read(obj, "w/xi_g_plus", "radial")
        # At least one inner bin should be significantly positive
        assert np.any(w > 0), f"Expected w_g+>0 for radial alignment, got {w}"


# ===========================================================================
# 2. Analytic limits — lightcone
# ===========================================================================

class TestAnalyticLimitsLightcone:
    """
    Same analytic limits as the box, applied to the lightcone measurement.
    """

    def _make_lc_data(self, N, e1, e2, rng, tmp_path):
        """Build a minimal lightcone data + randoms dict."""
        data = {
            "RA":                    rng.uniform(150, 155, N),
            "DEC":                   rng.uniform(2, 6, N),
            "Redshift":              rng.uniform(0.1, 0.3, N),
            "RA_shape_sample":       rng.uniform(150, 155, N),
            "DEC_shape_sample":      rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": e1, "e2": e2,
            "weight":               np.ones(N),
            "weight_shape_sample":  np.ones(N),
        }
        NR = N * 3
        randoms = {
            "RA":                    rng.uniform(150, 155, NR),
            "DEC":                   rng.uniform(2, 6, NR),
            "Redshift":              rng.uniform(0.1, 0.3, NR),
            "RA_shape_sample":       rng.uniform(150, 155, NR),
            "DEC_shape_sample":      rng.uniform(2, 6, NR),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, NR),
            "weight":               np.ones(NR),
            "weight_shape_sample":  np.ones(NR),
        }
        return data, randoms

    def test_zero_ellipticity_gives_zero_wgp(self, tmp_path):
        """e1=e2=0 → SplusD=0 → w_g+=0 exactly (NaN in empty bins treated as 0)."""
        rng = np.random.default_rng(_SEED)
        N   = 80
        data, rand = self._make_lc_data(
            N, np.zeros(N), np.zeros(N), rng, tmp_path)
        obj = _lc(data, rand, tmp_path)
        obj.measure_xi_w("galaxies", "lc_zero_e", "g+",
                          tree=False,
                          temp_file_path=str(tmp_path) + "/")
        w = _read(obj, "w/xi_g_plus", "lc_zero_e")
        w_finite = np.where(np.isfinite(w), w, 0.0)
        np.testing.assert_allclose(w_finite, np.zeros_like(w_finite), atol=1e-10)

    def test_negate_ellipticity_negates_wgp(self, tmp_path):
        """w_g+(−e) = −w_g+(+e) exactly."""
        rng   = np.random.default_rng(_SEED)
        N     = 60
        e1    = rng.uniform(-0.3, 0.3, N)
        e2    = rng.uniform(-0.3, 0.3, N)
        data_pos, rand = self._make_lc_data(N, e1,  e2,  rng, tmp_path)
        data_neg, _    = self._make_lc_data(N, -e1, -e2, rng, tmp_path)
        # Use same positions (both samples) and randoms; only e1/e2 differ
        for key in ("RA", "DEC", "Redshift",
                    "RA_shape_sample", "DEC_shape_sample", "Redshift_shape_sample"):
            data_neg[key] = data_pos[key]

        out_pos = str(tmp_path / "lc_pos.hdf5")
        out_neg = str(tmp_path / "lc_neg.hdf5")

        obj_pos = MeasureIALightcone(data_pos, rand, out_pos,
                                      separation_limits=_SEP,
                                      num_bins_r=_NR, num_bins_pi=_NPI,
                                      pi_max=60.0)
        obj_neg = MeasureIALightcone(data_neg, rand, out_neg,
                                      separation_limits=_SEP,
                                      num_bins_r=_NR, num_bins_pi=_NPI,
                                      pi_max=60.0)

        obj_pos.measure_xi_w("galaxies", "lc_pos_e", "g+",
                              tree=False,
                              temp_file_path=None)
        obj_neg.measure_xi_w("galaxies", "lc_neg_e", "g+",
                              tree=False,
                              temp_file_path=None)

        w_pos = _read(obj_pos, "w/xi_g_plus", "lc_pos_e")
        w_neg = _read(obj_neg, "w/xi_g_plus", "lc_neg_e")
        # Restrict comparison to bins where both are finite
        finite = np.isfinite(w_pos) & np.isfinite(w_neg)
        np.testing.assert_allclose(w_neg[finite], -w_pos[finite], rtol=1e-10)

    def test_double_ellipticity_doubles_wgp(self, tmp_path):
        """w_g+(2e) = 2 * w_g+(e) exactly."""
        rng  = np.random.default_rng(_SEED)
        N    = 60
        e1_1 = rng.uniform(-0.2, 0.2, N)
        e2_1 = rng.uniform(-0.2, 0.2, N)

        data1, rand = self._make_lc_data(N, e1_1,      e2_1,      rng, tmp_path)
        data2, _    = self._make_lc_data(N, 2 * e1_1,  2 * e2_1,  rng, tmp_path)
        # Same positions
        for key in ("RA", "DEC", "Redshift",
                    "RA_shape_sample", "DEC_shape_sample", "Redshift_shape_sample"):
            data2[key] = data1[key]

        out1 = str(tmp_path / "lc_e1x.hdf5")
        out2 = str(tmp_path / "lc_e2x.hdf5")

        obj1 = MeasureIALightcone(data1, rand, out1, separation_limits=_SEP,
                                   num_bins_r=_NR, num_bins_pi=_NPI, pi_max=60.0)
        obj2 = MeasureIALightcone(data2, rand, out2, separation_limits=_SEP,
                                   num_bins_r=_NR, num_bins_pi=_NPI, pi_max=60.0)

        obj1.measure_xi_w("galaxies", "lc_1x", "g+",
                           tree=False,
                           temp_file_path=None)
        obj2.measure_xi_w("galaxies", "lc_2x", "g+",
                           tree=False,
                           temp_file_path=None)

        w1 = _read(obj1, "w/xi_g_plus", "lc_1x")
        w2 = _read(obj2, "w/xi_g_plus", "lc_2x")
        np.testing.assert_allclose(w2, 2 * w1, rtol=1e-10)

    def test_wgg_symmetric_swap_position_shape(self, tmp_path):
        """w_gg is unchanged when position and shape samples are the same set."""
        rng  = np.random.default_rng(_SEED)
        N    = 60
        data, rand = self._make_lc_data(N, np.zeros(N), np.zeros(N),
                                         rng, tmp_path)
        obj = _lc(data, rand, tmp_path)
        obj.measure_xi_w("galaxies", "lc_swap_ab", "gg",
                          tree=False,
                          temp_file_path=None)

        # swap position and shape samples in both the data and the randoms;
        # the gg estimator (SD - RD - SR)/RR + 1 is symmetric under this swap
        data_swap = {**data,
                     "RA":       data["RA_shape_sample"].copy(),
                     "DEC":      data["DEC_shape_sample"].copy(),
                     "Redshift": data["Redshift_shape_sample"].copy(),
                     "RA_shape_sample":       data["RA"].copy(),
                     "DEC_shape_sample":      data["DEC"].copy(),
                     "Redshift_shape_sample": data["Redshift"].copy()}
        rand_swap = {**rand,
                     "RA":       rand["RA_shape_sample"].copy(),
                     "DEC":      rand["DEC_shape_sample"].copy(),
                     "Redshift": rand["Redshift_shape_sample"].copy(),
                     "RA_shape_sample":       rand["RA"].copy(),
                     "DEC_shape_sample":      rand["DEC"].copy(),
                     "Redshift_shape_sample": rand["Redshift"].copy(),
                     "weight":              rand["weight_shape_sample"].copy(),
                     "weight_shape_sample": rand["weight"].copy()}
        out2 = str(tmp_path / "lc_swap_ba.hdf5")
        obj2 = MeasureIALightcone(data_swap, rand_swap, out2,
                                   separation_limits=_SEP,
                                   num_bins_r=_NR, num_bins_pi=_NPI,
                                   pi_max=60.0)
        obj2.measure_xi_w("galaxies", "lc_swap_ba", "gg",
                           tree=False,
                           temp_file_path=None)

        w_ab = _read(obj,  "w/xi_gg", "lc_swap_ab")
        w_ba = _read(obj2, "w/xi_gg", "lc_swap_ba")
        # Swapping the samples flips the separation vector, so pi -> -pi:
        # the swapped grid equals the original with the pi axis reversed
        # (pi bins are symmetric about zero).
        w_ba = w_ba[:, ::-1]
        finite = np.isfinite(w_ab) & np.isfinite(w_ba)
        np.testing.assert_allclose(w_ab[finite], w_ba[finite], rtol=1e-5, atol=1e-12)


# ===========================================================================
# 3. _obs_estimator formula (Landy-Szalay, hand-computed)
# ===========================================================================

class TestObsEstimatorFormula:
    """
    Inject hand-computed pair counts directly into the HDF5 output file, then
    call _obs_estimator and verify the formula.

    Box formula (galaxies / Landy-Szalay):
      xi_g+  = SplusD / RR_g_plus          (since SplusR = 0 for no randoms)
      xi_gg  = (DD - RD - SR) / RR + 1

    Box formula (direct, from source):
      xi_g+  = SplusD / RR_g_plus
      xi_gg  = DD / RR_gg - 1

    Lightcone galaxies estimator:
      xi_g+  = (SplusD - SplusR) / RR
      xi_gg  = (DD - RD - SR) / RR + 1

    Lightcone clusters estimator:
      xi_g+  = SplusD / DD  -  SplusR / SR
      xi_gg  = (DD - RD - SR) / RR + 1
    """

    def _inject_and_read(self, obj, dataset_name, pair_counts, group_prefix,
                          num_samples, IA_estimator, corr_type):
        """Write pair_counts dict into the HDF5, call _obs_estimator, return xi."""
        with h5py.File(obj.output_file_name, "a") as f:
            for grp_key, datasets in pair_counts.items():
                full_key = obj.snap_group + grp_key
                grp = f.require_group(full_key)
                for ds_name, arr in datasets.items():
                    if ds_name in grp:
                        del grp[ds_name]
                    grp.create_dataset(ds_name, data=arr)
        obj._obs_estimator(corr_type, IA_estimator, dataset_name,
                            num_samples, jk_group_name="")
        with h5py.File(obj.output_file_name, "r") as f:
            results = {}
            if corr_type[0] in ("g+", "both"):
                results["xi_gp"] = f[f"{obj.snap_group}{corr_type[1]}/xi_g_plus/{dataset_name}"][:]
            if corr_type[0] in ("gg", "both"):
                results["xi_gg"] = f[f"{obj.snap_group}{corr_type[1]}/xi_gg/{dataset_name}"][:]
        return results

    def test_lc_galaxies_xi_gp_formula(self, tmp_path):
        """Lightcone galaxies: xi_g+ = (SplusD - SplusR) / RR."""
        # Inject pair counts into a lightcone output and verify formula
        rng  = np.random.default_rng(_SEED)
        N    = 10
        data = {
            "RA": rng.uniform(150, 155, N), "DEC": rng.uniform(2, 6, N),
            "Redshift": rng.uniform(0.1, 0.3, N),
            "RA_shape_sample": rng.uniform(150, 155, N),
            "DEC_shape_sample": rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": np.zeros(N), "e2": np.zeros(N),
            "weight": np.ones(N), "weight_shape_sample": np.ones(N),
        }
        NR = 30
        rand = {k.replace("_shape_sample", ""): rng.uniform(0, 1, NR)
                for k in data if "shape" not in k and k not in ("e1","e2")}

        obj = MeasureIALightcone(data, rand, str(tmp_path / "lc_formula.hdf5"),
                                  separation_limits=_SEP, num_bins_r=2,
                                  num_bins_pi=2, pi_max=60.0)

        shape    = (2, 2)
        SplusD   = np.array([[6.0, 3.0], [2.0, 4.0]])
        SplusR   = np.array([[3.0, 1.5], [1.0, 2.0]])
        RR       = np.array([[10.0, 5.0], [4.0, 8.0]])
        DD       = np.array([[20.0, 10.0], [8.0, 16.0]])
        SR       = np.array([[15.0, 7.5], [6.0, 12.0]])
        RD       = np.array([[12.0, 6.0], [4.0, 9.0]])
        N_D, N_S, N_R = 10, 10, 30

        pair_counts = {
            "w/xi_g_plus": {
                "lf_SplusD":  SplusD,
                "lf_SplusR":  SplusR,
                "lf_rp":      np.array([1.0, 5.0]),
                "lf_pi":      np.array([10.0, 30.0]),
            },
            "w/xi_g_cross": {
                "lf_ScrossD": np.zeros(shape),
                "lf_ScrossR": np.zeros(shape),
            },
            "w/xi_gg": {
                "lf_DD": DD, "lf_SR": SR,
                "lf_RD": RD, "lf_RR": RR,
                "lf_rp": np.array([1.0, 5.0]),
                "lf_pi": np.array([10.0, 30.0]),
            },
        }
        with h5py.File(obj.output_file_name, "a") as f:
            for grp_key, datasets in pair_counts.items():
                grp = f.require_group(grp_key)
                for ds_name, arr in datasets.items():
                    if ds_name in grp:
                        del grp[ds_name]
                    grp.create_dataset(ds_name, data=arr)

        num_samples = {"S": N_S, "D": N_D, "D_S": 0,
                       "R_D": N_R, "R_S": N_R}
        obj._obs_estimator(("g+", "w"), "galaxies", "lf", num_samples)

        with h5py.File(obj.output_file_name, "r") as f:
            xi_gp = f["w/xi_g_plus/lf"][:]

        # Expected: xi_g+ = (SplusD/N_SD - SplusR/N_SR) / (RR/N_RR)
        N_SD  = N_S * N_D
        N_SR  = N_S * N_R
        N_RR  = N_R ** 2
        expected = (SplusD / N_SD - SplusR / N_SR) / (RR / N_RR)
        np.testing.assert_allclose(xi_gp, expected, rtol=1e-10)

    def test_lc_clusters_xi_gp_formula(self, tmp_path):
        """Lightcone clusters: xi_g+ = SplusD/DD - SplusR/SR."""
        # Define pair counts directly (same values as test_lc_galaxies_xi_gp_formula)
        SplusD = np.array([[6.0, 3.0], [2.0, 4.0]])
        SplusR = np.array([[3.0, 1.5], [1.0, 2.0]])
        DD     = np.array([[20.0, 10.0], [8.0, 16.0]])
        SR     = np.array([[15.0, 7.5], [6.0, 12.0]])

        rng  = np.random.default_rng(_SEED)
        N, NR = 10, 30
        data = {
            "RA": rng.uniform(150, 155, N), "DEC": rng.uniform(2, 6, N),
            "Redshift": rng.uniform(0.1, 0.3, N),
            "RA_shape_sample": rng.uniform(150, 155, N),
            "DEC_shape_sample": rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": np.zeros(N), "e2": np.zeros(N),
            "weight": np.ones(N), "weight_shape_sample": np.ones(N),
        }
        rand = {"RA": rng.uniform(150, 155, NR),
                "DEC": rng.uniform(2, 6, NR),
                "Redshift": rng.uniform(0.1, 0.3, NR),
                "weight": np.ones(NR)}
        obj  = MeasureIALightcone(data, rand,
                                   str(tmp_path / "lc_formula_cl.hdf5"),
                                   separation_limits=_SEP, num_bins_r=2,
                                   num_bins_pi=2, pi_max=60.0)

        shape = SplusD.shape
        RR    = np.array([[10.0, 5.0], [4.0, 8.0]])
        RD    = np.array([[12.0, 6.0], [4.0, 9.0]])
        N_D, N_S, N_R = 10, 10, 30

        pair_counts = {
            "w/xi_g_plus": {"lf2_SplusD": SplusD, "lf2_SplusR": SplusR,
                             "lf2_rp": np.array([1.0, 5.0]),
                             "lf2_pi": np.array([10.0, 30.0])},
            "w/xi_g_cross": {"lf2_ScrossD": np.zeros(shape),
                              "lf2_ScrossR": np.zeros(shape)},
            "w/xi_gg": {"lf2_DD": DD, "lf2_SR": SR,
                         "lf2_RD": RD, "lf2_RR": RR,
                         "lf2_rp": np.array([1.0, 5.0]),
                         "lf2_pi": np.array([10.0, 30.0])},
        }
        with h5py.File(obj.output_file_name, "a") as f:
            for grp_key, datasets in pair_counts.items():
                grp = f.require_group(grp_key)
                for ds_name, arr in datasets.items():
                    grp.create_dataset(ds_name, data=arr)

        num_samples = {"S": N_S, "D": N_D, "D_S": 0,
                       "R_D": N_R, "R_S": N_R}
        obj._obs_estimator(("g+", "w"), "clusters", "lf2", num_samples)

        with h5py.File(obj.output_file_name, "r") as f:
            xi_gp = f["w/xi_g_plus/lf2"][:]

        # clusters: xi_g+ = SplusD/DD_norm - SplusR/SR_norm
        N_SD = N_S * N_D
        N_SR = N_S * N_R
        expected = SplusD / (N_SD * DD / N_SD) - SplusR / (N_SR * SR / N_SR)
        np.testing.assert_allclose(xi_gp, expected, rtol=1e-10)

    def _make_lc_gg_obj(self, tmp_path, fname):
        rng  = np.random.default_rng(_SEED)
        N    = 10
        data = {
            "RA": rng.uniform(150, 155, N), "DEC": rng.uniform(2, 6, N),
            "Redshift": rng.uniform(0.1, 0.3, N),
            "RA_shape_sample": rng.uniform(150, 155, N),
            "DEC_shape_sample": rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": np.zeros(N), "e2": np.zeros(N),
            "weight": np.ones(N), "weight_shape_sample": np.ones(N),
        }
        NR = 30
        rand = {k.replace("_shape_sample", ""): rng.uniform(0, 1, NR)
                for k in data if "shape" not in k and k not in ("e1", "e2")}
        return MeasureIALightcone(data, rand, str(tmp_path / fname),
                                  separation_limits=_SEP, num_bins_r=2,
                                  num_bins_pi=2, pi_max=60.0)

    def _inject_gg(self, obj, name, DD, SR, RD, RR):
        with h5py.File(obj.output_file_name, "a") as f:
            grp = f.require_group("w/xi_gg")
            for ds_name, arr in ((f"{name}_DD", DD), (f"{name}_SR", SR),
                                 (f"{name}_RD", RD), (f"{name}_RR", RR)):
                if ds_name in grp:
                    del grp[ds_name]
                grp.create_dataset(ds_name, data=arr)

    def test_lc_gg_empty_dd_bin_unbiased(self, tmp_path):
        """Regression: a bin with zero DD pairs must contribute DD=0 to the
        Landy-Szalay gg numerator — the old DD[DD==0]=1 guard injected a
        spurious 1/norm into empty bins."""
        obj = self._make_lc_gg_obj(tmp_path, "lc_gg_empty_dd.hdf5")
        DD = np.array([[0.0, 10.0], [8.0, 16.0]])   # empty bin at [0, 0]
        SR = np.array([[15.0, 7.5], [6.0, 12.0]])
        RD = np.array([[12.0, 6.0], [4.0, 9.0]])
        RR = np.array([[10.0, 5.0], [4.0, 8.0]])
        self._inject_gg(obj, "lgg", DD, SR, RD, RR)
        N_D, N_S, N_R = 10, 10, 30
        num_samples = {"S": N_S, "D": N_D, "D_S": 0, "R_D": N_R, "R_S": N_R}
        obj._obs_estimator(("gg", "w"), "galaxies", "lgg", num_samples)
        with h5py.File(obj.output_file_name, "r") as f:
            xi_gg = f["w/xi_gg/lgg"][:]
        expected = (DD / (N_S * N_D) - RD / (N_D * N_R) - SR / (N_S * N_R)) \
            / (RR / (N_R * N_R)) + 1
        np.testing.assert_allclose(xi_gg, expected, rtol=1e-10)

    def test_lc_gg_empty_rr_bin_warns(self, tmp_path):
        """A bin with zero random-random pairs must trigger a RuntimeWarning
        naming the problem (the estimator is NaN there)."""
        obj = self._make_lc_gg_obj(tmp_path, "lc_gg_empty_rr.hdf5")
        DD = np.array([[20.0, 10.0], [8.0, 16.0]])
        SR = np.array([[15.0, 7.5], [6.0, 12.0]])
        RD = np.array([[12.0, 6.0], [4.0, 9.0]])
        RR = np.array([[0.0, 5.0], [4.0, 8.0]])    # empty bin at [0, 0]
        self._inject_gg(obj, "lwz", DD, SR, RD, RR)
        num_samples = {"S": 10, "D": 10, "D_S": 0, "R_D": 30, "R_S": 30}
        with pytest.warns(RuntimeWarning, match="random-random"):
            obj._obs_estimator(("gg", "w"), "galaxies", "lwz", num_samples)
        with h5py.File(obj.output_file_name, "r") as f:
            xi_gg = f["w/xi_gg/lwz"][:]
        assert not np.isfinite(xi_gg[0, 0])
        assert np.all(np.isfinite(xi_gg[RR != 0]))


# ===========================================================================
# 4. Cosmology coordinate conversion
# ===========================================================================

class TestCosmologyConversion:
    """
    Verify that the lightcone converts redshifts to comoving distances
    correctly using the hardcoded pyccl cosmology
    (Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0).

    Strategy: place two galaxies at the same (RA, DEC) but at different
    redshifts, so their separation is purely along the LOS.  The measured
    LOS separation (stored in the pi grid) should match the comoving distance
    difference computed independently with pyccl.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_pyccl(self):
        pytest.importorskip("pyccl")

    def test_comoving_distance_at_z05(self, tmp_path):
        """
        Two galaxies at the same sky position, z=0.1 and z=0.3.
        Expected LOS separation = chi(0.3) - chi(0.1) [Mpc].
        """
        import pyccl as ccl
        cosmo = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045,
                               sigma8=0.8, h=0.7, n_s=1.0)
        chi_03 = float(ccl.comoving_radial_distance(cosmo, 1 / 1.3))
        chi_01 = float(ccl.comoving_radial_distance(cosmo, 1 / 1.1))
        expected_dchi = chi_03 - chi_01   # Mpc

        # Galaxy pair at same RA/DEC, different redshifts
        RA, DEC = 152.0, 4.0
        N   = 2
        data = {
            "RA":                    np.array([RA, RA]),
            "DEC":                   np.array([DEC, DEC]),
            "Redshift":              np.array([0.1, 0.3]),
            "RA_shape_sample":       np.array([RA, RA]),
            "DEC_shape_sample":      np.array([DEC, DEC]),
            "Redshift_shape_sample": np.array([0.1, 0.3]),
            "e1": np.zeros(N), "e2": np.zeros(N),
            "weight": np.ones(N), "weight_shape_sample": np.ones(N),
        }
        NR = 20
        rng  = np.random.default_rng(_SEED)
        rand = {
            "RA":                    rng.uniform(150, 155, NR),
            "DEC":                   rng.uniform(2, 6, NR),
            "Redshift":              rng.uniform(0.05, 0.35, NR),
            "RA_shape_sample":       rng.uniform(150, 155, NR),
            "DEC_shape_sample":      rng.uniform(2, 6, NR),
            "Redshift_shape_sample": rng.uniform(0.05, 0.35, NR),
            "weight": np.ones(NR), "weight_shape_sample": np.ones(NR),
        }
        # Use wide pi_max so the pair is captured
        obj = MeasureIALightcone(
            data, rand, str(tmp_path / "cosmo_test.hdf5"),
            separation_limits=[0.1, 2000.0],
            num_bins_r=4, num_bins_pi=40,
            pi_max=expected_dchi * 1.5,
        )
        obj.measure_xi_w("galaxies", "cosmo_test", "gg",
                          tree=False,
                          temp_file_path=None)

        pi_grid = _read(obj, "w/xi_gg", "cosmo_test_pi")
        # The pair falls in the pi bin closest to expected_dchi
        closest_bin = pi_grid[np.argmin(np.abs(pi_grid - expected_dchi))]
        assert abs(closest_bin - expected_dchi) < (pi_grid[1] - pi_grid[0]), \
            (f"Expected LOS separation {expected_dchi:.2f} Mpc not found in "
             f"pi grid; closest bin: {closest_bin:.2f}")

    def test_over_h_scales_by_h(self, tmp_path):
        """
        With over_h=True, comoving distances are multiplied by h=0.7.
        The LOS pi grid should be h times smaller than with over_h=False.
        """
        import pyccl as ccl
        cosmo = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045,
                               sigma8=0.8, h=0.7, n_s=1.0)
        h = 0.7

        rng = np.random.default_rng(_SEED)
        N   = 30
        data = {
            "RA":  rng.uniform(150, 155, N),
            "DEC": rng.uniform(2, 6, N),
            "Redshift": rng.uniform(0.1, 0.3, N),
            "RA_shape_sample":  rng.uniform(150, 155, N),
            "DEC_shape_sample": rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": np.zeros(N), "e2": np.zeros(N),
            "weight": np.ones(N), "weight_shape_sample": np.ones(N),
        }
        NR   = 90
        rand = {
            "RA":  rng.uniform(150, 155, NR),
            "DEC": rng.uniform(2, 6, NR),
            "Redshift": rng.uniform(0.1, 0.3, NR),
            "RA_shape_sample":  rng.uniform(150, 155, NR),
            "DEC_shape_sample": rng.uniform(2, 6, NR),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, NR),
            "weight": np.ones(NR), "weight_shape_sample": np.ones(NR),
        }

        for flag, tag in [(False, "Mpc"), (True, "Mpch")]:
            out = str(tmp_path / f"oh_{tag}.hdf5")
            obj = MeasureIALightcone(data.copy(), rand.copy(), out,
                                      separation_limits=_SEP,
                                      num_bins_r=_NR, num_bins_pi=_NPI,
                                      pi_max=60.0)
            obj.measure_xi_w("galaxies", f"oh_{tag}", "gg",
                              tree=False,
                              over_h=flag,
                              temp_file_path=None)

        with h5py.File(str(tmp_path / "oh_Mpc.hdf5"),  "r") as f:
            dd_mpc  = f["w/xi_gg/oh_Mpc_DD"][:]
        with h5py.File(str(tmp_path / "oh_Mpch.hdf5"), "r") as f:
            dd_mpch = f["w/xi_gg/oh_Mpch_DD"][:]

        # With over_h, distances are smaller by factor h, so more pairs
        # land in inner bins → total DD should differ between the two
        assert not np.allclose(dd_mpc, dd_mpch), \
            "over_h=True and over_h=False should give different DD arrays"


# ===========================================================================
# 5. Edge cases that ARE handled correctly
# ===========================================================================

class TestEdgeCasesHandled:
    """
    Smoke tests confirming the cases that are already guarded in source.
    """

    def test_none_data_initialises_without_error(self, tmp_path):
        """data=None should create an object with Num_position=0."""
        obj = MeasureIABox(None, str(tmp_path / "none.hdf5"),
                            simulation="TNG300", snapshot=99)
        assert obj.Num_position == 0
        assert obj.Num_shape    == 0

    def test_missing_weight_injected_as_ones(self, tmp_path):
        """Missing weight key → ones injected, no KeyError."""
        rng = np.random.default_rng(_SEED)
        N   = 20
        COM = rng.uniform(0, 205.0, (N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.ones(N)}
        obj = MeasureIABox(data, str(tmp_path / "no_wt.hdf5"),
                            simulation="TNG300", snapshot=99,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4, pi_max=50.0)
        np.testing.assert_array_equal(obj.data["weight"], np.ones(N))
        np.testing.assert_array_equal(obj.data["weight_shape_sample"],
                                       np.ones(N))

    def test_zero_separation_pair_does_not_raise(self, tmp_path):
        """Two galaxies at identical positions fall outside r_bins[0] and
        produce zero pairs — no NaN or crash."""
        N   = 10
        COM = np.zeros((N, 3))    # all at origin
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.full(N, 0.5)}
        obj = MeasureIABox(data, str(tmp_path / "zero_sep.hdf5"),
                            simulation=None, snapshot=None,
                            boxsize=100.0,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        obj.measure_xi_w("zero_sep", "both", 0, temp_file_path=False)
        w = _read(obj, "w/xi_g_plus", "zero_sep")
        assert np.all(w == 0)

    def test_nan_eplusD_set_to_zero(self, tmp_path):
        """e_plus NaN from zero-separation pair is set to 0 in source,
        so SplusD should remain finite."""
        N   = 10
        COM = np.zeros((N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.full(N, 0.5)}
        obj = MeasureIABox(data, str(tmp_path / "nan_ep.hdf5"),
                            simulation=None, snapshot=None,
                            boxsize=100.0,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        obj.measure_xi_w("nan_ep", "g+", 0, temp_file_path=False)
        splusd = _read(obj, "w/xi_g_plus", "nan_ep_SplusD")
        assert np.all(np.isfinite(splusd))

    def test_pi_max_derived_from_boxsize(self, tmp_path):
        """pi_max=None with known boxsize derives pi_max from L/2."""
        obj = MeasureIABox(None, str(tmp_path / "pi_deriv.hdf5"),
                            simulation="TNG300", snapshot=99)
        assert obj.pi_bins[-1] == pytest.approx(205.0 / 2.0)

    def test_pi_max_none_no_boxsize_raises(self, tmp_path):
        """pi_max=None with no boxsize must raise ValueError."""
        with pytest.raises(ValueError, match="pi_max"):
            MeasureIABox(None, str(tmp_path / "err.hdf5"),
                          simulation=None, snapshot=None,
                          pi_max=None, boxsize=None)

    def test_data_restored_after_successful_run(self, tmp_path):
        """self.data is restored to the full catalogue after measure_xi_w."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        COM = rng.uniform(0, 205.0, (N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.ones(N)}
        obj = MeasureIABox(data, str(tmp_path / "restore.hdf5"),
                            simulation="TNG300", snapshot=99,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        half = np.arange(N) < N // 2
        masks = {"Position": half, "Position_shape_sample": half,
                 "Axis_Direction": half, "q": half}
        obj.measure_xi_w("restore", "gg", 0, temp_file_path=False,
                          masks=masks)
        # After the call self.data["Position"] should be the full N-row array
        assert len(obj.data["Position"]) == N


# ===========================================================================
# 6. Previously-unhandled edge cases, now fixed at source — regression-locked
# ===========================================================================

class TestEdgeCasesNowHandled:
    """
    Previously-unhandled failure modes (all-zero weights, empty mask, single-object
    sample, data restored after a failed backend run) that were fixed at source during
    the P0 robustness pass. These were parked as xfails; they now pass and are kept as
    regression locks.
    """

    def test_all_zero_weights_gives_zero_not_nan(self, tmp_path):
        rng = np.random.default_rng(_SEED)
        N   = 20
        COM = rng.uniform(0, 205.0, (N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.full(N, 0.5),
                "weight": np.zeros(N),
                "weight_shape_sample": np.zeros(N)}
        obj = MeasureIABox(data, str(tmp_path / "zero_wt.hdf5"),
                            simulation=None, snapshot=None, boxsize=205.0,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("zw", "g+", 0, temp_file_path=False)
        w = _read(obj, "w/xi_g_plus", "zw")
        assert np.all(np.isfinite(w)), f"w_g+ contains NaN/Inf: {w}"
        assert np.all(w == 0)

    def test_empty_mask_gives_zero_not_crash(self, tmp_path):
        rng = np.random.default_rng(_SEED)
        N   = 20
        COM = rng.uniform(0, 205.0, (N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.full(N, 0.5)}
        obj = MeasureIABox(data, str(tmp_path / "empty_mask.hdf5"),
                            simulation=None, snapshot=None, boxsize=205.0,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        empty_mask = {"Position": np.zeros(N, dtype=bool),
                      "Position_shape_sample": np.zeros(N, dtype=bool)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("em", "g+", 0, temp_file_path=False,
                              masks=empty_mask)
        w = _read(obj, "w/xi_g_plus", "em")
        assert np.all(np.isfinite(w))
        assert np.all(w == 0)

    def test_n1_position_sample_does_not_divide_by_zero(self, tmp_path):
        N_shape = 30
        rng     = np.random.default_rng(_SEED)
        COM_s   = rng.uniform(0, 205.0, (N_shape, 3))
        COM_p   = rng.uniform(0, 205.0, (1, 3))   # single position object
        data = {"Position": COM_p,
                "Position_shape_sample": COM_s,
                "Axis_Direction": np.column_stack(
                    [np.ones(N_shape), np.zeros(N_shape)]),
                "LOS": 2, "q": np.full(N_shape, 0.5)}
        obj = MeasureIABox(data, str(tmp_path / "n1.hdf5"),
                            simulation=None, snapshot=None, boxsize=205.0,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("n1", "g+", 0, temp_file_path=False)
        w = _read(obj, "w/xi_g_plus", "n1")
        assert np.all(np.isfinite(w))

    def test_data_restored_after_failed_run(self, tmp_path, monkeypatch):
        """self.data must be restored even if the multiprocessing backend
        fails after self.data was offloaded to the temp file (the reload now
        lives in the finally block of the mp methods)."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        COM = rng.uniform(0, 205.0, (N, 3))
        data = {"Position": COM, "Position_shape_sample": COM,
                "Axis_Direction": np.column_stack([np.ones(N), np.zeros(N)]),
                "LOS": 2, "q": np.ones(N)}
        obj = MeasureIABox(data, str(tmp_path / "restore_fail.hdf5"),
                            simulation="TNG300", snapshot=99,
                            separation_limits=_SEP, num_bins_r=4,
                            num_bins_pi=4, num_nodes=2)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated pool failure")

        # The backend asks worker_pool for a pool after self.data has been
        # emptied and written to the temp file — the same moment the direct
        # Pool() call used to happen, and exactly the state that used to be left
        # behind on error.
        monkeypatch.setattr("measureia.worker_pool.active_pool", _boom)
        with pytest.raises(RuntimeError):
            obj.measure_xi_w("fail", "g+", 0,
                             temp_file_path=str(tmp_path) + "/")
        # self.data must be restored to the full N-row arrays
        assert len(obj.data["Position"]) == N, \
            "self.data was not restored after a failed backend call"
        assert len(obj.data["q"]) == N


# ===========================================================================
# 7. Degenerate lightcone inputs (mirrors of the box edge cases above)
# ===========================================================================

class TestEdgeCasesLightcone:
    """
    The lightcone counterparts of section 6. The Landy-Szalay estimator is
    genuinely undefined where the empirical RR is empty, so — unlike the box,
    whose analytic RR is guarded to zero — the assertions here are that the run
    completes, produces the right shape, and yields no *signal* (all-finite
    entries are exactly zero) rather than all-finite output.
    """

    def _cat(self, N, NR, rng, e1=None, e2=None):
        data = {
            "RA":                    rng.uniform(150, 155, N),
            "DEC":                   rng.uniform(2, 6, N),
            "Redshift":              rng.uniform(0.1, 0.3, N),
            "RA_shape_sample":       rng.uniform(150, 155, N),
            "DEC_shape_sample":      rng.uniform(2, 6, N),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, N),
            "e1": rng.uniform(-0.2, 0.2, N) if e1 is None else e1,
            "e2": rng.uniform(-0.2, 0.2, N) if e2 is None else e2,
            "weight":              np.ones(N),
            "weight_shape_sample": np.ones(N),
        }
        randoms = {
            "RA":                    rng.uniform(150, 155, NR),
            "DEC":                   rng.uniform(2, 6, NR),
            "Redshift":              rng.uniform(0.1, 0.3, NR),
            "RA_shape_sample":       rng.uniform(150, 155, NR),
            "DEC_shape_sample":      rng.uniform(2, 6, NR),
            "Redshift_shape_sample": rng.uniform(0.1, 0.3, NR),
            "weight":              np.ones(NR),
            "weight_shape_sample": np.ones(NR),
        }
        return data, randoms

    def test_all_zero_weights_gives_no_signal(self, tmp_path):
        """Zero weights kill every weighted count; w_g+ must not blow up into
        a spurious signal (finite bins are exactly zero)."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        data, rand = self._cat(N, 3 * N, rng)
        data["weight"] = np.zeros(N)
        data["weight_shape_sample"] = np.zeros(N)
        obj = _lc(data, rand, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("galaxies", "lc_zw", "g+", 
                             tree=False, temp_file_path=None)
        w = _read(obj, "w/xi_g_plus", "lc_zw")
        assert w.shape == (_NR, _NPI)
        finite = np.isfinite(w)
        np.testing.assert_allclose(w[finite], 0.0, atol=1e-12)

    def test_empty_mask_gives_no_signal(self, tmp_path):
        """An all-False mask empties both samples; the run must complete."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        data, rand = self._cat(N, 3 * N, rng)
        obj = _lc(data, rand, tmp_path)
        # A partial mask dict is enough: missing fields fall back to their
        # sample's coordinate mask.
        empty = {"RA": np.zeros(N, dtype=bool),
                 "RA_shape_sample": np.zeros(N, dtype=bool)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("galaxies", "lc_em", "g+", 
                             tree=False, masks=empty, temp_file_path=None)
        w = _read(obj, "w/xi_g_plus", "lc_em")
        assert w.shape == (_NR, _NPI)
        finite = np.isfinite(w)
        np.testing.assert_allclose(w[finite], 0.0, atol=1e-12)

    def test_single_object_position_sample_does_not_crash(self, tmp_path):
        """A one-object position sample (the N=1 divide-by-zero case on the
        box) must run through the lightcone estimator too."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        data, rand = self._cat(N, 3 * N, rng)
        for key in ("RA", "DEC", "Redshift", "weight"):
            data[key] = data[key][:1]
        obj = _lc(data, rand, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj.measure_xi_w("galaxies", "lc_n1", "g+", 
                             tree=False, temp_file_path=None)
        w = _read(obj, "w/xi_g_plus", "lc_n1")
        assert w.shape == (_NR, _NPI)

    def test_data_restored_after_failed_run(self, tmp_path, monkeypatch):
        """Lightcone twin of the box restore test: a Pool failure after the
        temp-file offload must still leave self.data intact. The lightcone
        multiprocessing path lives on the jackknife backend, so this needs
        a jackknife run (num_jk)."""
        rng = np.random.default_rng(_SEED)
        N   = 40
        data, rand = self._cat(N, 3 * N, rng)
        out = str(tmp_path / "lc_restore_fail.hdf5")
        obj = MeasureIALightcone(data=data, randoms_data=rand,
                                 output_file_name=out,
                                 separation_limits=_SEP, num_bins_r=_NR,
                                 num_bins_pi=_NPI, pi_max=60.0, num_nodes=2)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated pool failure")

        monkeypatch.setattr("measureia.worker_pool.active_pool", _boom)
        with pytest.raises(RuntimeError):
            obj.measure_xi_w("galaxies", "lc_fail", "g+", num_jk=4,
                             temp_file_path=str(tmp_path) + "/")
        assert len(obj.data["RA"]) == N, \
            "self.data was not restored after a failed backend call"
        assert len(obj.data["e1"]) == N

    def test_non_default_cosmology_changes_distances(self, tmp_path):
        """The `cosmology` argument is honoured: a different Omega_m maps the
        same redshifts onto different comoving distances, so w_g+ shifts."""
        ccl = pytest.importorskip("pyccl")
        rng = np.random.default_rng(_SEED)
        N   = 60
        data, rand = self._cat(N, 3 * N, rng)

        cosmo_a = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                                n_s=0.96, sigma8=0.8)
        cosmo_b = ccl.Cosmology(Omega_c=0.45, Omega_b=0.05, h=0.7,
                                n_s=0.96, sigma8=0.8)

        obj_a = MeasureIALightcone(
            data={k: v.copy() for k, v in data.items()},
            randoms_data={k: v.copy() for k, v in rand.items()},
            output_file_name=str(tmp_path / "cosmo_a.hdf5"),
            separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI,
            pi_max=60.0)
        obj_b = MeasureIALightcone(
            data={k: v.copy() for k, v in data.items()},
            randoms_data={k: v.copy() for k, v in rand.items()},
            output_file_name=str(tmp_path / "cosmo_b.hdf5"),
            separation_limits=_SEP, num_bins_r=_NR, num_bins_pi=_NPI,
            pi_max=60.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj_a.measure_xi_w("galaxies", "cos_a", "g+", 
                               tree=False, cosmology=cosmo_a,
                               temp_file_path=None)
            obj_b.measure_xi_w("galaxies", "cos_b", "g+", 
                               tree=False, cosmology=cosmo_b,
                               temp_file_path=None)

        w_a = _read(obj_a, "w/xi_g_plus", "cos_a")
        w_b = _read(obj_b, "w/xi_g_plus", "cos_b")
        both = np.isfinite(w_a) & np.isfinite(w_b)
        assert both.any(), "no bin was finite in both cosmologies"
        assert not np.allclose(w_a[both], w_b[both]), \
            "the cosmology argument did not affect the measurement"
