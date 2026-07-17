# Cross-package validation

This directory validates measureia's correlation-function estimators against
independent, publicly available implementations. It complements the unit and
analytic tests in `tests/` (which check internal consistency and brute-force
expectations) by checking full pipelines against external codes.

## How it works

- **Mock data** (`mock_catalogues.py`): all comparisons run on a seeded
  synthetic catalogue with a strong, non-null IA signal — centrals placed
  uniformly in a periodic box, satellites scattered around them with a
  Gaussian profile, and satellite projected major axes pointing at their own
  central plus Gaussian angle noise ("radial alignment"). Both codes read
  byte-identical inputs, so ratios are meaningful in every bin.
- **Runnable scripts** (`run_*.py`): one script per comparison. Each script
  always computes the measureia side. If the external package is installed,
  it also computes the external side and writes it to
  `reference_outputs/*.hdf5`; otherwise it compares against the committed
  reference file.
- **Reference outputs** (`reference_outputs/`): small committed HDF5 files
  holding the *external* package's results (plus version metadata). The
  pytest layer (`tests/test_validation_references.py`) compares measureia
  against these, so CI and reviewers get a genuine cross-package check
  without installing the external packages.

To install the external packages:

```bash
pip install measureia[validation]   # halotools + treecorr
```

## Comparisons and known convention differences

### Box w_gg / w_g+ vs halotools (`run_box_halotools.py`)

Compares `MeasureIABox.measure_xi_w` against
`halotools.mock_observables.ia_correlations.gi_plus_projected` (w_g+) and
`halotools.mock_observables.wp` (w_gg), both with analytic randoms in a
periodic box, distant-observer LOS along z.

- **Responsivity**: measureia divides the S+D sum by 2R with
  R = 1 − ⟨e²⟩/2; halotools does not. Therefore
  `w_g+^measureia × 2R = w_g+^halotools`. This is the only expected
  difference; the agreement is exact up to floating-point noise because both
  codes use analytic RR counts (verified: the RR volume factors match
  analytically).
- **Result** (halotools 0.9.4, 2026-07-16): with the 2R factor applied,
  w_g+ agrees to a maximum relative difference of 2×10⁻¹³ and w_gg to
  4×10⁻¹⁵ across all bins — machine precision. Enforced at rtol=1e-10 in
  `tests/test_validation_references.py`.

### Lightcone w_gg / w_g+ vs treecorr (`run_lightcone_treecorr.py`)

Compares `MeasureIALightcone.measure_xi_w` ('galaxies' estimator) against
the same estimator reconstructed from raw treecorr `NN`/`NG` counts, one
treecorr run per signed pi slab (`min_rpar`/`max_rpar`, `metric='Rperp'`,
`bin_slop=0`), on the lightcone radial-alignment mock.

- **Ellipticity convention**: measureia's lightcone e1/e2 input follows
  the standard survey shear-catalogue convention — the same components
  treecorr and lensfit/metacal-style catalogues use — and returns
  radial-positive w_g+ (the IA-literature sign, opposite to the lensing
  tangential shear: e_+ = −γ_t). The treecorr comparison therefore only
  needs the single, standard IA flip **`g1 = −e1, g2 = −e2`**, after
  which the ratio is +1. Beware when experimenting with conventions:
  getting the *chirality* (relative sign of e1 vs e2) wrong does not
  flip w_g+ — it washes the signal out to incoherent noise (the
  projection becomes cos(2(φ_axis+φ_sep)) instead of
  cos(2(φ_axis−φ_sep))), which is easy to misdiagnose as a code bug.
- **Separation definitions**: treecorr's `Rperp` (FisherRperp) and signed
  r_par differ from measureia's midpoint-LOS definitions by curvature
  terms, so a few pairs near bin edges migrate bins; treecorr also
  projects shears in the great-circle frame of each pair while measureia
  uses the (east, north) frame of the position-sample galaxy.
- **Result** (treecorr 5.1.3, 2026-07-16): w_g+ agrees to ~1×10⁻⁵
  (relative) in all high-signal bins; w_gg is exact in most bins and
  within 0.4% in the smallest-rp bins (bin migration). The near-zero
  outer w_g+ bins differ by <0.04 in absolute terms. Enforced at
  rtol=5e-3, atol=0.05 in `tests/test_validation_references.py`.

### Box vs treecorr (planned)

treecorr has no periodic-boundary support, so measureia must be run without
analytic randoms and the estimator reconstructed from separate DD/RR/SD
runs; only one pi bin is practical.

### Plane-parallel consistency (`run_plane_parallel.py`)

The identical radial-alignment box catalogue is measured with
`MeasureIABox` (periodic, analytic randoms; halotools-anchored) and, after
exact embedding at comoving distance 12000 Mpc, with `MeasureIALightcone`
('galaxies' estimator, empirical randoms filling the embedded cube). The
mock uses a margin so no periodically wrapped pair enters the measured
separation range.

**Result** (2026-07-16), reported at three levels which fully attribute
every difference:

1. **Raw pair counts**: DD grids agree to <1%; S+D grids agree to <1%
   after dividing by the responsivity 2R. This is the actual
   plane-parallel test — geometry, binning, and the survey-convention
   e1/e2 shape projection are consistent between the two pipelines.
2. **w with matched RR**: rebuilding the lightcone estimator with the
   box's analytic RR gives w_gg within ~1% and w_g+ within ~2% (bins
   with adequate pair counts).
3. **w as measured**: differs additionally by a few % at rp ≳ 5 Mpc/h
   because analytic RR assumes a periodic box while empirical randoms
   live in a bounded window (boundary pair loss grows with separation) —
   an understood estimator-design difference, not an error.

**Convention finding**: the box estimator divides S+ terms by the
responsivity 2R; the lightcone estimator does not (its e1/e2 are treated
as calibrated shear estimates). This cross-pipeline difference was
discovered by this check. Both `measure_xi_w` and `measure_xi_multipoles`
now expose it as the `responsivity` parameter (box default `True`,
lightcone default `False`); toggling it rescales w_g+ by exactly 2R
(enforced by tests) and leaves w_gg untouched. Set `responsivity=True`
on the lightcone when feeding raw distortions rather than calibrated
shears.

### Box multipoles vs corr_pc (`run_box_multipoles_corrpc.py`)

Compares `MeasureIABox.measure_xi_multipoles` against
[corr_pc](https://github.com/sukhdeep2/corr_pc) ([Singh
2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1632S/abstract)), the
C++ code used for the original multipole validation of this package.
corr_pc's periodic-box mode with `coordinates=7` measures ξ_gg(r, μ) and
ξ_g+(r, μ) on the identical log-r / linear-μ grid with the same natural
estimators and analytic RR; measureia's own associated-Legendre integration
is then applied to the corr_pc grid, so the comparison covers both the grid
and the multipole integration.

- **Responsivity**: as with halotools, measureia divides S+ terms by 2R,
  corr_pc does not: `ξ_g+^measureia × 2R = ξ_g+^corr_pc`.
- **Analytic RR normalisation**: measureia's `get_random_pairs_r_mur`
  uses (N_pos − 1) · N_shape while corr_pc uses N_pos · N_shape — a
  deterministic (N_pos − 1)/N_pos factor applied in the comparison.
- **Ellipticity chirality**: corr_pc rotates components as
  e+ = cos(2θ)e1 − sin(2θ)e2, i.e. its expected input convention is
  e1 = e·cos(2φ_axis), **e2 = −e·sin(2φ_axis)** (opposite chirality to
  the survey shear convention used by measureia's lightcone pipeline and
  treecorr). With that input its e+ is radial-positive, matching
  measureia. As with treecorr, getting the chirality wrong washes the
  signal out rather than flipping its sign.
- **Result** (2026-07-16): ξ(r, μ) grids agree to ≤5×10⁻⁶ in every bin
  and the multipoles ξ_g+,2 / ξ_gg,0 to ~10⁻⁶ (a few ×10⁻⁵ in near-zero
  bins) — limited purely by corr_pc's 6-significant-digit text output,
  i.e. machine-precision agreement. Enforced at rtol=1e-4 in
  `tests/test_validation_references.py`.

**Building corr_pc without MPI**: corr_pc's Makefile asks for `mpic++`,
but its only MPI usage is `MPI::Init/Finalize` and
`MPI::COMM_WORLD.Get_size/Get_rank` via the deprecated C++ bindings. The
single-process stub header in `corrpc_mpi_stub/mpi.h` replaces the whole
dependency. On macOS with homebrew gsl + libomp:

```bash
git clone https://github.com/sukhdeep2/corr_pc && cd corr_pc
make compiler=clang++ \
  CFLAGS="-c -I/path/to/measure_IA/validation/corrpc_mpi_stub \
          -I/opt/homebrew/opt/gsl/include -Xpreprocessor -fopenmp \
          -I/opt/homebrew/opt/libomp/include" \
  LDFLAGS="-L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas \
           -L/opt/homebrew/opt/libomp/lib -lomp"
CORR_PC_BIN=$PWD/corr_pc python run_box_multipoles_corrpc.py
```

### Jackknife covariance vs treecorr (`run_lightcone_treecorr_cov.py`)

The identical seeded kmeans patch assignment (from measureia's
`assign_jackknife_patches`) is supplied to both codes on the lightcone
mock, so both compute the same deterministic delete-one-patch statistic
with the standard (N−1)/N formula. Because measureia's RR-normalised
w_g+ estimator is not what treecorr's compensated NG `calculateXi`
computes, treecorr's built-in covariance machinery cannot reproduce it
directly; instead the jackknife loop is explicit — each patch-deleted
sample is re-processed with treecorr and the estimator rebuilt from raw
counts, mirroring measureia's internal definition exactly.

- **Result** (treecorr 5.1.3, 2026-07-16, 9 patches): jackknife standard
  deviations agree to ≤5×10⁻⁵ for w_g+ in all high-signal bins (≤2% in
  the near-zero outer bins) and ≤0.6% for w_gg; full correlation-matrix
  elements agree to ≤0.02 absolute. Enforced at rtol=3%, atol=0.05 in
  `tests/test_validation_references.py`.

### Box jackknife covariance (`run_box_cov_bridge.py`)

Two-level validation of the box jackknife (subbox partition, delete-one
reconstruction by count subtraction, analytic RR rescaled to the retained
counts and volume). No external package is needed — the reference here is
measureia's own validated full-sample estimators.

1. **Delete-one identity (machine precision, the rigorous test)**: every
   reconstructed realisation is compared against an independent direct
   measurement in which the subbox galaxies are physically removed and the
   plain (non-jackknife) box estimator is rerun. The retained DD grids
   match exactly, S+D (including the per-realisation responsivity 2R_i)
   to ≤10⁻¹², the jackknife RR equals the direct analytic RR times exactly
   V/V_del = L³/(L³−1), and the per-realisation w vectors follow to
   ≤10⁻¹³. This locks the count-subtraction machinery: pairs are removed
   when *either* member is in the deleted subbox, identical to physically
   deleting the patch from both samples. Enforced at <10⁻¹⁰ in
   `tests/test_validation_references.py` (runs in CI, no reference file
   needed).

2. **Cross-pipeline bridge (loose by expectation)**: the identical subbox
   partition is fed as `jk_patches` to the treecorr-validated lightcone
   jackknife on the plane-parallel embedding (`responsivity=True`).
   Jackknife std ratios come out at 0.68–1.05 and correlation-matrix
   elements differ by up to ~0.7 — and this is the *expected* result, not
   a defect. Realization-level forensics (2026-07-16) attributed it fully:
   rebuilding the box-style estimator from the lightcone run's own
   retained counts reproduces the *lightcone* stds to ~0.8–1.1, so the
   jackknife machinery is consistent and the residual comes from
   (a) plane-parallel-vs-sky geometry migrating ~0.1–1% of pairs per
   realisation between bins — amplified because 8-patch delete-one
   deviations are only a few % of the mean; (b) genuinely different
   estimator definitions (box: natural DD/RR−1, S+D/RR_analytic,
   per-realisation responsivity; lightcone: LS-compensated
   (DD−RD−SR)/RR+1, (S+D−S+R)/RR_empirical, full-sample responsivity) —
   different estimators have different covariances; and (c) the
   analytic-RR-under-deletion approximation, whose count/volume amplitude
   is exact and whose missed hole-boundary bin-shape is ~2% (moving stds
   ≲15%). The committed reference file stores both covariances, the
   box-style reconstruction, and these metrics; the tests enforce the
   documented bands.

**Known approximation (by design)**: under deletion the analytic RR is
rescaled by retained counts and volume but keeps the full-box bin shape;
the ~2% hole-boundary shape effect above shrinks as the number of patches
grows (the deleted hole gets smaller). Users needing that last few percent
should use more patches or the lightcone pipeline with explicit randoms.
