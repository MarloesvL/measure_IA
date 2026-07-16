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

### Multipoles (planned)

Compared against reference outputs from a private code (not redistributable;
outputs only), plus internal consistency with the validated xi(rp, pi) grid.

Covariance validation is out of scope for now; correlations first.
