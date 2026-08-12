# Changelog

All notable changes to MeasureIA are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0.0 onwards
MeasureIA follows [semantic versioning](https://semver.org/spec/v2.0.0.html): breaking changes to the
public API mean a major version bump.

## [Unreleased]

### Added

- `MeasureIABox.measure_galaxy_contributions`, which resolves the box estimators **per shape
  galaxy**: for each galaxy and radial bin it returns the projected alignment contribution `Y` and
  the pair count `P`, such that `Y.sum(axis=0)` is the ordinary `xi_g+,2(r)` (`statistic=
  "multipoles"`) or `w_g+(r_p)` (`statistic="w"`). Runs on `num_nodes` cores. With `num_jk > 0` it
  also returns both decomposed by the jackknife patch of the position-sample partner, which
  rebuilds every delete-one realisation without re-counting any pairs. This is the input needed to
  regress the alignment signal on per-galaxy properties: any number of properties can be fitted
  from one pair traversal instead of one correlation-function run per weighting.
- `per_galaxy`, `per_galaxy_proj` and `per_galaxy_jk` options on `pair_kernel.accumulate`, which
  provide the above. All default to off, and when off the pair loop, its iteration order and its
  float summation order are unchanged, so existing measurements are bit-identical and no slower.
- A warning when the sky k-means fit does not converge within its iteration limit. The patches
  remain usable, but the regions are less settled than usual.

### Fixed

- Jackknife patch labels are now always computed against the centres the fit returns. When the
  k-means hit its iteration limit, `assign_jackknife_patches` mixed labels from two different
  centre sets across its four samples, so the delete-one jackknife removed non-corresponding sky
  regions and biased the covariance without raising anything. Reachable only on a non-converging
  fit, which the default of 100 iterations made rare.

## [0.4.0] - 2026-08-11

The first release that installs with a plain `pip install measureia`: every earlier published version
declared a dependency that is not on PyPI, so the resolver failed for everyone.

### Added

- `measureia.mocks`: the seeded radial-alignment mock catalogues now ship with the package, so the
  examples and the validation comparisons can be reproduced from a plain `pip install`.
- A warning when jackknife patch assignment leaves any sample with patches holding fewer than ten
  objects, naming the thin samples and advising more randoms or a lower `num_jk`.
- `LICENSE` (MIT). The licence was claimed in the README and `CITATION.cff` but the text was missing
  from the repository entirely.
- PyPI metadata: licence, classifiers, keywords and project URLs for the documentation, repository,
  issues, changelog and the Zenodo archive.
- Stored jackknife patch labels inside the committed validation reference files, so cross-code
  covariance comparisons no longer depend on how patches happen to be generated.

### Changed

- **Breaking:** `MeasureIALightcone.measure_xi_w` and `measure_xi_multipoles` no longer accept
  `measure_cov`. The jackknife covariance is measured when `jk_patches` is given or `num_jk > 0`, and
  skipped otherwise — the same rule the box already used. Migration is to delete the argument.
  The previous default (`measure_cov=True` with no patch information) raised, so no working code
  depended on it.
- Jackknife sky patches are now built by MeasureIA's own spherical k-means (`measureia.kmeans_sphere`)
  rather than `kmeans_radec`. Patch assignments differ slightly from previous versions, so jackknife
  covariances shift correspondingly; the signal itself is unaffected.
- `assign_jackknife_patches` requires `num_jk` to be no larger than the number of position randoms.
  The previous "at least ten randoms per patch" limit was an artefact of the old backend's sampling.
- `assign_jackknife_patches` no longer touches the global NumPy or stdlib random state.

### Removed

- The `kmeans-radec` dependency. It was not on PyPI, which made `pip install measureia` fail to
  resolve for every published version, and it is GPL-2 licensed.

### Fixed

- `pip install measureia` now works in one command, with every dependency available from PyPI.
- `requirements.txt` listed pre-NumPy-2 pins (`numpy~=1.26.2`) and `pathos`, which is no longer a
  dependency.
- The example scripts and notebooks ran against test fixtures that had been deleted, or against empty
  placeholder arrays; they now run as-is on the bundled mocks.
- The lightcone example passed `rp_cut` to `measure_xi_multipoles`, which that method does not accept.

## [0.3.0] - 2026-01-09

- Lightcone methods and their cross-code validation against halotools, treecorr and corr_pc.
- Multiprocessing for the lightcone `w` and multipole measurements.
- Optional responsivity factor for the shape calibration.
- NumPy 2 support, with the supported range widened to Python 3.10 – 3.14 and tested across all of
  them in CI.
- Documentation site (mkdocs + mkdocstrings) covering input, output, conventions, estimator
  definitions and validation.

## [0.2.1] - 2025-10-27

## [0.2.0] - 2025-10-24

## [0.1.0] - 2025-10-02

Initial release: `MeasureIABox` with the projected correlations `w_gg` and `w_g+`, the multipole
moment estimator, and sub-box jackknife covariances.

<!--
  0.1.0 - 0.2.1 are listed for completeness; their entries were reconstructed from the tags rather
  than kept at the time. Fill in the 0.2.x lines if the detail is worth having.
-->

[Unreleased]: https://github.com/MarloesvL/measure_IA/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/MarloesvL/measure_IA/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MarloesvL/measure_IA/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/MarloesvL/measure_IA/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MarloesvL/measure_IA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MarloesvL/measure_IA/releases/tag/v0.1.0
