# Changelog

All notable changes to MeasureIA are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0.0 onwards
MeasureIA follows [semantic versioning](https://semver.org/spec/v2.0.0.html): breaking changes to the
public API mean a major version bump.

## [Unreleased]

## [0.5.0] - 2026-08-27

### Added

- `MeasureIABox.measure_galaxy_contributions`, which resolves the box estimators **per shape
  galaxy**: for each galaxy and radial bin it returns the projected alignment contribution `Y` and
  the pair count `P`, such that `Y.sum(axis=0)` is the ordinary `xi_g+,2(r)` (`statistic=
  "multipoles"`) or `w_g+(r_p)` (`statistic="w"`). Runs on `num_nodes` cores. With `num_jk > 0` it
  also returns both decomposed by the jackknife patch of the position-sample partner, which
  rebuilds every delete-one realisation without re-counting any pairs. This is the input needed to
  regress the alignment signal on per-galaxy properties: any number of properties can be fitted
  from one pair traversal instead of one correlation-function run per weighting.
- `per_galaxy_jk_sparse` on `pair_kernel.accumulate` (used by default from
  `measure_galaxy_contributions`), storing the per-galaxy jackknife decomposition only for the
  patches each galaxy actually has pairs in. A galaxy's neighbours span a ball of `r_max`, so it
  reaches only a few sub-boxes however many there are, and the rest of the patch axis is
  structurally zero. Pure change of representation — the stored values are identical and a test
  asserts bit-equality with the dense form. For a COLIBRE-L400-sized run (301k shape galaxies,
  125 patches, 12 bins) this is 3.6 GB -> 0.78 GB per array.
- `delete_one_estimator` and `jk_columns` helpers, which rebuild a delete-one realisation from
  the per-galaxy output without the caller needing to know the storage convention.
- `per_galaxy`, `per_galaxy_proj` and `per_galaxy_jk` options on `pair_kernel.accumulate`, which
  provide the above. All default to off, and when off the pair loop, its iteration order and its
  float summation order are unchanged, so existing measurements are bit-identical and no slower.
- A warning when the sky k-means fit does not converge within its iteration limit. The patches
  remain usable, but the regions are less settled than usual.

- `benchmarks/`, a cross-package speed comparison against halotools (periodic box) and TreeCorr
  (lightcone), plus measureia-only profiling. Repo-only, not shipped in the wheel, and it uses the
  existing `[validation]` extra, so no new dependencies. It reuses the binning and catalogue
  builders from `validation/` so a timed run measures the configuration whose cross-package
  agreement is already established, and every timing point is gated on reproducing the reference
  result before it is reported. `benchmarks/FINDINGS.md` records the measured results and, where a
  first analysis turned out to be wrong, says so.

### Changed

- **Pair counting is 1.4x - 1.7x faster.** `pair_kernel.accumulate` selected each galaxy's
  candidate partners with *two* KDTree queries — one at `r_min`, one at `r_max` — and a per-galaxy
  `np.setdiff1d` between them. The inner query was redundant: every `bin_pairs` already discards
  pairs below `r_bins[0]`, and `r_bins[0] == r_min` exactly, on the same metric its tree is built
  on. The set difference nevertheless sorted each galaxy's full candidate list twice (`setdiff1d`
  calls `unique` on both arguments), costing 20-30% of a box measurement to remove 0.6-5% of the
  candidates, while the actual grid accumulation (`np.add.at`) was 2%. Both branches now make a
  single query at the outer radius and let the existing mask do the work.

  Measured on a 12-core laptop, single thread: box `w` and multipoles 1.42x - 1.53x faster at
  `separation_limits=[0.5, 20]`, 1.56x - 1.62x at `[5, 20]`, up to 1.97x at 100k galaxies with one
  pi bin. The gain grows with catalogue size. Peak memory is unchanged or 12-19% lower — the old
  code held three chunk-level index structures at once where the new code holds one.

  Results are unchanged: 2,697 output arrays across 45 configurations — box and lightcone, `w` and
  multipoles, brute / tree / multiprocessing, jackknife on and off, `rp_cut`, masks,
  `measure_galaxy_contributions`, the lightcone `clusters` estimator — are bit-identical to the
  previous implementation, and the halotools and TreeCorr cross-validations reproduce their
  documented agreement exactly.

  **One edge case does change.** A pair whose separation is *exactly* `r_min` was previously
  dropped (`KDTree.query_ball_tree` is inclusive, so such a pair landed in the inner list and the
  set difference removed it) and is now counted, because the binning mask keeps
  `separation >= r_bins[0]`. With floating-point coordinates this is a measure-zero case and no
  test or validation moved. It is *not* measure-zero for gridded or quantised positions: if your
  catalogue sits on a lattice and `r_min` coincides with a lattice separation, pair counts in the
  first radial bin will differ slightly from 0.4.0. Choose an `r_min` that is not an exact lattice
  distance if this matters.

- **Multiprocessing on the lightcone now works, and is much faster where it already did.**
  Two defects, found by the new benchmark suite (`benchmarks/FINDINGS.md` F4):

  1. `MeasureIALightcone.measure_xi_w` and `measure_xi_multipoles` accepted `num_nodes > 1`
     on the **full-sample** path (no jackknife) and then silently ignored it — measured
     speedup at 2, 4 and 8 nodes was exactly 1.00x. Only the jackknife branch had a
     multiprocessing implementation; the full-sample branch dispatched to the tree or brute
     backend regardless. Both classes now have real `_multiprocessing` backends for the
     full-sample path, and the dispatch honours `num_nodes`.
  2. Every pair-count run created its own process pool. Under the `spawn` start method each
     worker re-imports numpy, scipy, h5py, pyccl and measureia, which costs ~0.9 s for a pool
     of 8 *per pool*. A lightcone `corr_type="both"` measurement performs six pair-count runs
     (S+D, S+R, DD, SR, RD, RR), so it paid that startup six times — about 5.5 s of fixed
     overhead, which is why multiprocessing measured **slower** than single-process below
     roughly 40,000 galaxies. A measurement now opens one pool and shares it across all of its
     pair-count runs (`measureia.worker_pool`).

  Effect on the lightcone jackknife path at 8 nodes: **0.19x -> 0.55x** at 9,600 galaxies and
  **0.85x -> 1.82x** at 38,400 — i.e. from *slower than one node* to genuinely faster. The box
  is unchanged, because it only ever performed one pair-count run and so only ever created one
  pool.

  **Numerical note.** Full-sample lightcone measurements run with `num_nodes > 1` now differ
  from 0.4.0 by ~1e-14 relative, in the `S+R` terms only. That is not a correction: it is the
  float summation order that comes from summing partial grids from separate position chunks,
  the same difference the jackknife multiprocessing path has always had against the tree
  backend. Results at `num_nodes = 1` are bit-identical to 0.4.0.

  Multiprocessing is still a net loss below roughly 40,000 galaxies — one pool costs ~0.9 s
  against less work than that — so `num_nodes > 1` is worth setting above that size and not
  below it.

- **Pair counting scales linearly again, and is up to 6x faster at large N.** The
  kernel chunks the sample 100 at a time, builds a KDTree of each chunk and queries it
  against the full tree of the other sample. A dual-tree query can only prune when the
  chunk occupies a small region, and nothing ordered the catalogue to make it so: a sample
  stored grouped by halo, or by id, or shuffled, gave chunks whose bounding box spanned
  essentially the whole volume, and the query degenerated towards brute force. On the
  package's own mock at 100,000 galaxies the chunk extent was 619 Mpc inside a 711 Mpc box.

  The kernel now visits the chunked sample in Morton (Z-order) spatial order, computed on
  the same coordinates the KDTree is built on. The pairs counted are identical; only the
  order in which they are visited changes.

  Effect at fixed number density, where the pair count is linear in N so the ideal scaling
  exponent is 1.0. Measured single-threaded, cumulative with the candidate-selection change
  above:

  | measurement | 9,600 | 38,400 | 100,000 | scaling exponent |
  |---|---:|---:|---:|---:|
  | box multipoles | 1.8x | 2.7x | **4.3x** | 1.28 -> **1.00** |
  | box `w` | 1.7x | 2.0x | **2.6x** | 1.25 -> 1.12 |
  | lightcone multipoles | 1.9x | 3.5x | **6.2x** | 1.46 -> **1.02** |
  | lightcone `w` | 1.3x | 1.8x | 2.8x | 1.37 -> **1.03** |

  (The lightcone `w` row is the effect of this change alone; unlike the other three its
  0.4.0 baseline was not captured, so no cumulative figure is quoted for it.)

  The gains grow with catalogue size because this fixes a scaling exponent rather than a
  constant. `measure_xi_w` on the box retains an exponent of 1.12: its KDTree is built on
  the 2D projection, so its query returns a cylinder through the full box depth and the
  number of candidate pairs genuinely grows as the box does. That is real pair work, not
  overhead, and is tracked separately (`benchmarks/FINDINGS.md` F2).

  **This also removes a performance trap.** Because the effect depended on how the input
  catalogue happened to be sorted, two users running identical science on identically sized
  catalogues could see very different runtimes — a spatially ordered file (some simulations
  store Peano-Hilbert ordered) was fast, a halo-ordered or shuffled one slow, with nothing
  to explain it. Runtime no longer depends on input ordering.

  **Numerical note.** Results are unchanged in content but not bit-for-bit: visiting
  galaxies in a different order changes the order in which their contributions are summed,
  so floating-point grids differ from 0.4.0 by ~1e-14 relative. Integer pair counts (`DD`,
  `SR`) are exactly unchanged, which is the check that the same pairs are being counted.
  Per-galaxy outputs from `measure_galaxy_contributions` are still indexed by the caller's
  galaxy order — the kernel maps its internal ordering back, and two tests assert this
  against an independent brute-force count and against a shuffled input, because the
  existing per-galaxy tests sum over the galaxy axis and so could not have caught a
  permutation.

### Removed

- `MeasureIABase.setdiff2D` and `MeasureIABase.setdiff_omit`. They existed only to serve the
  candidate-selection set difference removed above and had no remaining caller. They were generic
  nested-list helpers rather than part of the measurement API; if you were calling them,
  `np.setdiff1d` per row is the direct replacement.

- The analytic `RR` in the box is now normalised by `Num_position * Num_shape - num_overlap`, one formula
  replacing four ad-hoc branches, where `num_overlap` is the number of objects present in both samples. The
  two `get_random_pairs*` functions previously disagreed with each other about this: the (rp, pi) form
  assumed the samples were independent while the (r, mu_r) form assumed the shape sample was drawn from the
  position sample, so `w_g+` and `xi_g+,2` measured from one catalogue differed by `N/(N-1)` for no stated
  reason. The overlap is now measured from the coordinates instead of assumed, matching what the lightcone
  already did through `num_samples["D_S"]`. `MeasureIABox` gains a `num_overlap` argument to override it;
  `num_overlap=0` reproduces the convention of external codes such as halotools and corr_pc, and the
  cross-code validation runs now pass it explicitly rather than correcting for the difference afterwards.
  Box `w_gg`/`w_g+` values shift by `N/(N-1)` when the samples overlap; `xi_gg`/`xi_g+,2` are unchanged.

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

[Unreleased]: https://github.com/MarloesvL/measure_IA/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/MarloesvL/measure_IA/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/MarloesvL/measure_IA/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MarloesvL/measure_IA/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/MarloesvL/measure_IA/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MarloesvL/measure_IA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MarloesvL/measure_IA/releases/tag/v0.1.0
