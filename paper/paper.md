---
title: 'MeasureIA: intrinsic alignment correlation functions for simulation boxes and lightcones'
tags:
  - Python
  - astronomy
  - cosmology
  - large-scale structure
  - intrinsic alignments
  - weak lensing
authors:
  - name: M. L. van Heukelum
    orcid: 0009-0008-3780-1617
    affiliation: 1
  - name: N. E. Chisari
    orcid: 0000-0003-4221-6718
    affiliation: 1, 2
affiliations:
  - name: Utrecht University, The Netherlands
    index: 1
  - name: Leiden University, The Netherlands
    index: 2
date: 28 September 2026
bibliography: paper.bib
---

# Summary

Galaxies are not randomly oriented. Tidal fields in the large-scale matter distribution
stretch and twist galaxies as they form, so that their shapes tend to point towards other
galaxies and towards overdensities. These *intrinsic alignments* (IA; see @chisari2025 for a review) are an astrophysical
signal in their own right, carrying information about galaxy formation and the cosmic web,
but they are best known as the leading astrophysical contaminant of weak gravitational
lensing. Weak lensing infers the matter distribution from the coherent distortion of
background galaxy shapes by intervening mass; intrinsic alignments produce a correlated
shape pattern that is difficult to separate from this signal, and if ignored they bias the 
inferred cosmological parameters of current and upcoming surveys [@Krause_2015,@Paopiamsap2024]. 
Modelling IA is therefore part of every weak-lensing cosmology analysis [@Li2023,@DES2026,@Wright2025].
To inform the priors for these analyses, dedicated IA measurements on spectroscopic samples are performed.

`MeasureIA` is a Python package that measures the two-point correlation functions used to
quantify intrinsic alignments: galaxy clustering, $w_{gg}$; the position–shape correlation,
$w_{g+}$; the shape–shape correlations $w_{++}$ and $w_{\times\times}$; and the multipole
moments of these correlations introduced by @singh2024. Every statistic comes with
a jackknife estimate of its covariance matrix. The package works in the two data regimes in
which IA is measured: periodic simulation boxes, in Cartesian coordinates with analytic
random counts, and lightcones or survey-like catalogues, in sky coordinates (RA, Dec,
redshift) with an explicit random catalogue. Both are reached through a single interface with
a single, documented set of shape and sign conventions.

# Statement of need

Intrinsic alignments are measured for many reasons, but two settings are the most common.
In the periodic boxes of hydrodynamical simulations, IA is measured to understand its
physical origin and to calibrate models; on the sky, from real or mock galaxy catalogues, it
is measured to constrain its contribution to a lensing analysis. As both these settings inform 
each other, it is important to use cohesive estimators and conventions.

Few codes measure IA correlations directly: `halotools` [@hearin2017; @vanalfen2025] does so for periodic
boxes only, without multipoles or covariances. In practice, IA correlations are therefore
often reconstructed from general-purpose pair counters, most commonly `TreeCorr`
[@jarvis2004], with which a single projected $w_{g+}$ requires a separate run for
every term of the estimator — data–data, data–random, random–random and shape–density — after
which the user combines, normalises and integrates the counts themselves. Where the data do
not match the geometry a tool was built for, the data are often transformed to fit the tool
rather than the other way around: a periodic box is recast as a patch of sky, or a lightcone
is flattened into a box. Such workarounds lose information the measurement depends on, such
as periodicity or the true line of sight, and they add approximations that are rarely
checked or reported.

Each reconstruction is also error-prone in ways that are specific to IA. A position–shape
estimator depends on a chain of conventions that are rarely written down together: the definition of
the tangential and cross ellipticity components $e_+$ and $e_\times$ and their sign, the
handedness of the $e_1$/$e_2$ components on the sky, the definition of the ellipticity and
its shear responsivity, and the normalisation of the random pairs. Several of these fail
silently. A wrong chirality of $e_1$ relative to $e_2$, for example, does not flip the sign
of $w_{g+}$ but washes the signal out into noise, which is easily mistaken for a physical
null result or a numerical problem. Box and lightcone measurements, moreover, are usually
made with different codes that follow different conventions, which makes a like-for-like
comparison between simulation and survey awkward.

`MeasureIA` addresses this by providing validated IA estimators including jack-knife covariance for both 
regimes behind one interface, with the conventions fixed, documented and tested. A measurement is a single
method call on a catalogue, and the result — correlation function and covariance, including their building 
blocks — is written to a documented HDF5 file. The target audience is anyone measuring
alignments, in particular in cosmological boxes, on the sky, or in both.

# State of the field

Several mature packages compute correlation functions that overlap with `MeasureIA`, and it
is validated against all of them.

`halotools` [@hearin2017; @vanalfen2025] provides projected position–shape and shape–shape correlations
(`gi_plus_projected`, `ii_plus_projected`) and projected clustering for periodic boxes. It does
not provide multipole estimators, jackknife covariances for these statistics, or lightcone
geometry. With the shear responsivity accounted for, `MeasureIA` reproduces its $w_{g+}$,
$w_{gg}$ and $w_{++}$ to machine precision.

`TreeCorr` is a fast, general-purpose correlation engine for catalogues on the sky, with no
periodic-box mode; as described above, IA estimators must be assembled from its raw pair
counts. `MeasureIA` agrees with estimators reconstructed from
`TreeCorr` counts to better than 0.5% in $w_{g+}$ and $w_{gg}$, with the residual traced to the
slightly different definitions of projected separation, and its lightcone jackknife
covariance agrees with an explicit `TreeCorr` jackknife loop to a few percent or better.

`corr_pc` [@singh2021] is the C++ code with which the multipole estimator was
first developed. It is an independent implementation of both the box and sky estimators and
of the jackknife, but it is not packaged or documented for general use. `MeasureIA` matches
its multipoles to the precision of its text output, and its box and lightcone jackknife
realisations once the codes' differing normalisation conventions are aligned.

`MeasureIA` is a new package rather than an extension of one of these because its purpose
is to cover both regimes with one set of IA conventions. `halotools` is built around periodic
boxes and `TreeCorr` is intentionally agnostic about the physics of the correlation it
computes; adding IA estimators, multipoles, responsivity and jackknife covariances to either
would sit outside that package's design. The comparisons are shipped with the package: each
is a runnable script in `validation/`, the external results are committed as small reference
files, and the test suite checks `MeasureIA` against them in continuous integration without
the external codes installed. The agreements, and every convention difference found along the
way, are documented on the validation page of the documentation.

# Software design

**Native geometry for each regime.** Boxes are measured in Cartesian coordinates with
periodic boundaries and analytic random counts, which are exact and free of shot noise.
Lightcones are measured in (RA, Dec, redshift), with comoving distances, which factor the universe's
expansion out of the distance for a given cosmology, from `CCL`
[@chisari2019], a per-pair midpoint line of sight and an explicit random catalogue, which the survey
footprint requires. Neither regime is converted into the other, so neither loses periodicity
or picks up curvature errors.

**One pair kernel.** Every statistic, including multipoles and jackknife realisations, is
accumulated by a single pair-counting kernel parameterised by geometry and binning, so shape
projections and binning are written once and cannot drift between statistics or regimes. It
uses `SciPy` [@virtanen2020] KD-trees, is checked against a brute-force implementation, runs
on multiple processes, and scales linearly with catalogue size.

**One set of conventions**, documented and protected by tests: parity null tests, invariance
under flipping any galaxy's axis direction, and a positive signal for radial alignment.

**Jackknife and output.** The covariance uses sub-boxes for boxes and k-means patches on the
sphere for lightcones; realisations are formed by subtracting each region's pairs from the full
counts, which is verified to equal physically deleting the region. Results, pair counts and
realisations are written to documented HDF5 files, and `measureia.mocks` generates seeded
catalogues with a known signal so that every code path can be run without data.

# Research impact statement

`MeasureIA` was developed for measuring intrinsic alignments in hydrodynamical simulations,
and all measurements in @vanHeukelum2026a and @vanHeukelum2026b were made with it: the
former studies alignments in multiple projections, the latter compares disk and elliptical
galaxies across several simulations. Beyond these, the package is used by several researchers
in the IA community and by students, who have used it in their research projects over
the years. It has been run on the IllustrisTNG, EAGLE, Horizon-AGN, FLAMINGO and COLIBRE cosmological 
hydrodynamical and N-body simulations, for each of which it ships a preset with the box size 
and cosmology.

Its near-term significance lies in comparing simulations with surveys. Stage-IV lensing
surveys will need IA models calibrated on simulations and tested against survey measurements,
and `MeasureIA` makes both measurements with the same estimators and conventions. The `python` package
is installable from PyPI, archived on Zenodo (doi:10.5281/zenodo.17252215), documented with
runnable example notebooks, and validated against three independent codes.

# AI usage disclosure

`MeasureIA` was written by the authors without generative AI from 2023 onwards, and released
three times (v0.1.0 to v0.3.0, October 2025 to January 2026). By then the box estimators,
multipoles and jackknife covariances were working and validated, in a separate, unpublished
set of scripts: $w_{g+}$ and $w_{gg}$ against `halotools` and the multipoles against
`corr_pc`. The lightcone pipeline was at an earlier stage, but its $w_{g+}$ had been
validated against `TreeCorr`. From
July 2026, generative AI (Anthropic's Claude, through Claude Code) was used to prepare the
package for wider use: to refactor it, most notably consolidating the pair-counting code into
a single kernel; to rebuild the validation as the reproducible suite shipped with the package;
to complete the lightcone pipeline; to write tests and benchmarks; for documentation; and to
draft this paper. The scientific design, the estimators and all conventions were decided by the
authors. AI-generated code was reviewed by the authors and is covered by the test suite,
which includes brute-force cross-checks and the external validation against `halotools`,
`TreeCorr` and `corr_pc` described above; results were required to be unchanged by refactoring
before it was accepted. The authors reviewed and edited the paper text and take responsibility
for its content.

# Acknowledgements

We thank Elizabeth Gonzales, Casper Vedder, Dennis Neumann, Paul Rogozenski, Henk Hoekstra and Aniruddh Herle for their
valuable discussions and help in validating and debugging this package. This publication is part of the project
''A rising tide: Galaxy intrinsic alignments as a new probe of cosmology and galaxy evolution'' 
(with project number VI.Vidi.203.011) of the Talent programme Vidi which is (partly) financed by the Dutch Research 
Council (NWO).
`MeasureIA` is built on `NumPy` [@harris2020], `SciPy` [@virtanen2020], `Astropy`
[@astropy2022], `h5py` and `CCL` [@chisari2019].

# References
