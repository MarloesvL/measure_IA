# MeasureIA Documentation

Welcome to the documentation site of **MeasureIA**, a tool to measure intrinsic alignment correlation
functions in hydrodynamic simulations, for both periodic simulation boxes and lightcones.
Please feel free to contact me (m.l.vanheukelum@uu.nl) if you have any questions, or create an issue on
[GitHub](https://github.com/MarloesvL/measure_IA).

## Pages

- [Getting started](getting_started.md) — what MeasureIA does and when to use the box vs the lightcone
- [Installation](installation.md)
- **Guides**
    - [Input](input.md) — the data dictionaries expected on initialisation
    - [Usage](usage.md) — worked examples for the box and lightcone
    - [Output](output_structure.md) — the structure of the output file
    - [Included simulations](simulations.md) — box sizes and cosmologies known to `SimInfo`
- **Concepts**
    - [Conventions](conventions.md) — the shape and sign conventions (incl. the `e1`/`e2` convention) and binning
    - [Estimator definitions](estimator_definitions.md) — the mathematics of the estimators
- [Validation](validation.md) — cross-package validation and how to run it yourself
- [Roadmap](#roadmap) — recently completed and planned developments

The **API Reference** (see the navigation sidebar) documents `MeasureIABox`, `MeasureIALightcone` and the
supporting classes.

## Roadmap

Recently completed:

- Lightcone methods & cross-code validation (against halotools, treecorr and corr_pc)
- Multiprocessing support for the lightcone version (both `w` and multipoles)
- Optional responsivity factor for the shape calibration
- NumPy 2 support and testing across Python 3.10 – 3.14
- More exhaustive docstrings and internal method docs

Planned developments include:

- Non-periodic versions of box methods
- e1,e2 input for box methods
- look into extra speed up options

The issues on [GitHub](https://github.com/MarloesvL/measure_IA) are also used as To Do's. Feel free to request
features or comment on those already there to let me know you would like them to have a higher priority.

## Contributions

Bug reports, feature requests and questions are all welcome as
[GitHub issues](https://github.com/MarloesvL/measure_IA/issues).

See [CONTRIBUTING.md](https://github.com/MarloesvL/measure_IA/blob/main/CONTRIBUTING.md) for what makes a
bug report easy to act on (`measureia.mocks` usually reproduces a problem without your data), how feature
requests are prioritised, and the workflow for contributing code. Pull requests that have not been
discussed in an issue beforehand will not be accepted.

Issues carry a priority label. If you would like something that is already listed, comment on it — enough
interest is a good reason to raise its priority.

## Citation

Please use the [CITATION](https://github.com/MarloesvL/measure_IA/blob/main/CITATION.cff) file to cite this package
properly. MeasureIA is archived on Zenodo under the DOI
[10.5281/zenodo.17252215](https://doi.org/10.5281/zenodo.17252215), which always resolves to the latest
released version. If you need to cite the exact version you used, take that version's own DOI from the
[Zenodo record](https://doi.org/10.5281/zenodo.17252215) instead.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)