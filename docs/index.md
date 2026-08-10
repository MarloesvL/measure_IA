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
- **Concepts**
    - [Conventions](conventions.md) — the shape and sign conventions (incl. the `e1`/`e2` convention) and binning
    - [Estimator definitions](estimator_definitions.md) — the mathematics of the estimators
- [Validation](validation.md) — cross-package validation and how to run it yourself
- [Roadmap](roadmap.md)

The **API Reference** (see the navigation sidebar) documents `MeasureIABox`, `MeasureIALightcone` and the
supporting classes.

## Contributions
### Bugs

If you find a bug, please report it in a GitHub [issue](https://github.com/MarloesvL/measure_IA/issues).

### Features

If you would like a feature added that is not already on the [Roadmap](roadmap.md) or in an
[issue](https://github.com/MarloesvL/measure_IA/issues) on Github,
please create an issue with the request.
Within the issue, we can discuss how best to proceed and what the timeline will be. 
Pull requests that have not been discussed beforehand will not be accepted.

Note that the issues on GitHub contain a priority. Please comment on those you would like to have added, if there is 
enough interest, I will consider increasing the priority status.

## Citation

Please use the [CITATION](https://github.com/MarloesvL/measure_IA/blob/main/CITATION.cff) file to cite this package 
properly. Note that a DOI will be added to this soon. If you would like to cite this package and the DOI is not there,
please contact me.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)