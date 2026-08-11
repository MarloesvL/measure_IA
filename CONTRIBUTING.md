# Contributing to MeasureIA

Thank you for your interest in MeasureIA. This page covers how to get support, report a bug, request a
feature, and contribute code.

## Getting support

If something is unclear, start with the [documentation site](https://marloesvl.github.io/measure_IA/):
the [Getting started](https://marloesvl.github.io/measure_IA/getting_started/) page orients you between the
box and lightcone classes, [Input](https://marloesvl.github.io/measure_IA/input/) describes the data
dictionaries, and [Conventions](https://marloesvl.github.io/measure_IA/conventions/) documents the shape
and sign conventions, which are the usual source of confusion.

If that does not answer it, open a [GitHub issue](https://github.com/MarloesvL/measure_IA/issues) with the
question, or contact Marloes van Heukelum at <m.l.vanheukelum@uu.nl>.

## Reporting a bug

Please open a [GitHub issue](https://github.com/MarloesvL/measure_IA/issues). A report is much easier to act
on when it includes:

- what you expected to happen and what happened instead;
- the MeasureIA version (`python -c "import measureia; print(measureia.__version__)"` if available, or the
  version you installed), your Python version and your operating system;
- a minimal example that reproduces it. `measureia.mocks` is useful here — a mock catalogue reproduces most
  problems without needing your data, and lets anyone else run your example exactly:

  ```python
  from measureia import MeasureIABox
  from measureia.mocks import radial_alignment_box_mock

  mock = radial_alignment_box_mock()
  ...  # the call that goes wrong
  ```

- the full traceback, if there is one.

## Requesting a feature

Please open an issue describing what you would like and, importantly, the measurement you are trying to
make — that often changes what the right feature is. Check the
[roadmap](https://marloesvl.github.io/measure_IA/#roadmap) and the existing
[issues](https://github.com/MarloesvL/measure_IA/issues) first, since it may already be planned.

Issues carry a priority label. If you would like something already listed, comment on it: enough interest is
a good reason to raise its priority.

## Contributing code

**Please open an issue and agree an approach before writing code.** Pull requests that have not been
discussed beforehand will not be accepted. This is not to discourage contributions — it is to avoid anyone
spending time on a change that turns out to conflict with planned work or with the conventions the package
maintains.

Once an approach is agreed:

1. Fork the repository and create a branch off `dev` (not `main`).
2. Set the project up with [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```bash
   git clone https://github.com/<your-fork>/measure_IA.git
   cd measure_IA
   uv sync
   ```

3. Make your change, and add tests for it in `tests/`.
4. Run the suite. It must pass with no warnings — the suite runs with `filterwarnings = ["error", ...]`, so a
   new warning is a failure:

   ```bash
   uv run pytest
   ```

5. If you touched anything users see, update the documentation under `docs/` and check it still builds
   cleanly:

   ```bash
   uv run --group docs mkdocs build --strict
   ```

6. Add an entry to `CHANGELOG.md` under `Unreleased`.
7. Open the pull request against `dev`, referencing the issue.

### Things worth knowing

- **Conventions are load-bearing.** The shape, sign and ellipticity conventions are documented in
  [Conventions](https://marloesvl.github.io/measure_IA/conventions/) and pinned by tests. If a change
  alters them, say so explicitly in the pull request — it changes users' published results.
- **Correctness is checked against other codes.** `validation/` compares MeasureIA against halotools,
  TreeCorr and corr_pc, and `tests/test_validation_references.py` enforces the agreement against committed
  reference outputs. If your change moves any of those numbers, that needs explaining rather than
  re-generating the references.
- **The box and lightcone paths are peers.** A feature added to one is usually expected on the other; if it
  cannot be, the asymmetry should be documented.

## Code of conduct

Please be respectful and constructive in issues and pull requests. Reports of unacceptable behaviour can be
sent to <m.l.vanheukelum@uu.nl>.
