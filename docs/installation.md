# Installation

## Requirements

MeasureIA supports **Python 3.10 – 3.14** and requires **NumPy 2**. The remaining
dependencies (astropy, scipy, h5py, matplotlib, pyccl, sympy) are resolved
automatically; their versions are pinned in `pyproject.toml` / `uv.lock`.

## Installing with pip

```bash
pip install measureia
```

That is the whole installation — everything MeasureIA needs is on PyPI.

## Installing with uv

If you manage your project with [uv](https://docs.astral.sh/uv/getting-started/installation/),
add MeasureIA as a dependency:

```bash
uv add measureia
```

Your scripts then run inside that environment:

```bash
uv run my_script.py
```

To install into an existing environment instead of a uv project, use:

```bash
uv pip install measureia
```

## Installing an unreleased version

Released versions are on PyPI; work in progress lives on the `dev` branch of the
[repository](https://github.com/MarloesvL/measure_IA). To install straight from a branch,
without cloning:

```bash
pip install "git+https://github.com/MarloesvL/measure_IA.git@dev"
uv add "measureia @ git+https://github.com/MarloesvL/measure_IA.git@dev"   # uv equivalent
```

Clone the repository instead if you want to edit the code, run the test suite, or contribute:

```bash
git clone https://github.com/MarloesvL/measure_IA.git
cd measure_IA
uv sync              # creates .venv with the locked dependencies
uv run pytest        # optional: check the suite passes
uv run my_script.py
```

`uv sync` also installs a compatible Python interpreter, so no separate setup is needed.
If you use neither uv nor pip, the dependencies are listed in `requirements.txt`.
See [Contributing](https://github.com/MarloesvL/measure_IA/blob/main/CONTRIBUTING.md) for the
full development workflow.

## Optional extras

The cross-package [validation](validation.md) scripts compare MeasureIA against external
codes, and the [performance](performance.md) benchmarks time it against those same codes —
halotools for the periodic box, TreeCorr for the lightcone. Both use one extra:

```bash
pip install measureia[validation]   # halotools + treecorr
uv add "measureia[validation]"      # uv equivalent
```

Neither package is needed to use MeasureIA or to run its test suite. The scripts themselves
live in `validation/` and `benchmarks/` in the repository rather than in the installed
package, so running them means cloning it as described above:

```bash
uv sync --extra validation                                    # or: pip install -e ".[validation]"
uv run python validation/run_box_halotools.py
uv run python benchmarks/run_sweep.py --smoke --machine laptop
```
