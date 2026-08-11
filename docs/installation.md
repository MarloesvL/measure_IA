# Installation

## Requirements

MeasureIA supports **Python 3.10 – 3.14** and requires **NumPy 2**. The remaining
dependencies (astropy, scipy, h5py, matplotlib, pyccl, sympy) are resolved
automatically; their versions are pinned in `pyproject.toml` / `uv.lock`.

## Installing

You can install MeasureIA via:

```bash
pip install measureia
```

That is the whole installation — everything MeasureIA needs is on PyPI.

Alternatively, you can use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management:

```bash
git clone https://github.com/MarloesvL/measure_IA.git
cd measure_IA
uv sync
uv run [script_name].py
```

If not using uv or pip, install the dependencies via requirements.txt.

## Optional extras

The cross-package [validation](validation.md) scripts compare MeasureIA against external
codes. Those packages are not needed to use MeasureIA, or to run its test suite, and are
installed separately:

```bash
pip install measureia[validation]   # halotools + treecorr
```
