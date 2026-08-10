# Installation

## Requirements

MeasureIA supports **Python 3.10 – 3.14** and requires **NumPy 2**. The remaining
dependencies (astropy, scipy, h5py, matplotlib, pyccl, sympy, kmeans_radec) are resolved
automatically; their versions are pinned in `pyproject.toml` / `uv.lock`.

## Installing

You can install MeasureIA via:

```bash
pip install measureia
```

Note: the package depends on kmeans_radec, which is not pip‑installable. You need to install it manually
(see https://github.com/esheldon/kmeans_radec).

Alternatively, you can use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management:

```bash
git clone https://github.com/MarloesvL/measure_IA.git
cd measure_IA
uv sync
uv run [script_name].py
```

If not using uv or pip, install dependencies via requirements.txt, remembering the kmeans_radec external dependency.