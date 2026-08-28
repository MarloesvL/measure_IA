# Validation

Beyond the internal unit and analytic tests, MeasureIA's estimators are cross-validated against several
**independent, publicly available** correlation-function codes. This gives an end-to-end check that the full
pipelines — pair counting, estimators, multipole integration and jackknife covariance — agree with established
implementations, for both the box and the lightcone.

## Approach

- **Mock catalogues** (`measureia.mocks`): all comparisons run on a seeded synthetic catalogue
  with a strong, known radial-alignment signal, so ratios are meaningful in every bin. Both codes read
  byte-identical inputs.
- **Runnable scripts** (`validation/run_*.py`): one per comparison. Each always computes the MeasureIA side; if
  the external package is installed it also computes the external side and writes it to
  `validation/reference_outputs/`, otherwise it compares against the committed reference file.
- **Enforced in CI** (`tests/test_validation_references.py`): the committed reference outputs are compared against
  MeasureIA at fixed tolerances, so the cross-package agreement is checked on every test run without needing the
  external packages installed.

## What has been validated

| Comparison | External code | Quantities | Agreement |
|---|---|---|---|
| Box $w_{gg}$, $w_{g+}$ | halotools | projected $w$ | machine precision ($2\times10^{-13}$ / $4\times10^{-15}$)¹ |
| Box multipoles | corr_pc | $\xi(r,\mu)$ grid + $\tilde\xi_{gg,0}$, $\tilde\xi_{g+,2}$ | $\le5\times10^{-6}$ grid, $\sim10^{-6}$ multipoles¹ |
| Lightcone $w$ | treecorr | $w_{gg}$, $w_{g+}$ | $\sim10^{-5}$ ($g+$), $\le0.4\%$ ($gg$) |
| Lightcone $w$ | corr_pc | $w_{gg}$, $w_{g+}$ | $\le0.15\%$ ($g+$), $\le0.4\%$ ($gg$) |
| Lightcone multipoles | corr_pc | $\tilde\xi_{gg,0}$, $\tilde\xi_{g+,2}$ | $\le0.2\%$ / $\le0.3\%$ |
| Box ↔ lightcone (plane-parallel) | — (self-consistency) | pair counts, $w$ | DD $<1\%$; residuals fully attributed² |
| Box jackknife (delete-one identity) | — (self-consistency) | realisations, cov | machine precision ($\le10^{-12}$) |
| Box jackknife covariance | corr_pc | $w$ + multipole cov | realisations $\le5\times10^{-5}$, std $\le5\times10^{-7}$ |
| Lightcone jackknife covariance | treecorr | $w$ cov | std $\le5\times10^{-5}$ ($g+$), $\le0.6\%$ ($gg$) |
| Lightcone jackknife covariance | corr_pc | $w$ + multipole cov | realisations $\le3\times10^{-4}$ |

¹ Agreement is exact up to floating point / the external code's output precision, once the responsivity
$2\mathcal{R}$ factor is accounted for (MeasureIA divides $S_+$ terms by $2\mathcal{R}$; halotools and corr_pc do
not — see [Conventions](conventions.md)).
² The plane-parallel box↔lightcone difference is understood: analytic randoms (periodic box) versus empirical
randoms (bounded window), plus the box/lightcone estimator and responsivity differences.

The lightcone comparisons also confirm the **`e1`/`e2` shear convention** and chirality documented on the
[Conventions](conventions.md) page (treecorr needs only the standard IA flip $g \to -g$).

## Running the validations yourself

The comparison scripts live in `validation/` in the repository rather than in the installed package, so start
from a clone (see [Installation](installation.md#installing-an-unreleased-version)) and install the
pip-available external packages with it:

```bash
git clone https://github.com/MarloesvL/measure_IA.git
cd measure_IA
uv sync --extra validation            # or: pip install -e ".[validation]"  -- halotools + treecorr
```

Then run the enforced cross-package checks (these use the committed reference outputs):

```bash
uv run pytest tests/test_validation_references.py
```

Or run an individual comparison script, which will use the external package if it is installed and otherwise
compare against the committed reference:

```bash
uv run python validation/run_box_halotools.py
uv run python validation/run_lightcone_treecorr.py
```

The **corr_pc** comparisons ([Singh 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1632S/abstract),
[corr_pc on GitHub](https://github.com/sukhdeep2/corr_pc)) require building the C++ code separately. The full
build recipe (including a no-MPI stub and two small patches) and the detailed per-comparison convention notes are
in [`validation/README.md`](https://github.com/MarloesvL/measure_IA/blob/main/validation/README.md) in the
repository.
