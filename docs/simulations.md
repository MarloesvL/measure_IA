# Included simulations

MeasureIA can fill in the box size and cosmology of several simulations automatically through the `SimInfo`
class. Pass the **tag** as the `simulation` argument of `MeasureIABox` (or `ReadData`) together with a
`snapshot`, and the box size and $h$ are set for you. When a preset is used, positions and separation limits
must be in the stored units, **cMpc/$h$**.

For a simulation that is not listed here, pass `simulation=None` and provide `boxsize` (and optionally `h`)
manually, in the same units as your coordinates.

| Simulation | Tag(s) | Box size [cMpc/$h$] | $h$ | Project |
|---|---|---|---|---|
| IllustrisTNG | `TNG100`, `TNG100_2` | 75.0 | 0.6774 | [tng-project.org](https://www.tng-project.org/) |
| IllustrisTNG | `TNG300` | 205.0 | 0.6774 | [tng-project.org](https://www.tng-project.org/) |
| EAGLE | `EAGLE` | 67.77 (100 cMpc) | 0.6777 | [icc.dur.ac.uk/Eagle](https://icc.dur.ac.uk/Eagle/) |
| Horizon-AGN | `HorizonAGN` | 100.0 | 0.704 | [horizon-simulation.org](https://www.horizon-simulation.org/) |
| FLAMINGO | `FLAMINGO_L1` | 681.0 (1000 cMpc) | 0.681 | [flamingo.strw.leidenuniv.nl](https://flamingo.strw.leidenuniv.nl/) |
| FLAMINGO | `FLAMINGO_L2p8` | 1906.8 (2800 cMpc) | 0.681 | [flamingo.strw.leidenuniv.nl](https://flamingo.strw.leidenuniv.nl/) |
| COLIBRE | `COLIBRE_L400` | 272.4 (400 cMpc) | 0.681 | [Schaye et al. 2025](https://arxiv.org/abs/2508.21126) |
| COLIBRE | `COLIBRE_L200` | 136.2 (200 cMpc) | 0.681 | [Schaye et al. 2025](https://arxiv.org/abs/2508.21126) |

Notes:

- The box size is stored in cMpc/$h$. The IllustrisTNG boxes are natively defined in cMpc/$h$; for EAGLE,
  FLAMINGO and COLIBRE the commonly-cited comoving size (in cMpc, shown in parentheses) is multiplied by $h$.
- `TNG100_2` is the lower-resolution run of the same volume as `TNG100`.
- The `FLAMINGO` and `COLIBRE` tags require the box-length suffix (`_L1`/`_L2p8`, `_L400`/`_L200`) so that the
  correct box size is selected.

The stored specifications are hard-coded in `SimInfo` (see the [API reference](api/SimInfo.md)) and are
straightforward to extend for additional simulations.

## References

- **IllustrisTNG** — [Nelson et al. 2019, *The IllustrisTNG Simulations: Public Data Release*](https://arxiv.org/abs/1812.05609);
  project site: <https://www.tng-project.org/>
- **EAGLE** — [Schaye et al. 2015, *The EAGLE project*](https://ui.adsabs.harvard.edu/abs/2015MNRAS.446..521S/abstract);
  project site: <https://icc.dur.ac.uk/Eagle/>
- **Horizon-AGN** — Dubois et al. 2014, *MNRAS* 444, 1453; project site: <https://www.horizon-simulation.org/>
- **FLAMINGO** — [Schaye et al. 2023, *The FLAMINGO project*](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4978S/abstract);
  project site: <https://flamingo.strw.leidenuniv.nl/>
- **COLIBRE** — [Schaye et al. 2025, *The COLIBRE project*](https://arxiv.org/abs/2508.21126)
