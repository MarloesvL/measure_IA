# Pre-release task list

Prioritized task list from the full package review (2026-07-17). P0 = confirmed bugs,
P1 = features, P2 = input validation, P3 = test suite, P4 = cleanup & docs.

## P0 — Confirmed bugs (fix before release)

- [x] **NEW (found during P0 work): lightcone backend weight-mask fallback misalignment.**
  *Investigated and resolved. Verdict: NOT reachable through the public API — `_merged_masks`
  always injects `weight`/`weight_shape_sample` keys defaulting to the slot's coordinate mask,
  so the backend fallback was dead code on that path. The fallbacks were still misaligned
  landmines for direct backend calls; all 22 blocks now default to `masks["RA"]` /
  `masks["RA_shape_sample"]` (aligned) with explicit `not in` checks. Also removed every
  remaining bare `except:` in the package: debug try/except around `np.add.at` (which
  swallowed real errors and printed garbage) deleted, `write_data.py`/`read_data.py`
  delete-if-exists idioms rewritten as `in` checks, one-vs-two-randoms detection narrowed to
  `(KeyError, TypeError)`. `grep 'except:'` over src/ is now clean.*
- [x] **Weight-mask misalignment in Box backends.** When a `Position` (or `Position_shape_sample`)
  mask is given without a `weight` mask, the fallback keeps the *first* `sum(mask)` weights
  (`masks["weight"][sum(pos_mask):] = 0`) instead of selecting `weight[pos_mask]`, so weights only
  align with positions if the mask selects a contiguous prefix — silently wrong weighted results.
  Fix: default `masks["weight"] = pos_mask` and `masks["weight_shape_sample"] = shape_mask`.
  Locations: `src/measureia/measure_w_box.py:112-117`, `:281-286`, `:555-560` and the
  equivalents in `measure_m_box.py`, `measure_w_box_jk.py`, `measure_m_box_jk.py`.
- [x] **Unguarded divisions in Box produce silent NaN/inf.** Analytic `RR` is 0 when a sample is
  empty or fully masked; `Splus_D / RR_g_plus` and `(DD / RR_gg) - 1` then emit NaN/inf without
  warning (`measure_w_box.py:203, 225, 383, 405, 669, 691` + multipoles-box equivalents). The
  lightcone already guards this (`measure_w_lightcone.py:218-219`); port the guard to all Box
  divisions and to the non-DD denominators in `_obs_estimator` (`measure_IA_base.py:571-592`).
  *Done: guarded all full-sample and jk-realisation RR divisions in the four Box backend files
  (`xi_gg` set to 0 where analytic RR is 0), guarded the responsivity `R`/`R_jk` against zero
  total weight (falls back to 0.5), and guarded the scalar sample-count denominators in
  `_obs_estimator`. Deliberately NOT guarded: the empirical `RR`/`SR` array bins in
  `_obs_estimator` — a zero empirical-RR bin means the lightcone estimator is undefined there,
  and NaN is the honest output (committed validation references rely on this). The two
  matching xfail edge-case tests (all-zero weights, empty mask) now pass and were unmarked.*
- [x] **Bare `except:` silently zeroes samples.** A typo'd data key or malformed weight array is
  swallowed and the run proceeds with `Num_position = 0` or unit weights
  (`measure_IA_base.py:134-153`; lightcone weight-defaulting at
  `measure_IA_lightcone.py:483-517`, `:696-730`). Narrow to `except KeyError`.
- [x] **`exit()` in library code.** The "lightcone data passed to Box" guard prints and calls
  `exit()` inside a `try/except` that swallows the resulting `SystemExit`, so the guard is
  dead (`measure_IA.py:162-167`, `:263-268`). Replace with `raise TypeError(...)`.
- [x] **`ReadData` undefined-variable fallthrough.** On `KeyError` the readers print
  "Variable not found" and then use the undefined `data` variable, raising `UnboundLocalError`
  (`read_data.py:140-143`, `:195-198`, `:262-266`). Raise a clear error instead.

## P1 — Features

- [x] **Lightcone multipoles multiprocessing (mirror `w` exactly).**
  *Done. Also found and fixed while cloning: the `w` mp batch worker
  (`_measure_xi_rp_pi_lightcone_jk_batch`) clamped its index range with `Num_shape_masked`
  while iterating the position sample — for position samples larger than the shape sample
  (the S+R term: positions are the randoms) trailing chunks were silently dropped. The
  count-pairs twin had already been corrected. All previous lc "multiprocessing" tests ran
  `measure_cov=False`, which has no mp path, so they compared tree-vs-tree and never caught
  it. New `test_multiproc_vs_tree_jk` tests (w + multipoles, `measure_cov=True`) verified to
  fail with the bug present and lock both mp paths now.* No multiprocessing exists for
  lightcone multipoles today: `num_nodes`/`chunk_size`/`temp_file_path` are threaded through the
  signatures but ignored, and the dispatcher (`measure_IA_lightcone.py:761-775`) lacks the
  `num_nodes` branch that `measure_xi_w` has (`:549-570`).
  - Add to `measure_m_lightcone_jk.py` (jackknife path only, exactly like `w`):
    `_measure_xi_r_mur_lightcone_jk_multiprocessing` + `_measure_xi_r_mur_lightcone_jk_batch`
    (clones of `measure_w_lightcone_jk.py:650` / `:523` with `(r, mu_r)` binning as in
    `_measure_xi_r_mur_lightcone_jk_tree`), and
    `_count_pairs_xi_r_mur_lightcone_jk_multiprocessing` +
    `_count_pairs_xi_r_mur_lightcone_jk_batch` (clones of `:1306` / `:1215`).
  - Same pattern: SharedMemory blocks + temp-HDF5 offload + `Pool(num_nodes).map` + per-batch
    grid summation; add the missing imports (`os`, `multiprocessing`/`Pool`/`shared_memory`,
    `ReadData`).
  - Add the `if self.num_nodes == 1: ... else:` split in `measure_xi_multipoles()`, routing
    through the existing binning-agnostic `measure_xi_jk_helper`. No base-class changes needed.
  - Test: extend the n1-vs-n8 equality tests (pattern in `test_lc_measure_xi_w.py`) to
    lightcone multipoles.
- [x] **Box `count_pairs` methods for `corr_type='gg'` (non-jackknife paths).**
  *Done for `num_jk=0`: `_count_pairs_xi_rp_pi_box_brute/tree/batch/multiprocessing` in
  `measure_w_box.py` and `_count_pairs_xi_r_mur_box_*` in `measure_m_box.py`, dispatched from
  `measure_IA.py` when `corr_type='gg'`. DD grids bit-identical to the full loop (locked by
  tests, mp to 1e-10). Also found and fixed: `rp_cut` was accepted by `measure_xi_multipoles`
  but never forwarded to any backend — silently ignored; now forwarded in both branches, with
  a regression test. Remaining follow-up (open item below): jackknife count_pairs twins.*
- [x] **Box `count_pairs` jk twins (gg with covariance).**
  *Done: `_count_pairs_xi_rp_pi_box_jk_{brute,tree,batch,multiprocessing}` in
  `measure_w_box_jk.py` and `_count_pairs_xi_r_mur_box_jk_*` in `measure_m_box_jk.py`
  (DD, DD_jk, analytic RR_gg/RR_jk and xi_gg realisation writes only — no ellipticity, R,
  R_jk, Splus or sigmasq work), dispatched from both jk branches of `measure_IA.py` when
  `corr_type='gg'`. Tests compare final w_gg/multipoles, realisations and covariance against
  the full-loop jk path for all three backends.* The Box currently always computes
  `arccos` + `get_ellipticity` and accumulates `Splus_D`/`Scross_D` even when only `gg` is
  requested (`measure_w_box.py:159-186`); `corr_type` is only consulted at the reduction stage.
  - Add DD-only `_count_pairs_*` variants (brute/tree/batch + multiprocessing, and jk twins) to
    `measure_w_box.py`, `measure_m_box.py`, `measure_w_box_jk.py`, `measure_m_box_jk.py`,
    mirroring the lightcone pattern (`measure_w_lightcone.py:652`, `:919`); dispatch from
    `measure_IA.py` when `corr_type == 'gg'`.
  - Preserve: periodic KDTree (`boxsize=`), minimum-image wrapping
    (`measure_w_box.py:149-151`), analytic `RR_gg` from `get_random_pairs(_r_mur)` unchanged.
  - Test: `gg` result from the count_pairs path bit-compatible with the current full-loop result.

- [x] **Lightcone multiprocessing shared-memory name collision.**
  *Fixed: lightcone mp methods now use the Box `ID_shm` random-suffix pattern (jk-index keys
  shortened to `jk_region_indices_*` — macOS caps POSIX shm names at 31 chars). Also fixed the
  data-restore-on-failure gap in the same batch: the temp-file reload of `self.data` now lives
  in the `finally` block of all 12 mp methods (guarded by `os.path.exists`), so a mid-run
  worker failure no longer leaves `self.data` empty. The last xfail test was rewritten to
  monkeypatch `Pool` to fail after offloading and now passes as a regular test — the suite has
  zero xfails.*

- [x] **`_obs_estimator` empty-bin semantics (user-approved 2026-07-18).**
  *Empty-DD bins: the `DD[DD==0]=1` guard was applied to raw counts, biasing the gg
  Landy-Szalay numerator by 1/norm in empty bins; now the guard applies only to a denominator
  copy used by the clusters division — empty bins contribute 0 to gg (formula test added).
  Empty empirical-RR bins: stay NaN/inf (estimator undefined; references depend on it) but a
  RuntimeWarning now fires on the full-sample pass naming the number of affected bins and
  advising more randoms (test added).*

## P2 — Input validation & error messages

- [ ] **Symmetrize validation: add Lightcone type/shape checks.** The Box got
  `check_type_input_data`/`check_units_coordinates` in PR #61; the Lightcone only got
  key-existence + path checks (`measure_IA_lightcone.py:117-127`). Add a lightcone variant:
  RA/DEC/redshift/e1/e2 are ndarrays, consistent lengths (e1/e2 vs shape sample),
  RA ∈ [0, 360], DEC ∈ [-90, 90], finite values.
- [ ] **Harden `check_type_input_data`** (`check_input.py:78-104`): replace bare `assert`s
  (stripped under `python -O`, no message) with `TypeError`/`ValueError` + messages; accept
  `np.integer` for `LOS` (currently `np.int64(2)` fails); add length-consistency checks
  (`q`, `Axis_Direction`, `weight*` vs their samples) and NaN/inf checks.
- [ ] **Validate option strings at method entry.** Box `corr_type` currently fails only *after*
  the full pair count (`measure_IA_base.py:418/499`; jk path `measure_IA.py:196/297`);
  `ellipticity` only inside the backends. Check both at the top of
  `measure_xi_w`/`measure_xi_multipoles`, mirroring the lightcone's up-front `IA_estimator`
  check (`measure_IA_lightcone.py:450-466`). Unify the exception type for unknown options
  (currently a mix of `KeyError`/`ValueError`).
- [ ] **Validate numeric constructor params** (`measure_IA_base.py:154-166`):
  `num_bins_r`/`num_bins_pi` >= 1; `separation_limits` has 2 elements with
  `0 < r_min < r_max`; `boxsize > 0`; `num_nodes >= 1`; lightcone `num_jk` bounds vs sample
  size before `kmeans_sample` (`measure_jackknife.py:94`); Box negative `num_jk` is currently
  silently treated as 0.
- [ ] **`check_units_coordinates`** (`check_input.py:60-76`): handle 1-D coordinate arrays
  gracefully instead of a confusing indexing error.

## P3 — Test suite

- [ ] **Test infra**: add `[tool.pytest.ini_options]` to `pyproject.toml` (`testpaths`,
  `xfail_strict = true`, `filterwarnings`); add a CI workflow
  (`.github/workflows/tests.yml`, Python version matrix; optionally `pytest-cov`).
- [ ] **Resolve parked xfails.** `TestEdgeCasesUnhandled` cites a non-existent
  `fixes_measure_IA_base.py`; the three known bugs behind the xfails (all-zero weights → NaN,
  empty mask → crash, data not restored after a failed backend) are largely fixed by the P0
  items — flip the xfails to real assertions afterwards.
- [ ] **Dead/stale test scaffolding**: remove the unused `_out`/`_ref`/`PROC_PATH` constants in
  `test_lc_measure_xi_w.py` and `test_lc_measure_xi_multipoles.py`; rename `TestRegressionW/M`
  (they test determinism, not saved references) or commit real reference outputs; fix stale
  `conftest_lc.py` docstring references; decide the fate of the unused
  `tests/data/**/*.hdf5` files.
- [ ] **New coverage** (mirror existing box tests to the lightcone and fill gaps):
  - lightcone invalid `num_jk` / invalid `ellipticity` tests;
  - lightcone empty-catalogue / empty-mask / all-zero-weights / single-object tests;
  - lightcone `measure_covariance_multiple_datasets` / `create_full_cov_matrix_projections`
    tests (currently box-only);
  - unit tests for `assign_jackknife_patches` (output structure, seed determinism, coverage);
  - small-sample jackknife: `num_jk > N`, empty patch → finite covariance or a clear error;
  - exercise `rp_cut` (multipoles) and non-default `cosmology` arguments;
  - a multiprocessing test whose catalogue size does not divide evenly by chunk size;
  - synthetic-HDF5 tests for `read_subhalo`, `read_snapshot`, `read_snapshot_multiple`,
    `read_modelling_outputs` (zero coverage today);
  - tighten the `rel=0.5` tolerance in `test_rr_gg_consistent_with_formula`.

## P4 — Cleanup & docs

- [ ] **Dead code**: remove `_measure_xi_rp_pi_lightcone_brute_old` and
  `_count_pairs_xi_rp_pi_lightcone_brute_old` (`measure_w_lightcone.py:247`, `:788`); remove or
  implement the commented `auto`-correlation branches (Box hardcodes `corrtype="cross"`);
  remove the commented variance lines in `_measure_w_g_i`.
- [ ] **Docstring fixes**: phantom `randoms_data` parameter in the lightcone measure methods;
  undocumented `responsivity`/`tree`/`temp_file_path` parameters; "chunck_size" typo
  (`measure_IA.py:235`); sync the `SimInfo` simulation lists (class docstring vs `get_specs`
  vs `get_file_info` disagree on COLIBRE/FLAMINGO variants); add a docstring to
  `read_modelling_outputs`.
- [x] **Outstanding TODOs in code** — triaged (user-approved 2026-07-18):
  *`min_patch=1`: won't-fix — 1-based patch indices now raise a clear `ValueError` telling the
  user to renumber (test added); `auto` corrtype: dead commented-out branches deleted from the
  Box backends, DD documented as cross-count-only (`get_random_pairs`' tested `auto` utility
  branch kept); `++` correlation: deferred post-JOSS, comment updated to say so.* Still open
  from this list: "deal with masks" in the lightcone dispatchers (`measure_IA_lightcone.py`,
  the `# ToDo: deal with masks` sites) — belongs with the P2 lightcone-masks work.
- [ ] **`check_paths`**: also verify writability and input-file existence; give a clear error
  for a missing HDF5 file in `ReadData` (currently a raw `OSError`).
