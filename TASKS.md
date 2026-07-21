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

- [x] **Symmetrize validation: add Lightcone type/shape checks.** *Done: new
  `check_type_input_data_lightcone` (finite 1-D ndarrays; density- and shape-sample
  coordinate lengths internally consistent; RA ∈ [0,360], DEC ∈ [-90,90]), wired into the
  MeasureIALightcone constructor. Tests added.*
- [x] **Harden `check_type_input_data`** *Done: bare asserts → TypeError/ValueError with
  messages; `np.integer` LOS accepted; length-consistency (Axis_Direction/q vs shape sample)
  and NaN/inf checks added. Tests updated + added.*
- [x] **Validate option strings at method entry.** *Done: `MeasureIABox._validate_measure_options`
  called at the top of `measure_xi_w`/`measure_xi_multipoles` — corr_type and ellipticity
  checked up front with a uniform ValueError; negative num_jk now raises. Box invalid-corr_type
  tests updated to ValueError.*
- [x] **Validate numeric constructor params.** *Done: num_bins_r/num_bins_pi ≥ 1,
  separation_limits length-2 with 0 < r_min < r_max, boxsize > 0 (when set), pi_max > 0 (when
  set) in MeasureIABase; num_nodes ≥ 1 in both subclasses; lightcone num_jk bounded by the
  number of randoms in assign_jackknife_patches; negative Box num_jk raises. Tests added.*
- [x] **`check_units_coordinates`** *Done: rejects non-(N,3) input with a clear message
  instead of a confusing indexing error on 1-D arrays.*

## P3 — Test suite

- [x] **Test infra**: *Done: `[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths`,
  `xfail_strict = true`) and `.github/workflows/tests.yml` — uv-based, push to main/dev +
  PRs + manual, Python 3.10/3.11/3.12 (3.13 excluded: numpy 1.26.4 has no cp313 wheels;
  reason recorded in the workflow). Suite verified locally on 3.10 and 3.12 too. Not added:
  `filterwarnings` (the suite raises ~1000 deliberate RuntimeWarnings about empty RR bins —
  worth a policy pass of its own) and `pytest-cov`.*
- [x] **Resolve parked xfails.** *Done: the P0 fixes resolved all of them; the class was
  renamed `TestEdgeCasesNowHandled`, the markers removed, and the docstring references to the
  non-existent `fixes_measure_IA_base.py` dropped. The suite has zero xfails.*
- [x] **Dead/stale test scaffolding**: *Done: the unused `_out`/`_ref`/`PROC_PATH` constants
  and duplicate `NUM_JK` removed from the two lightcone test modules; `TestRegressionW/M`
  renamed to `TestDeterminismW/M` (they compare two runs of the same configuration, not a
  committed reference); stale `conftest_lc.py` docstring references fixed; the 6 unused
  `tests/data/**/*.hdf5` files (9.5 MB, referenced by no test since the suite moved to
  synthetic fixtures) removed.*
- [x] **New coverage** (mirror existing box tests to the lightcone and fill gaps):
  - [x] lightcone invalid `num_jk` / invalid `ellipticity`: covered by the existing
    invalid-estimator / invalid-corr_type / one-based-patch tests plus the new
    `assign_jackknife_patches` num_jk validation;
  - [x] lightcone empty-mask / all-zero-weights / single-object tests
    (`TestEdgeCasesLightcone`);
  - [x] lightcone `measure_covariance_multiple_datasets` /
    `create_full_cov_matrix_projections` (`TestCovarianceUtilitiesLightcone`);
  - [x] unit tests for `assign_jackknife_patches` (structure, label range, patch coverage,
    seed determinism, global random-state restoration, invalid num_jk);
  - [x] small-sample jackknife. *The `assign_jackknife_patches` guard was too loose:
    `kmeans_radec.kmeans_sample` draws `max(2*sqrt(N), 10*num_jk)` points **without
    replacement**, so anything below ~10 randoms per patch died with a bare
    `ValueError: Sample larger than population` from the stdlib. The guard now requires
    `10 * num_jk <= len(randoms)` and says so. Tests: below the limit raises, exactly ten
    per patch works, `num_jk` may still exceed the number of data objects.*
  - [x] non-default `cosmology` argument (lightcone); `rp_cut` (box multipoles) already has
    a regression test from the P1 work;
  - [x] a multiprocessing test whose catalogue size does not divide evenly by chunk size
    (box and lightcone, `chunk_size=37`);
  - [x] synthetic-HDF5 tests for `read_subhalo`, `read_snapshot`, `read_snapshot_multiple`,
    `read_modelling_outputs` (`TestReadDataSimulationFiles`);
  - [x] tightened the `rel=0.5` tolerance in `test_rr_gg_consistent_with_formula` to
    `rel=1e-10` (the per-bin analytic values telescope exactly).

## P4 — Cleanup & docs

- [x] **Dead code**: *Done. The `*_old` lightcone methods and the commented auto-correlation
  branches were removed during the kernel-consolidation refactor (`*_old`) and P0 triage (`auto`);
  the commented variance lines in `_measure_w_g_i` are now removed too.*
- [x] **Docstring fixes**: *Done: "chunck_size" typo fixed; SimInfo simulation lists synced across
  the class/__init__/get_specs docstrings and the get_file_info error (COLIBRE + TNG100_2 added,
  misleading FLAMINGO mass variants and the ",," typo removed); docstring added to
  read_modelling_outputs. The phantom `randoms_data` param was already removed from the lightcone
  backend methods by the refactor. (Remaining: audit undocumented responsivity/tree/temp_file_path
  params across the public methods.)*
- [x] **Outstanding TODOs in code** — triaged (user-approved 2026-07-18):
  *`min_patch=1`: won't-fix — 1-based patch indices now raise a clear `ValueError` telling the
  user to renumber (test added); `auto` corrtype: dead commented-out branches deleted from the
  Box backends, DD documented as cross-count-only (`get_random_pairs`' tested `auto` utility
  branch kept); `++` correlation: deferred post-JOSS, comment updated to say so.* Still open
  from this list: "deal with masks" in the lightcone dispatchers (`measure_IA_lightcone.py`,
  the `# ToDo: deal with masks` sites) — *now closed: both dispatcher mask blocks were
  replaced by `_sample_coordinates()` / `_field_mask()` during the P3 pass.*
- [x] **`check_paths`**: *Done: check_paths rejects a non-writable output folder
  (PermissionError); ReadData.read_cat raises a clear FileNotFoundError for a missing data file
  instead of a raw h5py OSError. Test added.*

## P5 — Next session (queued 2026-07-21)

- [ ] **Warnings policy.** Make a plan before touching `filterwarnings`. The suite emits
  ~1000 warnings per run, nearly all the deliberate `RuntimeWarning` from
  `measure_IA_base.py` about bins with zero empirical random-random pairs (plus
  `invalid value encountered in divide/sqrt` from the same NaN bins). Decide: which of
  these are genuine user-facing signals worth keeping, which should be raised once per run
  rather than per call, whether the numpy divide warnings should be suppressed at the
  division sites (`np.errstate`) now that the NaN is intentional and documented, and only
  then what `filterwarnings` belongs in `[tool.pytest.ini_options]` (e.g. `error` with
  targeted ignores, so new warnings cannot appear unnoticed).
- [ ] **Support newer numpy and Python.** `pyproject.toml` pins `numpy~=1.26.2`, which has
  wheels for cp310-cp312 only — that is why `.github/workflows/tests.yml` stops at 3.12
  while `requires-python` advertises `>=3.10`. Move to `numpy>=2.1` (check `astropy`,
  `scipy`, `pyccl`, `h5py`, `matplotlib`, `sympy` pins at the same time; several are `~=`
  pins that will need widening), fix any numpy-2 API breakage, re-run the validation
  reference tests to confirm the committed tolerances still hold, then add 3.13 (and 3.14?)
  to the CI matrix and state the supported range honestly in `requires-python` and the docs.
- [ ] **Documentation review.** Take a proper look at the current docs (mkdocs site + the
  docstrings) before the JOSS submission: what is missing, what is stale after the kernel
  refactor and the P0-P3 changes, and whether the public API is documented end to end
  (incl. the `responsivity`, `tree`, `temp_file_path` params flagged in the P4 docstring
  item above).
