# Pre-release task list

Prioritized task list from the full package review (2026-07-17). P0 = confirmed bugs,
P1 = features, P2 = input validation, P3 = test suite, P4 = cleanup & docs.

## P0 — Confirmed bugs (fix before release)

- [ ] **OPEN (raised 2026-08-12): the two analytic `RR` functions disagree on whether the
  cross count carries `Num_position - 1`.** Not yet resolved — the investigation below found
  contradictory evidence, so this is an investigation task, not a ready fix.

  **The inconsistency.** Three of the four analytic-`RR` branches exclude a self-pair, one
  does not (`measure_IA_base.py`):

  | function | corrtype | normalisation |
  |---|---|---|
  | `get_random_pairs_r_mur` (r, mu_r) | `auto`  | `(N_position - 1)/2 * N_shape` |
  | `get_random_pairs_r_mur` (r, mu_r) | `cross` | `(N_position - 1) * N_shape` |
  | `get_random_pairs` (rp, pi)        | `auto`  | `(N_position - 1) * N_shape / 2` |
  | `get_random_pairs` (rp, pi)        | `cross` | `N_position * N_shape` ← odd one out |

  **Why it looks wrong.** In the standard IA setup the shape sample is drawn *from* the
  position sample, so a shape galaxy cannot pair with itself and `N_position - 1` partners
  are available. That is true of the package's own mock: `radial_alignment_box_mock` returns
  `Position` = centrals + satellites and `Position_shape_sample` = satellites, a strict
  subset (`mocks.py:88-92`). On that reasoning the `-1` belongs in all four branches.

  **Why it is not that simple.** Both branches are currently pinned by cross-code
  validations that fail if changed, on that same mock:

  - Adding `-1` to `get_random_pairs` cross → **6 failures**:
    `TestGetRandomPairs::test_formula_cross` (hard-codes the current formula, expected),
    `TestBoxAgainstHalotools::{test_w_g_plus_matches_halotools_up_to_2R, test_w_gg_matches_halotools}`,
    `TestBoxCovarianceBridge::test_box_covariance_regression`,
    `TestBoxJackknifeAgainstCorrPC::{test_full_sample_w_signal, test_delete_one_realisations}`.
  - Removing `-1` from `get_random_pairs_r_mur` cross → **4 different failures**:
    `TestBoxMultipolesAgainstCorrPC::{test_xi_grids_match_corrpc, test_multipoles_match_corrpc}`,
    `TestBoxJackknifeAgainstCorrPC::{test_full_sample_multipole_signal, test_delete_one_realisations}`.

  So each convention is validated *as written* against an external code, on the same
  catalogue. They cannot both be describing the same normalisation correctly. The likely
  explanation is that at least one comparison is matching how that reference code normalises
  its own counts rather than testing the normalisation — which would make that validation
  self-fulfilling on this point.

  **Where to look.** How the reference counts are set up in
  `validation/run_box_halotools.py` and `validation/run_box_multipoles_corrpc.py` (and the
  `corr_pc` jackknife bridge), specifically whether the reference is given
  `N_position * N_shape` or `(N_position - 1) * N_shape` pairs, and whether the shape sample
  is passed as a distinct catalogue or as a subset flag.

  **Impact — small but real.** The two differ by a uniform factor `N/(N-1)`: ~4.8e-4 on the
  mock (`N_position = 2100`), ~1e-5 to 1e-6 at simulation sample sizes. Note the corr_pc
  delete-one comparison runs at a 5e-4 tolerance, so on the mock the effect sits *just* under
  the threshold — the current tests cannot distinguish the two conventions at that size. A
  discriminating test would need a deliberately small sample, where `1/N` is large enough to
  exceed the tolerance.

  **Already insulated.** `measure_galaxy_box._galaxy_rr_ratio` derives the jackknife `RR`
  amplitude ratio by calling these functions twice and dividing, rather than restating the
  normalisation, so per-galaxy contributions stay consistent under either convention and
  will not need changing when this is settled. Any fix does need to update
  `TestGetRandomPairs::test_formula_cross`, which hard-codes the current formula.

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

- [x] **Warnings policy.** *Done (2026-08-10). The ~1000 warnings/run were two families:
  (a) the deliberate zero-empirical-RR `RuntimeWarning` from `_obs_estimator` — a genuine,
  informative user signal, kept as-is (fires once per full-sample dataset, the right
  granularity); (b) redundant numpy `invalid value`/`divide by zero` noise from the same
  intentional NaN/inf bins. Family (b) is now silenced at its source with `np.errstate` so
  end users (not just the test suite) stop seeing it: the estimator divisions in
  `_obs_estimator`, the `w` pi-integral and the multipole integral in `measure_IA_base.py`,
  the jackknife `sqrt` in `measure_jackknife.py`, and the `mu_r` division in
  `pair_kernel.SkyRMuR.bin_pairs` (matching its already-guarded siblings). Pytest policy:
  `filterwarnings = ["error", ...]` in `pyproject.toml` — every warning is now an error so a
  new numpy/scipy deprecation on any Python in the 3.10-3.14 matrix fails CI instead of
  hiding — with three targeted `ignore`s for the package's own deliberate warnings (the
  zero-RR `RuntimeWarning` and the two `check_input` UserWarnings, all asserted in tests via
  `pytest.warns`). Suite is now warning-clean: 565 passed / 0 warnings, verified in strict
  mode on all of 3.10/3.11/3.12/3.13/3.14.*
- [x] **Support newer numpy and Python.** *Done (2026-08-10). All `~=` pins widened to `>=`
  lower bounds (library best practice; `uv.lock` still pins exact versions): `numpy>=2.1`,
  `scipy>=1.15`, `h5py>=3.11`, `matplotlib>=3.9`, `sympy>=1.12`. Two dependency subtleties
  required more than a version bump:
  - **astropy split by Python:** 6.1.7 (last release supporting 3.10) calls the removed
    `np.in1d` and only works on numpy where `in1d` survives (≤2.2, which is what 3.10 resolves
    to); astropy 7+ dropped 3.10. Pinned `astropy>=6.1; python_version<'3.11'` and
    `astropy>=7.0; python_version>='3.11'` (resolves to 8.x, handles numpy≥2.3).
  - **pyccl bumped to `>=3.3.4`:** 3.2.x is sdist-only (compiled from source on every Python)
    and its SWIG bindings **segfault at import on Python 3.14**. 3.3.4 is the first release with
    cp310–cp314 binary wheels — fixes 3.14 and removes source builds everywhere.
  One source-code break: **scipy removed `scipy.special.lpmn` in 1.17.** Replaced with the
  vectorised `assoc_legendre_p(sab, l, mu_r)[0]` in `measure_IA_base._measure_multipoles` (and
  the `legendre_multipole` cross-check in `validation/run_box_multipoles_corrpc.py`); output is
  bit-identical (verified, and the Condon–Shortley phase is +1 for the m=0,2 used here). No other
  numpy-2/scipy API breakage in the source.
  CI matrix now `["3.10","3.11","3.12","3.13","3.14"]` (3.15 excluded — not a stable release).
  Full suite (565 tests) verified green locally on all five versions with the final lock;
  committed validation-reference tolerances hold under numpy 2 + pyccl 3.3.6. `requires-python`
  stays `>=3.10` (honest). Remaining: mention the supported range in the docs (folds into the
  docs-review item below).*
- [x] **Documentation review.** *Done (2026-08-10). mkdocs (readthedocs theme) + mkdocstrings
  pulling numpy-style docstrings from src/; docs are fully merged into dev (127 commits ahead of
  origin/docs), so the work is on dev. Fixes:
  - **Docstrings / griffe:** the site now builds with zero griffe warnings (was 3). Documented
    `responsivity` on all four public measure methods; removed the phantom `randoms_data` param
    from both lightcone docstrings; documented `tree` and `temp_file_path` on both lightcone
    methods; removed the empty Returns section in `read_data.read_MeasureIA_output`.
  - **Narrative staleness:** `installation.md` now states Python 3.10-3.14 + NumPy 2; `roadmap.md`
    moves completed items (lightcone methods/validation, lightcone multiprocessing, responsivity)
    out of "planned"; `index.md` drops the placeholder tone and replaces the broken absolute-link
    Pages list (which also omitted Input + Estimator definitions) with relative links; "adn" typo
    fixed in `estimator_definitions.md`.
  - **Coverage:** `input.md` gained a lightcone-input section (RA/DEC/Redshift + e1/e2 + randoms)
    and a custom-key-names section; `usage.md` reorganised into Box/Lightcone with a lightcone
    example.
  Verified: `mkdocs build --strict` exits 0 (clean, only the informational note that the internal
  docs/REFACTOR_PLAN.md is not in the nav); suite still 565 passed. Leftover (optional): docs/
  REFACTOR_PLAN.md is an internal planning doc lingering in the tree (not published) — could be
  moved out of docs/ later.*
  - [x] **Full site restructure** *(2026-08-10, follow-up pass). Made box and lightcone peers throughout
    and added the missing conceptual material, treecorr-inspired. Corrected the responsivity definition in
    `estimator_definitions.md` to match the code (R = 1 - <e^2>/2 = <w(1-e^2/2)>/<w>, dividing S+D by 2R;
    was wrongly R = 1 - <e^2>). New `conventions.md` page (separation vector, phi, e+/ex, ellipticity defs,
    responsivity, and the lightcone e1/e2 survey shear-catalogue convention incl. chirality, radial-positive
    w_g+ / e+ = -gamma_t, and the treecorr g -> -g relation). New `getting_started.md` (box-vs-lightcone
    orientation + minimal example of each). De-boxed `input.md` (Box input / Lightcone input as peer
    sections), `estimator_definitions.md` (added a Lightcone estimators section: explicit randoms, clusters
    vs galaxies g+ forms, parity null test; lightcone kmeans jackknife paragraph), `output_structure.md`
    (box vs lightcone note: Snapshot group box-only, analytic vs empirical RR). Nav regrouped into
    Guides / Concepts / API, with the four lightcone backend classes added to the API reference (the two
    jackknife ones referenced by module path since they are not re-exported from the package). Verified:
    mkdocs build --strict exits 0, all lightcone API pages render class docs.*
  - [x] **Lightcone JK backends exported + refactor plan moved** *(2026-08-10). Added
    MeasureWLightconeJackknife / MeasureMultipolesLightconeJackknife to `src/measureia/__init__.py`
    (mirroring the box JK exports); their API pages now use the clean `measureia.X` path. Moved
    `docs/REFACTOR_PLAN.md` → new `plans/` folder (git mv), so the strict build is fully clean with no
    "not in nav" note. Suite 565 passed.*
  - [x] **Validation page + binning conventions** *(2026-08-10). New `docs/validation.md` distilled from
    `validation/README.md`: the cross-package validation approach, a results table (halotools / treecorr /
    corr_pc, w + multipoles + jackknife covariance, with agreements), and how to run it yourself
    (`pip install measureia[validation]`, `pytest tests/test_validation_references.py`, the run_*.py scripts;
    corr_pc build recipe left in the repo README, linked). Added a **Binning** section to `conventions.md`
    (log r/r_p over separation_limits, linear signed pi over ±pi_max, linear mu_r over [-1,1], midpoint bin
    coordinates, units). Nav gained a top-level Validation entry. Verified mkdocs build --strict clean.*

## P6 — Performance (opened 2026-08-21)

New `benchmarks/` directory: cross-package speed comparison against halotools (box)
and treecorr (lightcone), plus measureia-only profiling. Methodology in
`benchmarks/README.md`; **all measured results and conclusions in
`benchmarks/FINDINGS.md`**, written to be picked up cold in a later session.

- [x] **DONE (2026-08-25). The annulus set-difference costs 20–30% of a box measurement
  to discard 0.6–5% of the candidates** — the single clearest optimisation target found.
  `pair_kernel.accumulate` makes two `query_ball_tree` calls (at `r_min` and
  `r_max`) and `setdiff2D`s them (`pair_kernel.py:872-875` box,
  `:616-619` lightcone). `setdiff2D` calls `np.setdiff1d` per galaxy with the
  default `assume_unique=False`, which sorts *both* full candidate lists — two
  sorts per galaxy, 76,802 sort calls for 38,400 galaxies. Profiled at 38,400
  shape galaxies: `setdiff1d` cumulative **1.524 s of 4.953 s (30.8%)** for
  `measure_xi_w` and **0.982 s of 4.838 s (20.3%)** for `measure_xi_multipoles`,
  while `np.add.at` — the actual accumulation — is 2%.

  It looks redundant: `r_bins[0] == r_min` exactly (`measure_IA_base.py:168,172`)
  and every `bin_pairs` already masks `separation_len >= r_bins[0]` on the same
  metric its tree is built on, so the mask removes a superset of what the
  set-difference removes. `r_min = 0` is **not** reachable
  (`measure_IA_base.py:138` enforces `0 < r_min < r_max`), so the inner query is
  not removing self-pairs either.

  Two remedies, measured with `benchmarks/micro_candidate_selection.py`. They are
  **alternatives, not cumulative**. Memory cost of both is negligible (peak
  allocation per chunk never above 0.92 MB; B allocates less than today).

  - **(A) `np.setdiff1d(..., assume_unique=True)`** — saves ~13% of
    `measure_xi_w` and ~4% of `measure_xi_multipoles`. **Measured
    output-identical**: zero of 38,400 candidate lists differed, in both
    binnings. Risk is that it rests on *undocumented* scipy behaviour
    (`query_ball_tree` promises no ordering); assert the precondition or
    `np.sort` explicitly. Failure mode if duplicates ever appeared is silently
    double-counted pairs.
  - **(B) drop the inner query and the set-difference**, letting the existing
    mask do the work — saves **~31% of both**. The inner query alone costs
    0.39 s / 0.74 s (31% / 58% of the outer query), which is why B wins. Extra
    rows through `bin_pairs` are only +0.01 s / +0.08 s. Changes behaviour at
    exactly `d == r_min` (currently dropped, would be kept) and must be run
    against the bit-identity tests — the superset-in-same-order argument says it
    should be identical, but that is reasoning, not a test result.

  Both are constrained by `accumulate`'s frozen iteration and float-summation
  order (`plans/REFACTOR_PLAN.md` §4). Full write-up, including what is *not*
  established and one corrected earlier estimate, in `benchmarks/FINDINGS.md` §F1.

  *Applied remedy (B) on 2026-08-25: one `query_ball_tree` at the outer radius in
  both the box and lightcone branches of `pair_kernel.accumulate`; the inner
  `r_min` query and `setdiff2D` are gone, and `MeasureIABase.setdiff2D` /
  `setdiff_omit` were removed with them (no remaining caller). Result: **1.42x–1.72x
  faster** (up to 1.97x on the benchmark config at 100k), peak memory flat or 12–19%
  lower, and **2,697 arrays across 45 configurations bit-identical** — including
  `rp_cut`, masks, `measure_galaxy_contributions`, the lightcone `clusters`
  estimator, and `separation_limits=[5,20]` where the removed query used to discard
  up to 32.5% of candidates. Suite green (589 after the 8 setdiff tests went with
  the methods); halotools and TreeCorr validations reproduce exactly. One
  behavioural change, in the changelog: a pair at exactly `d == r_min` was dropped
  before and is counted now. A first reading suggested the gain shrank at large
  `r_min`; that was a `tracemalloc` artifact — re-measured untraced, it is larger.*

- [ ] **Not yet reproduced: the large `measure_xi_w` vs `measure_xi_multipoles`
  timing gap** reported on a run with jackknife + multiprocessing enabled. On
  the full-sample single-threaded box path the two are within ~10% at every size
  measured (see `FINDINGS.md` §F2, which also confirms and then *rules out* the
  2D-projected-tree candidate-count asymmetry as the cause). A sweep crossing
  `num_jk` with `num_nodes` is implemented (`run_sweep.py --sweeps jackknife`).

- [ ] **Lightcone `num_nodes` is silently ignored on the full-sample path.** Measured
  speedup at 2/4/8 nodes for `MeasureIALightcone.measure_xi_w` with `num_jk=0`:
  exactly 1.00x. `measure_IA_lightcone.py:662-676` dispatches that case only to
  `_count_pairs_xi_rp_pi_lightcone_tree` / `..._brute`, and none of those four
  methods uses `num_nodes` in its body — the `_multiprocessing` variant is reachable
  only from the jackknife branch (`:629`). A user setting `num_nodes=8` gets
  single-process execution with no warning. Either wire the mp path in or raise/warn
  when the request cannot be honoured. `measure_xi_multipoles` (`:818`) has the same
  dispatch shape and was not audited. See `benchmarks/FINDINGS.md` F4.

- [ ] **Multiprocessing is a net loss below ~40k galaxies, and always on the lightcone
  jackknife path.** Box at 8 nodes: 0.51x at N=9,600 (i.e. 2x *slower* than one node),
  2.06x at 38,400 — a best-case parallel efficiency of 26%. Lightcone `w` + jackknife
  at 8 nodes: 0.19x at 9,600 and 0.85x at 38,400, i.e. slower than single-process at
  both sizes despite having four pair-count runs and 5x randoms to distribute. Needs
  profiling of the spawn/`SharedMemory` setup cost, a finer measurement of the box
  crossover, and user-facing guidance in the docs. See `benchmarks/FINDINGS.md` F4.

- [x] **DONE (2026-08-25). The superlinear scaling was the KDTree
  query, caused by spatially incoherent chunks.** `accumulate` chunks the shape sample
  by array order and queries each chunk's tree against the full position tree; nothing
  makes those chunks spatially compact, so the dual-tree traversal cannot prune. Measured
  chunk extent at N=100,000 is 619 Mpc inside a 711 Mpc box — essentially the whole
  volume. Per-stage slopes show `tree_query` at **1.52 and 59% of runtime** while every
  other stage is sublinear (~0.85). Spatially sorting the sample first: **slope 1.63 →
  1.01, query 18x faster at 100k, identical candidate counts.** Estimated ~2.2x
  end-to-end on top of F1, growing with N. Blocked on the same bit-identity constraint
  as F1, and unlike F1 the summation order genuinely changes (~1e-14). Note the
  per-galaxy outputs are indexed by shape-sample position, so a reorder must be undone
  before returning or per-galaxy results are silently permuted — that is the risky part.
  *Fixed by visiting the chunked sample in Morton spatial order
  (`pair_kernel.spatial_order`), in both `accumulate` and `_accumulate_lightcone`.
  Scaling exponent now **1.00** (box multipoles), **1.02** (lightcone multipoles) and
  **1.03** (lightcone w), against an ideal of 1.0. End-to-end vs 0.4.0 at 100k galaxies:
  **4.3x** box multipoles, **6.2x** lightcone multipoles, **2.6x** box w. The per-galaxy
  hazard was handled by writing dense rows at the caller's galaxy id and applying an
  inverse permutation to the sparse arrays, guarded by two order-sensitive tests written
  before the change (the pre-existing tests sum over the galaxy axis and are
  permutation-invariant, so they could not have caught it). 593 tests pass. Four
  brute-vs-tree assertions loosened from exact to allclose after verifying the floats
  differ by ~1e-14 while integer pair counts stay exactly equal.*

  Full write-up in `benchmarks/FINDINGS.md` F5.

- [ ] **Box `w` still has a scaling exponent of 1.12** — the only path that does. Its
  KDTree is built on the 2D projection (`BoxRpPi.tree_coords`), so `query_ball_tree`
  returns a cylinder through the full box depth and candidates per galaxy genuinely grow
  as `L ∝ N^(1/3)` (measured 88.8 → 134.4 → 210.4 at fixed density). That is real pair
  work, so spatial ordering cannot remove it. The lightcone equivalent `SkyRpPi` already
  queries a 3D ball of radius `sqrt(r_max^2 + pi_max^2)` and is linear (1.03), which is
  the evidence for this diagnosis. Making `BoxRpPi` do the same is the remaining lead;
  the surviving pair set would be unchanged (the mask already applies the rp and pi
  windows), but the candidate list and hence summation order would change. See
  `benchmarks/FINDINGS.md` F2 and F5.

  *(superseded description: slopes 1.25-1.46
  where the pair count is linear and halotools measures 1.00). Partly explained for
  `box_w` by the full-depth cylinder query (F2 predicts 4/3), but `box_multipoles` has
  flat candidates-per-galaxy and still shows 1.28. Cause unknown; it is why the
  cross-package ratios widen with N rather than holding flat, so it should be found
  before the paper quotes those ratios. Suggested start: diff
  `profile_measureia.py` stage breakdowns at N=9,600 and N=300,000. See
  `benchmarks/FINDINGS.md` F3.)*

- [ ] **Re-run the cross-package comparison at realistic number density.** Every
  halotools and treecorr ratio recorded so far was measured on the reference mock, at
  n = 3.1e-4 and ~11-19 candidates per galaxy. A real catalogue sits near n = 1e-2 with
  ~345, and measureia's fixed per-galaxy cost (~33.8 us, F6) amortises as that rises,
  so the ratios will move — plausibly in measureia's favour. **The numbers currently in
  `FINDINGS.md` F3 and in the CHANGELOG are therefore density-specific and must not go
  into the paper as-is.** `bench_lib.box_mock_for(..., density=)` now takes an absolute
  density; the sweep needs a density axis to match. Note the absolute runtimes grow
  ~20-30x with the pair count, so the size ceiling will need lowering.

- [x] **RESOLVED (2026-08-26): the lightcone memory ceiling is ~23 GB at 1e7 with 10x
  randoms, so 64 GB of RAM is sufficient and no cluster is needed.** Measured RSS at
  constant density: 216 MB / 722 MB / 2.34 GB at 100k / 300k / 1e6 shape galaxies with
  10x randoms. The overhead over the analytic array model is a stable **~2.9x** across a
  decade (2.71, 3.02, 2.94) rather than shrinking, i.e. it is proportional to the arrays
  (per-chunk transients and intermediates in prepare_lightcone_samples) rather than a
  fixed cost -- which makes the extrapolation to 1e7 (~23 GB) trustworthy. Memory
  optimisation is therefore a nice-to-have, not a blocker for the 1e7 target. Caveat:
  num_nodes>1 transiently doubles the derived arrays during shared-memory setup, so the
  parallel peak is higher (~30 GB at 1e7) and should be measured before a production run.
  Earlier estimates of 55-77 GB and then 13-48 GB were both wrong; superseded.

- [ ] **The multi-core cross-package comparison, which is the unfavourable one.** The
  ratios recorded so far are single-thread. All three codes parallelise but not
  comparably: halotools and treecorr use OpenMP threads with negligible startup, while
  measureia uses processes via 'spawn' with ~0.9 s pool startup and a best-measured
  parallel efficiency of 26% (2.06x from 8 workers), and is a net loss below ~40k
  galaxies. So single-thread numbers flatter measureia, and the multi-core comparison
  must be run rather than omitted. Roughly doubles the sweep (~40 min to 100k, ~2 h with
  300k). This is also the strongest argument for the forkserver work, since cluster runs
  are multi-core.

  (superseded item: settle the lightcone memory ceiling at 1e6-1e7 with 10x randoms.)
  <!-- The analytic
  array model gives ~1.3 GB at 1e6 and ~13.3 GB at 1e7, but measured RSS at 100k shape /
  5x randoms is 3.6x the model (156 MB against 44 MB) because of per-chunk transients,
  HDF5 buffers and RSS being a high-water mark over four sequential pair-count runs. If
  that ratio persists 1e7 needs ~48 GB; if it shrinks as arrays come to dominate, ~13 GB.
  **The spread is too wide to plan against** — one run at ~1e6 total points, where arrays
  dominate the overheads, would settle it. Needs a cluster node, not the laptop.

  Reducible items, measured or derived (F6):
  - **retained user input: 6.6 GB at 1e7, the largest single item.** `self.data` stays
    resident alongside the derived arrays on the single-process path; the multiprocessing
    path already offloads it to a temp HDF5 and clears it. Contained, no numerical effect.
  - float32 for derived positions: 3.4 GB, but **risky** — `s = s_shape - s_pos` cancels
    ~2500 Mpc down to ~1 Mpc, so float32 gives ~2.5e-4 Mpc absolute error on a 1 Mpc
    separation. Must be tested against the smallest bin edge before being considered.
  - computing the east/north/n_pos sky basis per chunk instead of storing it: 0.7 GB.
  - **not** the KDTree: measured at 8 bytes/point (it stores an index permutation, not a
    copy of the coordinates), so it is negligible at any of these scales. -->
