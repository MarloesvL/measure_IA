# Kernel consolidation plan (phase 1 design)

Status: **approved design, not yet implemented**. Written 2026-07-18 after the P0/P1
robustness campaign (see `TASKS.md`). This document is the specification for implementing
sessions — it deliberately contains enough detail that the implementation does not require
re-deriving any of the analysis below. Read it fully before writing code.

## 1. Problem

The measurement code exists in ~45 near-identical method variants spread over eight backend
files. Every variant is a permutation of **one algorithm** along five axes:

| Axis | Values |
|---|---|
| Geometry | periodic Cartesian box / lightcone sky (RA, DEC, z → 3D via pyccl) |
| Binning | `(rp, pi)` / `(r, mu_r)` / box `(r, signed-pi(r))` (the `_measure_xi_r_pi_box_brute` oddity) |
| Backend | brute / tree (chunked KDTree) / multiprocessing (Pool over chunks) |
| Jackknife | off / on (per-realisation `*_jk` grids, union deletion) |
| Shape terms | full (S+D, SxD, DD) / DD-only (`_count_pairs_*`) |

The cost is not aesthetic: the `w`-mp batch-clamp bug (fixed 2026-07-18, commit `4fa61b5`)
existed because a fix landed in the count-pairs twin but not the shape twin. Every future fix
currently needs up to 16 landing sites.

**Goal:** one pair-accumulation kernel parameterized over the axes, with thin wrappers that
keep every existing method name, signature, and HDF5 output byte-layout, so the dispatchers
(`measure_IA.py`, `measure_IA_lightcone.py`), the test suite, and the committed validation
references in `validation/reference_outputs/` are untouched.

## 2. Method inventory (what gets consolidated)

Counting loops to be replaced by kernel calls:

- `measure_w_box.py`: `_measure_xi_rp_pi_box_{brute,tree,batch,multiprocessing}`,
  `_count_pairs_xi_rp_pi_box_{brute,tree,batch,multiprocessing}`
- `measure_m_box.py`: same 8 for `_xi_r_mur_`, plus `_measure_xi_r_pi_box_brute`
- `measure_w_box_jk.py` / `measure_m_box_jk.py`: `_measure_*_jk_{brute,tree,batch,multiprocessing}`
  and `_count_pairs_*_jk_{brute,tree,batch,multiprocessing}` each
- `measure_w_lightcone.py` / `measure_m_lightcone.py`: `_measure_*_lightcone_{brute,tree}` and
  `_count_pairs_*_lightcone_{brute,tree}` (plus two dead `_old` methods — delete, do not migrate)
- `measure_w_lightcone_jk.py` / `measure_m_lightcone_jk.py`:
  `_measure_*_lightcone_jk_{brute,tree,batch,multiprocessing}` and count twins

**Not** consolidated (unchanged): `get_random_pairs`, `get_random_pairs_r_mur`,
`_measure_w_g_i`, `_measure_multipoles`, `_obs_estimator`, `_get_jackknife_region_indices`,
`_merged_masks`, the dispatchers' public signatures, `MeasureJackknife`, I/O helpers.

## 3. Kernel design

New module `src/measureia/pair_kernel.py`, pure functions + two small dataclass-like
containers. No class inheritance games — the existing backend classes keep their methods as
thin wrappers calling into this module.

### 3.1 Sample preparation (per geometry)

```python
def prepare_box_samples(data, masks, Num_position, Num_shape, *, shapes: bool,
                        ellipticity: str) -> SampleSet
def prepare_lightcone_samples(data, masks, *, shapes: bool, cosmology, over_h,
                              responsivity_correction) -> SampleSet
```

`SampleSet` fields: `pos` (N,3 float64), `pos_shape` (M,3), `weight` (N,), `weight_shape`
(M,), and for `shapes=True`: box → `axis_direction` (M,2 normalized), `e` (M,); lightcone →
`e` (M,2 = e1,e2 pre-scaled by 1/(2R) if responsivity), `east`/`north` (N,3), `n_pos` (N,3).
Optional `jk_pos`/`jk_shape` (int patch indices). Box also carries `LOS_ind`, `not_LOS`.

Mask semantics must reproduce exactly the current rules: box uses
`masks.get("Position", ones)` etc. with weight defaulting to the coordinate mask; the box jk
variants index `masks["Position"]` directly (KeyError on partial dicts — preserve); lightcone
uses `masks["RA"]`-style direct indexing with weight defaulting to the coordinate mask.

### 3.2 Accumulation

```python
def accumulate(sample_set, binning, *, chunk_axis, chunk_size_outer=100,
               shapes: bool, jk: bool, num_jk: int | None,
               pool: PoolSpec | None) -> Grids
```

- `binning` is one of five small objects implementing
  `bin_pairs(sep_vectors, aux) -> (mask, ind_r, ind_2nd)` plus the query radius:
  - `BoxRpPi` (query radius `r_max`; wraps minimum-image; `LOS = sep[:, LOS_ind]`)
  - `BoxRMuR` (query radius `r_max`; also applies `rp_cut`)
  - `BoxRPi` (the per-r-bin signed-pi grid; brute only today — migrate last)
  - `SkyRpPi` (query radius `sqrt(r_max² + pi_max²)`; `n_LOS = (s1+s2)/|s1+s2|`)
  - `SkyRMuR` (query radius `r_max`)
  The clamping conventions differ and must be preserved per family: box clamps with
  `ind >= num_bins → -= 1`; lightcone clamps with `ind == shape → shape - 1`.
- `chunk_axis` preserves the existing float-summation order (§4): `"shape"` for box
  tree/mp, `"position"` for lightcone tree/mp.
- `jk=True` adds the union-deletion accumulation: increment `X_jk[patch_of_chunked(n)]` for
  every pair, and `X_jk[patch_of_other]` where the two patches differ (`pos_mask`/`shape_mask`
  logic — copy verbatim, it is validated against corr_pc).
- `pool` handles the mp path: SharedMemory blocks named `f"{key}_{ID_shm}"` with
  `ID_shm = np.random.randint(100000)` (POSIX names ≤ 31 chars on macOS — keep keys short:
  `jk_region_indices_*` not `jackknife_region_indices_*`), temp-HDF5 offload of `self.data`,
  `spawn` start method, and the reload-into-`finally` contract (commit `4e2e96c`): the temp
  file must be reloaded and removed in a `finally` guarded by `os.path.exists`.

`Grids`: `Splus_D`, `Scross_D`, `DD` (always), `DD_jk`, `Splus_D_jk` when `jk`. For
`shapes=False` only `DD`(+`DD_jk`).

Box S+ grids divide by `(2 R)` inline; lightcone pre-scales `e1,e2` instead and applies
responsivity later ("responsivity added later" in jk grids). These conventions are locked by
`TestResponsivityOption` — do not "unify" them; parameterize.

### 3.3 Reduction & writing

Keep the existing per-family reduction/writer code, extracted into
`write_w_box(...)`, `write_m_box(...)`, `write_w_box_jk(...)`, etc. — these produce the
exact dataset names (`_SplusD`, `_RR_g_plus`, `_sigmasq`, `_rp`/`_pi` vs `_r`/`_mu_r`,
realisation `_{i}` datasets, `RR_jk` with `volume_jk = L3 (n-1)/n`) that the estimator layer
and tests read. The analytic-RR loops and all `*_denom` zero-guards stay as they are today.
Simplest extraction: keep them inside the existing wrapper methods; only the counting loop
is replaced by a kernel call.

## 4. Float-summation-order rule (the critical constraint)

Bit-identity with the current code is required for the **tree** and **mp** paths per family
(tests use `assert_array_equal` in places, and the committed reference files were generated
from these paths). That fixes the kernel's iteration order:

- Box tree/mp: outer loop over **shape** chunks of 100, KDTree of the chunk queried against
  the full-position tree, inner loop `n` over the chunk, vectorized `np.add.at` per `n`.
- Lightcone tree/mp: outer loop over **position** chunks (mp: `chunk_size` slabs then inner
  100s), per-chunk `KDTree(s_pos_chunk).query_ball_tree(shape_tree, ...)`.

> **Note added 2026-08-25.** This section, and the mention of "the r-window setdiff" below,
> describe candidate selection as it was during the refactor: two KDTree queries (at `r_min`
> and `r_max`) with a per-galaxy `setdiff1d` between them. That inner query and the set
> difference were removed as redundant — `bin_pairs` already applies the same lower bound —
> which made pair counting 1.4x-1.7x faster with every output bit-identical across 45
> configurations. See `benchmarks/FINDINGS.md` F1. The iteration and float-summation order
> this section fixes is unchanged and still binding; only the *size* of each galaxy's
> candidate list changed, and the extra entries are masked out in the same order. Interesting
> in hindsight: the observation below that the r-window "changes which pairs enter the
> log10/binning arithmetic — NaN-safe but different masks" is exactly why the removal turned
> out to be safe.

The **brute** paths iterate in a different order (box brute loops positions, vectorizing
over all shapes). Decision for the implementer: implement brute as `tree` behavior minus the
KDTree radius pre-filter is NOT equivalent (the r-window setdiff changes which pairs enter
the log10/binning arithmetic — NaN-safe but different masks). Instead: **drop the separate
brute order** and implement `backend="brute"` as the tree path with a single all-sky radius
query replaced by a full cross-join per chunk. Equality vs legacy brute is then only
`allclose(rtol=1e-10)`, which is all the existing brute-vs-tree tests demand anyway. Flag any
test that needs loosening in the PR description; never loosen a validation-reference
tolerance.

## 5. A/B equality harness (build FIRST)

Before migrating anything, add `tests/test_kernel_equivalence.py`:

1. Rename nothing. For each migration step, the legacy method is copied to
   `_legacy_<name>` (temporary, private) and the wrapper switched to the kernel.
2. The harness runs both on identical fixture objects (reuse `conftest.py` fixtures:
   `IA_mock_TNG300_n1/n8`, `IA_mock_lc_n1/n8`, `lc_jk_patches`) writing to two dataset
   names, then compares **every dataset in the affected HDF5 groups** by iterating keys —
   not a hand-picked subset:
   ```python
   with h5py.File(obj.output_file_name) as f:
       grp_a, grp_b = f[path_a], f[path_b]
       assert set(map(strip_name, grp_a)) == set(map(strip_name, grp_b))
       for k in grp_a: np.testing.assert_array_equal(grp_a[k][:], grp_b[k][:])
   ```
   `assert_array_equal` for tree/mp comparisons, `assert_allclose(rtol=1e-10, atol=1e-13)`
   for brute (§4). Use `equal_nan=True` where jk grids can contain NaN (empty RR cells).
3. When a path's equivalence test has passed in CI/local full-suite runs, delete the
   `_legacy_` copy and its harness entry in the same commit that migrates the next path.

## 6. Migration order

Each step = one commit: kernel change + wrapper switch + equivalence green + full suite
green (`uv run python -m pytest tests/ -q`, expect 511+ passed, 0 xfail).

1. Harness + `pair_kernel.py` skeleton with `BoxRpPi` binning; migrate
   `_measure_xi_rp_pi_box_tree` only. (Proves the design.)
2. `_measure_xi_rp_pi_box_{batch,multiprocessing}` (same order guarantees; mp reuses the
   chunked function via Pool).
3. Box w brute + count twins (4 methods).
4. Box multipoles (`BoxRMuR`, `rp_cut`): 8 methods + `BoxRPi` (`_measure_xi_r_pi_box_brute`).
5. Box jk (w + multipoles, 16 methods) — the union-deletion accumulation and `R_jk`
   handling move into the kernel's `jk=True` path; RR_jk/reduction stays in the wrappers.
6. Lightcone non-jk (8 methods, `SkyRpPi`/`SkyRMuR`).
7. Lightcone jk (16 methods incl. mp).
8. Cleanup: delete `_old` methods, drop the harness scaffolding, update `docs/api`.

Steps 1–2 are the Fable-recommended starting point; steps 3–7 are mechanical against the
harness and suitable for cheaper models; step 8 is trivial.

## 7. Known hazards (read before each step)

- `measure_xi_jk_helper` (lightcone) **mutates `self.data`** to swap S/D/R sample
  combinations between kernel calls, and `_merged_masks` always injects aligned
  `weight`/`weight_shape_sample` masks. The kernel must read `self.data` at call time, not
  capture it.
- The mp batch functions are **bound methods pickled into spawned workers**: everything the
  batch needs must be on `self` before `Pool` (`shm_infos`, `ID_shm`, `chunk_size`,
  `num_jk`/`num_box`, `sub_box_len_*`, `shape_tree`/`pos_tree`, `Num_*_masked`). Keep this
  contract; a kernel-side closure will not survive `spawn`.
- Batch index clamp is against the **chunked** sample's masked length
  (`Num_position_masked` for lightcone, `Num_shape_masked` for box) — regression locked by
  `test_multiproc_vs_tree_jk`.
- `pi_bins`/`mu_r_bins` conventions: lightcone `(rp,pi)` uses signed pi in `[-pi_max, pi_max]`
  (`num_bins_pi` bins); `mu_r` sub-box length is `2.0/num_bins_pi`; box `BoxRPi` builds
  per-r-bin pi edges. Copy the arithmetic verbatim — it is validated against
  treecorr/corr_pc at the 1e-5 level.
- Temp-file names are part of the contract (`w_temp_data_*`, `multipoles_temp_data_*`,
  `w_{simname}_temp_data_*`, `w_gg_{...}`, `multipoles_gg_{...}`): keep them so concurrent
  w/multipole runs don't clobber each other.
- Zero-`sum(weight_shape)` responsivity guard (`R → 0.5`) and all RR `*_denom` guards are
  behavior, not noise — preserve.

## 8. Definition of done (whole refactor)

- All counting loops live in `pair_kernel.py`; each of the eight backend files is wrappers +
  reduction/writing only.
- Full suite green, 0 xfail, no validation-reference tolerance changed.
- `grep -rn "_old" src/measureia` returns nothing; the harness `_legacy_` copies are gone.
- Net line count of `src/measureia` reduced by roughly half.
