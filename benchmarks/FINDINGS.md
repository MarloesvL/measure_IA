# Benchmark findings

Measured results and the conclusions drawn from them. Written to be picked up
in a later session by someone with no memory of the one that produced it, so
each entry states what was measured, on what, what it means, and what it does
**not** yet establish.

Nothing in `src/measureia/` has been changed on the basis of anything here.

Environment for every number below unless stated otherwise: Apple M-series
laptop, 12 logical cores, macOS (Darwin 25.6.0), Python 3.11.13, numpy 2.4.6,
scipy 1.17.1, measureia 0.4.0 (branch `feat/per-galaxy-contributions`,
`c3ccf2b`), halotools 0.9.4, treecorr 5.1.3. Single thread
(`OMP_NUM_THREADS=1`, `num_nodes=1`) unless stated otherwise.

---

## F1 — The annulus set-difference cost 20–30% of a box measurement to discard 0.6–5% of the candidates  ✅ FIXED

**Status: APPLIED (2026-08-25), remedy (B).** `pair_kernel.accumulate` now makes
a single `query_ball_tree` at the outer radius in both the box and lightcone
branches; the inner `r_min` query and the `setdiff2D` it fed are gone.
`MeasureIABase.setdiff2D` and `setdiff_omit` were removed with it (they had no
remaining caller). Measured result: **1.42x - 1.72x faster**, peak memory flat
or lower, and every output bit-identical across 45 configurations. Details in
"What happened when it was applied" at the end of this section. The analysis
below is kept as written, because the reasoning is what justified the change --
with one conclusion corrected, flagged inline.

### What the code does

`pair_kernel.accumulate` selects each galaxy's candidate partners with **two**
KDTree queries and a set difference, in both the box branch
(`src/measureia/pair_kernel.py:872-875`, shown below) and the lightcone branch
(`src/measureia/pair_kernel.py:616-619`, same shape with `query_r_min` /
`query_r_max` and the trees the other way round):

```python
shape_tree = KDTree(binning.tree_coords(positions_shape_sample_i, not_LOS), boxsize=base.boxsize)
ind_min_i = shape_tree.query_ball_tree(pos_tree, binning.r_min)
ind_max_i = shape_tree.query_ball_tree(pos_tree, binning.r_max)
ind_rbin_i = base.setdiff2D(ind_max_i, ind_min_i)
```

`setdiff2D` (`src/measureia/measure_IA_base.py:352-373`) is a Python loop
calling `np.setdiff1d(a1[i], a2[i])` once per galaxy. With
`assume_unique=False` — the default, and what is used — numpy's `setdiff1d`
calls `unique()` on **both** arrays before the membership test. `unique()`
sorts. So every galaxy pays two sorts of its full candidate list.

The profile confirms the call counts exactly: 38,400 galaxies produced 38,400
`setdiff1d` calls and 76,802 `sort` calls — two per galaxy.

### What it costs

`cProfile`, box, 38,400 shape galaxies, `fixed_density` (L = 516.6 Mpc/h),
`separation_limits=[0.5, 20.0]`, `num_bins_r=10`, `num_bins_pi=20`:

| | `measure_xi_w` | `measure_xi_multipoles` |
|---|---:|---:|
| `accumulate` cumulative | 4.953 s | 4.838 s |
| `setdiff1d` **cumulative** | **1.524 s (30.8%)** | **0.982 s (20.3%)** |
| ↳ of which `unique` | 0.842 s | 0.324 s |
| ↳ of which `isin` | 0.608 s | 0.594 s |
| `query_ball_tree` (both radii) | 1.268 s | 2.043 s |
| `bin_pairs` cumulative | 0.731 s | 0.857 s |
| `np.add.at` | 0.110 s | 0.104 s |

Note `np.add.at` — the actual grid accumulation, and the thing one would guess
is the bottleneck — is **2%**. The set-difference costs ten to fifteen times
as much as the arithmetic it is feeding.

### What it is discarding

**Measured** (`benchmarks/micro_candidate_selection.py`, 38,400 shape galaxies,
L = 516.6 Mpc/h, `r` ∈ [0.5, 20]):

| | `BoxRpPi` (w) | `BoxRMuR` (multipoles) |
|---|---:|---:|
| candidates from the outer query | 8,126,861 (211.6/gal) | 752,108 (19.6/gal) |
| candidates from the inner query | 48,748 | 38,989 |
| **fraction discarded** | **0.60%** | **5.18%** |

⚠️ **Correction to an earlier draft of this note.** A first pass estimated the
discarded fraction geometrically — `(r_min/r_max)²` ≈ 6.3 × 10⁻⁴ and
`(r_min/r_max)³` ≈ 1.6 × 10⁻⁵ — and concluded "one candidate in 1,600 to
60,000". That is wrong for any clustered catalogue: the mock's satellites sit
in σ = 2 Mpc groups, so the r < 0.5 Mpc inner ball is full of clump-mates, not
empty as a uniform-density argument assumes. The measured fractions above are
1–3 orders of magnitude larger than the geometric estimate. The conclusion
(the set-difference is disproportionately expensive) survives; the specific
"one in 60,000" figure did not, and should not be requoted.

### Why it appears to be redundant

`r_bins[0] == r_min` exactly — `measure_IA_base.py:168,172` sets
`self.r_min = separation_limits[0]` and
`self.r_bins = np.logspace(np.log10(self.r_min), ...)`, so the first bin edge
*is* `r_min`.

And every binning class's `bin_pairs` **already applies that lower bound** to
the same quantity the corresponding tree is built on:

| binning | tree built on | inner query radius | mask in `bin_pairs` |
|---|---|---|---|
| `BoxRpPi` | 2D projection (`:342`) | `r_min`, projected | `:374` — `separation_len` is the **projected** length |
| `BoxRMuR` | full 3D (`:415`) | `r_min`, 3D | `:438` — `separation_len` is the **3D** length |
| `SkyRpPi` | full 3D | `query_r_min = r_min` (`:480`), 3D | `:494` — `separation_len` is projected; projected ≤ 3D, so anything the query cuts the mask cuts too |
| `SkyRMuR` | full 3D | `query_r_min = r_min` (`:530`), 3D | `:542` — 3D |

In every case the mask removes a superset of what the inner query + set
difference removes. The one exception is the **boundary**: `query_ball_tree`
returns neighbours at distance **≤ r**, so `setdiff` drops pairs at exactly
`d == r_min`, whereas the mask (`>= r_bins[0]`) **keeps** them. For continuous
coordinates this is a measure-zero case, but it is a real behavioural
difference and must be stated in any change.

### Two remedies — measured

`benchmarks/micro_candidate_selection.py` isolates the four lines of candidate
selection and times each part separately (no tracemalloc; `--repeats 3`).
Whole-measurement times for comparison: `measure_xi_w` 4.08 s,
`measure_xi_multipoles` 3.89 s at this size.

| step (one pass over the shape sample) | `BoxRpPi` | `BoxRMuR` |
|---|---:|---:|
| shape-tree build | 0.017 s | 0.015 s |
| query @ `r_min` (inner) | 0.390 s | 0.740 s |
| query @ `r_max` (outer) | 0.858 s | 1.271 s |
| `setdiff2D` as written | 1.041 s | 0.499 s |
| `setdiff2D` with `assume_unique=True` **[A]** | 0.522 s | 0.356 s |
| `np.asarray` only **[B]** | 0.168 s | 0.021 s |
| **selection total, current** | **2.307 s** | **2.525 s** |

Note the inner query is **not** free — 0.39 s / 0.74 s, i.e. 31% / 58% of the
outer query's cost despite returning 0.6% / 5.2% as many candidates. That
answers open question 2 from the earlier draft, and it is most of why (B) beats
(A) so decisively.

| | saves, selection | saves, whole measurement |
|---|---:|---:|
| **A** `assume_unique=True` — `BoxRpPi` | 0.519 s (22.5%) | ~**13%** |
| **A** — `BoxRMuR` | 0.143 s (5.7%) | ~**4%** |
| **B** drop inner query + setdiff — `BoxRpPi` | 1.264 s (54.8%) | ~**31%** |
| **B** — `BoxRMuR` | 1.218 s (48.2%) | ~**31%** |

**A and B are alternatives, not cumulative** — B removes the `setdiff1d` call
that A optimises.

#### Memory

Negligible for both. Peak allocation per chunk, traced separately
(`--memory`), never exceeded **0.92 MB** for any step in either binning, and
the two remedies allocate the same as or less than the current code:

- **A** allocates strictly less — it skips two `unique()` temporaries per galaxy.
- **B** drops the entire `ind_min` list (48,748 / 38,989 indices) and produces
  the same `ndarray` candidate list. Its only increase is downstream: the
  per-galaxy `separation` array grows by the discarded fraction, i.e. 0.60% /
  5.47%, on a transient of order 5 KB per galaxy.

Neither remedy changes the size of any persistent array (the grids, the
jackknife realisations, the per-galaxy decomposition).

#### Risk to the measurements

**A — `assume_unique=True`.** *Measured to be output-identical.* Across all
384 chunks × 38,400 galaxies, in both binnings, **zero** candidate lists
differed from the current output (checked elementwise with `np.array_equal`).
`query_ball_tree` was verified to return sorted, duplicate-free lists, and
numpy's `setdiff1d` preserves `ar1`'s order in the `assume_unique` branch, so
both branches agree.

The residual risk is that this rests on **undocumented scipy behaviour** —
`query_ball_tree` has no `return_sorted` parameter and makes no ordering
promise (unlike `query_ball_point`). Failure modes if a future scipy changed:

- *unsorted but still unique* → same pair set, different iteration order →
  bit-identity breaks, results change in the last few ulp. Detectable by the
  existing tests.
- *duplicated indices* → pairs counted twice → **silently wrong results**.
  This would be a scipy bug rather than a licence change, but it is the reason
  to assert the precondition or call `np.sort` explicitly rather than trust it.

**B — drop the inner query and the set difference.** Larger win, more surface.

- *Bit-identity:* the candidate list becomes a superset **in the same sorted
  order**; the extra entries are removed by `bin_pairs`' existing mask; so the
  surviving pairs reach `np.add.at` as the same values in the same order. That
  is an argument for bit-identity, and a strong one, but it has **not** been
  run against the bit-identity tests. Do that first.
- *One real behavioural change:* pairs at **exactly** `d == r_min`.
  `query_ball_tree` is inclusive (`≤ r`), so today they are dropped; the mask
  is `>= r_bins[0]`, so afterwards they would be kept. With float coordinates
  this is measure-zero, but it is not zero for gridded or quantised input, and
  it should be stated in the changelog rather than discovered.
- *The extra work is not free but is small:* 0.60% / 5.47% more rows flow
  through `separation`, the periodicity wrap and `bin_pairs`. Against those
  stages' measured cost (`accumulate` 0.950 s / 0.588 s plus `bin_pairs`
  0.731 s / 0.857 s) that is about **+0.01 s / +0.08 s**, versus 1.26 s / 1.22 s
  saved. It does not change the conclusion, but the multipole case is the one
  where it matters more, and on a much more strongly clustered catalogue than
  this mock the discarded fraction — and so this offset — would grow.
- *Not checked:* whether any other consumer of `ind_rbin_i` assumes the inner
  ball has been removed. The jackknife and per-galaxy branches index it the
  same way, but they were not audited line by line.

### What happened when it was applied

Remedy (B) was implemented on 2026-08-25: one `query_ball_tree` at `r_max`,
`np.asarray` in place of `setdiff2D`, in both `accumulate` (box) and
`_accumulate_lightcone`. Two sites, +17/-6 lines.

**Speed** (untraced, best of 3, single thread, `num_bins_r=10`,
`num_bins_pi=20`):

| task | limits | N | before | after | speedup |
|---|---|---:|---:|---:|---:|
| box multipoles | [0.5, 20] | 38,400 | 3.69 s | 2.51 s | 1.47x |
| box multipoles | [0.5, 20] | 100,000 | 14.64 s | 9.56 s | 1.53x |
| box w | [0.5, 20] | 38,400 | 3.68 s | 2.58 s | 1.42x |
| box w | [0.5, 20] | 100,000 | 14.19 s | 9.43 s | 1.50x |
| box multipoles | **[5, 20]** | 100,000 | 15.19 s | 9.35 s | **1.62x** |
| box w | **[5, 20]** | 100,000 | 14.92 s | 9.35 s | **1.60x** |

On the benchmark's own configuration (`num_bins_pi=1` for the halotools
comparison) the gain reaches **1.68x - 1.97x** at 100k, and it grows with N in
every series: the sorting the change removes scales worse than the loop it
sits in.

⚠️ **Correction.** An earlier reading of this, taken with `tracemalloc` active,
suggested the gain *shrank* at large `r_min` (where more candidates now survive
into `bin_pairs`). That was an artifact -- allocation tracing taxes the extra
candidates far more than real execution does. Re-measured without tracing, the
gain at `[5, 20]` is **larger**, not smaller: a query at `r = 5` is much more
expensive to execute than one at `r = 0.5`, so deleting it saves more than the
extra downstream work costs. Never quote timings taken under `tracemalloc`.

**Memory.** Flat or better, including at the stressing end. At `[5, 20]` the
change pushes 48.9% more candidates through `bin_pairs` for `BoxRMuR` and 10.6%
more for `BoxRpPi`, yet peak Python allocation is unchanged for the multipoles
and **12-19% lower** for `w`. The reason: the per-galaxy `separation` array is
a few KB, negligible beside the chunk-level structures, and the old code held
*three* of those at once (`ind_min_i`, `ind_max_i`, and the `setdiff` output).
Dropping two of them more than pays for the larger kept list. Peak RSS moves
within +-6 MB on a ~170 MB baseline, i.e. noise.

**Correctness.** Verified before the change was trusted:

- `bitidentity_matrix.py`, **45 configurations / 2,697 arrays compared
  bitwise: 0 differing.** Box and lightcone, w and multipoles, brute / tree /
  multiprocessing backends, jackknife on and off, `rp_cut`, non-contiguous
  masks, `measure_galaxy_contributions`, the lightcone `clusters` estimator,
  and `separation_limits=[5, 20]` where the removed query used to discard
  12.8% (`BoxRpPi`) to 32.5% (`BoxRMuR`) of all candidates. Every dataset in
  the output file is compared, not just the final w -- raw grids, jackknife
  realisations, bin coordinates. The brute backend, which bypasses the changed
  code entirely, served as a control.
- Full test suite green (589 after the 8 `setdiff` tests were removed with the
  methods; 597 before).
- The halotools and TreeCorr cross-validations reproduce their documented
  agreement exactly.

**The one behavioural change**, as predicted: a pair at *exactly*
`d == r_min` was previously dropped (`query_ball_tree` is inclusive, so it
landed in the inner list and the set difference removed it) and is now counted
(`bin_pairs` masks on `>= r_bins[0]`). Demonstrated on a constructed catalogue
with a pair at exactly `r_min`: 0 pairs before, 1 after. Measure-zero for float
coordinates -- which is why 2,697 arrays came out identical on real mocks --
but not zero for gridded or quantised input. Recorded in the changelog.

### The constraint that governs both

`pair_kernel.accumulate`'s iteration order and float summation order are
deliberately frozen for bit-identity against the legacy tree/mp paths — see
the `accumulate` docstring and `plans/REFACTOR_PLAN.md` section 4. Any change
here must either preserve them or re-derive the guarantee, and the repo has
tests that will say which.

### How to reproduce

```bash
uv pip install -e ".[validation]"
python benchmarks/profile_measureia.py --n-shape 38400 --paths box_w box_multipoles
python benchmarks/micro_candidate_selection.py --repeats 3            # the tables above
python benchmarks/micro_candidate_selection.py --memory               # peak allocation
```

---

## F2 — `measure_xi_w` and `measure_xi_multipoles` cost about the same on the full-sample box path

**Status:** measured. Refutes the hypothesis that motivated the internal
comparison; the mechanism behind that hypothesis is real but does not produce
the effect.

The two paths select candidates very differently:

| path | tree built on | query radius | candidate region |
|---|---|---|---|
| box `w` (`BoxRpPi`) | **2D projection** | `r_max` projected | a cylinder through the **entire box depth** |
| box multipoles (`BoxRMuR`) | full 3D | `r_max` | a 3D ball |

So box `w` hands `bin_pairs` every neighbour within `r_max` in projection at
*any* line-of-sight separation, and the π window throws most of them away. The
candidate counts confirm this quantitatively, and they grow with the box as
predicted by `3L / (4 r_max)`:

| N_shape | L (Mpc/h) | `w` cand/gal | multipoles cand/gal | ratio | uniform-density prediction |
|---:|---:|---:|---:|---:|---:|
| 2,400 | 205.0 | 88.8 | 18.6 | 4.78 | 7.69 |
| 9,600 | 325.4 | 134.4 | 18.3 | 7.33 | 12.20 |
| 38,400 | 516.6 | 210.4 | 18.6 | 11.33 | 19.37 |

(The prediction over-estimates because the mock is strongly clumped —
satellites sit in σ = 2 Mpc groups — so a fixed ~8 clump-mates per galaxy sit
inside both regions and dilute the ratio. Multipole cand/gal is flat at ~18.6,
exactly as a fixed-density 3D ball should be.)

**But the wall-clock ratio stays flat at ~1.05–1.10** across all three sizes
(0.150 s vs 0.139 s; 0.723 s vs 0.651 s; 4.08 s vs 3.84 s). The reason is
visible in F1's profile table: box `w` processes ~11× more candidates but its
**2D** tree query is much cheaper than the multipoles' **3D** query — 1.268 s
vs 2.043 s. The two effects very nearly cancel.

Lightcone, same comparison: candidate ratio 2.62 against a predicted 2.83
(`(r_max² + π_max²)^{3/2} / r_max³`), time ratio 1.22.

**Consequence:** "box `w` queries a full-depth cylinder" is a true and possibly
worth-fixing inefficiency, but it is **not** the explanation for any large
w-vs-multipoles timing gap, and optimising it alone would buy little. F1 is
the better target.

**The jackknife + multiprocessing case: also matched.** The original report was
of a large gap on a run with jackknife enabled and multiprocessing on, which the
numbers above do not exercise. Block B (2026-08-25) crossed `num_jk` ∈ {0, 27
(box) / 9 (lightcone)} with `num_nodes` ∈ {1, 2, 4, 8} at N = 9,600 and 38,400:

| N | jk | nodes | box `w`/mult | lightcone `w`/mult |
|---:|---:|---:|---:|---:|
| 9,600 | 0 | 1 | 1.07x | 1.24x |
| 9,600 | 27/9 | 8 | 1.01x | 0.98x |
| 38,400 | 0 | 1 | 1.04x | 1.18x |
| 38,400 | 27/9 | 8 | 1.03x | 0.94x |

Across all 32 combinations the ratio stays within **0.94x - 1.26x**. Neither the
jackknife nor the multiprocessing path opens a gap between the two estimators.

**So the reported gap is not reproduced by anything measured here**, and its
cause is still unknown. What has *not* been tried: a real simulation catalogue
(this mock's clustering is a specific, mild choice), N well above 38,400 with
jackknife on, and configurations where the two calls do not share binning. The
next step is to get the actual script or its parameters rather than widen the
grid further — the grid has now covered the plausible space and come back
negative twice.

---

## F3 — Cross-package speed and memory, single thread

**Status:** complete for the size sweep (block A, 59 records, 2026-08-25). Every
point passed its correctness gate — halotools at `rtol=1e-10` after the `2R`
responsivity factor, TreeCorr at `rtol=5e-3, atol=0.05` — so no timing below is
reported for a configuration that computes the wrong answer.

All numbers post-F1, single thread, fixed number density (the box grows as
`N^(1/3)`, so pairs per galaxy stay constant).

### Wall time

Final numbers, after F1, the multiprocessing pool fix and F5 (121-point sweep,
2026-08-26). Every point passed its correctness gate.

| N | box `w` vs halotools | lightcone `w` vs TreeCorr |
|---:|---:|---:|
| 2,400 | 12.9x slower | **0.6x — measureia faster** |
| 9,600 | 14.5x | **1.0x — parity** |
| 38,400 | 17.0x | 1.9x |
| 100,000 | 18.9x | 2.4x |
| 300,000 | 22.3x | **2.6x** |

For comparison, the same table before the optimisations: the box ratio ran
14.0x → 47.7x and the lightcone 0.7x → 18.8x. Both now flatten instead of
diverging, which is the visible consequence of fixing the scaling exponent.

The lightcone comparison starts in measureia's favour because obtaining the
same w_g+ and w_gg from TreeCorr takes 24 correlation runs plus user-side
estimator assembly, which the benchmark counts.

### Peak memory — where measureia wins

| N | measureia (lightcone) | TreeCorr | ratio |
|---:|---:|---:|---:|
| 9,600 | 176 MB | 295 MB | 1.7x |
| 38,400 | 209 MB | 692 MB | 3.3x |
| 100,000 | 293 MB | 1,357 MB | 4.6x |
| 300,000 | **527 MB** | **3,163 MB** | **6.0x** |

Both include a ~160 MB interpreter and import baseline, so the ratio in
*measurement* memory is larger still. TreeCorr builds trees for all four
catalogues -- data, shapes, and both randoms at 5x -- and holds them across the
24 runs. halotools is closer to measureia (307 MB vs 252 MB at 300,000 on the
box) but above it at every size.

Worth stating plainly in the paper: on the lightcone, measureia is within a
factor of ~2.6 on time at the largest size measured while using ~6x less
memory, and memory is what decides whether a catalogue runs at all.

### TreeCorr's `bin_slop` does not matter here

The benchmark records TreeCorr at `bin_slop=0` (accuracy-matched to measureia,
what the validation uses) and at its own default, because quoting only the
default would overstate its speed against an accuracy-matched measureia. On this
comparison the two are indistinguishable — 0.98x to 1.04x at every size, 1.00x at
300,000 — so the concern the design guarded against does not arise, and the
numbers above are already accuracy-matched.

⚠️ An earlier note here claimed a 2.1x penalty for `bin_slop=0` at N=38,400. That
was an error: it compared a `fixed_volume` point against a `fixed_density` one.
Like-for-like there is no penalty.

### The scaling defect — the finding that matters most

Slopes of `d(log t) / d(log N)` at fixed number density. The pair count is linear
in N in this regime, so **1.0 is the ideal**:

| code / task | slope |
|---|---:|
| halotools — box `w` | 1.00 |
| **measureia — box multipoles** | **1.00** |
| **measureia — lightcone `w`** | **1.02** |
| **measureia — lightcone multipoles** | **1.02** |
| measureia — box `w` | 1.12 |
| TreeCorr — lightcone `w` | 0.73 |

**Resolved.** When first measured the measureia exponents were 1.25-1.46; F5
traced that to the KDTree query and fixed it. Three of the four paths are now
at the ideal 1.0 within measurement scatter. Box `w` remains at 1.12 for a
understood reason -- see F5 -- and is the one outstanding lead.

The original diagnosis, kept because it is what motivated F5:

**measureia is superlinear where it should be linear.** This is why the ratios in
the tables above widen with N instead of holding flat: the gap at 300,000 is not
a constant-factor Python penalty, it is a scaling defect on top of one. A
constant factor and a bad slope are different problems and want different fixes,
and only the slope gets worse as catalogues grow.

Partial explanation, for `box_w` only: its KDTree is built on the 2D projection,
so the query is a cylinder through the full box depth and candidates per galaxy
grow as `L ∝ N^(1/3)` (see F2). That predicts slope 4/3 ≈ 1.33, close to the
observed 1.25.

**But it does not explain the rest.** `box_multipoles` has *flat* candidates per
galaxy at fixed density (measured: 18.6, 18.3, 18.6 across three decades) and
still shows slope 1.28. The lightcone is worse at 1.42-1.46. Tree-build cost
(`N log N`) contributes only about 0.10 to a slope over this range, nowhere near
enough. **The cause is not known and should be found before the paper quotes any
of these ratios** — "47x slower at 300k" and "constant-factor slower plus a
fixable scaling bug" are very different claims, and only one of them is
actionable.

Suggested next step: profile `box_multipoles` at 9,600 and 300,000 and diff the
stage breakdown (`profile_measureia.py` already groups by stage). Whatever grows
its share between those two points is the culprit.

Context for the box ratio: `src/measureia/` is pure Python/NumPy — no Cython, no
C extension, no numba — while halotools' `ia_correlations` is Cython + OpenMP and
TreeCorr is C++ + OpenMP.


---

## F4 — Multiprocessing is often a net loss, and on one lightcone path it does nothing at all

**Status:** measured (block B, 2026-08-25), **not fixed**. Two separate
problems; the first is a documentation/API issue, the second a performance one.

### `num_nodes` is silently ignored on the full-sample lightcone path

Measured speedup of `MeasureIALightcone.measure_xi_w` with `num_jk=0`, relative
to `num_nodes=1`:

| N | 2 nodes | 4 nodes | 8 nodes |
|---:|---:|---:|---:|
| 9,600 | 0.99x | 1.01x | 1.00x |
| 38,400 | 1.00x | 1.00x | 1.00x |

Exactly no effect. The code explains it: `measure_IA_lightcone.py:662-676`
dispatches the `num_jk == 0` case to `_count_pairs_xi_rp_pi_lightcone_tree` or
`..._brute` only — there is no multiprocessing branch. `num_nodes` is passed
into those methods but **none of the four uses it in its body** (verified by
scanning each function body: zero references). The `_multiprocessing` variant
exists but is reachable only from the jackknife branch at line 629, which *does*
switch on `self.num_nodes == 1`.

So a user who sets `num_nodes=8` for a full-sample lightcone measurement gets
single-process execution, no speedup, and no warning. Either wire the
multiprocessing path in, or raise/warn when `num_nodes > 1` cannot be honoured.
The same dispatch shape should be checked for `measure_xi_multipoles`
(`measure_IA_lightcone.py:818`), which was not audited.

### Where it does engage, it frequently loses

Speedup relative to `num_nodes=1`:

| path | N = 9,600 | N = 38,400 |
|---|---:|---:|
| box `w`, 8 nodes | **0.51x** | 2.06x |
| box `w` + jk27, 8 nodes | **0.58x** | 1.88x |
| box multipoles + jk27, 8 nodes | **0.55x** | 1.85x |
| lightcone `w` + jk9, 8 nodes | **0.19x** | **0.85x** |

- **Below roughly N = 38,000 the box is about twice as slow** with 2-8 workers as
  with one. `spawn` start-up plus the `SharedMemory` copies cost more than the
  pair work saved. The crossover sits between 9,600 and 38,400 and should be
  measured more finely before being documented as user guidance.
- **The lightcone jackknife path is slower than single-process at both sizes
  tested** — 5x slower at 9,600, and still 15% slower at 38,400, where it should
  be winning. It has four pair-count runs (DD, SR, RD, RR) and 5x randoms, so it
  has more work to distribute than the box, not less. Worth profiling.
- Best parallel efficiency observed anywhere is **26%** (2.06x from 8 workers).

Practical guidance for users, pending a fix: `num_nodes > 1` is worth setting on
the **box** above a few tens of thousands of galaxies, and currently not worth
setting on the lightcone at all.


---

## F5 — The superlinear scaling is the KDTree query, and its cause is chunk spatial incoherence

**Status: APPLIED (2026-08-25).** `pair_kernel.spatial_order` computes a Morton
key on `tree_coords`, and both `accumulate` and `_accumulate_lightcone` now visit
their chunked sample in that order. Measured outcome: **total scaling exponent
1.12 -> 1.00** for `box_multipoles`, `tree_query` share 59% -> 9%, and end-to-end
speed-ups of 2.6x - 6.2x at 100,000 galaxies cumulative with F1. See "What
happened when it was applied" at the end of this section. The analysis below is
kept as written, because the reasoning is what justified the change.

### Where the superlinearity lives

Per-stage slopes for `box_multipoles` at fixed number density, N = 9,600 →
100,000 (`profile_measureia.py --scaling`). This is the cleanest test case: its
candidates per galaxy are *flat* at fixed density (18.6, 18.3, 18.6), so the
pair work really is linear and any excess slope is overhead.

| stage | 9,600 | 38,400 | 100,000 | slope | share at 100k |
|---|---:|---:|---:|---:|---:|
| **tree_query** | 0.175 s | 1.250 s | **6.244 s** | **1.52** | **59%** |
| accumulate | 0.196 s | 0.568 s | 1.475 s | 0.86 | 14% |
| binning | 0.165 s | 0.439 s | 1.142 s | 0.83 | 11% |
| other | 0.161 s | 0.470 s | 1.223 s | 0.87 | 11% |
| shapes | 0.044 s | 0.125 s | 0.325 s | 0.85 | 3% |
| add_at | 0.033 s | 0.098 s | 0.252 s | 0.87 | 2% |
| io | 0.003 s | 0.003 s | 0.003 s | — | 0% |
| **TOTAL** | 0.777 s | 2.952 s | 10.662 s | **1.12** | |

Every stage except the tree query is *sublinear* (~0.85, per-call overheads
amortising). The KDTree work is the whole defect. This kills the hypotheses
that were on the table — cache locality of the gather, Python loop overhead,
`np.add.at` — none of which could produce this even in principle at 2-14% share.

### Why the tree query is superlinear

`accumulate` chunks the shape sample by **array order**, 100 at a time, builds a
`KDTree` of each chunk and queries it against the full position tree. Whether
that dual-tree traversal can prune depends entirely on how compact the chunk is
in space, and nothing orders the catalogue to make it so.

On the mock, satellites are stored grouped by their central and the centrals are
placed at random, so 100 consecutive satellites come from ~12 centrals scattered
across the entire box. Measured chunk extent vs box size:

| N | chunking | mean chunk extent | box L | query time | candidates |
|---:|---|---:|---:|---:|---:|
| 9,600 | array order (current) | 290 | 325 | 0.137 s | 185,723 |
| 9,600 | spatially sorted | 144 | 325 | **0.032 s** | 185,723 |
| 38,400 | array order | 452 | 517 | 1.234 s | 752,108 |
| 38,400 | spatially sorted | 190 | 517 | **0.129 s** | 752,108 |
| 100,000 | array order | 619 | 711 | 6.196 s | 1,933,419 |
| 100,000 | spatially sorted | 243 | 711 | **0.344 s** | 1,933,419 |

**Slope 1.63 → 1.01. At 100,000 galaxies the query is 18x faster, and the
candidate counts are identical** (1,933,419 either way) — it is the same work,
reorganised so the traversal can prune. The chunk's bounding box currently
covers essentially the whole volume, which is the worst possible input to a
dual-tree query.

The sort used in the experiment was deliberately crude — `lexsort` on a coarse
20 Mpc grid cell. A proper Morton or Hilbert key would do at least as well.

### Why this matters beyond the benchmark

- **Performance currently depends on how the user's file happens to be sorted.**
  A catalogue already in spatial order (some simulations store Peano-Hilbert
  ordered) gets the good behaviour; one ordered by halo, by ID, or shuffled gets
  the bad one. Identical science, wildly different runtime, with nothing in the
  documentation to explain it.
- **It plausibly explains the lightcone's worse slopes** (1.42-1.46 against the
  box's 1.25-1.28): `_accumulate_lightcone` chunks the *position* sample the same
  way, and the lightcone additionally has 5x randoms.
- Estimated end-to-end effect at N = 100,000 for `box_multipoles`: total 10.66 s →
  ~4.8 s, i.e. **~2.2x on top of F1**, growing with N.

### What a fix has to deal with

Reordering the sample changes the iteration order and therefore the float
summation order, so it **breaks bit-identity** (`plans/REFACTOR_PLAN.md` §4) —
the same constraint F1 had to satisfy, but unlike F1 this one cannot be argued
around: the pairs really are summed in a different sequence. Expect differences
at the ~1e-14 level, exactly like the multiprocessing path already produces
against the tree backend.

Open questions, none answered yet:

1. **Sort the sample, or sort only within chunks?** Sorting the whole sample once
   is simplest and makes chunks compact. Sorting *within* a chunk does not help —
   the chunk's extent is what matters.
2. **Where to apply it** — inside `accumulate`, or in `prepare_box_samples` /
   `prepare_lightcone_samples`? The latter is tidier but affects more callers.
3. **The per-galaxy outputs are indexed by shape-sample position**
   (`measure_galaxy_contributions` returns arrays whose galaxy axis matches
   `sample_set.pos_shape`). A reorder must either be undone before returning or
   be documented, or per-galaxy results would silently be permuted. **This is the
   dangerous part of the change** and needs a test.
4. Does the box `w` path benefit as much? Its tree is 2D, so pruning behaves
   differently; F5 was measured on `BoxRMuR` (3D).

### What happened when it was applied

Implemented 2026-08-25. `spatial_order` builds a Morton (Z-order) key on a
1024-per-axis grid over `binning.tree_coords(...)` — the same metric the KDTree
uses, so it works for the 2D-projected box `w` binning as well as the 3D ones.
The brute backend deliberately keeps array order: it has no tree to prune, and
leaving it alone preserves it as a control in the bit-identity matrix.

**Per-stage effect** (`box_multipoles`, fixed density, 9,600 → 100,000):

| | before | after |
|---|---:|---:|
| `tree_query` slope | 1.52 | **1.02** |
| `tree_query` share at 100k | 59% | **9%** |
| `tree_query` time at 100k | 6.244 s | **0.419 s** |
| **total slope** | **1.12** | **1.00** |

**End-to-end, cumulative with F1**, against 0.4.0:

| task | 9,600 | 38,400 | 100,000 | slope |
|---|---:|---:|---:|---|
| box multipoles | 1.84x | 2.68x | **4.28x** | 1.26 → **1.00** |
| box `w` | 1.67x | 2.00x | **2.58x** | 1.22 → 1.12 |
| lightcone multipoles | 1.88x | 3.53x | **6.17x** | 1.41 → **1.02** |
| lightcone `w`* | 1.25x | 1.75x | 2.78x | 1.37 → **1.03** |

\* F5 alone. The lightcone `w` points were not measured before F1 (that sweep was
stopped before reaching them), so no cumulative figure against 0.4.0 exists for
this row; the slope, which is the claim that matters, is unaffected by that gap.

Lightcone `w` becoming linear (1.03) while box `w` does not (1.12) is itself
evidence for the cylinder diagnosis: `SkyRpPi` already queries a 3D ball of
radius `sqrt(r_max^2 + pi_max^2)`, so it has no cylinder to pay for, whereas
`BoxRpPi` builds its tree on the 2D projection. The only path left with a
superlinear exponent is the only one that queries a cylinder.

**`box_w` keeps an exponent of 1.12, and that closes the loop on F2.** Its tree
is built on the 2D projection, so the query returns a cylinder through the full
box depth and candidates per galaxy genuinely grow with the box (F2 measured
88.8 → 134.4 → 210.4 at fixed density). That is real pair work, so no amount of
ordering removes it. F2 concluded the cylinder did not explain the *timing*
because tree overhead dominated; with that overhead gone, the cylinder is exactly
what remains. Making `BoxRpPi` query a 3D ball of radius `sqrt(r_max^2 +
pi_max^2)` — which `SkyRpPi` already does — is the remaining lead.

**The per-galaxy hazard, and how it is handled.** Open question 3 above was the
dangerous one. Two mechanisms:

- the dense per-galaxy arrays are written at `gj = sel[n]`, the caller's galaxy
  id, rather than `i + n`, the visit position;
- the sparse jackknife arrays cannot be, because they are built per chunk and
  concatenated, so `_restore_galaxy_order` applies the inverse permutation with
  a `fill` matching each array's padding convention (-1 for patch ids, 0 for
  values).

Both are guarded by tests written **before** the implementation, because every
pre-existing per-galaxy test asserts on `Y.sum(axis=0)` — which is invariant
under permutation and would have passed while each galaxy's alignment signal was
attributed to a different galaxy. The new tests brute-force count pairs for seven
probe galaxies (including both ends and a chunk boundary) and separately assert
that shuffling the input shuffles the output identically.

**Test changes.** Four `brute_equals_tree` assertions moved from exact equality
to `allclose(rtol=1e-10)`. Verified before changing them: the float grids differ
by ~1e-14 while the integer pair counts `DD` and `SR` remain *exactly* equal, so
the pair sets are identical and only the summation order moved.
`plans/REFACTOR_PLAN.md` §4 already specifies `allclose` for brute-vs-tree; those
four were stricter than the contract and had passed by coincidence.


---

## F6 — The cost model, and why the benchmark's density misleads

**Status:** measured 2026-08-26, gating the next round of optimisation. No code
change.

### The reference mock is ~30x less dense than real data

Candidates per galaxy within `r_max` is `n * (4/3) pi r_max^3`, and it sets the
length of the arrays every per-galaxy NumPy call operates on. It is therefore
the single number that decides where time goes:

| configuration | n [/(Mpc/h)^3] | candidates/galaxy |
|---|---:|---:|
| the benchmark mock | 3.1e-4 | **11-19** |
| TNG300-like (1e5 in 205 Mpc/h) | 1.2e-2 | **389** |
| 1e6 in 500 Mpc/h | 8.0e-3 | 268 |
| 1e7 in 1000 Mpc/h | 1.0e-2 | 335 |

**Any profile taken only at mock density describes the mock.** `bench_lib` now
takes an absolute `density=` so the same code can be measured in the regime real
catalogues occupy (`SIM_BOX_DENSITY`).

### The cost model

Box multipoles, `r` in [0.5, 20], single thread, tracemalloc off:

| regime | cand/gal | N=9,600 | N=38,400 | µs per candidate |
|---|---:|---:|---:|---:|
| mock | 19-20 | 0.34 s | 1.42 s | **1.84 / 1.89** |
| simulation | 344-346 | 0.88 s | 3.37 s | **0.264 / 0.255** |

Fitting `cost = A + B * candidates` to the two regimes:

> **A ≈ 33.8 µs fixed per galaxy, B ≈ 0.157 µs per candidate.**

which splits the runtime as:

| regime | cand/gal | fixed overhead | per-pair work |
|---|---:|---:|---:|
| mock | 19 | **92%** | 8% |
| **simulation** | **345** | **38%** | **62%** |
| dense or large r_max | 1000 | 18% | 82% |

The fixed 33.8 µs is per-galaxy Python and NumPy call overhead: ~10 NumPy calls
each on a short array, 100,000 calls apiece to `bin_pairs` and `get_ellipticity`,
405k ufunc `reduce`, 200k `errstate.__enter__`. Batching the inner loop across a
chunk attacks that term and nothing else, so **its payoff is ~1.6x at simulation
density, not the ~3x a mock-density profile suggests**. The other 62% is genuine
per-pair arithmetic, which only cheaper per-pair operations (or compiled code)
can reduce.

### Memory

Box, simulation density, above a 156 MB interpreter baseline: 8 / 18 / 42 / 100
MB at 9,600 / 38,400 / 100,000 / 300,000 shape galaxies — linear at **~330 bytes
per galaxy**. Extrapolated: 1e6 → ~330 MB, 1e7 → ~3.3 GB. Comfortable.

Lightcone is the constraint, because the randoms dominate the point count:

| N_shape | randoms | total points | RSS above baseline | bytes/point |
|---:|---:|---:|---:|---:|
| 9,600 | 5x | 64,800 | 15 MB | 242 |
| 38,400 | 5x | 259,200 | 58 MB | 236 |
| 100,000 | 5x | 675,000 | 156 MB | 243 |
| 38,400 | **10x** | 475,200 | 162 MB | **357** |

Projected at 10x randoms (shape + position, data + randoms):

| sample | total points | projected RSS |
|---:|---:|---:|
| 1e6 shape | 2.2e7 | **~7 GB** |
| 1e7 shape | 2.2e8 | **~55-77 GB** |

**At 1e7 with 10x randoms the lightcone is memory-bound, not compute-bound.**
All four catalogues and their KDTrees are resident at once, and the
multiprocessing path transiently doubles the peak while copying arrays into
shared memory (the originals are freed afterwards, so it is a spike rather than
a permanent doubling). Making the pair loop faster does not help a run that
cannot allocate.

Analytic or subsampled randoms are not available as an escape: lightcones are
not periodic, the randoms encode the survey mask and must be user-supplied, and
at 10x density subsampling them would degrade the estimator.

Directions worth measuring, in memory rather than speed terms: float32 for
positions where precision allows (halves the largest arrays), tiling the RR
computation so both randoms catalogues need not be fully resident, and avoiding
the transient copy in the shared-memory setup.
