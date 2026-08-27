# Speed benchmarks

This directory measures how fast measureia is, against the same external codes
`validation/` already checks it agrees with: **halotools** for the periodic box
and **TreeCorr** for the lightcone. It complements `validation/`, which answers
"is the answer right"; this answers "what does the answer cost".

Two audiences:

- the JOSS paper's "Comparison to related software" section, which needs
  defensible numbers rather than adjectives;
- optimisation work, which needs to know where the time actually goes before
  anyone changes `src/`.

## Quick start

```bash
uv pip install -e ".[validation]"          # halotools + treecorr

python benchmarks/run_sweep.py --smoke --machine laptop      # ~2 min, checks the harness
python benchmarks/run_sweep.py --machine laptop              # the full sweep
python benchmarks/plot_results.py benchmarks/results/laptop_local.jsonl
python benchmarks/profile_measureia.py                       # where the time goes
```

Before and after a change to `pair_kernel.accumulate`:

```bash
python benchmarks/bitidentity_matrix.py capture --out baseline.npz   # on the old code
# ... make the change ...
python benchmarks/bitidentity_matrix.py compare --ref baseline.npz   # must be 0 differing
```

`--skip-mp` keeps that on one CPU; `--only-mp` runs just the multiprocessing
configurations, for topping up a baseline captured with `--skip-mp`.

Sweeps accept `--max-cpus N`, which caps the thread / `num_nodes` axis so a run
cannot exceed a CPU budget.

## Files

| file | what it does |
|---|---|
| `bench_lib.py` | timing with warmup/repeats, environment capture, size-scaled mocks, KDTree candidate instrumentation, the JSONL store |
| `bench_runner.py` | runs **one** benchmark point in a fresh process and emits one JSON record |
| `run_sweep.py` | builds the grid, spawns runners, resumable; runs the correctness gates |
| `machines.py` | named machine profiles — scratch paths, thread lists, size caps |
| `profile_measureia.py` | cProfile hot spots and a stage breakdown for the four measurement paths |
| `plot_results.py` | figures plus a markdown table for the paper |
| `job_template.sh` | cluster batch template |
| `micro_candidate_selection.py` | isolates the four lines of candidate selection and times/measures each part |
| `memcheck.py` | peak RSS and peak Python allocation for one measurement, one config per process |
| `bitidentity_matrix.py` | captures every output array across a config matrix and compares bitwise — the safety net for changes to `pair_kernel` |
| `FINDINGS.md` | **the measured results and what they mean**, written to be picked up cold in a later session |
| `results/` | committed JSONL results, one file per machine and scratch location |

## Methodology

### One subprocess per point

Every timing point runs in its own process. This is not tidiness — without it
the numbers would be wrong:

- `OMP_NUM_THREADS` and friends only bite if they are set **before** numpy and
  TreeCorr are imported. TreeCorr's `num_threads` otherwise silently defaults
  to every core on the machine, and `validation/run_lightcone_treecorr.py`
  never sets it.
- measureia's multiprocessing paths call
  `mp.set_start_method("spawn", force=True)`, which does not survive being
  driven repeatedly in one long-lived process.
- `resource.getrusage` peak RSS only means something for a process that did
  one job.

`run_sweep.py` appends one JSON line per point and skips keys already present,
so a long laptop run or a walltime-killed cluster job resumes without redoing
work.

### Fairness rules

These are enforced in code, not just intended.

- **Byte-identical inputs.** Both codes get the same seeded mock arrays,
  converted by the package's own helpers (`measureia.mocks.halotools_inputs`,
  and the catalogue builder from `validation/run_lightcone_treecorr.py`).
- **Identical binning**, imported straight from the validation scripts
  (`RP_LIMS`, `NUM_BINS_RP`, `PI_MAX`, `NUM_BINS_PI`, `COSMOLOGY`), so the
  timed configuration is the one the cross-package agreement was established
  on.
- **Correctness is gated.** After the sweep, every measureia point is compared
  against the reference measured on the identical catalogue, at the tolerances
  from `tests/test_validation_references.py` (halotools `rtol=1e-10` after the
  `2R` responsivity factor; TreeCorr `rtol=5e-3, atol=0.05`). Failures are
  reported and excluded by `plot_results.py`. A timing for a configuration
  that computes the wrong answer is worse than no timing.
- **Threads pinned on both sides** — measureia's `num_nodes`, halotools' and
  TreeCorr's `num_threads`, and the OpenMP environment variables.
- **Warmup and repeats.** One untimed warmup, then five timed runs. `t_min` is
  the headline (least contaminated by scheduler noise), `t_median` is stored
  next to it so an unstable point is visible rather than hidden. In practice
  the two agree to within about 1%.
- **Console output is not timed.** Both codes' stdout/stderr go to devnull
  while the clock runs; measureia is chatty, and timing its printing would be
  meaningless.
- **The same science product.** One `measure_xi_w(..., "both")` call returns
  w_gg *and* w_g+ with the responsivity applied. halotools needs two calls
  (`gi_plus_projected` + `wp`). TreeCorr needs **six correlation runs per
  signed π slab** — `NG(D,S)`, `NG(R_D,S)`, `NN(D,S)`, `NN(D,R_S)`,
  `NN(R_D,S)`, `NN(R_D,R_S)` — i.e. 24 runs at `NUM_BINS_PI=4`, plus the
  user-side estimator assembly, all of which the benchmark includes because
  that is what it costs to obtain the same numbers.
- **TreeCorr at two accuracies.** `bin_slop=0` (the accuracy-matched setting
  the validation uses) and TreeCorr's own default. Quoting only the default
  would be an unfair speed claim built on approximate binning. (Measured: the
  two turn out to be indistinguishable on this comparison — see `FINDINGS.md`
  F3 — but the guard stays, because that is a result rather than an assumption.)
- **Memory is reported, not just time.** Every point records peak RSS from
  `resource.getrusage` in its own process. It is a first-class axis: memory is
  what decides whether a catalogue runs at all on a given machine, and it is
  the axis on which measureia currently does best (roughly 8x smaller than
  TreeCorr at 300k galaxies).

### What is and is not counted

`t_min`/`t_median` cover the whole measurement: catalogue objects in memory to
result arrays in hand. `peak_rss_mb` is the high-water mark of the whole
process, so it includes the ~160 MB interpreter-and-imports baseline; compare
differences and ratios rather than treating it as the measurement's own
footprint. For measureia that **includes writing the HDF5 output**,
which neither reference code does. That cost is not unpicked from inside the
call — instead `bench_runner.io_probe` measures an HDF5 write and read of a
representative array in the same scratch directory, so the filesystem's
contribution is visible as its own number rather than being either hidden or
guessed at. Mock generation is never timed.

### Scaling regimes

Two, because they answer different questions:

- **`fixed_density`** — the box grows as `N^(1/3)` from the validation mock's
  205 Mpc/h at 2400 shape galaxies, so pairs-per-galaxy stays constant. This is
  what running a bigger simulation actually looks like, and it is the headline
  regime.
- **`fixed_volume`** — N grows inside the same 205 Mpc/h box, so density and
  pairs-per-galaxy rise and the total cost goes as `N²`. Capped at 38,400
  galaxies (`MAX_N_FIXED_VOLUME`): enough to fix the slope, and running it to
  300k would take hours to say nothing new.

Together the two slopes separate *cost per galaxy* from *cost per pair*.

On the lightcone the same idea applies to the sky window (area ∝ N at fixed
comoving shell), and `n_randoms_factor` is held at 5 so the randoms:data ratio
never shifts underneath the comparison — it matters, because the randoms
dominate the pair count.

### Machines and scratch filesystems

Every path in `machines.py` is written out explicitly. Nothing is auto-detected
and no temporary directory is inferred from the environment, because on a
cluster the difference between shared NFS and node-local disk is one of the
things being measured — guessing it would destroy the measurement. `run_sweep.py`
refuses to start if the scratch location it was asked for is still blank.

measureia has two code paths with no halotools/TreeCorr equivalent that touch
the filesystem: the `temp_file_path` HDF5 offload used by the tree and
multiprocessing paths, and the HDF5 output write. On a cluster, run the same
sweep once per named scratch location:

```bash
python benchmarks/run_sweep.py --machine cluster --scratch nfs
python benchmarks/run_sweep.py --machine cluster --scratch local
```

The two result files are kept separate and never merged into one curve. Each
record stores the literal scratch path and the filesystem type it resolved to.

## The internal comparison: `w` vs multipoles

`measure_xi_w` and `measure_xi_multipoles` differ in how they select candidate
pairs, and the benchmark records the KDTree candidate count for every measureia
point (folded into the warmup call, so it is free) alongside the time:

| path | KDTree built on | query radius | candidate region |
|---|---|---|---|
| box `w` | the **2D projection** | `r_max` (projected) | a cylinder through the **full box depth** |
| box multipoles | full 3D | `r_max` | a 3D ball |
| lightcone `w` | full 3D | `sqrt(r_max² + π_max²)` | a ball enclosing the (rp, π) cylinder |
| lightcone multipoles | full 3D | `r_max` | a 3D ball |

The box `w` path therefore hands `bin_pairs` every neighbour within `r_max` in
projection *at any line-of-sight separation*, and the π window then discards
most of them. The `internal` sweep measures the consequence directly, including
a boxsize series at fixed sample (`L` = 205, 400, 800 Mpc/h) — the candidate
cost of the (rp, π) query should grow with box depth while the (r, μ) query's
does not.

See "Results" below for what this turns out to cost in wall time, which is not
what the candidate counts alone would suggest.

## Results

Measured results, and the reasoning behind them, live in **`FINDINGS.md`** —
including one case where a first analysis was wrong and had to be corrected.
Regenerate the figures and the summary table from the raw records with:

```bash
python benchmarks/plot_results.py benchmarks/results/laptop_local.jsonl
```

`results/laptop_local_pre_F1.jsonl` is kept as a before/after archive: it holds
the same measurements taken before the candidate-selection speedup
(`FINDINGS.md` F1) was applied.
