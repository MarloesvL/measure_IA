# Performance

Alongside the correctness checks in [Validation](validation.md), MeasureIA is **benchmarked against the same
external codes** — halotools for the periodic box and TreeCorr for the lightcone. Validation answers "is the
answer right"; this answers "what does the answer cost", so you can size a run before starting it.

## How it is measured

- **Identical inputs and binning.** Both codes read byte-identical seeded mocks, and the binning is imported
  from the validation scripts, so a timed run measures the configuration whose cross-package agreement is
  already established.
- **Every timing is gated on correctness.** A point is only reported if MeasureIA reproduces the reference
  result at the validation tolerances. A timing for a configuration that computes the wrong answer is worse
  than no timing.
- **Threads pinned on both sides**, one untimed warm-up then best-of-five, and each point in its own process.
- **At a realistic number density** — $n \simeq 10^{-2}\,h^3\,\mathrm{Mpc}^{-3}$, about 345 neighbours per
  galaxy within $r_\mathrm{max}=20$. This matters more than anything else in this list: the package's own test
  mock is ~30× sparser, and MeasureIA's fixed per-galaxy cost amortises over more pairs as density rises. The
  same comparison measured on the sparse mock reports the box gap as 22× rather than 3.6×.¹

## How MeasureIA compares

Single thread, box $w_{gg}$/$w_{g+}$ against halotools and lightcone against TreeCorr:

| $N_\mathrm{shape}$ | box vs halotools | lightcone vs TreeCorr | lightcone memory |
|---:|---:|---:|---:|
| 2,400 | 2.7× | 2.0× | 1.0× |
| 9,600 | 3.6× | 2.4× | 1.3× |
| 38,400 | 3.6× | 2.5× | 2.2× |
| 100,000 | **3.6×** | **2.6×** | **4.5× less than TreeCorr** |

The ratios are **flat**, not growing: scaling exponents are 1.05 for MeasureIA against 0.98 for halotools, and
1.16 against 1.09 for TreeCorr, where 1.0 is ideal at fixed number density.

With all cores in use, each code at **its own best setting** (12-core laptop, $N=38{,}400$):

| | MeasureIA | reference | ratio |
|---|---:|---:|---:|
| box vs halotools | 2.59 s (8 workers) | 1.78 s (1 thread²) | **1.45×** |
| lightcone vs TreeCorr | 4.83 s (8 workers) | 1.03 s (8 threads) | **4.7×** |

## Trade-offs

**In MeasureIA's favour**

- **No compiler, no build step.** `pip install measureia` needs nothing but Python and NumPy. It is pure
  Python/NumPy throughout — no Cython, no C extension, no OpenMP toolchain to get working.
- **Substantially less memory on the lightcone** — 4.5× less than TreeCorr at 100,000 galaxies, and the gap
  widens with size. Memory is often what decides whether a catalogue runs at all.
- **One call gives the whole measurement.** `measure_xi_w` returns $w_{gg}$, $w_{g+}$, the responsivity
  correction and the jackknife covariance together. Obtaining the same from TreeCorr means assembling the IA
  estimator yourself from 24 separate correlation runs.

**Against it**

- **A constant factor behind compiled codes**, as the tables above show. That factor is now stable with
  catalogue size rather than growing, but it is real.
- **Parallel efficiency is the weakest point.** MeasureIA reaches 31–49% on 8 cores where TreeCorr reaches
  91%: it parallelises with processes rather than threads, and each pool costs ~0.9 s to start.
- **Multiprocessing does not pay below ~40,000 galaxies.** Below that, `num_nodes > 1` is *slower* than
  leaving it at 1. Set it for large runs only.

## Roughly how long will my measurement take?

Single thread, realistic density, on one core of an Apple M2 Max. Peak resident memory in brackets, of which
~160 MB is the Python interpreter and imports.

| $N_\mathrm{shape}$ | box $w$ | box multipoles | lightcone $w$ | lightcone multipoles |
|---:|---:|---:|---:|---:|
| 2,400 | 0.3 s (170 MB) | 0.2 s (164 MB) | 0.7 s (188 MB) | 0.4 s (175 MB) |
| 9,600 | 1.6 s (176 MB) | 0.9 s (168 MB) | 4.1 s (228 MB) | 1.9 s (194 MB) |
| 38,400 | 6.4 s (188 MB) | 3.4 s (181 MB) | 19.1 s (281 MB) | 8.8 s (233 MB) |
| 100,000 | 16.6 s (212 MB) | 9.0 s (194 MB) | 53.3 s (345 MB) | 23.4 s (304 MB) |

**Extrapolating** is safe, because cost is now linear in $N$ at fixed density: ten times the galaxies costs
about ten times the time. Memory grows at roughly **330 bytes per galaxy** for the box and **240–360 bytes per
point** for the lightcone (counting randoms, which dominate) — so a 10⁷-galaxy lightcone with 10× randoms
needs of order 20 GB.

Two things change the cost substantially, both through the number of neighbours per galaxy: **number density**
and **$r_\mathrm{max}$**. A sample twice as dense, or a maximum separation $2^{1/3}$ times larger, roughly
doubles the pair count and so the runtime. Use `num_nodes` above ~40,000 galaxies for a further 2–4×.

### Why $w$ costs about twice what the multipoles cost

The projected statistics are consistently slower than the multipoles in the table above, and the reason is
geometry rather than implementation. $w_{gg}$ and $w_{g+}$ are binned in $(r_p, \pi)$, which selects a
*cylinder*, so the neighbour search has to cover a ball large enough to contain it — radius
$\sqrt{r_\mathrm{max}^2 + \pi_\mathrm{max}^2}$. The multipoles are binned in $(r, \mu)$, which selects a
*sphere*, so $r_\mathrm{max}$ is enough. The projected run therefore examines
$\left(1 + (\pi_\mathrm{max}/r_\mathrm{max})^2\right)^{3/2}$ times as many candidate pairs, which makes
**$\pi_\mathrm{max}$ a cost knob in its own right** — and a fairly sharp one:

| $\pi_\mathrm{max}/r_\mathrm{max}$ | candidate pairs, relative to the multipoles |
|---:|---:|
| 0.5 | 1.4× |
| 1 (the default, $\pi_\mathrm{max}=r_\mathrm{max}$) | 2.8× |
| 2 | 11× |

At the default this predicts 2.8× the candidates, and the measured ratio is 2.78×; it shows up as ~1.9× in wall
time because the per-galaxy overhead is common to both. None of those extra pairs is wasted work — the $w$
estimator integrates over $|\pi| \le \pi_\mathrm{max}$, so they carry signal. But if you do not need a large
$\pi_\mathrm{max}$, lowering it is one of the cheapest speed-ups available.

## Running the benchmarks yourself

```bash
pip install measureia[validation]           # halotools + treecorr
python benchmarks/run_sweep.py --density 1e-2
python benchmarks/plot_results.py benchmarks/results/laptop_local.jsonl
```

The methodology, the full result records and the reasoning behind each optimisation — including analyses that
turned out to be wrong and were corrected — are in
[`benchmarks/README.md`](https://github.com/MarloesvL/measure_IA/blob/main/benchmarks/README.md) and
[`benchmarks/FINDINGS.md`](https://github.com/MarloesvL/measure_IA/blob/main/benchmarks/FINDINGS.md).

---

¹ Measured on the package's own mock at $n \simeq 3\times10^{-4}$, where each galaxy has ~19 neighbours instead
of ~345. Benchmarks quoted at that density understate MeasureIA roughly sixfold and suggest a scaling problem
that does not exist.

² halotools' `num_threads` uses `multiprocessing`, not OpenMP, so on a problem this size more workers make it
slower; 1 thread is its best setting here. TreeCorr uses OpenMP and scales to 91% efficiency — note that its
macOS wheels are often built without OpenMP, in which case it silently runs single-threaded.
