# Box jackknife with analytic randoms: why the errors inflate, and how to fix it

Working notes for the box jackknife investigation. The code is uncommitted:
`jk_cross_pairs_lib.py`, `run_box_jk_cross_pairs.py` and `run_box_jk_corrpc.py`. There are no
changes to `src/` yet.

**Status (2026-10-14, reviewed 2026-10-27):** validation runs done for w and multipoles; the
review's open points are listed in section 8.
- The analytic DR and S+R terms (option 2) reproduce the random-catalogue jackknife to within a
  few percent in every configuration tested (sections 5 and 6).
- Today's analytic jackknife overestimates large-scale errors by up to about 4× (g+) and 6×
  (gg) at realistic density.

---

## 1. The problem in one paragraph

`MeasureIABox` measures w_g+ and w_gg in a periodic box with analytic randoms: RR is a
formula, not a pair count. It estimates errors with a delete-one sub-box jackknife. On large
scales, these jackknife errors come out much larger than the errors you get by running the same
measurement with a real random catalogue.

The comparison is measureia run on data and randoms, with its pair counting as it is today.
The size of the excess (range over 8, 27 and 64 sub-boxes):

| | w_g+, largest rp | w_gg, largest rp |
|---|---|---|
| sparse mocks (n ≈ 10⁻³ (h/Mpc)⁻³, 100 mocks) | 1.4–1.6× | 1.9–2.4× |
| realistic density (n = 10⁻² (h/Mpc)⁻³, 40 mocks) | 3.4–4.0× | 4.8–5.7× |

At realistic density, w_gg is already 1.09–1.26× too large at rp ≈ 1 Mpc/h. The multipoles
(ξ_g+,2 and ξ_gg,0) show the same pattern (section 6).

---

## 2. What goes wrong, intuitively

Picture one jackknife realisation. Sub-box k has been cut out of the box, leaving a cube-shaped
hole. Every galaxy in the hole is gone, and so is every pair with a member in the hole.

Now look at a galaxy that sits **just outside the hole**. Part of its neighbourhood (its "partner
shell" at separation r) lies inside the hole, so it has lost some of its partners. How many it
lost depends on *where exactly it sits* relative to the hole, and on *how many galaxies happen to
sit near the hole* in this particular realisation.

- **With a random catalogue,** the DR term (data × randoms) counts the same thing for uniform
  points: for each galaxy, how much of its partner shell survives the cut. The DR term therefore
  moves up and down with the data from realisation to realisation, and in the estimator
  (DD − DR − RD + RR)/RR those fluctuations cancel. The S+R term does the same for the
  alignment signal. For a galaxy with a full shell, S+R averages to zero. For a galaxy with a
  truncated shell it does not, and it removes the spurious "alignment" that the truncation
  creates in S+D.
- **With analytic randoms,** there is no DR or S+R term. RR is one number per bin and
  realisation; it knows the hole's *volume*, but not *which galaxies sit next to it*. The
  fluctuation in DD (and S+D) caused by the galaxies near the hole edges is left uncancelled,
  and it shows up as extra scatter between realisations. Hence the inflated errors. It grows with
  scale, because larger shells reach the hole from further away, and with density, because
  shot noise no longer hides it.

This was tested directly by **swapping one term at a time**:

| combination | result |
|---|---|
| analytic RR + the random catalogue's DR and S+R terms | **matches** the randoms reference |
| the random catalogue's RR, but no DR or S+R terms | as bad as today's analytic version |

So the cause is the missing DR and S+R terms, not RR.

What did *not* help: removing cross-region pairs with weight ½ (the paper's rule; treecorr
`mean`), the Mohammad & Percival "match" weight, corr_pc's per-galaxy random draws, or a
per-pair random draw, each combined with any RR normalisation. None of them reproduces the randoms
reference, because none of them supplies the missing DR and S+R information.

---

## 3. The fix: DR and S+R in the limit of infinitely many randoms

A uniform random catalogue in a periodic box has a known density everywhere. So we never need
the random *points*: we only need the **volume** of each galaxy's partner shell that falls in
the hole. That is pure geometry.

### Notation

- $V = L^3$: the box. $C_k$: sub-box $k$, with volume $V_k = V/N_{\rm jk}$, and $v = V_k/V$.
- Bin $b$: $r_p \in [r_1, r_2)$, $\pi \in [p_1, p_2)$, with volume
  $V_b = \pi (r_2^2 - r_1^2)(p_2 - p_1)$.
- Separation $\mathbf{s} = \mathbf{x}_{\rm shape} - \mathbf{x}_{\rm position}$ (minimum image), as in
  measureia. $\phi_s$ is its azimuth on the sky plane.
- A shape galaxy $j$ has weight $w_j$, ellipticity $e_j$ and major-axis angle $\theta_j$:
  $e_+ = e_j \cos 2(\phi_s - \theta_j)$.
- Retained weight sums: $W_{D,k} = \sum_{i \notin k} w_i$ and $W_{S,k} = \sum_{j \notin k} w_j$.
- Union deletion throughout: realisation $k$ drops every galaxy and every random point in $C_k$.

### The two geometric integrals

How much of bin $b$, centred on a galaxy at $\mathbf{x}$, lands inside the hole:

$$
O_k(\mathbf{x}; b) = \int_b d^3 s \; \mathbb{1}[\mathbf{x} + \mathbf{s} \in C_k] \quad \text{(periodic)}
$$

The same, weighted by the tangential projection of a shape with angle $\theta$:

$$
Q_k(\mathbf{x}, \theta; b) = \int_b d^3 s \; \mathbb{1}[\mathbf{x} + \mathbf{s} \in C_k]\, \cos 2(\phi_s - \theta)
$$

$\tilde b$ denotes the bin with $\pi$ reflected, $\pi \in [-p_2, -p_1)$. It is needed when the
partner sits at $\mathbf{x} - \mathbf{s}$ rather than $\mathbf{x} + \mathbf{s}$. Reflecting
$\mathbf{s} \to -\mathbf{s}$ changes $\phi$ by $\pi$, which leaves $\cos 2\phi$ unchanged.

### The terms (each normalised by its own retained pair count)

**D·R_S**: a position galaxy $i$; its random partner sits at $\mathbf{x}_i + \mathbf{s}$.

$$
\widehat{DR}_k(b) = \frac{\sum_{i \in D,\, i \notin k} w_i \,[\,V_b - O_k(\mathbf{x}_i; b)\,]}{W_{D,k}\,(V - V_k)}
$$

*Intuition:* each galaxy contributes its full bin volume, minus the part that was cut away
by the hole.

**R_D·S**: a shape galaxy $j$; its random partner sits at $\mathbf{x}_j - \mathbf{s}$.

$$
\widehat{RD}_k(b) = \frac{\sum_{j \in S,\, j \notin k} w_j \,[\,V_b - O_k(\mathbf{x}_j; \tilde b)\,]}{W_{S,k}\,(V - V_k)}
$$

**S+R_D**: shapes against random positions.

$$
\widehat{S_+R}_k(b) = -\,\frac{\sum_{j \in S,\, j \notin k} w_j\, e_j\, Q_k(\mathbf{x}_j, \theta_j; \tilde b)}{W_{S,k}\,(V - V_k)}
$$

*Intuition:* around a full shell, $\cos 2(\phi - \theta)$ averages to zero, so a galaxy far
from the hole contributes nothing. Near the hole, the missing part of the shell would have
contributed $Q$, so what is left is $-Q$. This is exactly the fake alignment that the cut
imprints, and the estimator subtracts it.

**RR (exact, union)**: this replaces today's per-realisation analytic RR.

$$
\widehat{RR}_k(b) = \frac{V_b}{V}\;\frac{1 - 2v + v\, f(b)}{(1 - v)^2},
\qquad
f(b) = \frac{1}{V_b}\int_b d^3 s\; \frac{V_{\rm ov}(\mathbf{s})}{V_k},
\qquad
V_{\rm ov}(\mathbf{s}) = \prod_{a=x,y,z} \max(0,\, L_{\rm sub} - |s_a|)
$$

*Intuition:* $V_{\rm ov}$ is the overlap of the hole with itself shifted by $\mathbf{s}$, i.e.
the volume where *both* pair members land in the hole. At small $s$, $f \to 1$ and this
reduces to today's $V_b/(V - V_k)$. Beyond the sub-box size, $f \to 0$: the pairs lost are
"one end in the hole" from either end, which today's formula does not account for.
Section 5 shows the measured effect.

"Exact" refers to the formula: it is the infinite-randoms limit of the counted union RR. The
prototype estimated $f(b)$ by Monte Carlo (2×10⁵ samples, error ~10⁻⁴, the same value for every
realisation). The implementation computes it by deterministic quadrature.

### Validity condition

The periodic image sums and the $V_{\rm ov}$ formula assume that every separation component is
at most $L/2$: $r_{p,\max} \le L/2$ and $\pi_{\max} \le L/2$ in rp–π mode, $r_{\max} \le L/2$ in
r–μ mode. Otherwise minimum-image pair counting and the analytic integrals disagree. The sub-box
condition $L_{\rm sub} \le L/2$ holds automatically, since $N_{\rm jk} \ge 8$. measureia's
minimum-image counting and today's analytic RR need the same condition, but no input check
enforces it yet.

### The realisation estimators

$$
\xi_{gg,k} = \frac{\widehat{DS}_k - \widehat{DR}_k - \widehat{RD}_k + \widehat{RR}_k}{\widehat{RR}_k},
\qquad
\xi_{g+,k} = \frac{\widehat{S_+D}_k - \widehat{S_+R}_k}{\widehat{RR}_k \cdot 2R_k}
$$

- $\widehat{DS}_k$ and $\widehat{S_+D}_k$ are today's retained counts divided by the retained
  pair weight $W_{D,k} W_{S,k} - \sum_{i \in D \cap S,\, i \notin k} w_{D,i}\, w_{S,i}$; the
  subtracted sum removes the self-pairs of galaxies that are in both samples.
- $R_k$ is today's per-realisation responsivity.
- **The full-sample measurement is unchanged**: in the full periodic box, DR and RD equal RR
  and S+R vanishes, so the estimator reduces to today's.

---

## 4. How the integrals are computed (and what it costs)

The sub-box is an axis-aligned cube and the bin is a cylindrical shell, so both integrals split
into a line-of-sight part and a sky-plane part:

$$
O_k = \ell_z \cdot A_k, \qquad Q_k = \ell_z \cdot \left[\cos 2\theta\, A^c_k + \sin 2\theta\, A^s_k\right]
$$

- **$\ell_z$** is the overlap length of $[p_1, p_2]$ with the hole's $z$-range relative to the
  galaxy, summed over periodic images. It has a closed form.
- **$A_k$** is the area of the annulus $r_1 \le r < r_2$ that falls inside the hole's
  $(x, y)$ rectangle; $A^c$ and $A^s$ are the same area weighted by $\cos 2\phi$ and
  $\sin 2\phi$:
  $A = \int_{r_1}^{r_2} r\, dr\, I_0(r)$, and likewise for $A^c$ and $A^s$.
- At fixed $r$, $I_0$, $I_c$ and $I_s$ are integrals of $1$, $\cos 2\phi$ and $\sin 2\phi$ over
  the arcs of the circle inside the rectangle. They are **exact**: the arc endpoints are where
  the circle crosses the rectangle's edges.
- The $r$-integral uses 16-point Gauss–Legendre quadrature, accurate to about $3\times10^{-5}$
  of $V_b$. It is not exact, and splitting cannot make it exact: between kinks, $r\,I_0(r)$
  contains arccos/arcsin terms. Kinks sit where the circle first touches each rectangle edge
  ($r$ = distance to that edge) and where it passes each corner. Splitting the integral at all of
  these radii gives fast convergence. Exact values would need the closed forms for the
  disc ∩ rectangle area and its $\cos 2\phi$ / $\sin 2\phi$ moments.

**Cost:** only galaxies within $r_{p,\max}$ of a sub-box (in projection) and within
$\pi_{\max}$ along the line of sight contribute. There are no random points and no extra pair
counts. A random catalogue at 10× instead costs two extra pair-counting passes, each about
10× the DD cost.

Measured prototype timings (plain numpy, one core, vectorised over galaxies only, with Python
loops over sub-boxes, periodic images and quadrature nodes):

| | analytic terms, per N_jk | all pair counting incl. 10× randoms, all three N_jk together |
|---|---|---|
| rp–π (w), sparse mock | ~7 s | — |
| rp–π (w), dense mock | ~90 s | ~27 min |
| r–μ (multipoles), sparse mock | ~80 s | — |
| r–μ (multipoles), dense mock | ~12 min | ~10 min |

The two columns are not like for like: the analytic column is per N_jk, the counting column
covers all three. For all three N_jk, the dense analytic terms take ~4.5 min (rp–π) and ~36 min
(r–μ): r–μ is about 3.6× slower than all the counting combined. Why the rp–π counting took
longer than the r–μ counting (27 vs 10 min) was not established. The implementation benchmark
re-measures both like for like.

The r–μ version is too slow as it stands. The implementation plan keeps it pure
Python/numpy/scipy (no build step, no numba) and requires:
- vectorising over galaxies × quadrature nodes × images, in bounded-memory chunks;
- a test that the vectorised result equals the prototype;
- a benchmark on a dense mock, reported before merging.

If r–μ is still too slow, the fallback is to tabulate the overlaps against the galaxy's position
relative to a sub-box and interpolate. All sub-boxes have the same shape, so one table serves
every sub-box.

---

## 5. Evidence (w)

All runs finished on 2026-10-13. Each entry is σ_jk(method) / σ_jk(randoms reference), averaged
over mocks. The reference is a 10× random catalogue run through measureia as it is today (union
deletion, counted RR). "Option 1" = random-catalogue DR and S+R + exact RR. "Option 2" = the
analytic terms above + exact RR. Ranges are over the 10 rp bins (1.2–49 Mpc/h) and over 8, 27
and 64 sub-boxes unless stated.

Mocks:
- **sparse:** the clustered box mock, n ≈ 1.1×10⁻³ (h/Mpc)⁻³, L = 150 Mpc/h;
- **strong clustering:** fewer, richer and more compact haloes (160 centrals, 20 satellites,
  σ = 1 Mpc/h);
- **dense:** n = 10⁻² (h/Mpc)⁻³, realistic for current surveys, also L = 150 Mpc/h.

All mocks have a fixed number of haloes and satellites per mock.

### Machinery checks (all exact unless stated)

| # | check | result |
|---|---|---|
| 1 | independent counter vs measureia's current jackknife | DD exact, S+D ≤ 2×10⁻¹¹, RR_jk 3×10⁻¹⁶ |
| 2 | treecorr `simple` / `mean` / `match` patch reductions | ≤ 7×10⁻¹⁶ |
| 3 | corr_pc's box jackknife (its random draws regenerated) | counts exact; w ≤ 10⁻⁶ (g+), 2×10⁻⁴ (gg) |
| 4 | randoms reference from four separate measureia runs vs the counter | exact |
| 5 | masked runs vs pre-filtered runs | bit for bit |
| 6 | predicted bias (V − 2V_k + V_ov)/(V − V_k) of today's realisations vs Poisson boxes | matches, no free parameters |
| 7 | analytic geometry vs Monte Carlo | within MC noise; quadrature ≤ 3×10⁻⁵ of V_b |

### Diagnosis

| # | test | result |
|---|---|---|
| 8 | cross-pair rules (½, MP22 match, per-pair draws, corr_pc draws), any RR amplitude | none reproduces the reference |
| 9 | term swap: analytic RR + randoms DR and S+R | matches the reference |
| 9b | term swap: randoms RR, no DR or S+R | as bad as today |

### Options against the reference

| # | configuration | statistic | today (largest rp) | option 1 | option 2 |
|---|---|---|---|---|---|
| 10 | sparse, 100 mocks | w_g+ | 1.44–1.61 | 1.00 | 0.98–1.00 |
| | | w_gg | 1.85–2.38 | 1.00–1.01 | 0.99–1.00 |
| 11 | strong clustering, 100 mocks | w_g+ | 1.15–1.21 | 1.00 | 0.99–1.00 |
| | | w_gg | 1.32–1.52 | 1.00 | 0.99–1.00 |
| 12 | dense, 40 mocks | w_g+ | 3.35–4.04 | 1.00 | 0.96–1.00 |
| | | w_gg | 4.77–5.72 (1.09–1.26 at rp ≈ 1) | **1.00 at small rp → 1.10–1.19** | 0.98–1.00 |
| 13 | dense, second random catalogue, 10 mocks | w_g+ | 3.33–3.89 | 0.99–1.00 | 0.96–1.01 |
| | | w_gg | 4.73–7.41 | 1.00 → 1.02–1.12 | 0.95–1.03 |
| 14 | dense, 20× randoms, 5 mocks | w_gg | as above | ≤ 1.04–1.17 (10× on the same mocks: 1.10–1.35) | 0.95–1.02 |

### Robustness of the reference itself

σ_jk(reference with an alternative catalogue) / σ_jk(reference with the original):

| # | check | result |
|---|---|---|
| 15 | sparse, other seed, 100 mocks | 0.98–1.01 |
| 16 | sparse, 20× randoms, 100 mocks | 0.99–1.01 |
| 17 | dense, other seed, 10 mocks | 0.96–1.04 |
| 18 | dense, 20× randoms, 5 mocks | 0.94–1.06, no systematic shift |

**19. The option 2 vs option 1 differences are random-catalogue noise.** The realisation-level
rms(option 2 − option 1) shrinks by about 1/√2 from 10× to 20× randoms:
- sparse: 0.08–0.16 → 0.05–0.12 σ_jk;
- dense w_g+ at small rp: 0.15–0.38 → 0.13–0.18 σ_jk.

(A term-by-term "difference / realisation scatter" metric rises with N_jk. An earlier version
of this note explained that by saying the random-catalogue noise is nearly common to all
delete-one realisations. That cannot be fully true if the same noise also explains the open
point below, which needs a realisation-to-realisation part. The variance decomposition listed
under the open point settles it; see section 8.)

### Why option 1 drifts at high density and option 2 does not

Option 1 mixes noisy random-catalogue DR terms with a noiseless exact RR. In the reference, the
noise in DR and RR comes from the same random catalogue and partly cancels; mixing breaks that
cancellation. The leftover noise grows with density relative to the signal, which is why it
shows up in dense w_gg (and more strongly in dense ξ_gg,0, section 6). It shrinks with more
randoms (test 14). Option 2 has no random noise at all.

So option 1 at the default of 10× would need about 20× randoms or a counted R_D·R_S pass to
avoid this.

### Against the true mock-to-mock scatter

Every randoms-style jackknife (the reference, option 1 and option 2) comes out too large
compared with the scatter between mocks:
- sparse: 0.9–1.7×;
- dense: 1.0–1.8×;
- strong clustering, smallest scales: up to 3.4×.

A random catalogue has this excess too, so it is not caused by analytic randoms. Whether it is
"ordinary" jackknife behaviour is **not shown**:
- The mocks fix the number of haloes and satellites, so the mock-to-mock scatter lacks
  halo-number variance while the jackknife includes it. The "true" scatter is then biased low.
- With 160 haloes, a 64-sub-box jackknife deletes only about 2.5 haloes per realisation, which
  breaks the jackknife's smoothness assumption.

None of this affects the option 2 vs reference comparison. Matching a random catalogue, our
target, removes the *analytic-randoms* problem; it does not make jackknife errors equal the true
errors.

### Open point

Dense w_g+ at rp < 3 Mpc/h: option 2 sits 2–4% below the reference (40 mocks).
- This is most likely the reference's own random-catalogue noise, which the noise-free analytic
  term lacks; option 1 shows 1.00 there only because it shares the reference's randoms.
- The 20× check on 5 mocks is too noisy to settle it: option 2 gives 0.96–1.00 with 20× and
  0.94–1.03 with 10× on the same mocks.
- Test 19 shows the analytic-vs-randoms difference behaves like noise. That supports the
  explanation but does not prove it.
- Rough consistency: an uncorrelated noise of rms ε·σ_jk in the reference lowers the ratio to
  1/√(1+ε²). With ε = 0.2–0.3, that gives 0.96–0.98, the size of the deficit. Whether the
  test 19 rms is jackknife-scaled in that sense was not recorded.
- Measurement that settles it (section 8): per realisation, Δ_k = ref_k − opt2_k, and
  σ²_ref = σ²_opt2 + σ²_Δ + 2 cov(opt2, Δ), all as jackknife-scaled variances. The identity
  itself holds by construction; the information is in its terms. If cov ≈ 0 and σ²_Δ accounts
  for the gap, the noise explanation is confirmed.

---

## 6. Multipoles (r–μ binning)

In r–μ mode the bin is a spherical-shell sector, $r \in [r_1, r_2)$ and
$\mu = s_\parallel/r \in [\mu_1, \mu_2)$. It does not split into "z-interval × sky-plane area".
The prototype (`analytic_partial_shell_rmu`) uses cylindrical coordinates $(\rho, h)$ around
the galaxy instead:
- At fixed projected radius $\rho$, the allowed heights $h$ are an intersection of intervals,
  all analytic: the spherical shell $r_1^2 \le \rho^2 + h^2 < r_2^2$, the μ-bin
  ($h = \rho\mu/\sqrt{1-\mu^2}$ at the edges), and the hole's z-range (periodic images).
- At fixed $\rho$, the sky-plane part is the same exact arc integrals $I_0$, $I_c$, $I_s$
  as in rp–π mode.
- The $\rho$-integral uses Gauss–Legendre quadrature on segments split at the r-bin edges, where
  the h-intervals change form. The integrand has further kinks that the prototype does not split
  at: $\rho = r_i\sqrt{1-\mu_j^2}$, where the μ-cone meets the shell; the radii where the shell
  meets the hole's z-faces; and the sky-plane kinks of section 4. These likely explain why this
  mode is accurate to 1.6×10⁻⁴ rather than 3×10⁻⁵.

Other points:
- measureia's $e_+$ in r–μ mode also uses the projected angle, so $\cos 2\phi$ is unchanged.
- An `rp_cut` adds the condition $\rho \ge r_{p,\rm cut}$ (not yet in the prototype).
- The exact RR's $f(b)$ becomes the same average over the spherical bin.
- The multipoles are linear combinations of the $\xi(r, \mu)$ bins, so the corrected
  realisations give their covariance directly.

### Checks

| # | check | result |
|---|---|---|
| 20 | counter vs measureia's `measure_xi_multipoles` jackknife | DD exact, S+D ≤ 3×10⁻¹², RR 4×10⁻¹⁶ |
| 21 | analytic r–μ geometry vs Monte Carlo | within MC noise; quadrature ≤ 1.6×10⁻⁴ of V_b |

### Options against the reference (ξ_g+,2 and ξ_gg,0, r from 1.2 to 49 Mpc/h)

| # | configuration | statistic | today (largest r) | option 1 | option 2 |
|---|---|---|---|---|---|
| 22 | sparse, 100 mocks, N_jk 8/27/64 | ξ_g+,2 | 1.48–1.63 | 0.98–1.01 | 0.98–1.00 |
| | | ξ_gg,0 | 1.92–2.20 | 0.96–1.01 | 0.95–1.00 |
| 23 | dense, 5 mocks, N_jk 8/27/64 | ξ_g+,2 | 3.59–5.04 | 0.98–1.01 | 0.93–1.04 |
| | | ξ_gg,0 | 4.93–6.12 | **1.00 at small r → 1.14–1.32** | 0.94–1.03 |

Notes:
- N_jk = 64 has sub-boxes of 37.5 Mpc/h, smaller than r_max = 60 Mpc/h, a regime w never
  reaches along the line of sight. The analytic geometry holds there.
- In the smallest r bin, ξ_gg,0 is 0.95–0.98 for **both** options (which configuration was not
  recorded). An earlier version blamed the exact RR. That is not supported:
  - Option 1 uses the reference's own DR terms, so option 1 vs the reference differs **only** in
    RR: exact vs counted.
  - At small r, f → 1, so the exact RR equals today's V_b/(V − V_k) to high accuracy. The
    (N−1)/N r–μ normalisation is a ~1/N effect. Neither can plausibly move σ by 2–5%.
  - What both options share is the **absence of the counted RR's random noise**, the same
    explanation as the w_g+ open point.

  Test (section 8): a term swap in that bin (option 2 with the reference's counted RR_k), plus
  the reference at 20× randoms.
- The dense multipoles show option 1's drift more strongly than w does (up to 1.32).

### Inferred from w, not repeated for multipoles

The reference robustness checks (other seed, 20× randoms) and the strong-clustering mock test
properties of the random catalogue and of the clustering, not of the binning. They were run for
w only (tests 11, 15–18).

---

## 7. Not covered yet

- **Weights:** the equations include weights, but the prototype currently accepts unit
  weights only.
- **Shape–shape (++, ××):** needs its own random terms.
- **The (N−1)/N factor:** check how it interacts with the r–μ-mode analytic RR factor noted in
  `validation/README.md`.
- **`rp_cut` in r–μ mode:** not in the prototype yet.
- **Speed:** the r–μ prototype is too slow for production (section 4).
- **Masks and weights in option 2:** covered by implementation tests, written before the source
  change, not by this validation.
- **Decision (2026-10-20):** option 2 (analytic).
- **Line-of-sight axis:** the prototype hard-codes z as the line of sight and θ in x–y. The
  implementation tests check LOS = 0 against a permuted LOS = 2.

---

## 8. Open after review (2026-10-27)

An external review of this note raised the points below. The per-realisation outputs of the
October runs were in a temporary directory that has since been cleared, so these need reruns.
Rerun outputs are kept under `validation/`.

1. **Smallest-r ξ_gg,0 deficit:** term swap and 20× reference (section 6 notes).
2. **Dense w_g+ open point:** variance decomposition (section 5, open point).
3. **Uncertainties:** mean ± s.e.m. across mocks for every σ ratio, at least in the extreme
   bins. Until then, the ranges above have no error bars; this matters most for the 5-mock
   dense multipole test 23.
4. **Quadrature accuracy (done, 2026-10-27).** Per galaxy, the prototype's r–μ quadrature
   error is up to 2.6×10⁻³ of V_b (95th percentile 4×10⁻⁴), compared with an adaptive
   reference. Its effect on the error bars was measured with
   `run_box_jk_quadrature_check.py` on the sparse mock: realisation terms with 16 (prototype),
   64 and 256 Gauss–Legendre nodes per segment.

   | N_jk | change in σ_jk, ξ_gg,0 | change in σ_jk, ξ_g+,2 | largest realisation shift / σ_jk |
   |---|---|---|---|
   | 8 | ≤ 0.08% | ≤ 0.08% | ≤ 0.0025 |
   | 27 | ≤ 0.01% | ≤ 0.06% | ≤ 0.002 |
   | 64 | ≤ 0.03% | ≤ 0.07% | ≤ 0.001 |

   The 64-node results are about 10× closer to the 256-node ones, so the reference has
   converged. The effect grows with r and is largest in the last bin. The implementation
   therefore keeps the prototype's quadrature for r–μ, and uses the exact closed form for rp–π.

   Correction to the earlier figures: the "3×10⁻⁵" (rp–π) and "1.6×10⁻⁴" (r–μ) quadrature
   accuracies in sections 4 and 6 were after summing over galaxies. Per galaxy, the prototype
   is off by up to 5×10⁻⁴ of V_b (rp–π, where a bin contains a tangent or corner radius) and
   3.6×10⁻³ (r–μ, the full-bin integral).

   **Node layout (r–μ), `run_box_jk_node_layout.py`.** The prototype shares one set of ρ-nodes
   across all r bins (split at every bin edge), so a bin's value depends slightly (~10⁻⁴) on
   which other bins exist. Giving each bin its own nodes on [0, r_i] and [r_i, r_{i+1}] removes
   that dependence:

   | | shared (prototype) | per-bin |
   |---|---|---|
   | per-galaxy error / V_b: max, median | 9.6×10⁻⁴, 1.0×10⁻⁴ | 6.1×10⁻³, 1.7×10⁻⁴ |
   | change in σ_jk, N_jk 8 / 27 / 64 | ≤ 0.08 / 0.06 / 0.07% | ≤ 0.06 / 0.07 / 0.06% |
   | time, dense mock (34k galaxies), N_jk = 27, 1 core | 75 s | 37 s |
   | peak memory, dense | 1.0 GB | 2.1 GB |

   For scale: measureia's whole r–μ jackknife pair counting on the same mock takes 52 s; the
   loop prototype took ~12 min per N_jk.
5. **Dense mock size:** record the galaxy counts with the rerun (an earlier draft said 64k
   galaxies, which does not match n = 10⁻² at L = 150 Mpc/h for one sample).
