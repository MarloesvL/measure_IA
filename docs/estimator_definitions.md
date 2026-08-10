# Estimator definitions

This page contains the mathematical definitions of the estimators used in MeasureIA to measure the correlation functions
and the jackknife covariance.

## Projected correlation functions

First of all, we define $w_\mathrm{gg}$ and $w_{\mathrm{g}+}$ as follows:
\begin{align}
w_\mathrm{gk}(r_p) &= \int_{-\Pi_{\text{max}}}^{\Pi_{\text{max}}} \xi_\mathrm{gk}(r_p, \Pi) \text{d}\Pi \\\\
\mathrm{with} \, \mathrm{k} &\in [\mathrm{g},+]\,.
\end{align}
Here, $\Pi_\mathrm{max}$ is the line-of-sight limit you are integrating over, given by the 'pi_max' input value in
the initialisation of a `MeasureIABox` or `MeasureIALightcone` object.

The Landy-Szalay estimator $\xi_\mathrm{gg}$ is defined as follows:
\begin{align}
\xi_{gg} (r_p,\Pi) &= \frac{DD-2DR+RR}{RR}
\end{align}
where $DD$, $RR$ and $DR$ denote the number of object pairs found in an $(r_p,\Pi)$ bin in the data ($DD$),
random data ($RR$) or between the two ($DR$) and are defined as
\begin{align}
DD &= \sum_{\{i,j\}}^{(r_\mathrm{p},\Pi)} w_i w_j \,.
\end{align}
Both $w_i$ and $w_j$ denote the weights of objects in pair ${i,j}$, which are both set to $1$ if not provided.
The weights can be provided using the 'weight' and 'weight_shape_sample' keys in the data dictionary upon
initialisation.
In the **box**, the $RR$ term is calculated analytically using the number of objects in the samples and the
simulation volume. As $DR$ is expected to be equal to $RR$ in simulations, the equation for $\xi_\mathrm{gg}$
reduces to $\xi_\mathrm{gg}=DD/RR-1$. (For the **lightcone**, $RR$ and the $DR$ terms are counted from explicit
random catalogues instead — see [Lightcone estimators](#lightcone-estimators) below.)

The Landy-Szalay estimator $\xi_{\mathrm{g}+}$ is defined as:
\begin{align}
\xi_{g,+}(r_\mathrm{p},\Pi) &= \frac{S_+D - S_+R}{RR} \\
\end{align}
For $\xi_{g+}$, the $S_+R$ term is negligible in simulations, reducing the equation for $\xi_{g+}$ to $\xi_{g+}=S_+D/RR$,
where we define $S_+D$ as follows
\begin{align}
S_+D &= \sum_{\{i,j\}}^{(r_\mathrm{p},\Pi)} w_i w_j \frac{e_{+}(j|i)}{2\mathcal{R}}
\end{align}
The responsivity factor, $\mathcal{R}$, is the weighted average
$\mathcal{R}=\frac{\sum_i w_i\left(1-\epsilon_i^2/2\right)}{\sum_i w_i} = 1-\langle \epsilon^2 \rangle/2$,
with $\epsilon=\frac{1-q^2}{1+q^2}$, the distortion, by default.
The responsivity correction can be switched off with the `responsivity` argument of the measurement methods, in
which case $\mathcal{R}=0.5$ (so $2\mathcal{R}=1$ and no calibration is applied) — appropriate when the input
shapes are already calibrated shears. See the [Conventions](conventions.md) page for more on the shape definitions.
The definition of $\epsilon$ can be changed between 'distortion' and 'ellipticity' ($\epsilon=\frac{1-q}{1+q}$) using
the 'ellipticity' input to 'distortion' or 'ellipticity' when calling the measure_xi_w or measure_xi_multipoles methods
of the MeasureIABox class.
Furthermore, the $+$ and $\times$ components of the ellipticities are measured using:
\begin{align}
(e_{+},e_{\times}) &= \epsilon[\cos(2\phi),\sin(2\phi)]
\end{align}
where $\phi$ denotes the angle between the projected separation vector ($r_\mathrm{p}$) and the semi-major axis
direction of the object for each position-shape object pair.

Once measured in $(r_\mathrm{p},\Pi)$ bins, the estimators $\xi_{gg}$ and $\xi_{g+}$ are integrated over $\Pi$ to obtain
the projected correlation functions following the equations at the top of this Section.
This box case (analytic $RR$) is the default; the lightcone estimators with explicit randoms are described next.
Information on how the output is saved can be found on the [Output file structure](output_structure.md) page.

## Lightcone estimators

For lightcone data (`MeasureIALightcone`) the same projected correlation functions and multipoles are measured,
but the random terms are counted from explicit **random catalogues** rather than computed analytically. Comoving
distances are obtained from the redshifts using the chosen cosmology (see the `cosmology` argument), and the pair
counts $DD$, $S_+D$, $S_+R$, $RR$, $RD$, $SR$ are each normalised by the number of possible pairs in the
corresponding samples.

Two estimator definitions are available through the `IA_estimator` argument, differing in how the $g+$ signal is
normalised:

**`'galaxies'`** — the standard Landy–Szalay form, normalised by the random–random pairs:
\begin{align}
\xi_{gg}(r_p,\Pi) &= \frac{DD - RD - SR}{RR} + 1\,, \\\\
\xi_{g+}(r_p,\Pi) &= \frac{S_+D - S_+R}{RR}\,.
\end{align}

**`'clusters'`** — for cluster/halo shape samples, the $g+$ signal is instead normalised by the clustering pair
counts:
\begin{align}
\xi_{gg}(r_p,\Pi) &= \frac{DD - RD - SR}{RR} + 1\,, \\\\
\xi_{g+}(r_p,\Pi) &= \frac{S_+D}{DD} - \frac{S_+R}{SR}\,.
\end{align}

Here $RD$ and $SR$ are the density–random and shape–random cross pair counts, and $S_+D$/$S_+R$ use the same
$e_+$ (and responsivity) definition as above — note that for the lightcone the `responsivity` correction defaults
to off, since `e1`/`e2` are assumed to be already-calibrated shears (see the [Conventions](conventions.md) page).
In both cases MeasureIA also outputs a **parity null test**, $\xi_{g\times}$, built from the cross component
$e_\times$ in place of $e_+$ (galaxies: $(S_\times D - S_\times R)/RR$; clusters: $S_\times D/DD - S_\times R/SR$),
which is expected to be consistent with zero. The $\xi$ grids are then integrated over $\Pi$ (or $\mu_r$ for the
multipoles) exactly as in the box case.

## Multipoles

For the multipole moment expansion, first introduced in [(Singh et al. 2024)](https://arxiv.org/abs/2307.02545), the
estimators $\xi_\mathrm{gg}$ and $\xi_\mathrm{g+}$, described in the previous Section, are measured in $(r,\mu_r)$ bins.
Here, $r$ refers to the 3D separation length (not the projected as in the previous Section) and $\mu_r$ is defined as:
$$\mu_r = \frac{\Pi}{r}$$ with $\Pi$, the line of sight component of the separation vector as above.
As the $(r,\mu_r)$ bins are spaced differently from the $(r_\mathrm{p},\Pi)$ bins, the $\xi_\mathrm{gg}$
and $\xi_\mathrm{g+}$ are remeasured using the `measure_xi_multipoles` method (of either `MeasureIABox` or
`MeasureIALightcone`) to ensure accuracy. The $\xi_\mathrm{gg}$ and $\xi_\mathrm{g+}$ estimators are integrated over $\mu_r$ using the associated
Legendre polynomials $L^{\ell,s_{ab}}$ to obtain the correlation functions:

$$\tilde{\xi}_\mathrm{gk}^{\ell, s_{ab}} (r) = \frac{2\ell + 1}{2}\frac{(\ell - s_{ab})!}{(\ell + s_{ab})!}\int \text{d} \mu_
{r}L^{\ell,s_{ab}}(\mu_r)\xi_\mathrm{gk}(r, \mu_{r})$$

where we can obtain the correlation between position-position and position-shape samples by filling in the
corresponding $\ell$ and $s_{ab}$ for the prefactors and associated Legendre polynomials $L^{\ell,s_{ab}}$:

$$\tilde{\xi}_\mathrm{gg}^{0,0} (r) = \frac{1}{2}\int \text{d} \mu_{r}L^{0,0}(\mu_r)\xi_\mathrm{gg}(r, \mu_{r})$$

$$\tilde{\xi}_\mathrm{g+}^{2,2} (r) = \frac{5}{48}\int \text{d} \mu_{r}L^{2,2}(\mu_r)\xi_\mathrm{g+}(r, \mu_{r}) .$$

## Covariance

The covariance is estimated using the jack-knife method. The covariance is measured by combining the measurements of
$N_\mathrm{jk}$ jackknife realisations in the following way:

\begin{align}
C_{ij} &= \frac{N_{\mathrm{jk}}-1}{N_{\mathrm{jk}}} \sum_{n=1}^{N_{\mathrm{jk}}} (\psi^n_i - \bar{\psi_i})(\psi^n_j -
\bar{\psi_j}) \\
&\mathrm{with} \ \bar{\psi_i} = \frac{1}{N_{\mathrm{jk}}} \sum_{n=1}^{N_{\mathrm{jk}}}\psi_i^n\\
&\mathrm{and} \ \psi \in [w_{gg},w_{g+},\tilde{\xi}_{gg,0},\tilde{\xi}_{g+,2}] \,.
\end{align}

For **simulation boxes**, $N_\mathrm{jk}$ is the number of sub-boxes, which is related to the box-length $L$
via $N_\mathrm{jk}=x^3$ with $L_\mathrm{sub}=L/x$ and $x$ an integer.
The jack-knife realisations are created by omitting one sub-box from the full volume at a time. In practice, the code
saves the information about each sub-box so that the correlations do not need to be remeasured $N_\mathrm{jk}$ times.

For the **lightcone**, the jackknife regions are instead defined on the sky: the objects are grouped into
$N_\mathrm{jk}$ patches by k-means clustering of the random catalogue (using `kmeans_radec`), and one patch is
omitted at a time. The number of patches is set with the `num_jk` argument, or the patch assignment can be
supplied directly through `jk_patches`; the same delete-one covariance formula above then applies.