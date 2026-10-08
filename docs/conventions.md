# Conventions

This page collects the geometric and shape conventions used throughout MeasureIA, for both the box and
the lightcone. The mathematical estimators that use these quantities are defined on the
[Estimator definitions](estimator_definitions.md) page.

## Separation vector

For every position–shape pair the separation vector is defined as

$$\mathbf{s} = \mathbf{x}_\mathrm{shape} - \mathbf{x}_\mathrm{position}\,,$$

i.e. pointing *from* the position (density) object *to* the shape object. This ordering is used consistently
across all backends (box and lightcone, brute-force and tree). The line-of-sight separation $\Pi$ is the
component of $\mathbf{s}$ along the line of sight and is binned signed over $(-\Pi_\mathrm{max}, +\Pi_\mathrm{max})$;
the projected separation $r_p$ is the perpendicular component. In the box the line of sight is a fixed
coordinate axis; in the lightcone it is defined per pair, as described under
[Lightcone geometry](#lightcone-geometry).

## Binning

The separation bins are fixed at initialisation and shared by every measurement on the object:

- **Transverse / 3D separation** ($r_p$ for $w$, $r$ for the multipoles): `num_bins_r` **logarithmic** bins
  between `separation_limits[0]` and `separation_limits[1]` (i.e. $r_\mathrm{min}$ and $r_\mathrm{max}$).
  Pass `binning='linear'` to the constructor for **linear** bins instead (e.g. around the BAO peak).
- **Line of sight** $\Pi$: `num_bins_pi` **linear** bins spanning the *signed* range
  $[-\Pi_\mathrm{max}, +\Pi_\mathrm{max}]$, with $\Pi_\mathrm{max}$ set by `pi_max`.
- **$\mu_r = \Pi/r$** (used for the multipoles): `num_bins_pi` **linear** bins over $[-1, 1]$.

The bin coordinates written to the output (`*_rp`, `*_pi`, `*_r`, `*_mu_r`) are the bin **midpoints**.
Separations are in the units of the input coordinates: for a box initialised with the internal `simulation`
option these are Mpc/$h$; for the lightcone the comoving distances are computed from the redshifts using the
chosen cosmology and can be converted with the `over_h` argument.

## Lightcone geometry

The lightcone is measured on the curved sky in full 3D. MeasureIA makes **no flat-sky or small-angle
approximation**: there is no projection onto a tangent plane, and separations are never built from an
angle times a distance.

### Positions

Each object's `RA`, `DEC` and `Redshift` become a 3D comoving position vector

$$\mathbf{x} = \chi(z)\,\hat{\mathbf{n}}(\mathrm{RA}, \mathrm{DEC})\,,\qquad
\hat{\mathbf{n}} = (\cos\delta\cos\alpha,\ \cos\delta\sin\alpha,\ \sin\delta)\,,$$

where $\chi(z)$ is the comoving radial distance computed with
[CCL](https://ccl.readthedocs.io/) for the `cosmology` passed to the measurement method. Without one, a
default flat $\Lambda$CDM cosmology is used ($\Omega_c = 0.225$, $\Omega_b = 0.045$, $h = 0.7$,
$\sigma_8 = 0.8$, $n_s = 1$). Distances are in Mpc, or Mpc/$h$.

The redshift is converted to distance as given. If your redshifts include peculiar velocities
(observed redshifts), the redshift-space distortions carry through into $\Pi$ and $\mu_r$; if they are
cosmological redshifts, they do not.

### Line of sight

Each pair gets its own line of sight, the direction to the **comoving midpoint** of the pair:

$$\hat{\mathbf{n}}_\mathrm{LOS} = \frac{\mathbf{x}_\mathrm{position} + \mathbf{x}_\mathrm{shape}}
{\left|\mathbf{x}_\mathrm{position} + \mathbf{x}_\mathrm{shape}\right|}\,.$$

This is the midpoint of the two 3D positions, *not* the angular bisector
($\hat{\mathbf{n}}_1 + \hat{\mathbf{n}}_2$), and *not* the direction to either galaxy (the endpoint
line of sight). The three agree when the pair's angular separation is small and differ in the wide-angle
regime, so keep this in mind when comparing against a code that uses another choice.

### Separations

With $\mathbf{s} = \mathbf{x}_\mathrm{shape} - \mathbf{x}_\mathrm{position}$ as [above](#separation-vector),

$$\Pi = \mathbf{s}\cdot\hat{\mathbf{n}}_\mathrm{LOS}\,,\qquad
r_p = \sqrt{|\mathbf{s}|^2 - \Pi^2}\,,\qquad
\mathbf{s}_\perp = \mathbf{s} - \Pi\,\hat{\mathbf{n}}_\mathrm{LOS}\,.$$

- $\Pi$ is signed: it is positive when the shape object lies farther along the line of sight than the
  position object. The `pi_max` window is applied to this comoving $\Pi$.
- $r_p$ is a 3D comoving transverse distance, not an angular separation times a distance (such as
  $\theta\,\chi$ or TreeCorr's `Rperp`). Pairs near bin edges can therefore fall into different bins in
  MeasureIA and in an angle-based code; see [Validation](validation.md).
- For the multipoles, $r = |\mathbf{s}|$ is the full 3D separation and $\mu_r = \Pi / r$, with the same
  midpoint $\hat{\mathbf{n}}_\mathrm{LOS}$.

### Projection of the shapes

The orientation $\phi$ used to build $(e_+, e_\times)$ is the angle of the projected separation
$\mathbf{s}_\perp$ in a galaxy's local (east, north) sky basis,

$$\hat{\mathbf{e}}_\mathrm{east} = (-\sin\alpha,\ \cos\alpha,\ 0)\,,\qquad
\hat{\mathbf{e}}_\mathrm{north} = (-\sin\delta\cos\alpha,\ -\sin\delta\sin\alpha,\ \cos\delta)\,,$$

$$\phi = \operatorname{arctan2}\!\left(\mathbf{s}_\perp\cdot\hat{\mathbf{e}}_\mathrm{north},\
\mathbf{s}_\perp\cdot\hat{\mathbf{e}}_\mathrm{east}\right).$$

$\mathbf{s}_\perp$ is perpendicular to the *midpoint* line of sight, while each basis is tangent to the
sphere at *one galaxy*, so the two planes differ slightly for a pair at finite angular separation. Which
galaxy's basis is used differs between $w_{g+}$ and the shape–shape terms; see
[Which tangent frame each shape is projected in](#lightcone-measureialightcone) below.

MeasureIA applies no wide-angle correction. The measured statistic is exact for the line of sight
defined above, with no flat-sky term left to correct. If the model you compare against assumes the
plane-parallel limit, any wide-angle terms belong in that model, and they should use the same midpoint
line of sight.

### Comparison with the box

The box uses a single fixed line of sight, the coordinate axis given by `LOS`: the plane-parallel
(distant-observer) limit. The lightcone reduces to it when the angular extent of the pairs is small, and
the box ↔ lightcone row of the [Validation](validation.md) table checks that the two agree in that limit.

Lightcone jackknife patches are built on the sphere too: they are clustered by great-circle distance
(k-means on unit vectors), not on a flat (RA, DEC) plane, so the patches stay compact near the poles.

## Radial ($+$) and cross ($\times$) shape components

The alignment signal is built from the shape components measured relative to the separation vector,

$$(e_+,\, e_\times) = \epsilon\,[\cos 2\phi,\ \sin 2\phi]\,,$$

where $\phi$ is the orientation of the projected separation vector relative to the shape (see below) and
$\epsilon$ is the shape magnitude (the [ellipticity](#ellipticity-definitions) of the object).

**Sign convention (intrinsic alignment).** MeasureIA uses the intrinsic-alignment sign convention in which
$e_+ > 0$ means the major axis of the shape points *along* the separation vector (radial alignment). Radial
alignment therefore produces $w_{g+} > 0$. Note that this is the opposite sign to the weak-lensing tangential
shear: $e_+ = -\gamma_t$.

## Shape input: box vs lightcone

The two entry points differ in *how* the shapes are supplied, but both end up as $(e_+, e_\times)$ through the
relation above.

### Box (`MeasureIABox`)

Shapes are given as a projected **axis direction** and an **axis ratio**:

- `Axis_Direction` — the unit vector of the projected semi-major axis of each shape object;
- `q` — the projected axis ratio $q = b/a$ (with $a > b$).

Here $\phi$ is the angle between the projected separation vector $r_p$ and the semi-major axis direction of the
object, computed per position–shape pair. The magnitude $\epsilon$ follows from $q$ (see below). This branch is
radial-positive by construction.

!!! note "The sign of `Axis_Direction` does not matter"
    A semi-major axis has no head and no tail: $\hat a$ and $-\hat a$ describe the same shape. MeasureIA
    therefore never forms $\phi$ itself, and instead builds $\cos 2\phi$ and $\sin 2\phi$ directly from the
    dot and 2D cross products of $\hat a$ with the unit separation direction,

    $$\cos 2\phi = 2(\hat a\cdot\hat s)^2 - 1\,,\qquad \sin 2\phi = 2(\hat a\cdot\hat s)(\hat a\times\hat s)\,.$$

    Both are invariant under $\hat a\to-\hat a$, so you may supply whatever sign convention your shape code
    emits — canonicalised (say, a positive first component) or not — and every output is bit-for-bit
    identical. Recovering $\phi$ with `arccos` would **not** have this property: it folds the angle into
    $[0,\pi]$, which leaves $\cos 2\phi$ alone but flips $\sin 2\phi$, so $e_\times$ would depend on an
    arbitrary input choice. Versions before this fix did exactly that; see the changelog.

### Lightcone (`MeasureIALightcone`)

Shapes are given directly as the two **ellipticity/shear components** `e1` and `e2`. These must follow the
**standard survey shear-catalogue convention**: the components are defined on the local $(\mathrm{RA}, \mathrm{DEC})$
axes, exactly as delivered by e.g. *lensfit*/*metacal*-style catalogues and as expected by
[TreeCorr](https://rmjarvis.github.io/TreeCorr/). Internally the radial and cross components are then

$$
e_+ = e_1 \cos 2\phi - e_2 \sin 2\phi\,, \qquad
e_\times = e_1 \sin 2\phi + e_2 \cos 2\phi\,,
$$

where $\phi = \operatorname{arctan2}(\text{north}, \text{east})$ is the orientation of the projected separation
vector in the internal (east, north) sky frame. As in the box case, the output $w_{g+}$ is radial-positive
($e_+ > 0$ for radial alignment).

!!! note "Relation to TreeCorr"
    TreeCorr reports the tangential shear $\gamma_t$, so its $g$ has the opposite sign to the IA $e_+$
    ($e_+ = -\gamma_t$). Comparing a MeasureIA lightcone $w_{g+}$ against a TreeCorr `NG` measurement therefore
    needs only the standard IA flip $g \to -g$; no per-component sign change is applied to `e1`/`e2`.
    Note also that a *wrong chirality* (swapping the handedness of `e2`) does not simply flip the sign of
    $w_{g+}$ — it replaces $\cos 2(\phi_a - \phi_s)$ with $\cos 2(\phi_a + \phi_s)$ and washes the signal out
    to noise, which is a common cause of a "vague, noisy mismatch" against other codes.

!!! note "Which tangent frame each shape is projected in"
    On the curved sky the local (east, north) basis differs from galaxy to galaxy, so a pair has two
    of them. For $w_{g+}$ MeasureIA projects the shape galaxy's `e1`/`e2` in the tangent frame of its
    **position-sample partner** — a plane-of-the-pair approximation, and the source of part of the
    residual against TreeCorr documented in `validation/README.md`.

    The shape–shape terms instead project **each galaxy in its own frame**, because the pair is
    symmetric there and no single partner frame is privileged. The two conventions therefore coexist
    deliberately; they agree in the plane-parallel limit and differ by curvature terms of the same
    order as the other separation-definition differences. The box has no such ambiguity: its
    projection plane is fixed by `LOS`.

## Ellipticity definitions

The shape magnitude $\epsilon$ is derived from the axis ratio $q$, so this choice applies to the **box** only;
it is selected with the `ellipticity` argument of `MeasureIABox.measure_xi_w` and
`MeasureIABox.measure_xi_multipoles`. The lightcone takes `e1`/`e2` directly and has no such argument. The two
definitions are:

- `'distortion'` (default): $\epsilon = \dfrac{1 - q^2}{1 + q^2}$
- `'ellipticity'`: $\epsilon = \dfrac{1 - q}{1 + q}$

## Responsivity

When shapes are raw distortions/ellipticities (as in the box case, derived from axis ratios), the $g+$ signal is
calibrated by the responsivity factor $2\mathcal{R}$, with

$$\mathcal{R} = \frac{\sum_i w_i\left(1 - \epsilon_i^2/2\right)}{\sum_i w_i} = 1 - \langle \epsilon^2\rangle/2\,,$$

so that $S_+D = \sum w_i w_j\, e_+(j|i)/(2\mathcal{R})$ (see [Estimator definitions](estimator_definitions.md)).
The correction is controlled by the `responsivity` argument: it defaults to `True` for the box (raw shapes) and
`False` for the lightcone (where `e1`/`e2` are assumed to be already-calibrated shears). When switched off,
$\mathcal{R} = 0.5$ so that $2\mathcal{R} = 1$ and no calibration is applied. Only the $g+$ correlations are
affected; the clustering ($gg$) signal is unchanged.

### Shape-shape: one factor per sample

A shape–shape product has two spin-2 factors, so it carries **two** responsivities — one per sample,
each computed over its own weights:

$$S_+S_+ = \sum w_i w_j \frac{e_+(j|i)}{2\mathcal{R}_j}\frac{e_+(i|j)}{2\mathcal{R}_i}\,.$$

The two generally differ, since the density and shape samples have different shape-noise properties;
in the auto case (the same catalogue in both slots) they coincide and the factor reduces to
$(2\mathcal{R})^2$. Switching `responsivity` off sets both to $0.5$, rescaling the shape–shape
products by exactly $(2\mathcal{R}_i)(2\mathcal{R}_j)$ — quadratically rather than linearly as for
$g+$.

The differing defaults are deliberate rather than an oversight: they match what each input format usually
contains, so the common case needs no argument. Pass `responsivity` explicitly whenever your inputs do not
follow that pattern — for example `responsivity=True` on the lightcone when your `e1`/`e2` are raw
distortions rather than calibrated shears, or `responsivity=False` on the box when the shapes you supply
have already been calibrated.
