# Input

MeasureIA has two entry points, which expect different input dictionaries:

- **`MeasureIABox`** — for periodic simulation boxes, using Cartesian positions and projected shapes
  (axis direction + axis ratio). See [Box input](#box-input) below.
- **`MeasureIALightcone`** — for lightcone data, using sky coordinates and ellipticity/shear components,
  together with a random catalogue. See [Lightcone input](#lightcone-input) below.

Both accept optional per-object weights and allow you to rename the dictionary keys (see
[Custom key names](#custom-key-names)). The shape conventions common to both are described on the
[Conventions](conventions.md) page.

## Box input

The `MeasureIABox` object is initialised with a single `data` dictionary of the following structure:

```python
data_dict = {
	"Position": np.array([]),
	"Position_shape_sample": np.array([]),
	"Axis_Direction": np.array([]),
	"q": np.array([]),
	"LOS": 2,
	"weight": np.array([]),
	"weight_shape_sample": np.array([]),
}
```

Here, the 'Position' key contains an array with the coordinates of the position sample
(see [Estimator definitions](estimator_definitions.md) for the definition of this sample), where each row corresponds
to an object and the columns are the $x,y,z,$ coordinates.
Note that the coordinates are assumed to be $\in$ \[0,boxsize\] and that the units need to be the same as those of the
given boxsize. When using an initialisation of MeasureIABox with the internal 'simulation' input option, the
boxsize will be in Mpc/$h$.
In the same fashion, the 'Position_shape_sample' key contains the array with the coordinates of the shape sample,
or the second position sample in case only clustering is measured.

The 'Axis_Direction' key contains an array with the components of the unit vector corresponding to the direction of the
projected axis with respect to which the measurement is done.
This is the axis direction vector used in the calculation of $\phi$, as defined
in [Estimator definitions](estimator_definitions.md).
Once again, the array rows correspond to objects in the shape sample, assuming the same ordering as '
Position_shape_sample'.
The columns are the two components of the unit vector, corresponding for example to $x$ and $y$ if the projection axis
is $z$.
Normalisation is assumed, but also enforced in case the vector given does not have a unit length.

The 'q' key contains an array with the axis ratios of the objects in the shape sample.
As described in [Estimator definitions](estimator_definitions.md), $q$ is defined as $q=b/a$ with $a,b$ (with $a>b$) the
projected axis
lengths of the object.
Again, the same ordering as 'Position_shape_sample' is assumed.

The 'LOS' key contains an integer corresponding to the index of the line-of-sight axis of the
'Position' and 'Position_shape_sample' arrays.
Note that this needs to be consistent with the axis used to project the shapes.
For example, if the shapes are projected over the $z$ axis and the 'Position' and 'Position_shape_sample' array columns
are ordered $x,y,z$, 'LOS' needs to have a value of 2 (if $x$, it would be 0 etc.).

The 'weight' and 'weight_shape_sample' keys are optional array inputs where a weight per object for the position and
shape
samples, respectively, can be added.
The ordering is assumed to be the same as in the 'Position' and 'Position_shape_sample' arrays and normalisation is
not enforced. See the [Estimator definitions](estimator_definitions.md) page for how these weights are included in the
pair counts.

## Lightcone input

For lightcone data, use the `MeasureIALightcone` class instead. It is initialised with two dictionaries,
`data` and `randoms_data`, using sky coordinates rather than Cartesian positions.

The `data` dictionary has the following structure:

```python
data_dict = {
	"RA": np.array([]),                 # position (density) sample RA  [deg, 0-360]
	"DEC": np.array([]),                # position (density) sample DEC [deg, -90..90]
	"Redshift": np.array([]),           # position (density) sample redshift
	"RA_shape_sample": np.array([]),    # shape sample RA
	"DEC_shape_sample": np.array([]),   # shape sample DEC
	"Redshift_shape_sample": np.array([]),  # shape sample redshift
	"e1": np.array([]),                 # first ellipticity component of the shape sample
	"e2": np.array([]),                 # second ellipticity component of the shape sample
	"weight": np.array([]),                 # optional, position sample
	"weight_shape_sample": np.array([]),    # optional, shape sample
}
```

The `RA`/`DEC`/`Redshift` keys describe the position (density) sample and the `*_shape_sample` keys the
shape sample, each row corresponding to one object with the same ordering within a sample. `RA` is expected
in degrees $\in [0, 360]$ and `DEC` in degrees $\in [-90, 90]$. Comoving distances are computed internally
from the redshifts using the cosmology passed to the measurement method (a default $\Lambda$CDM cosmology is
used if none is given).

Instead of the axis direction and axis ratio used in the box case, the shapes are provided directly as the
two ellipticity (or shear) components `e1` and `e2`. See the [Estimator definitions](estimator_definitions.md)
page for the sign/chirality convention and the `responsivity` option that controls the $2R$ shape
calibration. The `weight` and `weight_shape_sample` keys are optional, as in the box case.

The `randoms_data` dictionary provides the random catalogues used for the pair counts:

```python
randoms_dict = {
	"RA": np.array([]),
	"DEC": np.array([]),
	"Redshift": np.array([]),
	# optionally also RA_shape_sample / DEC_shape_sample / Redshift_shape_sample
}
```

If only `RA`, `DEC` and `Redshift` are given, the same random sample is used for both the position and the
shape random terms; provide the `*_shape_sample` keys as well to use a separate random catalogue for the
shape sample.

## Custom key names

The default key names above can be overridden at initialisation, so you can pass dictionaries that already
use your own naming without copying the arrays. Every key has a corresponding `*_name` constructor argument;
for example, for the box:

```python
mi = MeasureIABox(
	data=data_dict,
	output_file_name="./outfile.hdf5",
	boxsize=205.0,
	positions_density_sample_name="pos",
	axis_ratio_name="axis_ratio",
	line_of_sight_index_name="los",
)
```

The `MeasureIALightcone` constructor accepts the analogous `RA_density_sample_name`, `DEC_shape_sample_name`,
`redshift_density_sample_name`, `e1_name`, `e2_name`, `weight_shape_sample_name`, and so on. The same custom
names are also understood in the `masks` (and `masks_randoms`) dictionaries passed to the measurement methods.