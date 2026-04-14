# Input

On this page you will find everything you need to know about the input in the data dictionary that needs to be
provided when the MeasureIABox object is initialised.

As shown on the [Usage](usage.md) page, the input dictionary has the following structure:

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