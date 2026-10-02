# Usage & Examples

The two example notebooks are rendered with their output on this site - [simulation
box](examples/example_MeasureIA_box.ipynb) and [lightcone](examples/example_MeasureIA_lightcone.ipynb) - and
each page has a download link at the top. To run one without installing anything, open it in Google Colab:

[![Box example in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarloesvL/measure_IA/blob/main/examples/example_MeasureIA_box.ipynb) simulation box &nbsp;
[![Lightcone example in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarloesvL/measure_IA/blob/main/examples/example_MeasureIA_lightcone.ipynb) lightcone

Their first cell installs MeasureIA if it is missing, so nothing else is needed: the mock catalogues come with
the package, and cloning this repository is never required.

You can see the same examples, plus the plain scripts, in the repository under `examples/`:

- `example_measure_IA_box.py` / `example_MeasureIA_box.ipynb` — periodic simulation box
- `example_measure_IA_lightcone.py` / `example_MeasureIA_lightcone.ipynb` — lightcone
- `example_read_and_plot.py` — reading a result back with `ReadData` and plotting it with jackknife errors

They run as-is, without any simulation or survey data: each one builds a seeded mock catalogue with a known
radial-alignment signal from `measureia.mocks` (the same mocks the
[validation](validation.md) suite uses), so a full measurement takes about a second and produces a real,
non-null signal. Run them from the `examples/` directory and swap the data dictionary entries for your own
arrays to measure your own data.

See the [Input](input.md) page for a full description of the data dictionaries. Measurements run on multiple
cores by passing `num_nodes > 1` (see `num_nodes` in the box example); the `if __name__ == "__main__":` guard
is required in that case.

## Box

```python
from measureia import MeasureIABox
import numpy as np

data_dict = {
	"Position": np.array([]),
	"Position_shape_sample": np.array([]),
	"Axis_Direction": np.array([]),
	"LOS": 2,
	"q": np.array([])
}

mi = MeasureIABox(
	data=data_dict,
	output_file_name="./outfile.hdf5",
	boxsize=205.0,
)

mi.measure_xi_w(dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path='./')
mi.measure_xi_multipoles(dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path='./')
```

Beyond `measure_xi_w` and `measure_xi_multipoles`, the box also offers
`measure_galaxy_contributions`, which resolves the same estimator per shape galaxy in one pair traversal —
see [Per-galaxy contributions](galaxy_contributions.md).

## Lightcone

For lightcone data, use `MeasureIALightcone`, which takes a `data` and a `randoms_data` dictionary of sky
coordinates. The measurement methods take an additional `IA_estimator` argument (`"clusters"` or
`"galaxies"`) and the jackknife regions are supplied through `jk_patches` (or generated internally with
`num_jk`):

```python
from measureia import MeasureIALightcone
import numpy as np

data = {
	"RA": np.array([]), "DEC": np.array([]), "Redshift": np.array([]),
	"RA_shape_sample": np.array([]), "DEC_shape_sample": np.array([]),
	"Redshift_shape_sample": np.array([]),
	"e1": np.array([]), "e2": np.array([]),
}
randoms_data = {
	"RA": np.array([]), "DEC": np.array([]), "Redshift": np.array([]),
}

mi = MeasureIALightcone(data, randoms_data, output_file_name="./outfile.hdf5")

mi.measure_xi_w("clusters", dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path='./')
mi.measure_xi_multipoles("clusters", dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path='./')
```


## Shape-shape correlations

`corr_type="++"` correlates the shapes of both samples, giving $w_{++}$, $w_{\times\times}$ and the
parity-odd null $w_{+\times}$ in one run, plus the $(4,4)$ multipole of $\xi_{++}$ from
`measure_xi_multipoles`. It needs shapes on the density sample as well, which are supplied through
extra keys in the data dictionary (see [Input](input.md#shape-shape-correlations)):

```python
data_dict = {
	"Position": np.array([]),
	"Position_shape_sample": np.array([]),
	"Axis_Direction": np.array([]),
	"LOS": 2,
	"q": np.array([]),
	"Axis_Direction_density_sample": np.array([]),   # shapes of the density sample
	"q_density_sample": np.array([]),
}

mi = MeasureIABox(data=data_dict, output_file_name="./outfile.hdf5", boxsize=205.0)
mi.measure_xi_w(dataset_name="ds1", corr_type="++", num_jk=27, temp_file_path='./')
```

Use `corr_type="all"` to measure $g+$, $gg$ and $++$ together. `"both"` keeps its original meaning of
$g+$ and $gg$ only, so existing scripts are unaffected. The lightcone works the same way, with
`e1_density_sample` and `e2_density_sample` instead, except that `IA_estimator="clusters"` is not
defined for shape-shape and raises.
