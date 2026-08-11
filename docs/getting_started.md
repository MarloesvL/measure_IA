# Getting started

**MeasureIA** measures intrinsic-alignment correlation functions — the projected correlations $w_{gg}$ and
$w_{g+}$ and their multipoles $\tilde\xi_{gg,0}$, $\tilde\xi_{g+,2}$ — together with their jackknife
covariance. It works on two kinds of data:

| | Class | Data | Randoms | Shapes | Jackknife |
|---|---|---|---|---|---|
| **Box** | `MeasureIABox` | Cartesian positions in a periodic box | analytic | axis direction + axis ratio `q` | sub-boxes ($x^3$) |
| **Lightcone** | `MeasureIALightcone` | sky coordinates (RA, DEC, redshift) | explicit random catalogue | ellipticity/shear `e1`, `e2` | sky patches (k-means) |

Use **`MeasureIABox`** for periodic hydrodynamic simulation snapshots, and **`MeasureIALightcone`** for
lightcone / survey-like data where you have a random catalogue and shear-style shape measurements.

## Install

See [Installation](installation.md). In short (Python 3.10–3.14):

```bash
pip install measureia
```

## A first measurement

**Box:**

```python
from measureia import MeasureIABox
import numpy as np

data = {
	"Position": np.array([]), "Position_shape_sample": np.array([]),
	"Axis_Direction": np.array([]), "q": np.array([]), "LOS": 2,
}
mi = MeasureIABox(data, output_file_name="./out.hdf5", boxsize=205.0)
mi.measure_xi_w(dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path="./")
```

**Lightcone:**

```python
from measureia import MeasureIALightcone
import numpy as np

data = {
	"RA": np.array([]), "DEC": np.array([]), "Redshift": np.array([]),
	"RA_shape_sample": np.array([]), "DEC_shape_sample": np.array([]),
	"Redshift_shape_sample": np.array([]), "e1": np.array([]), "e2": np.array([]),
}
randoms_data = {"RA": np.array([]), "DEC": np.array([]), "Redshift": np.array([])}
mi = MeasureIALightcone(data, randoms_data, output_file_name="./out.hdf5")
mi.measure_xi_w("galaxies", dataset_name="ds1", corr_type="both", num_jk=27, temp_file_path="./")
```

## Where to go next

- **[Input](input.md)** — the full data dictionaries for the box and the lightcone.
- **[Usage](usage.md)** — worked examples, including multipoles and multiprocessing.
- **[Conventions](conventions.md)** — the shape/sign conventions (especially the `e1`/`e2` convention).
- **[Estimator definitions](estimator_definitions.md)** — the mathematics of the estimators.
- **[Output structure](output_structure.md)** — how results are stored in the HDF5 output file.
