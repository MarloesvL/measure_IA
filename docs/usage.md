# Usage & Examples

You can see examples in the repository under `examples/`, e.g.:

- `example_measure_IA_box.py`
- `example_measureIA_box.ipynb`

Here is a minimal usage snippet:

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

mi.measure_xi_w(dataset_name="ds1", corr_type="both", num_jk=27)
mi.measure_xi_multipoles(dataset_name="ds1", corr_type="both", num_jk=27)
```