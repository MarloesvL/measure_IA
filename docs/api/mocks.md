# mocks

Seeded synthetic catalogues with a known, non-null intrinsic alignment signal. They need no
simulation or survey data, which makes them the quickest way to try MeasureIA out, and they are
what the [validation](../validation.md) comparisons and the `examples/` scripts run on.

```python
from measureia import MeasureIABox
from measureia.mocks import radial_alignment_box_mock

mock = radial_alignment_box_mock(n_centrals=600, n_sat=8)
data = {k: mock[k] for k in ("Position", "Position_shape_sample", "Axis_Direction", "q", "LOS")}
ia = MeasureIABox(data, "./out.hdf5", boxsize=mock["boxsize"], separation_limits=[0.3, 8.0])
ia.measure_xi_w("mock", "both", num_jk=27, temp_file_path="./")
```

::: measureia.mocks
    handler: python
    options:
      show_source: true
      members_order: source
      show_root_heading: true
      heading_level: 2
