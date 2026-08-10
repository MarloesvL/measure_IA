# pair_kernel

The consolidated pair-accumulation kernel. Every pair-counting loop in the package lives
here; the measurement backend classes are thin wrappers that prepare a sample, call
[`accumulate`][measureia.pair_kernel.accumulate], and do their own reduction / RR / HDF5
writing. See the module docstring for the geometry / binning / backend / jackknife structure.

::: measureia.pair_kernel
    handler: python
    options:
      show_source: true
      members_order: source
      show_root_heading: true
      heading_level: 2
