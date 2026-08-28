# Output file structure
Your output file with your own input of [output_file_name, snapshot, dataset_name, num_jk] will have the following structure:

```
[output_file_name]  
└── Snapshot_[snapshot]                                 Optional. If input [snapshot] is None, this group is omitted.
	├── w_gg
	│	├── [dataset_name]								w_gg values for each r_p bin
	│	├── [dataset_name]_rp							r_p mean bin values
	│	├── [dataset_name]_mean_[num_jk]				mean w_gg value of all jackknife realisations
	│	├── [dataset_name]_jackknife_cov_[num_jk]		jackknife estimate of covariance matrix
	│	├── [dataset_name]_jackknife_[num_jk]			sqrt of diagonal of covariance matrix (size of errorbars)
	│	└── [dataset_name]_jk[num_jk]					group containing all jackknife realisations for this dataset
	│		├── [dataset_name]_[i]						jackknife realisations with i running from 0 to num_jk - 1
	│		└── [dataset_name]_[i]_rp					r_p bin values of each jackknife realistation
	├── w_g_plus
	│	├── [dataset_name]								w_g+ values for each r_p bin
	│	├── [dataset_name]_rp							r_p mean bin values
	│	├── [dataset_name]_mean_[num_jk]				mean w_g+ value of all jackknife realisations
	│	├── [dataset_name]_jackknife_cov_[num_jk]		jackknife estimate of covariance matrix
	│	├── [dataset_name]_jackknife_[num_jk]			sqrt of diagonal of covariance matrix (size of errorbars)
	│	└── [dataset_name]_jk[num_jk]					group containing all jackknife realisations for this dataset
	│		├── [dataset_name]_[i]						jackknife realisations with i running from 0 to num_jk - 1
	│		└── [dataset_name]_[i]_rp					r_p bin values of each jackknife realistation
	└──  w
		├── xi_gg
		│	├── [dataset_name]							xi_gg grid in (r_p,pi)
		│	├── [dataset_name]_rp						r_p mean bin values
		│	├── [dataset_name]_pi						pi mean bin values
		│	├── [dataset_name]_RR_gg					RR grid in (r_p,pi)
		│	├── [dataset_name]_DD						DD grid in (r_p,pi) (pair counts)
		│	└── [dataset_name]_jk[num_jk]				group containing all jackknife realisations for this dataset
		│		├── [dataset_name]_[i] 					jackknife realisations with i running from 0 to num_jk - 1
		│		└── [dataset_name]_[i]_[x]				with x in [rp, pi, RR_gg, DD] as above
		├── xi_g_plus
		│	├── [dataset_name]							xi_g+ grid in (rp_,pi)
		│	├── [dataset_name]_rp						r_p mean bin values
		│	├── [dataset_name]_pi						pi mean bin values
		│	├── [dataset_name]_RR_g_plus				RR grid in (r_p,pi)
		│	├── [dataset_name]_SplusD					S+D grid in (r_p,pi)
		│	└── [dataset_name]_jk[num_jk]				group containing all jackknife realisations for this dataset
		│		├── [dataset_name]_[i] 					jackknife realisations with i running from 0 to num_jk - 1
		│		└── [dataset_name]_[i]_[x]				with x in [rp, pi, RR_g_plus, SplusD] as above
		└── xi_g_cross
			├── [dataset_name]							xi_gx grid in (r_p,pi)
			├── [dataset_name]_rp						r_p mean bin values
			├── [dataset_name]_pi						pi mean bin values
			├── [dataset_name]_RR_g_cross				RR grid in (r_p,pi)
			├── [dataset_name]_ScrossD					SxD grid in (r_p,pi) (pair counts)
			└── [dataset_name]_jk[num_jk]				group containing all jackknife realisations for this dataset
				├── [dataset_name]_[i] 					jackknife realisations with i running from 0 to num_jk - 1
				└── [dataset_name]_[i]_[x]				with x in [rp, pi, RR_g_cross, ScrossD] as above

```
The pair-count dataset names above (`RR_gg`, `RR_g_plus`, `RR_g_cross`, `DD`, `SplusD`, `ScrossD`) are the
**box** ones; the lightcone writes a different set, listed under [Box vs lightcone](#box-vs-lightcone) below.
The `xi_g_cross` group holds the parity null test $\xi_{g\times}$ and, unlike `xi_gg` and `xi_g_plus`, is
written for the full sample only - there are no per-realisation jackknife entries for it in either geometry.

If you choose to measure multipoles instead of wg+, all 'w' will be replaced by 'multipoles' - or both will appear, if you have measured both.
For the multipoles, all xi_g+, DD (etc) grids are in (r, mu_r), not in (r_p, pi) and the suffixes of the bin values are also replaced by '_r' and '_mu_r' accordingly.
In one file, multiple redshift (snapshot) measurements can be saved without being overwritten, as well as the jackknife
information for different numbers of jackknife realisations (num_jk) for the same dataset.

## Box vs lightcone

The structure above applies to both `MeasureIABox` and `MeasureIALightcone`. Two things differ:

- The `Snapshot_[snapshot]` group only appears for the box, when a `snapshot` label is given at initialisation;
  the lightcone has no snapshot grouping and this level is omitted.
- The random–random grids are computed **analytically** from the sample sizes and volume for the box, and
  counted from the **explicit random catalogues** for the lightcone. Bins with zero empirical random–random
  pairs are left as `NaN` (the estimator is undefined there) and trigger a warning advising more randoms —
  see the [Estimator definitions](estimator_definitions.md).
- **The pair-count datasets therefore differ by geometry.** The box stores one analytic grid per group, named
  after that group; the lightcone stores a single `RR` grid plus the density–random and shape–random terms of
  the Landy–Szalay estimators:

| group | box | lightcone |
|---|---|---|
| `xi_gg` | `[dataset_name]_DD`, `[dataset_name]_RR_gg` | `[dataset_name]_DD`, `[dataset_name]_RR`, `[dataset_name]_RD`, `[dataset_name]_SR` |
| `xi_g_plus` | `[dataset_name]_SplusD`, `[dataset_name]_RR_g_plus` | `[dataset_name]_SplusD`, `[dataset_name]_SplusR` |
| `xi_g_cross` | `[dataset_name]_ScrossD`, `[dataset_name]_RR_g_cross` | `[dataset_name]_ScrossD`, `[dataset_name]_ScrossR` |

## Per-galaxy contributions

`measure_galaxy_contributions` (box only, see [Per-galaxy contributions](galaxy_contributions.md)) writes to a
separate top-level `galaxy_contributions` group, under `w` or `multipoles` according to its `statistic`
argument. With `M` shape galaxies and `K` the largest number of jackknife patches any single galaxy has pairs
in:

```
galaxy_contributions
└── multipoles                                          'w' when statistic="w"
	├── [dataset_name]_Y							(M, num_bins_r)   per-galaxy contribution; sums over
	│												  galaxies to the ordinary estimator
	├── [dataset_name]_P							(M, num_bins_r)   pairs each galaxy contributed through
	├── [dataset_name]_r							(num_bins_r,)     r (or r_p) mean bin values
	└── [dataset_name]_jk[num_jk]					only when num_jk > 0
		├── Y_jk_values							(M, K, num_bins_r)  Y decomposed by the position partner's
		│											  patch, stored raw (no 2R, no RR amplitude)
		├── P_jk_values							(M, K, num_bins_r)  the same decomposition of P
		├── jk_patches							(M, K)              which patch each stored column is
		├── jk_shape							(M,)                the patch of each shape galaxy
		├── R_jk								(num_jk,)           delete-one responsivities
		├── rr_ratio							(num_jk,)           RR_jk / RR amplitude ratios
		└── attribute 'R'						full-sample responsivity
```

The jackknife arrays are sparse along the patch axis: only the patches a galaxy actually has pairs in are
stored, which is what `jk_patches` records. Use `delete_one_estimator` / `jk_columns` to read them rather
than indexing the patch axis directly.

## Reading the output file

The output file is a plain HDF5 file, so it can be read with `h5py` using the paths above. The package also
ships a [`ReadData`](api/ReadData.md) class that knows this structure and takes care of the optional
`Snapshot_[snapshot]` group for you:

```python
from measureia import ReadData

reader = ReadData(
    simulation=None,            # simulation tag used at measurement time (None if the boxsize was given directly)
    catalogue="example_IA_box",  # output file name *without* the .hdf5 extension
    snapshot=None,              # snapshot label; 99 would select the "Snapshot_99" group
    data_path="./",             # folder holding <catalogue>.hdf5
)
```

The file that is read is `data_path + catalogue + ".hdf5"`, and every dataset path is taken relative to
`Snapshot_[snapshot]/` when a `snapshot` is given (and relative to the file root when it is not).

### `read_MeasureIA_output(dataset_name, num_jk)`

The convenience route: it looks for all four correlation functions of one dataset and fills the corresponding
attributes on the object, leaving whatever is not in the file at `None`.

```python
reader.read_MeasureIA_output("mock", 27)   # dataset_name and num_jk of the run that wrote the file

plt.errorbar(reader.rp, reader.w_gp, yerr=reader.errors_w_gp)
```

| attribute | contents |
| --- | --- |
| `rp`, `r` | mean bin values of the projected (`w`) and 3D (`multipoles`) statistics |
| `w_gg`, `w_gp` | `w_gg` and `w_g+` per `r_p` bin |
| `multipoles_gg`, `multipoles_gp` | the monopole of `xi_gg` and the quadrupole of `xi_g+` per `r` bin |
| `cov_w_gg`, `cov_w_gp`, `cov_multipoles_gg`, `cov_multipoles_gp` | jackknife covariance matrices |
| `errors_w_gg`, `errors_w_gp`, `errors_multipoles_gg`, `errors_multipoles_gp` | sqrt of the diagonal of those covariances, i.e. the error bars |

Notes:

- `num_jk` must match the number of jackknife regions of the run that wrote the file, since it is part of the
  dataset names. Pass `num_jk=None` to read only the measurements and skip the covariance.
- Statistics that were never measured (e.g. the multipoles, if only `measure_xi_w` was called) stay `None`,
  and so do the covariance attributes when the run used `num_jk=0`. All attributes are reset at the start of
  every call, so the same object can be reused for several datasets.
- The method reads the final correlation functions only. The pair-count grids under `w`/`multipoles` and the
  individual jackknife realisations are not touched; use `read_cat` for those.

### `read_cat(dataset_name, cut=None, indices=None)`

The direct route: it returns one dataset as an array, by name. The group it sits in is given with `sub_group`
at initialisation, which lets you reach anything in the file, including the terms of the estimators and the
individual jackknife realisations:

```python
reader = ReadData(None, "example_IA_box", None, sub_group="w_g_plus/", data_path="./")
wgp = reader.read_cat("mock")                      # w_g+ values
rp = reader.read_cat("mock_rp")                    # r_p bin values
wgp_error = reader.read_cat("mock_jackknife_27")   # error bars

grids = ReadData(None, "example_IA_box", None, sub_group="w/xi_g_plus/", data_path="./")
splusd = grids.read_cat("mock_SplusD")             # S+D pair counts on the (r_p, pi) grid
realisation_0 = grids.read_cat("mock_jk27/mock_0")  # first jackknife realisation
```

Pass `cut=[start, stop]` to read a slice, or `indices` to read a selection of elements, instead of the whole
dataset.

### `read_modelling_outputs(catalogue)`

A helper for the results of a subsequent modelling step: it reads `A_IA`/`b_g` amplitudes and their errors
that were stored as HDF5 attributes on the `w` and/or `multipoles` groups of `catalogue`, into
`w_A_IA`, `w_A_IA_err`, `w_b_g`, `w_b_g_err` and their `multipoles_` counterparts. These attributes are not
written by the measurement code itself.

A worked example of both reading routes is in `examples/example_read_and_plot.py` and in the example
notebooks — see [Usage](usage.md).
