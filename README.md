# MeasureIA - The tool for measuring intrinsic alignment correlation functions in hydrodynamic simulations

MeasureIA is a tool that can be used to easily measure intrinsic alignment correlation functions and clustering in simulation boxes.
It includes measurement of wg+, wgg and the multipole moment estimator introduced in Singh et al (2024).
The correlation functions are measured for simulations in cartesian coordinates with periodic boundary conditions.
[Lightcone version is coming up, see Roadmap.]
Furthermore, the jackknife method is used to estimate the covariance matrix.
Outputs are saved in hdf5 files.

[//]: # (This package was developed for [link to paper].)

#### WARNING: This package is still in a development phase and this is therefore a beta-version.

## Installation

This package is currently still in a development phase and therefore not pip-installable. However, this will be
available in the future. The beta-version of this package can be installed.

### Install beta-version

The easiest way to install MeasureIA and its dependencies is using uv.

First, install uv (see https://docs.astral.sh/uv/getting-started/installation/).
Then clone the repository using either option:

```angular2html
git clone git@github.com:MarloesvL/measure_IA.git
git clone https://github.com/MarloesvL/measure_IA.git
```

Next, navigate into the directory in your terminal (using cd).
Then run the command:

```angular2html
uv sync
```

This will create a virtual environment with all the dependencies needed for this package.
Either activate the virtual environment created by uv, or run scripts directly using:

```angular2html
uv run [script_name].py
```

#### Installing without uv

If you do not want to use uv, you can also install dependencies the provided requirements.txt document.
Note that you need to also download the kmeans-radec repository (https://github.com/esheldon/kmeans_radec) in this case.
Also, make sure your Python version is compatible. This package has been set up to use Python 3.11.
Both the extra repository and the python version are handeled by uv automatically so please consider using this for
easy installation.

## Usage

See the example script 'example_measure_IA_box.py' or the jupyter notebook 'example_measureIA_box.ipynb' in the
examples directory for short examples on how this package can be used.
Explanations on various input parameters are explained in the comments (and more fully in the docstrings of the methods
and classes).
Given the data dictionary in the correct format, the methods (with all optional parameters as their default)
can be called as follows:

```angular2html
MeasureIA_test = MeasureIABox(data=data_dict, output_file_name="./outfile_name.hdf5", boxsize=205.)
# measure wgg, wg+
MeasureIA_test.measure_xi_w(dataset_name=dataset_name, corr_type="both", num_jk=27)
# measure multipoles
MeasureIA_test.measure_xi_multipoles(dataset_name=dataset_name, corr_type="both", num_jk=27)
```

It is advisable to check out all the optional inputs in the examples.

## Documentation

The documentation for this package is still under development (see roadmap). Currently, the methods meant for use and
the inits of all classes have docstrings that provide the information needed. Please feel free to contact me for any
additional questions.

## Output file structure
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
	│		└── [dataset_name]_[i]_rp					r_p bin values of each jackknife realisation
	├── w_g_plus
	│	├── [dataset_name]								w_g+ values for each r_p bin
	│	├── [dataset_name]_rp							r_p mean bin values
	│	├── [dataset_name]_mean_[num_jk]				mean w_g+ value of all jackknife realisations
	│	├── [dataset_name]_jackknife_cov_[num_jk]		jackknife estimate of covariance matrix
	│	├── [dataset_name]_jackknife_[num_jk]			sqrt of diagonal of covariance matrix (size of errorbars)
	│	└── [dataset_name]_jk[num_jk]					group containing all jackknife realisations for this dataset
	│		├── [dataset_name]_[i]						jackknife realisations with i running from 0 to num_jk - 1
	│		└── [dataset_name]_[i]_rp					r_p bin values of each jackknife realisation
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
If you choose to measure multipoles instead of wg+, all 'w' will be replaced by 'multipoles' - or both will appear, if you have measured both.
For the multipoles, all xi_g+, DD (etc) grids are in (r, mu_r), not in (r_p, pi) and the suffixes of the bin values are also replaced by '_r' and '_mu_r' accordingly.
In one file, multiple redshift (snapshot) measurements can be saved without being overwritten, as well as the jackknife
information for different numbers of jackknife realisations (num_jk) for the same dataset.

## Roadmap

Upcoming developments include adding docstrings for all (internal) methods; creating a documentation website; extending
the tests; validating the lightcone methods and adding the Landy-Salazy estimator for the lightcone code.
Once the lightcone code is sufficiently validated, multiprocessing methods will be added there too.
Further down the road, another speed update may be added for the box methods; along with more variability
in definitions (e.g. optional resposivity factor).

## Requests

### Bugs

If you find a bug, please report it in a GitHub issue.

### Features

If you would like a feature added, please create an issue with the request. Within the issue, we can discuss how best
to proceed and what the timeline will be. Pull requests that have not been discussed beforehand will not be accepted.

## License

[MIT](https://choosealicense.com/licenses/mit/)
