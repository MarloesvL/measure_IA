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
If you choose to measure multipoles instead of wg+, all 'w' will be replaced by 'multipoles' - or both will appear, if you have measured both.
For the multipoles, all xi_g+, DD (etc) grids are in (r, mu_r), not in (r_p, pi) and the suffixes of the bin values are also replaced by '_r' and '_mu_r' accordingly.
In one file, multiple redshift (snapshot) measurements can be saved without being overwritten, as well as the jackknife
information for different numbers of jackknife realisations (num_jk) for the same dataset.