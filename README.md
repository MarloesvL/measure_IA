# MeasureIA - The tool for measuring intrinsic alignment correlation functions in hydrodynamic simulations

MeasureIA is a tool that can be used to easily measure intrinsic alignement correlation functions and clustering in simulation boxes.
It includes measurement of wg+, wgg and the multipole moment estimator introduced in Singh et al (2024).
The correlation functions are measured for simulations in cathresian coordinates with periodic boundary conditions.
Furthermore, the jackknife method is used to estimate the covariance matrix.
Outputs are saved in hdf5 files.
This package was developed for [link to paper].

## Installation
Information about installation needs to be added once this is an actual installable package.

## Usage
See the example script 'example_measure_IA_sims.py' for a short example on how this package can be used.
Explanations on various input parameters are explained in the comments.
Note that this script is not meant to run; it does not include data.

## Output file structure
Your output file with your own input of [output_file_name, snapshot, dataset_name, num_jk] will have the following structure:

>[output_file_name]  
>>Snapshot_[snapshot]
>>w
>>>xi_gg
>>>>>[dataset_name]								xi grid in (rp,pi)
>>>>>[dataset_name]_rp						rp mean bin values
>>>>>[dataset_name]_pi						pi mean bin values
>>>>>[dataset_name]_RR_gg
>>>>>[dataset_name]_DD
>>>>>[dataset_name]_jk[num_jk]
>>>>>>[dataset_name]_[i] with i running from 0 to num_jk - 1
>>>>xi_g_plus
>>>>xi_g_cross
	 			
