import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from src.measure_IA_base import MeasureIABase
from src.read_data import ReadData
from src.plotting import *

file_path_randoms = ""
file_path_clusters = "/shared-scratch/Kasper/"
file_path_jk = ""
fig_path = ""

# randoms catalogue for S+R calculation
random_file = fits.open(f"{file_path_randoms}/redmapper_dr8_public_v6.3_randoms.fits", memmap=True)
cluster_catalog = Table(random_file[1].data)
RA_rand = cluster_catalog['RA']
DEC_rand = cluster_catalog['DEC']
lam_rand = cluster_catalog['LAMBDA']
z_rand = cluster_catalog['Z']
RA_rand[(RA_rand > 300)] -= 360.

# cluster data for S+D
datafile = np.loadtxt(f'{file_path_clusters}/redmapper_shapes_pmemthres0p2.dat', delimiter=" ")
RA = datafile[:, 0]
DEC = datafile[:, 1]
e1 = datafile[:, 3]
e2 = datafile[:, 4]
lam = datafile[:, 5]  # richness
z = datafile[:, 6]
RA[(RA > 300)] -= 360.

# Read jackknife patch numbers
jk_patches_file = h5py.File("../data/processed/redmapper/redmapper_jacknife_regions.hdf5", "r")
jk_patches_randoms = jk_patches_file["patch_numbers_randoms_45"][:]  # jackknife patches randoms
jk_patches_shape = jk_patches_file["patch_numbers_45"][:]  # jackknife patches clusters

# parameters
h = 0.7  # hubble
num_bins_r = 10  # number of r or rp bins
num_bins_pi = 20  # number of pi bins
separation_limits = [2.5 / h, 140.0 / h]  # [Mpc] limits of r or rp
LOS_lims = [100. / h, 125 / h, 150 / h]  # [Mpc] pi_max and -pi_min
data_path_out = ""
file_name = "testing_IA.hdf5"  # needs to be hdf5
rp_cut = None  # ignore
num_nodes = 9  # number of CPUs used in multiprocessing

# selections clusters matching van Uitert 2017
selection_z1 = (z > 0.08) * (z <= 0.16)
selection_z2 = (z > 0.16) * (z <= 0.35)
selection_z3 = (z > 0.35) * (z <= 0.6)
z_names = ["0p08z0p16", "0p16z0p35", "0p35z0p6"]

selection_shape1 = (lam > 19.8) * (lam <= 28)
selection_shape2 = (lam > 28) * (lam <= 40.5)
selection_shape3 = (lam > 40.5)
lam_names = ["lambda_leq28", "28lambda40p5", "lambda_gt40p5"]

# selections randoms matching van Uitert 2017
selection_z1_r = (z_rand > 0.08) * (z_rand <= 0.16)
selection_z2_r = (z_rand > 0.16) * (z_rand <= 0.35)
selection_z3_r = (z_rand > 0.35) * (z_rand <= 0.6)

selection_shape1_r = (lam_rand > 19.8) * (lam_rand <= 28)
selection_shape2_r = (lam_rand > 28) * (lam_rand <= 40.5)
selection_shape3_r = (lam_rand > 40.5)
zsels = [selection_z1_r, selection_z2_r, selection_z3_r]
szels = [selection_shape1_r, selection_shape2_r, selection_shape3_r]

for i, zsel in enumerate([selection_z1, selection_z2, selection_z3]):
	for j, shapesel in enumerate([selection_shape1, selection_shape2, selection_shape3]):
		for randoms_bool in [True, False]:  # measure S+R/DR, then S+D/DD
			if randoms_bool:
				randoms_suff = "_randoms"
			else:
				randoms_suff = ""

			dataset_name = f"{z_names[i]}_{lam_names[j]}{randoms_suff}"  # name of output

			# creating the masks for the data and the input directories for the measurements
			selection_shape = zsel * shapesel
			if randoms_bool:
				selection_pos = zsels[i] * szels[j]  #
				jk_patches_pos = jk_patches_randoms
				data = {"Redshift": z_rand[selection_pos],
						"Redshift_shape_sample": z[selection_shape],
						"RA": RA_rand[selection_pos],
						"RA_shape_sample": RA[selection_shape],
						"DEC": DEC_rand[selection_pos],
						"DEC_shape_sample": DEC[selection_shape],
						"e1": e1[selection_shape],
						"e2": e2[selection_shape]}
			else:
				selection_pos = selection_shape
				jk_patches_pos = jk_patches_shape

				data = {"Redshift": z[selection_pos],
						"Redshift_shape_sample": z[selection_shape],
						"RA": RA[selection_pos],
						"RA_shape_sample": RA[selection_shape],
						"DEC": DEC[selection_pos],
						"DEC_shape_sample": DEC[selection_shape],
						"e1": e1[selection_shape],
						"e2": e2[selection_shape]}
			print(
				f"Dataset name: {dataset_name}, with {sum(selection_shape)} clusters in the shape sample and {sum(selection_pos)} clusters in the position sample.")
			IA_Projected = MeasureIABase(
				data,
				simulation=False,
				num_bins_r=num_bins_r,
				num_bins_pi=num_bins_pi,
				separation_limits=separation_limits,
				LOS_lim=LOS_lims[i],
				output_file_name=data_path_out + file_name,
			)
			# measure xi_g+ for w_g+
			IA_Projected.measure_projected_correlation_obs_clusters(dataset_name=dataset_name, over_h=False)
			# measure w_g+
			corr_type = ['g+', 'w']  # there are multiple choices here: [0]: g+, gg or both and [1]: w or multipoles
			IA_Projected.measure_w_g_i(corr_type=corr_type[0], dataset_name=dataset_name)
			# jackknife error realisations
			if num_nodes == 1:
				IA_Projected.measure_jackknife_realisations_obs(jk_patches_pos[selection_pos],
																jk_patches_shape[selection_shape],
																corr_type=corr_type, dataset_name=dataset_name,
																over_h=False)
			else:
				IA_Projected.measure_jackknife_realisations_obs_multiprocessing(jk_patches_pos[selection_pos],
																				jk_patches_shape[selection_shape],
																				corr_type=corr_type,
																				dataset_name=dataset_name,
																				over_h=False, num_nodes=num_nodes)
			# measure xi_g+ for multipoles
			IA_Projected.measure_projected_correlation_multipoles_obs_clusters(dataset_name=dataset_name, over_h=False)
			# measure multipoles
			corr_type = ['g+',
						 'multipoles']  # there are multiple choices here: [0]: g+, gg or both and [1]: w or multipoles
			IA_Projected.measure_multipoles(corr_type=corr_type[0], dataset_name=dataset_name)
			# jackknife error realisations
			if num_nodes == 1:
				IA_Projected.measure_jackknife_realisations_obs(jk_patches_pos[selection_pos],
																jk_patches_shape[selection_shape],
																corr_type=corr_type, dataset_name=dataset_name,
																over_h=False)
			else:
				IA_Projected.measure_jackknife_realisations_obs_multiprocessing(jk_patches_pos[selection_pos],
																				jk_patches_shape[selection_shape],
																				corr_type=corr_type,
																				dataset_name=dataset_name,
																				over_h=False, num_nodes=num_nodes)
# combine jackknife error realisations into error bars and covariance
# Need to have both S+D and S+R runs
dataset_names = []
for i, zsel in enumerate([selection_z1, selection_z2, selection_z3]):
	for j, shapesel in enumerate([selection_shape1, selection_shape2, selection_shape3]):
		dataset_name = f"{z_names[i]}_{lam_names[j]}"  # name of output
		dataset_names.append(dataset_name)

		IA_Projected = MeasureIABase(
			None,
			simulation=False,
			num_bins_r=num_bins_r,
			num_bins_pi=num_bins_pi,
			separation_limits=separation_limits,
			LOS_lim=LOS_lims[i],
			output_file_name=data_path_out + file_name,
		)
		# wg+
		IA_Projected.measure_jackknife_errors_obs(max_patch=max(jk_patches_shape), min_patch=min(jk_patches_shape),
												  randoms_suf="_randoms", dataset_name=dataset_name,
												  corr_type=["g+", "w"])
		# multipoles
		IA_Projected.measure_jackknife_errors_obs(max_patch=max(jk_patches_shape), min_patch=min(jk_patches_shape),
												  randoms_suf="_randoms", dataset_name=dataset_name,
												  corr_type=["g+", "multipoles"])

# plotting results
IA_file = h5py.File(f'{data_path_out}/testing_IA.hdf5', 'r')
wg_group = IA_file["Snapshot_None/w_g_plus"]
multipole_group = IA_file["Snapshot_None/multipoles_g_plus"]

plt.rcParams.update({'figure.subplot.wspace': 0.18, 'figure.subplot.hspace': 0.15,
					 'figure.figsize': [3.32 * 3, 2.49 * 3],
					 'figure.subplot.left': 0.19,
					 'figure.subplot.bottom': 0.19,
					 'figure.subplot.top': 0.95,
					 'axes.labelsize': 9,
					 'xtick.labelsize': 8,
					 'ytick.labelsize': 8,
					 'legend.fontsize': 8,
					 'text.usetex': True,
					 'figure.dpi': 500,
					 'savefig.dpi': 500
					 })
fig = plt.figure()
axes = panelplot_flexible(9, 3, 3)
labels = [r"$\lambda\leq28, \\ 0.08<z\leq0.16$", r"$28<\lambda\leq40.5, \\ 0.08<z\leq0.16$",
		  r"$\lambda>40.5, \\ 0.08<z\leq0.16$",
		  r"$\lambda\leq28, \\ 0.16<z\leq0.35$", r"$28<\lambda\leq40.5, \\ 0.16<z\leq0.35$",
		  r"$\lambda>40.5, \\ 0.16<z\leq0.35$",
		  r"$\lambda\leq28, \\ 0.35<z\leq0.6$", r"$28<\lambda\leq40.5, \\ 0.35<z\leq0.6$",
		  r"$\lambda>40.5, \\ 0.35<z\leq0.6$"]
for i, ax in enumerate(axes):
	data = wg_group[dataset_names[i]][:] * h  # S+D/DD
	randoms = wg_group[dataset_names[i] + "_randoms"][:] * h  # S+R/DR
	rp = wg_group[dataset_names[i] + "_rp"][:] * h
	err = wg_group[dataset_names[i] + "_jackknife_45"][:] * h * rp
	ax.errorbar(rp, -(data - randoms) * rp, yerr=err, fmt='o')
	ax.set_xlim(2, 200)
	ax.set_ylim(-15, 80)
	ax.text(4, 60, labels[i])
	ax.set_xscale('log')
	ax.set_ylabel(r"$r_p w_{g+}$")
	ax.set_xlabel(r"$r_p$ [Mpc/h]")
plt.savefig(f"{fig_path}/test_redmapper_wgplus.png", bbox_inches='tight')
plt.close()

plt.rcParams.update({'figure.subplot.wspace': 0.18, 'figure.subplot.hspace': 0.15,
					 'figure.figsize': [3.32 * 3, 2.49 * 3],
					 'figure.subplot.left': 0.19,
					 'figure.subplot.bottom': 0.19,
					 'figure.subplot.top': 0.95,
					 'axes.labelsize': 9,
					 'xtick.labelsize': 8,
					 'ytick.labelsize': 8,
					 'legend.fontsize': 8,
					 'text.usetex': True,
					 'figure.dpi': 500,
					 'savefig.dpi': 500
					 })
fig = plt.figure()
axes = panelplot_flexible(9, 3, 3)
labels = [r"$\lambda\leq28, \\ 0.08<z\leq0.16$", r"$28<\lambda\leq40.5, \\ 0.08<z\leq0.16$",
		  r"$\lambda>40.5, \\ 0.08<z\leq0.16$",
		  r"$\lambda\leq28, \\ 0.16<z\leq0.35$", r"$28<\lambda\leq40.5, \\ 0.16<z\leq0.35$",
		  r"$\lambda>40.5, \\ 0.16<z\leq0.35$",
		  r"$\lambda\leq28, \\ 0.35<z\leq0.6$", r"$28<\lambda\leq40.5, \\ 0.35<z\leq0.6$",
		  r"$\lambda>40.5, \\ 0.35<z\leq0.6$"]
for i, ax in enumerate(axes):
	data = multipole_group[dataset_names[i]][:] * h
	randoms = multipole_group[dataset_names[i] + "_randoms"][:] * h
	r = multipole_group[dataset_names[i] + "_r"][:] * h
	err = multipole_group[dataset_names[i] + "_jackknife_45"][:] * h * r
	ax.errorbar(r, -(data - randoms) * r, yerr=err, fmt='o')
	ax.set_xlim(2, 200)
	if i in [0, 1, 2]:
		ax.set_ylim(-0.3, 0.4)
		ax.text(4, 0.3, labels[i])
	elif i in [3, 4, 5]:
		ax.set_ylim(-0.1, 0.2)
		ax.text(4, 0.15, labels[i])
	else:
		ax.set_ylim(-0.06, 0.1)
		ax.text(4, 0.08, labels[i])
	ax.set_xscale('log')
	ax.set_ylabel(r"$r \tilde{\xi}_{g+,2}$")
	ax.set_xlabel(r"$r$ [Mpc/h]")
plt.savefig(f"{fig_path}/test_redmapper_multipolesgplus.png", bbox_inches='tight')
plt.close()
