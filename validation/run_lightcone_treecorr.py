"""Cross-validation of MeasureIALightcone w_gg / w_g+ against treecorr.

Runs both codes on the synthetic lightcone radial-alignment mock (see
measureia.mocks). measureia uses its 'galaxies' estimator; the treecorr
side reconstructs exactly the same estimator from raw NN/NG counts:

	xi_g+(rp, pi) = ( S+D/(N_S N_D) - S+R/(N_S N_RD) ) / ( RR/(N_RD N_RS) )
	xi_gg(rp, pi) = ( DD/(N_D N_S) - RD/(N_D N_RS) - SR/(N_S N_RD) )
					/ ( RR/(N_RD N_RS) ) + 1
	w(rp) = sum_pi xi(rp, pi) * dpi

with one treecorr run per signed pi slab (min_rpar/max_rpar), using
metric='Rperp' and bin_slop=0.

Known convention differences (see README):
- treecorr uses the lensing shear sign convention: g1 = -e1, g2 = -e2.
- treecorr's tangential projection uses the great-circle frame of each
  pair; measureia evaluates the position angle in the (east, north) frame
  of the position-sample galaxy. These agree only up to curvature terms,
  so w_g+ matches at the sub-percent level, not machine precision.
- treecorr's Rperp (FisherRperp) separation definition differs slightly
  from measureia's midpoint-LOS definition; a few pairs near bin edges
  land in different bins.

Both codes see identical comoving distances: the mock is generated in
comoving space, converted to redshift with the fixed CCL cosmology below,
and treecorr receives the CCL-computed distances for those redshifts.
"""

import os
import sys

import h5py
import numpy as np
import pyccl as ccl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measureia.mocks import radial_alignment_lightcone_mock

from measureia import MeasureIALightcone

RP_LIMS = [0.5, 20.0]
NUM_BINS_RP = 8
PI_MAX = 20.0
NUM_BINS_PI = 4  # signed slabs spanning (-PI_MAX, +PI_MAX)
DATASET = "lc_treecorr_mock"
COSMOLOGY = ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=0.7, sigma8=0.8, n_s=0.96)
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs", "lightcone_treecorr.hdf5"
)


def _redshift_of_chi(chi):
	"""Invert the CCL comoving distance to redshift."""
	a = ccl.scale_factor_of_chi(COSMOLOGY, chi)
	return 1.0 / a - 1.0


def build_catalogues():
	"""Mock in MeasureIALightcone format plus the exact distances treecorr uses."""
	data, randoms, info = radial_alignment_lightcone_mock()
	for d in (data, randoms):
		d["Redshift"] = _redshift_of_chi(d.pop("r_com"))
		d["Redshift_shape_sample"] = _redshift_of_chi(d.pop("r_com_shape_sample"))
	# distances treecorr gets = distances measureia computes internally
	dist = {
		"D": ccl.comoving_radial_distance(COSMOLOGY, 1 / (1 + data["Redshift"])),
		"S": ccl.comoving_radial_distance(COSMOLOGY, 1 / (1 + data["Redshift_shape_sample"])),
		"R_D": ccl.comoving_radial_distance(COSMOLOGY, 1 / (1 + randoms["Redshift"])),
		"R_S": ccl.comoving_radial_distance(COSMOLOGY, 1 / (1 + randoms["Redshift_shape_sample"])),
	}
	return data, randoms, info, dist


def run_measureia(data, randoms, output_file, temp_path):
	data = {k: v for k, v in data.items()}
	ia = MeasureIALightcone(
		data=data,
		randoms_data={k: v for k, v in randoms.items()},
		output_file_name=output_file,
		separation_limits=RP_LIMS,
		num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI,
		pi_max=PI_MAX,
		num_nodes=1,
	)
	ia.measure_xi_w("galaxies", DATASET, "both", tree=True,
					cosmology=COSMOLOGY, over_h=False, temp_file_path=temp_path)
	with h5py.File(output_file, "r") as f:
		rp = f[f"w_g_plus/{DATASET}_rp"][:]
		w_g_plus = f[f"w_g_plus/{DATASET}"][:]
		w_gg = f[f"w_gg/{DATASET}"][:]
	return ia, rp, w_g_plus, w_gg


def run_treecorr(data, randoms, dist, r_bins, pi_bins):
	"""Reconstruct the measureia 'galaxies' estimator from treecorr counts."""
	import treecorr

	pcat = treecorr.Catalog(ra=data["RA"], dec=data["DEC"], r=dist["D"],
							w=data["weight"], ra_units="deg", dec_units="deg")
	# measureia and treecorr share the survey shear-catalogue component
	# convention; only the overall IA-vs-lensing sign differs (e+ = -gamma_t),
	# absorbed here by the standard flip g = -e so the ratio comes out +1
	scat = treecorr.Catalog(ra=data["RA_shape_sample"], dec=data["DEC_shape_sample"],
							r=dist["S"], w=data["weight_shape_sample"],
							g1=-data["e1"], g2=-data["e2"],
							ra_units="deg", dec_units="deg")
	rdcat = treecorr.Catalog(ra=randoms["RA"], dec=randoms["DEC"], r=dist["R_D"],
							 w=randoms["weight"], ra_units="deg", dec_units="deg")
	rscat = treecorr.Catalog(ra=randoms["RA_shape_sample"], dec=randoms["DEC_shape_sample"],
							 r=dist["R_S"], w=randoms["weight_shape_sample"],
							 ra_units="deg", dec_units="deg")

	N_D, N_S = pcat.sumw, scat.sumw
	N_RD, N_RS = rdcat.sumw, rscat.sumw

	config = dict(nbins=NUM_BINS_RP, min_sep=r_bins[0], max_sep=r_bins[-1],
				  bin_slop=0, metric="Rperp")

	n_pi = len(pi_bins) - 1
	xi_gp = np.zeros((NUM_BINS_RP, n_pi))
	xi_gg = np.zeros((NUM_BINS_RP, n_pi))
	for i in range(n_pi):
		slab = dict(min_rpar=pi_bins[i], max_rpar=pi_bins[i + 1])

		ng = treecorr.NGCorrelation(**config, **slab)
		ng.process(pcat, scat)
		SpD = ng.xi * ng.weight  # raw sum of w_p * w_s * gamma_T

		ngr = treecorr.NGCorrelation(**config, **slab)
		ngr.process(rdcat, scat)
		SpR = ngr.xi * ngr.weight

		def _nn(cat1, cat2):
			nn = treecorr.NNCorrelation(**config, **slab)
			nn.process(cat1, cat2)
			return nn.weight

		DD = _nn(pcat, scat)
		RD = _nn(pcat, rscat)   # D positions x shape-sample randoms
		SR = _nn(rdcat, scat)   # position randoms x S (same pass as S+R)
		RR = _nn(rdcat, rscat)

		norm_RR = RR / (N_RD * N_RS)
		xi_gp[:, i] = (SpD / (N_S * N_D) - SpR / (N_S * N_RD)) / norm_RR
		xi_gg[:, i] = (DD / (N_D * N_S) - RD / (N_D * N_RS) - SR / (N_S * N_RD)) / norm_RR + 1

	dpi = np.diff(pi_bins)
	w_g_plus = np.sum(xi_gp * dpi, axis=1)
	w_gg = np.sum(xi_gg * dpi, axis=1)
	return w_g_plus, w_gg


def main():
	import treecorr

	data, randoms, info, dist = build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))
	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	ia, rp, wgp_mia, wgg_mia = run_measureia(data, randoms, scratch, here + "/")
	os.remove(scratch)

	print(f"rp               = {rp}")
	print(f"w_g+ (measureia) = {wgp_mia}")
	print(f"w_gg (measureia) = {wgg_mia}")

	wgp_tc, wgg_tc = run_treecorr(data, randoms, dist, ia.r_bins, ia.pi_bins)
	os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
	with h5py.File(REFERENCE_FILE, "w") as f:
		f.attrs["treecorr_version"] = treecorr.__version__
		f.attrs["pi_max"] = PI_MAX
		f.attrs["mock_seed"] = info["seed"]
		f["r_bins"] = ia.r_bins
		f["pi_bins"] = ia.pi_bins
		f["w_g_plus"] = wgp_tc
		f["w_gg"] = wgg_tc
	print(f"treecorr {treecorr.__version__} results written to {REFERENCE_FILE}")

	print("\n--- w_g+ comparison ---")
	print(f"treecorr : {wgp_tc}")
	print(f"ratio    : {wgp_mia / wgp_tc}")
	print("\n--- w_gg comparison ---")
	print(f"treecorr : {wgg_tc}")
	print(f"ratio    : {wgg_mia / wgg_tc}")


if __name__ == "__main__":
	main()
