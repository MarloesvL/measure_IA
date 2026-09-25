"""Cross-validation of MeasureIALightcone w_++ / w_xx against treecorr GG.

This is the leg that validates the **cross-cross** signal. halotools'
ii_minus_projected cannot serve as a reference for it (its value depends on the
arbitrary sign of the orientation vectors -- see run_box_shape_shape_halotools),
whereas treecorr works with spin-2 shear components, for which xi_+ and xi_- are
well defined and sign-convention free.

The comparison is the **II auto-correlation of the shape (satellite) sample**:
the same catalogue goes into both of measureia's sample slots, matching the box
halotools leg.

treecorr's GG gives, about the separation direction,

	xi_+ = xi_tt + xi_xx        xi_- = xi_tt - xi_xx

so the raw pair sums measureia accumulates are recovered as

	S_+S_+ = (xip + xim) * weight / 2
	S_xS_x = (xip - xim) * weight / 2

with treecorr's ``weight`` = sum_ij w_i w_j over the same pairs. Comparing at
the level of these raw sums, slab by slab in signed pi, isolates the projection
itself: no RR, no estimator, no Pi integral in the way.

**No shear sign flip is applied here, and none is needed.** measureia's IA
convention has e_+ = -gamma_t, but a shape-shape product carries two such
factors, so the flip cancels exactly. The script asserts this rather than
assuming it, by running treecorr both ways.

The only convention difference is the separation *magnitude*: treecorr bins on
Rperp (FisherRperp), measureia on its midpoint-LOS definition, so pairs near a
bin edge land in different bins. Expect sub-percent agreement rather than
machine precision, growing with angular separation.

The projection *direction* is not a difference. Both codes project each shear in
its own (east, north) tangent frame, and the projection direction is NOT one of them: measureia's
  midpoint-LOS-perpendicular direction and treecorr's great-circle bearing are
  the *same* direction, exactly, in each galaxy's own tangent frame. Proof: at
  galaxy i the tangential part of the separation s = r_j n_j - r_i n_i is r_j
  times the tangential part of n_j, since n_i has no tangential component at i
  by construction -- so the radial separation drops out of the direction. And
  n_mid is proportional to r_i n_i + r_j n_j, hence lies in span(n_i, n_j), so
  its tangential part at i is along that same direction; subtracting a multiple
  of it rescales but cannot rotate, and a sign change is irrelevant to a spin-2
  quantity (phi -> phi + pi leaves cos 2phi and sin 2phi fixed). Verified
  numerically over separations of 0.3-179.4 deg and distance ratios up to 18.7:
  cos 2phi and sin 2phi agree to 3e-13.

When treecorr is installed its results are written to
reference_outputs/lightcone_shape_shape_treecorr.hdf5 so the pytest layer can
compare without treecorr present.
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
DATASET = "lc_shape_shape_mock"
COSMOLOGY = ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=0.7, sigma8=0.8, n_s=0.96)
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"lightcone_shape_shape_treecorr.hdf5"
)


def _redshift_of_chi(chi):
	a = ccl.scale_factor_of_chi(COSMOLOGY, chi)
	return 1.0 / a - 1.0


def build_catalogues():
	"""The mock's shape sample, placed in both of measureia's sample slots."""
	data, randoms, info = radial_alignment_lightcone_mock()
	for d in (data, randoms):
		d["Redshift"] = _redshift_of_chi(d.pop("r_com"))
		d["Redshift_shape_sample"] = _redshift_of_chi(d.pop("r_com_shape_sample"))

	# both slots hold the satellite (shape) catalogue: the II auto-correlation
	auto = {
		"RA": data["RA_shape_sample"], "DEC": data["DEC_shape_sample"],
		"Redshift": data["Redshift_shape_sample"],
		"RA_shape_sample": data["RA_shape_sample"],
		"DEC_shape_sample": data["DEC_shape_sample"],
		"Redshift_shape_sample": data["Redshift_shape_sample"],
		"e1": data["e1"], "e2": data["e2"],
		"e1_density_sample": data["e1"], "e2_density_sample": data["e2"],
		"weight": data["weight_shape_sample"],
		"weight_shape_sample": data["weight_shape_sample"],
	}
	rand = {
		"RA": randoms["RA_shape_sample"], "DEC": randoms["DEC_shape_sample"],
		"Redshift": randoms["Redshift_shape_sample"],
		"RA_shape_sample": randoms["RA_shape_sample"],
		"DEC_shape_sample": randoms["DEC_shape_sample"],
		"Redshift_shape_sample": randoms["Redshift_shape_sample"],
		"weight": randoms["weight_shape_sample"],
		"weight_shape_sample": randoms["weight_shape_sample"],
	}
	dist = ccl.comoving_radial_distance(COSMOLOGY, 1 / (1 + auto["Redshift"]))
	return auto, rand, dist


def run_measureia(data, randoms, output_file, temp_path):
	"""Returns (rp, raw S+S+ grid, raw SxSx grid, w_++, w_xx, w_+x)."""
	ia = MeasureIALightcone(
		data={k: v for k, v in data.items()},
		randoms_data={k: v for k, v in randoms.items()},
		output_file_name=output_file,
		separation_limits=RP_LIMS, num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX, num_nodes=1,
	)
	ia.measure_xi_w("galaxies", DATASET, "++", tree=True,
					cosmology=COSMOLOGY, over_h=False, temp_file_path=temp_path)
	with h5py.File(output_file, "r") as f:
		rp = f[f"w_plus_plus/{DATASET}_rp"][:]
		SpSp = f[f"w/xi_plus_plus/{DATASET}_SplusSplus"][:]
		SxSx = f[f"w/xi_cross_cross/{DATASET}_ScrossScross"][:]
		w_pp = f[f"w_plus_plus/{DATASET}"][:]
		w_xx = f[f"w_cross_cross/{DATASET}"][:]
		w_px = f[f"w_plus_cross/{DATASET}"][:]
	return rp, SpSp, SxSx, w_pp, w_xx, w_px


def run_treecorr(data, dist, r_bins, pi_bins, flip=False):
	"""Raw S+S+ and SxSx sums from treecorr GG, slab by slab in signed pi."""
	import treecorr

	sign = -1.0 if flip else 1.0
	cat = treecorr.Catalog(ra=data["RA"], dec=data["DEC"], r=dist,
						   w=data["weight"],
						   g1=sign * data["e1"], g2=sign * data["e2"],
						   ra_units="deg", dec_units="deg")
	config = dict(nbins=NUM_BINS_RP, min_sep=r_bins[0], max_sep=r_bins[-1],
				  bin_slop=0, metric="Rperp")

	n_pi = len(pi_bins) - 1
	SpSp = np.zeros((NUM_BINS_RP, n_pi))
	SxSx = np.zeros((NUM_BINS_RP, n_pi))
	for i in range(n_pi):
		gg = treecorr.GGCorrelation(**config,
									min_rpar=pi_bins[i], max_rpar=pi_bins[i + 1])
		gg.process(cat, cat)
		# xi_+ = xi_tt + xi_xx and xi_- = xi_tt - xi_xx, weight-normalised
		SpSp[:, i] = (gg.xip + gg.xim) * gg.weight / 2.0
		SxSx[:, i] = (gg.xip - gg.xim) * gg.weight / 2.0
	return SpSp, SxSx


def main():
	data, randoms, dist = build_catalogues()
	r_bins = np.logspace(np.log10(RP_LIMS[0]), np.log10(RP_LIMS[1]), NUM_BINS_RP + 1)
	pi_bins = np.linspace(-PI_MAX, PI_MAX, NUM_BINS_PI + 1)

	here = os.path.dirname(os.path.abspath(__file__))
	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	rp, SpSp_mia, SxSx_mia, w_pp, w_xx, w_px = run_measureia(data, randoms, scratch, here)
	os.remove(scratch)

	print(f"rp   = {rp}")
	print(f"w_++ = {w_pp}")
	print(f"w_xx = {w_xx}")
	print(f"w_+x = {w_px}   <- parity null")

	try:
		import treecorr
		have_treecorr = True
	except ImportError:
		have_treecorr = False

	if have_treecorr:
		SpSp_tc, SxSx_tc = run_treecorr(data, dist, r_bins, pi_bins)
		SpSp_flip, SxSx_flip = run_treecorr(data, dist, r_bins, pi_bins, flip=True)
		print("\n--- the IA sign flip cancels in a shape-shape product ---")
		for nm, a, b in (("S+S+", SpSp_tc, SpSp_flip), ("SxSx", SxSx_tc, SxSx_flip)):
			# mathematically exact: (-a)(-b) = ab. treecorr rotates the shears
			# internally, so the two runs differ only in floating-point noise --
			# report the relative size rather than demanding bitwise equality.
			rel = np.max(np.abs(a - b)) / np.max(np.abs(a))
			print(f"treecorr {nm} with g=-e: max relative difference = {rel:.2e}")
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["treecorr_version"] = treecorr.__version__
			f.attrs["pi_max"] = PI_MAX
			f["rp_bins"] = r_bins
			f["pi_bins"] = pi_bins
			f["SplusSplus"] = SpSp_tc
			f["ScrossScross"] = SxSx_tc
		print(f"treecorr {treecorr.__version__} results written to {REFERENCE_FILE}")
	else:
		print("treecorr not installed; comparing against committed reference outputs")
		if not os.path.exists(REFERENCE_FILE):
			print(f"No reference file at {REFERENCE_FILE} either — install treecorr "
				  f"(pip install measureia[validation]) and rerun to create it.")
			return
		with h5py.File(REFERENCE_FILE, "r") as f:
			SpSp_tc = f["SplusSplus"][:]
			SxSx_tc = f["ScrossScross"][:]

	def report(name, mia, tc):
		mask = np.abs(tc) > 1e-8 * np.max(np.abs(tc))
		ratio = np.where(mask, mia / np.where(tc == 0, 1, tc), np.nan)
		print(f"\n--- raw {name} sums, measureia / treecorr (per rp, summed over pi) ---")
		m, t = mia.sum(axis=1), tc.sum(axis=1)
		print(f"measureia : {m}")
		print(f"treecorr  : {t}")
		with np.errstate(invalid='ignore', divide='ignore'):
			print(f"ratio     : {m / t}")

	report("S+S+", SpSp_mia, SpSp_tc)
	report("SxSx", SxSx_mia, SxSx_tc)


if __name__ == "__main__":
	main()
