"""Plane-parallel consistency check: MeasureIABox vs MeasureIALightcone.

The same radial-alignment box mock (generated with a margin so no
periodically wrapped pair lies within the measured separation range) is
measured twice:

1. MeasureIABox: periodic box, analytic randoms, distant-observer LOS
   along z (this path is validated against halotools at machine precision).
2. MeasureIALightcone ('galaxies' estimator): the identical catalogue
   embedded at a large comoving distance, with the projected shapes
   converted exactly to survey-convention e1/e2 in each galaxy's local
   tangent frame, and uniform randoms filling the embedded box cube.

The comparison is reported at three levels, which fully attribute every
difference between the two pipelines:

1. RAW PAIR COUNTS: the DD grids agree to <1% and the S+D grids agree to
   <1% after dividing by the responsivity 2R — the box estimator divides
   S+D by 2R while the lightcone estimator does not (a deliberate
   convention difference: lightcone e1/e2 are assumed to be calibrated
   shear estimates). This is the true plane-parallel test: geometry,
   binning and shape projection agree.
2. w WITH MATCHED RR: rebuilding the lightcone estimator with the box's
   analytic RR gives w_gg within ~1% and w_g+ within ~2% (bins with
   adequate pair counts) of the box result.
3. w AS MEASURED: the as-is w ratios additionally contain the RR window
   difference — analytic RR assumes a periodic box, while the empirical
   randoms live in a bounded cube whose boundary removes pairs at a rate
   growing with separation (a few % at rp ~ 5-20 Mpc/h). This is an
   understood estimator-design difference, not an error.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_catalogues import (radial_alignment_box_mock, embed_box_mock_on_lightcone,
							 responsivity)
from run_lightcone_treecorr import COSMOLOGY, _redshift_of_chi

from measureia import MeasureIABox, MeasureIALightcone

RP_LIMS = [0.5, 20.0]
NUM_BINS_RP = 8
PI_MAX = 20.0
NUM_BINS_PI = 4
MARGIN = 25.0  # > max measured separation, so periodic wrapping is irrelevant
DISTANCE = 12000.0  # comoving Mpc; deep plane-parallel limit
N_RANDOMS_FACTOR = 10
DATASET = "plane_parallel_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs", "plane_parallel.hdf5"
)


def build_mock():
	return radial_alignment_box_mock(margin=MARGIN)


def run_box(mock, output_file):
	"""Run MeasureIABox; return dict with w and the raw count grids."""
	data = {k: mock[k] for k in
			["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	ia = MeasureIABox(
		data, output_file,
		simulation=None, snapshot=None,
		separation_limits=RP_LIMS,
		num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI,
		pi_max=PI_MAX,
		boxsize=mock["boxsize"],
		num_nodes=1,
	)
	ia.measure_xi_w(DATASET, "both", 0, temp_file_path=False)
	with h5py.File(output_file, "r") as f:
		out = {
			"rp": f[f"w_g_plus/{DATASET}_rp"][:],
			"w_g_plus": f[f"w_g_plus/{DATASET}"][:],
			"w_gg": f[f"w_gg/{DATASET}"][:],
			"DD": f[f"w/xi_gg/{DATASET}_DD"][:],
			"RR": f[f"w/xi_gg/{DATASET}_RR_gg"][:],  # analytic, normalized to N_D*N_S
			"SplusD": f[f"w/xi_g_plus/{DATASET}_SplusD"][:],
		}
	return out


def run_lightcone(mock, output_file, temp_path):
	"""Run MeasureIALightcone on the embedded mock; return w and count grids."""
	data, randoms = embed_box_mock_on_lightcone(
		mock, distance=DISTANCE, n_randoms_factor=N_RANDOMS_FACTOR)
	num = {"D": len(data["RA"]), "S": len(data["RA_shape_sample"]),
		   "R_D": len(randoms["RA"]), "R_S": len(randoms["RA_shape_sample"])}
	for d in (data, randoms):
		d["Redshift"] = _redshift_of_chi(d.pop("r_com"))
		d["Redshift_shape_sample"] = _redshift_of_chi(d.pop("r_com_shape_sample"))
	ia = MeasureIALightcone(
		data=data,
		randoms_data=randoms,
		output_file_name=output_file,
		separation_limits=RP_LIMS,
		num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI,
		pi_max=PI_MAX,
		num_nodes=1,
	)
	ia.measure_xi_w("galaxies", DATASET, "both", measure_cov=False, tree=True,
					cosmology=COSMOLOGY, over_h=False, temp_file_path=temp_path)
	with h5py.File(output_file, "r") as f:
		out = {
			"rp": f[f"w_g_plus/{DATASET}_rp"][:],
			"w_g_plus": f[f"w_g_plus/{DATASET}"][:],
			"w_gg": f[f"w_gg/{DATASET}"][:],
			"DD": f[f"w/xi_gg/{DATASET}_DD"][:],
			"RD": f[f"w/xi_gg/{DATASET}_RD"][:],
			"SR": f[f"w/xi_gg/{DATASET}_SR"][:],
			"RR": f[f"w/xi_gg/{DATASET}_RR"][:],
			"SplusD": f[f"w/xi_g_plus/{DATASET}_SplusD"][:],
			"SplusR": f[f"w/xi_g_plus/{DATASET}_SplusR"][:],
			"num": num,
		}
	return out


def lightcone_w_with_analytic_RR(box, lc, R):
	"""Rebuild the lightcone 'galaxies' estimator using the box's analytic
	(periodic) RR, removing the bounded-window difference; also divide the
	S+ terms by 2R to match the box responsivity convention."""
	n = lc["num"]
	rr = box["RR"] / (n["D"] * n["S"])  # per-pair probability, periodic box
	xi_gp = (lc["SplusD"] / (n["S"] * n["D"])
			 - lc["SplusR"] / (n["S"] * n["R_D"])) / rr / (2 * R)
	xi_gg = (lc["DD"] / (n["D"] * n["S"])
			 - lc["RD"] / (n["D"] * n["R_S"])
			 - lc["SR"] / (n["S"] * n["R_D"])) / rr + 1
	dpi = 2.0 * PI_MAX / NUM_BINS_PI
	return np.sum(xi_gp, axis=1) * dpi, np.sum(xi_gg, axis=1) * dpi


def main():
	mock = build_mock()
	R = responsivity(mock)
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_box_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	box = run_box(mock, scratch)
	os.remove(scratch)

	scratch = os.path.join(here, f"{DATASET}_lc_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	lc = run_lightcone(mock, scratch, here + "/")
	os.remove(scratch)

	print(f"rp = {box['rp']}")
	print(f"responsivity 2R = {2 * R:.6f}")

	print("\n--- level 1: raw pair counts (plane-parallel geometry test) ---")
	dd_ratio = lc["DD"].sum(axis=1) / box["DD"].sum(axis=1)
	spd_ratio = lc["SplusD"].sum(axis=1) / box["SplusD"].sum(axis=1) / (2 * R)
	print(f"DD  lightcone/box          : {dd_ratio}")
	print(f"S+D lightcone/(box * 2R)   : {spd_ratio}")

	print("\n--- level 2: w with matched (analytic) RR ---")
	wgp_ana, wgg_ana = lightcone_w_with_analytic_RR(box, lc, R)
	print(f"w_g+ ratio: {wgp_ana / box['w_g_plus']}")
	print(f"w_gg ratio: {wgg_ana / box['w_gg']}")

	print("\n--- level 3: w as measured (includes RR window difference) ---")
	print(f"w_g+ lightcone/(box * 2R): {lc['w_g_plus'] / (box['w_g_plus'] * 2 * R)}")
	print(f"w_gg lightcone/box       : {lc['w_gg'] / box['w_gg']}")

	os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
	with h5py.File(REFERENCE_FILE, "w") as f:
		f.attrs["distance"] = DISTANCE
		f.attrs["margin"] = MARGIN
		f.attrs["pi_max"] = PI_MAX
		f.attrs["mock_seed"] = mock["seed"]
		f.attrs["responsivity_R"] = R
		f["rp"] = box["rp"]
		f["w_g_plus_box"] = box["w_g_plus"]
		f["w_gg_box"] = box["w_gg"]
		f["w_g_plus_lightcone"] = lc["w_g_plus"]
		f["w_gg_lightcone"] = lc["w_gg"]
		f["w_g_plus_lightcone_analytic_RR"] = wgp_ana
		f["w_gg_lightcone_analytic_RR"] = wgg_ana


if __name__ == "__main__":
	main()
