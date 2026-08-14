"""Cross-validation of MeasureIABox w_gg / w_g+ against halotools.

Runs both codes on the synthetic radial-alignment mock (see
measureia.mocks) and compares:

- w_g+ : MeasureIABox.measure_xi_w vs halotools gi_plus_projected
  (halotools.mock_observables.ia_correlations, available in halotools >= 0.9).
- w_gg : MeasureIABox.measure_xi_w vs halotools wp.

Known convention difference (documented in the README): measureia divides
the S+D sum by the responsivity 2R with R = 1 - <e^2>/2; halotools does not.
So the comparison is  w_g+^measureia * 2R  ==  w_g+^halotools.

When halotools is installed, its results are written to
reference_outputs/box_halotools.hdf5 so that the pytest layer can compare
measureia against them without halotools present. Running this script never
requires halotools: without it, only the measureia side is (re)computed and
compared against the committed reference file.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measureia.mocks import radial_alignment_box_mock, halotools_inputs, responsivity

from measureia import MeasureIABox

# --- comparison configuration (must match the committed reference file) ---
RP_LIMS = [0.5, 20.0]
NUM_BINS_RP = 10
PI_MAX = 20.0
NUM_BINS_PI = 1  # halotools integrates 0..pi_max in one go
DATASET = "box_halotools_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs", "box_halotools.hdf5"
)


def run_measureia(mock, output_file):
	"""Run MeasureIABox on the mock; return (rp, w_g_plus, w_gg)."""
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
		# halotools and corr_pc both normalise a cross count by N_1 * N_2, i.e. they
		# treat the two samples as independent. The mock's shape sample is drawn from
		# its position sample, so MeasureIA would otherwise measure an overlap and
		# subtract the self-pairs; num_overlap=0 states the reference codes' convention
		# explicitly so the comparison comes out like for like.
		num_overlap=0,
	)
	ia.measure_xi_w(DATASET, "both", 0, temp_file_path=False)
	with h5py.File(output_file, "r") as f:
		rp = f[f"w_g_plus/{DATASET}_rp"][:]
		w_g_plus = f[f"w_g_plus/{DATASET}"][:]
		w_gg = f[f"w_gg/{DATASET}"][:]
	return rp, w_g_plus, w_gg


def run_halotools(mock, rp_bins):
	"""Run halotools on the mock; return (w_g_plus, w_gg)."""
	from halotools.mock_observables.ia_correlations import gi_plus_projected
	from halotools.mock_observables import wp

	shapes, orientations, e, density, period = halotools_inputs(mock)
	w_g_plus = gi_plus_projected(
		shapes, orientations, e, density, rp_bins, PI_MAX,
		period=period, num_threads=1,
	)
	w_gg = wp(
		density, rp_bins, PI_MAX, sample2=shapes,
		period=period, num_threads=1, do_auto=False, do_cross=True,
	)
	return w_g_plus, w_gg


def main():
	mock = radial_alignment_box_mock()
	R = responsivity(mock)
	rp_bins = np.logspace(np.log10(RP_LIMS[0]), np.log10(RP_LIMS[1]), NUM_BINS_RP + 1)

	scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	rp, wgp_mia, wgg_mia = run_measureia(mock, scratch)
	os.remove(scratch)
	print(f"responsivity R = {R:.8f}  (2R = {2 * R:.8f})")
	print(f"rp           = {rp}")
	print(f"w_g+ (measureia) = {wgp_mia}")
	print(f"w_gg (measureia) = {wgg_mia}")

	try:
		import halotools
		have_halotools = True
	except ImportError:
		have_halotools = False

	if have_halotools:
		wgp_ht, wgg_ht = run_halotools(mock, rp_bins)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["halotools_version"] = halotools.__version__
			f.attrs["responsivity_R"] = R
			f.attrs["pi_max"] = PI_MAX
			f.attrs["mock_seed"] = mock["seed"]
			f["rp_bins"] = rp_bins
			f["w_g_plus"] = wgp_ht
			f["w_gg"] = wgg_ht
		print(f"halotools {halotools.__version__} results written to {REFERENCE_FILE}")
	else:
		print("halotools not installed; comparing against committed reference outputs")
		if not os.path.exists(REFERENCE_FILE):
			print(f"No reference file at {REFERENCE_FILE} either — install halotools "
				  f"(optional extra: pip install measureia[validation]) and rerun to create it.")
			return
		with h5py.File(REFERENCE_FILE, "r") as f:
			wgp_ht = f["w_g_plus"][:]
			wgg_ht = f["w_gg"][:]

	print("\n--- w_g+ comparison (measureia * 2R vs halotools) ---")
	ratio_gp = (wgp_mia * 2 * R) / wgp_ht
	print(f"halotools : {wgp_ht}")
	print(f"ratio     : {ratio_gp}")
	print("\n--- w_gg comparison ---")
	ratio_gg = wgg_mia / wgg_ht
	print(f"halotools : {wgg_ht}")
	print(f"ratio     : {ratio_gg}")


if __name__ == "__main__":
	main()
