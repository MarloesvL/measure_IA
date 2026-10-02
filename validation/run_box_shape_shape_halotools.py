"""Cross-validation of MeasureIABox w_++ / w_xx against halotools.

Runs both codes on the synthetic radial-alignment mock (see measureia.mocks)
and compares the shape-shape (II) correlations:

- w_++ : MeasureIABox.measure_xi_w(corr_type='++') vs halotools
  ii_plus_projected.
- w_xx : **not enforced against halotools.** halotools calls the cross-cross
  signal "minus" and its e_-(j|i) = e_j sin(2phi) is nominally measureia's
  e_x, but its value depends on the *sign* of the orientation vectors, which
  is physically meaningless (o and -o describe the same shape). Measured on
  this mock: flipping a random half of the orientation signs leaves
  ii_plus_projected bit-identical and changes ii_minus_projected by 113%,
  while both measureia products are bit-identical. That is the same
  unsigned-angle issue measureia fixed in its own box e_x (see the CHANGELOG
  entry for the axis-direction sign), so there is no single halotools number
  for w_xx to compare against. Both are still computed and printed here for
  inspection, and w_xx is validated against TreeCorr instead, where the
  spin-2 shear components make xi_- well defined.

The comparison is the **auto-correlation of the shape (satellite) sample**:
both members of every pair are satellites, which carry a real II signal
because satellites of a shared central are aligned with each other through
it. Feeding the same catalogue into both sample slots is exactly what
halotools does with sample2=sample1, and exercises measureia's auto path.

Known convention difference (as for the g+ leg): measureia divides each S+
factor by its sample's responsivity 2R, halotools divides by neither. A
shape-shape product carries two such factors, and here both samples are the
same catalogue, so the comparison is

    w_++^measureia * (2R)^2  ==  w_++^halotools

When halotools is installed, its results are written to
reference_outputs/box_shape_shape_halotools.hdf5 so the pytest layer can
compare measureia against them without halotools present. Running this
script never requires halotools: without it, only the measureia side is
(re)computed and compared against the committed reference file.
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
DATASET = "box_shape_shape_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"box_shape_shape_halotools.hdf5"
)


def run_measureia(mock, output_file, cross=False):
	"""Run MeasureIABox on the shape sample II auto-correlation, or -- with
	``cross=True`` -- on the genuine two-catalogue cross correlation between the
	density sample (centrals + satellites, which needs ``density_shapes=True``
	on the mock) and the shape sample (satellites).

	Returns (rp, w_plus_plus, w_cross_cross, w_plus_cross).
	"""
	shapes = mock["Position_shape_sample"]
	if cross:
		data = {
			"Position": mock["Position"],
			"Position_shape_sample": shapes,
			"Axis_Direction": mock["Axis_Direction"],
			"q": mock["q"],
			"Axis_Direction_density_sample": mock["Axis_Direction_density_sample"],
			"q_density_sample": mock["q_density_sample"],
			"LOS": mock["LOS"],
		}
	else:
		data = {
			# the same catalogue in both slots: the II auto-correlation
			"Position": shapes,
			"Position_shape_sample": shapes,
			"Axis_Direction": mock["Axis_Direction"],
			"q": mock["q"],
			"Axis_Direction_density_sample": mock["Axis_Direction"],
			"q_density_sample": mock["q"],
			"LOS": mock["LOS"],
		}
	ia = MeasureIABox(
		data, output_file,
		simulation=None, snapshot=None,
		separation_limits=RP_LIMS,
		num_bins_r=NUM_BINS_RP,
		num_bins_pi=NUM_BINS_PI,
		pi_max=PI_MAX,
		boxsize=mock["boxsize"],
		num_nodes=1,
		# halotools normalises a count by N_1 * N_2 regardless of overlap; here the two
		# slots hold the *same* catalogue, so measureia would otherwise subtract the
		# self-pairs. num_overlap=0 states halotools' convention explicitly so the
		# comparison is like for like.
		num_overlap=0,
	)
	ia.measure_xi_w(DATASET, "++", 0, temp_file_path=False)
	with h5py.File(output_file, "r") as f:
		rp = f[f"w_plus_plus/{DATASET}_rp"][:]
		w_pp = f[f"w_plus_plus/{DATASET}"][:]
		w_xx = f[f"w_cross_cross/{DATASET}"][:]
		w_px = f[f"w_plus_cross/{DATASET}"][:]
	return rp, w_pp, w_xx, w_px


def run_halotools(mock, rp_bins):
	"""Run halotools ii_plus_projected / ii_minus_projected on the same pairs."""
	from halotools.mock_observables.ia_correlations import (
		ii_plus_projected, ii_minus_projected)

	shapes, orientations, e, _density, period = halotools_inputs(mock)
	common = dict(rp_bins=rp_bins, pi_max=PI_MAX, period=period, num_threads=1)
	w_pp = ii_plus_projected(
		shapes, orientations, e, shapes, orientations, e, **common)
	w_xx = ii_minus_projected(
		shapes, orientations, e, shapes, orientations, e, **common)
	return w_pp, w_xx


def run_halotools_cross(mock, rp_bins):
	"""halotools ii_plus_projected on the same two-catalogue cross correlation.

	ii_minus_projected is NOT called here: as of halotools 0.9.4 it builds the
	second sample's marks from the *first* sample's orientations
	(ii_minus_projected.py, `marks2[:, 1] = orientations1[:, 0]`, where
	ii_plus_projected.py correctly uses orientations2). That raises outright when
	the two samples differ in length, and would silently use the wrong
	orientations if they happened to match. It is a separate defect from the
	orientation-sign dependence documented in the module docstring, and a second
	reason ii_minus cannot serve as a reference for w_xx.
	"""
	from halotools.mock_observables.ia_correlations import ii_plus_projected

	q_d = mock["q_density_sample"]
	e_d = (1 - q_d ** 2) / (1 + q_d ** 2)
	q_s = mock["q"]
	e_s = (1 - q_s ** 2) / (1 + q_s ** 2)
	common = dict(rp_bins=rp_bins, pi_max=PI_MAX, period=mock["boxsize"], num_threads=1)
	# sample1 = density (centrals + satellites), sample2 = shape (satellites)
	args = (mock["Position"], mock["Axis_Direction_density_sample"], e_d,
			mock["Position_shape_sample"], mock["Axis_Direction"], e_s)
	return ii_plus_projected(*args, **common)


def _responsivity_from_q(q):
	e = (1 - q ** 2) / (1 + q ** 2)
	return np.mean(1 - e ** 2 / 2.0)


def main():
	mock = radial_alignment_box_mock(density_shapes=True)
	R = responsivity(mock)
	rp_bins = np.logspace(np.log10(RP_LIMS[0]), np.log10(RP_LIMS[1]), NUM_BINS_RP + 1)

	scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	rp, wpp_mia, wxx_mia, wpx_mia = run_measureia(mock, scratch)
	os.remove(scratch)
	# both samples are the same catalogue, so both responsivity factors are this one
	resp = (2 * R) ** 2
	print(f"responsivity R = {R:.8f}  ((2R)^2 = {resp:.8f})")
	print(f"rp            = {rp}")
	print(f"w_++ (measureia) = {wpp_mia}")
	print(f"w_xx (measureia) = {wxx_mia}")
	print(f"w_+x (measureia) = {wpx_mia}   <- parity null, expected ~0")

	try:
		import halotools
		have_halotools = True
	except ImportError:
		have_halotools = False

	if have_halotools:
		wpp_ht, wxx_ht = run_halotools(mock, rp_bins)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["halotools_version"] = halotools.__version__
			f.attrs["responsivity_R"] = R
			f.attrs["pi_max"] = PI_MAX
			f.attrs["mock_seed"] = mock["seed"]
			f["rp_bins"] = rp_bins
			f["w_plus_plus"] = wpp_ht
			f["w_cross_cross"] = wxx_ht
		print(f"halotools {halotools.__version__} results written to {REFERENCE_FILE}")
	else:
		print("halotools not installed; comparing against committed reference outputs")
		if not os.path.exists(REFERENCE_FILE):
			print(f"No reference file at {REFERENCE_FILE} either — install halotools "
				  f"(optional extra: pip install measureia[validation]) and rerun to create it.")
			return
		with h5py.File(REFERENCE_FILE, "r") as f:
			wpp_ht = f["w_plus_plus"][:]
			wxx_ht = f["w_cross_cross"][:]

	print("\n--- w_++ comparison (measureia * (2R)^2 vs halotools) ---")
	print(f"halotools : {wpp_ht}")
	print(f"ratio     : {(wpp_mia * resp) / wpp_ht}")
	print("\n--- w_xx (NOT enforced; halotools ii_minus is orientation-sign dependent) ---")
	print(f"halotools : {wxx_ht}")
	print(f"ratio     : {(wxx_mia * resp) / wxx_ht}")
	print("  ^ this ratio is not expected to be 1; see the module docstring.")

	print("\n--- orientation-sign invariance ---")
	flipped = dict(mock)
	rng = np.random.default_rng(0)
	signs = rng.choice([-1.0, 1.0], len(mock["Axis_Direction"]))
	flipped["Axis_Direction"] = mock["Axis_Direction"] * signs[:, None]
	scratch2 = scratch.replace(".hdf5", "_flip.hdf5")
	if os.path.exists(scratch2):
		os.remove(scratch2)
	_, wpp_f, wxx_f, _ = run_measureia(flipped, scratch2)
	os.remove(scratch2)
	print(f"measureia w_++ unchanged under a per-galaxy axis flip: "
		  f"{np.array_equal(wpp_mia, wpp_f)}")
	print(f"measureia w_xx unchanged under a per-galaxy axis flip: "
		  f"{np.array_equal(wxx_mia, wxx_f)}")

	# ------------------------------------------------------------------
	# the genuine two-catalogue cross correlation: density (centrals +
	# satellites, with their own shapes) x shape sample (satellites)
	# ------------------------------------------------------------------
	scratch3 = scratch.replace(".hdf5", "_cross.hdf5")
	if os.path.exists(scratch3):
		os.remove(scratch3)
	_, wpp_x, wxx_x, wpx_x = run_measureia(mock, scratch3, cross=True)
	os.remove(scratch3)
	# two different samples now, so one responsivity factor each
	R_d = _responsivity_from_q(mock["q_density_sample"])
	resp_x = (2 * R) * (2 * R_d)
	print(f"\n=== two-catalogue CROSS correlation ===")
	print(f"R_shape = {R:.8f}, R_density = {R_d:.8f}, product = {resp_x:.8f}")
	print(f"w_++ (measureia) = {wpp_x}")
	print(f"w_+x (measureia) = {wpx_x}   <- parity null")
	if have_halotools:
		wpp_x_ht = run_halotools_cross(mock, rp_bins)
		with h5py.File(REFERENCE_FILE, "a") as f:
			if "cross_w_plus_plus" in f:
				del f["cross_w_plus_plus"]
			f["cross_w_plus_plus"] = wpp_x_ht
			f.attrs["responsivity_R_density"] = R_d
	else:
		with h5py.File(REFERENCE_FILE, "r") as f:
			if "cross_w_plus_plus" not in f:
				print("reference file predates the cross comparison; rerun with halotools")
				return
			wpp_x_ht = f["cross_w_plus_plus"][:]
	print(f"halotools        = {wpp_x_ht}")
	print(f"ratio            : {(wpp_x * resp_x) / wpp_x_ht}")


if __name__ == "__main__":
	main()
