"""Box jackknife covariance validation against corr_pc (explicit delete-one loop).

corr_pc has a built-in jackknife (do_jk=1), but in periodic-box mode it
assigns each galaxy a RANDOM jk_prob (read_dat.cpp) so cross-region pairs
are tallied stochastically (~exclusive deletion in expectation), while
measureia's box jackknife deletes deterministically (a pair is removed if
EITHER member lies in the deleted subbox) — different jackknife
definitions that cannot agree tightly. The tight external check used here
instead makes the delete-one loop explicit: each of the L^3 subboxes is
physically removed from both samples (union semantics by construction) and
corr_pc's full pipeline is run on every deleted catalogue, in both
periodic-box modes:

- coordinates=6 (rp-pi grid) -> w_gg, w_g+ per realisation;
- coordinates=7 (r-mu grid) + measureia's Legendre integration ->
  multipoles per realisation.

The delete-one identity (run_box_cov_bridge.py, machine precision) states
that measureia's reconstructed jackknife realisations equal direct
measurements on the physically deleted catalogues up to the exact analytic
volume factor VF = V/V_del = N/(N-1): xi_jk = xi_direct/VF for g+ and
xi_jk + 1 = (xi_direct + 1)/VF for gg. The corr_pc realisations are
therefore mapped into measureia's jackknife convention through this exact
affine relation (per-realisation, so covariances map exactly as well) and
compared realisation by realisation AND at the covariance level.

Convention differences (same as the signal-level corr_pc legs):

- Responsivity: corr_pc has none; measureia is run with responsivity=False
  so the comparison carries no 2R factor.
- Analytic RR normalisation, r-mu mode only: measureia uses
  (N_pos - 1) N_shape, corr_pc N_pos N_shape; applied per realisation with
  the retained N_pos. The rp-pi mode (coordinates=6) has no such factor
  (verified: full-sample w agrees to ~1e-6 with only the responsivity
  factor, which also makes this script the box-w signal validation against
  corr_pc — previously halotools-only).
- Ellipticity chirality: corr_pc's periodic-box rotation needs
  e2 = -e sin(2 phi) (opposite to the survey shear convention).

Set CORR_PC_BIN to the compiled binary (see README) to (re)generate the
reference outputs; without it the measureia side is compared against the
committed reference file.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measureia.mocks import radial_alignment_box_mock
from run_box_multipoles_corrpc import legendre_multipole

from measureia import MeasureIABox

# --- comparison configuration (must match the committed reference file) ---
NUM_JK = 8  # 2x2x2 subboxes; box jackknife requires x^3
VOLUME_FACTOR = NUM_JK / (NUM_JK - 1.0)  # V_box / V_delete-one
SEP_LIMS = [0.5, 20.0]  # rp limits (coordinates=6) and 3D r limits (=7)
NUM_BINS_R = 10
NUM_BINS_PI = 20  # pi in [-PI_MAX, PI_MAX]
PI_MAX = 20.0
NUM_BINS_MU = 20  # mu in [-1, 1]
DATASET = "box_corrpc_cov_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"box_corrpc_cov.hdf5"
)


def corrpc_ellipticities(mock):
	"""corr_pc periodic-box chirality: e1 = e cos(2 phi), e2 = -e sin(2 phi)."""
	e = (1 - mock["q"] ** 2) / (1 + mock["q"] ** 2)
	ax = mock["Axis_Direction"]
	phi = np.arctan2(ax[:, 1], ax[:, 0])  # projected axis angle, LOS = z
	return e * np.cos(2 * phi), -e * np.sin(2 * phi)


def run_corrpc(dens, shape, e1, e2, coordinates, boxsize, workdir, binary):
	"""One corr_pc periodic-box run; returns the xi grids of
	bins2D_cross_final.dat (rp-pi for coordinates=6, r-mu for 7)."""
	order_d = np.argsort(dens[:, 2])
	order_s = np.argsort(shape[:, 2])
	dens, shape = dens[order_d], shape[order_s]

	pref_s = os.path.join(workdir, "shape")
	pref_d = os.path.join(workdir, "density")
	np.savetxt(pref_s + "_pos.dat", shape[:, :2])
	np.savetxt(pref_s + "_z.dat", shape[:, 2])
	np.savetxt(pref_s + "_e.dat", np.column_stack((e1[order_s], e2[order_s])))
	np.savetxt(pref_s + "_jk.dat", np.zeros(len(shape)), fmt="%i")
	np.savetxt(pref_d + "_pos.dat", dens[:, :2])
	np.savetxt(pref_d + "_z.dat", dens[:, 2])
	np.savetxt(pref_d + "_jk.dat", np.zeros(len(dens)), fmt="%i")

	out_dir = os.path.join(workdir, "corr_data_out")
	os.makedirs(out_dir, exist_ok=True)
	out_pref = os.path.join(out_dir, "mia_")
	if coordinates == 6:
		n_p_bin, pmin, pmax = NUM_BINS_PI, -PI_MAX, PI_MAX
	else:
		n_p_bin, pmin, pmax = NUM_BINS_MU, -1, 1
	lines = [
		("which_corr", 1), ("coordinates", coordinates), ("estimator", 0),
		("data_sorted", 1), ("use_comoving", 1), ("do_jk", 0),
		("sig_crit", 0),
		("shape_pos", pref_s + "_pos.dat"), ("shape_z", pref_s + "_z.dat"),
		("shape_e", pref_s + "_e.dat"), ("shape_wt", 0),
		("shape_jk", pref_s + "_jk.dat"),
		("density_patch", 0),
		("density_pos", pref_d + "_pos.dat"), ("density_z", pref_d + "_z.dat"),
		("density_wt", 0), ("density_jk", pref_d + "_jk.dat"),
		("density_e", 0),
		("Srandom_pos", 0), ("Srandom_z", 0), ("Srandoms_wt", 0),
		("Srand_jk", 0),
		("Drandom_patch", 0), ("Drandom_pos", 0), ("Drandom_z", 0),
		("Drandoms_wt", 0), ("drand_jk", 0),
		("distances", 0), ("patch_file", 0),
		("out_file", out_pref),
		("n_threads", 1),
		("n_shape", len(shape)), ("n_density", len(dens)),
		("n_Srand", 0), ("n_Drand", 0), ("rand_subsample", 0),
		("n_jk", 0), ("n_patch", 0),
		("binR_min", SEP_LIMS[0]), ("binR_max", SEP_LIMS[1]),
		("n_bins", NUM_BINS_R), ("lin_bin", 0),
		("n_p_bin", n_p_bin), ("pmin", pmin), ("pmax", pmax),
		("z_min", 0), ("z_max", 2 * boxsize), ("dz", 0.0001),
		("z_sep_min", -2 * boxsize), ("z_sep_max", 2 * boxsize),
		("periodic_box", 1), ("box_size", boxsize),
	]
	inp = os.path.join(workdir, "mia.inp")
	with open(inp, "w") as f:
		for key, val in lines:
			f.write(f"{key}    {val}\n")
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	raw = np.loadtxt(out_pref + "bins2D_cross_final.dat")
	shape2d = (NUM_BINS_R, n_p_bin)
	return {
		"r": raw[:, 0].reshape(shape2d)[:, 0],
		"p": raw[:, 1].reshape(shape2d)[0],
		"xi_gg": raw[:, 3].reshape(shape2d),
		"xi_gp": raw[:, 5].reshape(shape2d),
	}


def corrpc_w(grids):
	dpi = 2.0 * PI_MAX / NUM_BINS_PI
	return (np.sum(grids["xi_gp"], axis=1) * dpi,
			np.sum(grids["xi_gg"], axis=1) * dpi)


def corrpc_multipoles(grids):
	return (legendre_multipole(grids["xi_gp"], grids["p"], 2, 2),
			legendre_multipole(grids["xi_gg"], grids["p"], 0, 0))


def run_corrpc_delete_one(mock, labels, binary):
	"""corr_pc full pipeline on the full catalogue and on each physically
	deleted catalogue, in both periodic-box modes."""
	import tempfile

	e1, e2 = corrpc_ellipticities(mock)
	lab_pos, lab_shape = labels
	out = {"w_gp": [], "w_gg": [], "mult_gp": [], "mult_gg": [], "n_pos": []}

	def one(dens, shape, e1_i, e2_i):
		with tempfile.TemporaryDirectory() as workdir:
			g6 = run_corrpc(dens, shape, e1_i, e2_i, 6, mock["boxsize"],
							workdir, binary)
		with tempfile.TemporaryDirectory() as workdir:
			g7 = run_corrpc(dens, shape, e1_i, e2_i, 7, mock["boxsize"],
							workdir, binary)
		return corrpc_w(g6), corrpc_multipoles(g7)

	(wgp, wgg), (mgp, mgg) = one(mock["Position"],
								 mock["Position_shape_sample"], e1, e2)
	out["w_gp_full"], out["w_gg_full"] = wgp, wgg
	out["mult_gp_full"], out["mult_gg_full"] = mgp, mgg
	out["n_pos_full"] = len(mock["Position"])

	for i in range(NUM_JK):
		keep_p, keep_s = lab_pos != i, lab_shape != i
		(wgp, wgg), (mgp, mgg) = one(mock["Position"][keep_p],
									 mock["Position_shape_sample"][keep_s],
									 e1[keep_s], e2[keep_s])
		out["w_gp"].append(wgp)
		out["w_gg"].append(wgg)
		out["mult_gp"].append(mgp)
		out["mult_gg"].append(mgg)
		out["n_pos"].append(int(np.sum(keep_p)))
	for k in ["w_gp", "w_gg", "mult_gp", "mult_gg"]:
		out[k] = np.array(out[k])
	out["n_pos"] = np.array(out["n_pos"])
	return out


def map_corrpc_to_jk(pc):
	"""Map the corr_pc direct delete-one measurements into measureia's
	jackknife-realisation convention via the exact affine relations
	(module docstring); covariances then compare directly."""
	total_pi = 2.0 * PI_MAX
	# No RR-normalisation term: measureia is constructed with num_overlap=0 above, so both
	# codes normalise the analytic RR by N_pos * N_shape in r-mu mode as well as rp-pi.
	return {
		"w_gp": pc["w_gp"] / VOLUME_FACTOR,
		"w_gg": (pc["w_gg"] + total_pi) / VOLUME_FACTOR - total_pi,
		"mult_gp": pc["mult_gp"] / VOLUME_FACTOR,
		"mult_gg": (pc["mult_gg"] + 1.0) / VOLUME_FACTOR - 1.0,
	}


def run_measureia_jk(mock, output_file, temp_path):
	"""measureia box jackknife for w and multipoles (responsivity=False);
	returns full-sample vectors, realisations, and covariances."""
	data = {k: mock[k] for k in
			["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	jk = f"{DATASET}_jk{NUM_JK}"
	out = {}

	ia = MeasureIABox(data, output_file, simulation=None, snapshot=None,
					  separation_limits=SEP_LIMS, num_bins_r=NUM_BINS_R,
					  num_bins_pi=NUM_BINS_PI, pi_max=PI_MAX,
					  boxsize=mock["boxsize"], num_nodes=1,
					  # corr_pc treats the two samples as independent; state that same
					  # convention explicitly so the comparison is like for like
					  # (see run_box_halotools.py).
					  num_overlap=0)
	ia.measure_xi_w(DATASET, "both", num_jk=NUM_JK, temp_file_path=temp_path,
					responsivity=False)
	with h5py.File(output_file, "r") as f:
		for grp, key in [("w_g_plus", "w_gp"), ("w_gg", "w_gg")]:
			out[f"{key}_full"] = f[f"{grp}/{DATASET}"][:]
			out[key] = np.array(
				[f[f"{grp}/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
	labels = ia._get_jackknife_region_indices(None, round(NUM_JK ** (1 / 3)))

	os.remove(output_file)
	ia = MeasureIABox(data, output_file, simulation=None, snapshot=None,
					  separation_limits=SEP_LIMS, num_bins_r=NUM_BINS_R,
					  num_bins_pi=NUM_BINS_MU, pi_max=SEP_LIMS[1],
					  boxsize=mock["boxsize"], num_nodes=1,
					  # corr_pc treats the two samples as independent; state that same
					  # convention explicitly so the comparison is like for like
					  # (see run_box_halotools.py).
					  num_overlap=0)
	ia.measure_xi_multipoles(DATASET, "both", num_jk=NUM_JK,
							 temp_file_path=temp_path, responsivity=False)
	with h5py.File(output_file, "r") as f:
		for grp, key in [("multipoles_g_plus", "mult_gp"),
						 ("multipoles_gg", "mult_gg")]:
			out[f"{key}_full"] = f[f"{grp}/{DATASET}"][:]
			out[key] = np.array(
				[f[f"{grp}/{jk}/{DATASET}_{i}"][:] for i in range(NUM_JK)])
			out[f"cov_{key}"] = f[f"{grp}/{DATASET}_jackknife_cov_{NUM_JK}"][:]
			out[f"std_{key}"] = f[f"{grp}/{DATASET}_jackknife_{NUM_JK}"][:]
	return out, labels


def jk_cov(samples):
	d = samples - samples.mean(axis=0)
	return (NUM_JK - 1.0) / NUM_JK * (d.T @ d)


def corrmat(cov):
	s = np.sqrt(np.diag(cov))
	return cov / np.outer(s, s)


def main():
	mock = radial_alignment_box_mock()
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	mia, labels = run_measureia_jk(mock, scratch, here + "/")
	os.remove(scratch)

	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if binary and os.path.exists(binary):
		pc = run_corrpc_delete_one(mock, labels, binary)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["source"] = "corr_pc github.com/sukhdeep2/corr_pc"
			f.attrs["mock_seed"] = mock["seed"]
			f.attrs["num_jk"] = NUM_JK
			f.attrs["pi_max"] = PI_MAX
			f.attrs["sep_lims"] = SEP_LIMS
			f.attrs["n_pos_full"] = pc["n_pos_full"]
			for k in ["w_gp_full", "w_gg_full", "mult_gp_full", "mult_gg_full",
					  "w_gp", "w_gg", "mult_gp", "mult_gg", "n_pos"]:
				f[k] = pc[k]
		print(f"corr_pc results written to {REFERENCE_FILE}")
	else:
		print("corr_pc binary not found (set CORR_PC_BIN); comparing against "
			  "committed reference outputs")
		if not os.path.exists(REFERENCE_FILE):
			print(f"No reference file at {REFERENCE_FILE} — build corr_pc and rerun.")
			return
		pc = {}
		with h5py.File(REFERENCE_FILE, "r") as f:
			for k in ["w_gp_full", "w_gg_full", "mult_gp_full", "mult_gg_full",
					  "w_gp", "w_gg", "mult_gp", "mult_gg", "n_pos"]:
				pc[k] = f[k][:]
			pc["n_pos_full"] = f.attrs["n_pos_full"]

	print("\n--- full-sample signal (measureia / corr_pc, responsivity off) ---")
	rr_norm_full = (pc["n_pos_full"] - 1.0) / pc["n_pos_full"]
	print(f"w_g+ ratio   : {mia['w_gp_full'] / pc['w_gp_full']}")
	print(f"w_gg ratio   : {mia['w_gg_full'] / pc['w_gg_full']}")
	print(f"mult_g+ ratio: {mia['mult_gp_full'] * rr_norm_full / pc['mult_gp_full']}")
	mult_gg_adj = (mia["mult_gg_full"] + 1) * rr_norm_full - 1
	print(f"mult_gg ratio: {mult_gg_adj / pc['mult_gg_full']}")

	mapped = map_corrpc_to_jk(pc)
	print("\n--- delete-one realisations: measureia jackknife reconstruction vs")
	print("    corr_pc on the physically deleted catalogues (max |rel diff|) ---")
	for key in ["w_gp", "w_gg", "mult_gp", "mult_gg"]:
		scale = np.maximum(np.abs(mapped[key]),
						   1e-3 * np.max(np.abs(mapped[key])))
		print(f"{key:8s}: {np.max(np.abs(mia[key] - mapped[key]) / scale):.2e}")

	print("\n--- jackknife covariance (std ratios measureia / corr_pc-mapped) ---")
	for key in ["w_gp", "w_gg", "mult_gp", "mult_gg"]:
		cov_pc = jk_cov(mapped[key])
		std_pc = np.sqrt(np.diag(cov_pc))
		print(f"{key:8s}: {mia[f'std_{key}'] / std_pc}")
		print(f"{'':8s}  corr-matrix max |diff|: "
			  f"{np.max(np.abs(corrmat(mia[f'cov_{key}']) - corrmat(cov_pc))):.2e}")


if __name__ == "__main__":
	main()
