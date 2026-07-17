"""Cross-validation of MeasureIABox.measure_xi_multipoles against corr_pc.

corr_pc (https://github.com/sukhdeep2/corr_pc, Singh 2021) is the C++ code
used for the original multipole validation of this package. In periodic-box
mode with coordinates=7 it measures xi_gg(r, mu) and xi_g+(r, mu) on the
same log-r / linear-mu grid measureia's multipole estimator uses, with the
same natural estimators and analytic RR:

	xi_gg = SD / (N_S N_D V_bin/V) - 1
	xi_g+ = Sum(w e+) / (N_S N_D V_bin/V)

The multipoles are then obtained here by applying measureia's own
associated-Legendre integration to the corr_pc grid (corr_pc itself stops
at the grid), so the comparison covers both the grid and the integration.

Known convention differences (documented in the README):

- Responsivity: measureia divides S+ terms by 2R (box default); corr_pc
  does not. Compared as xi_g+^measureia * 2R == xi_g+^corr_pc.
- Analytic RR normalisation: measureia's get_random_pairs_r_mur uses
  (N_pos - 1) * N_shape; corr_pc uses N_pos * N_shape. Deterministic
  factor (N_pos - 1)/N_pos applied to the measureia side.
- Ellipticity components: corr_pc rotates with
  e+ = cos(2 theta) e1 - sin(2 theta) e2, so its input convention is
  e1 = e cos(2 phi_axis), e2 = -e sin(2 phi_axis) (opposite chirality to
  the survey shear convention); with that input its e+ is radial-positive,
  matching measureia.

Building corr_pc needs no MPI installation: the two MPI calls are
satisfied by a stub header (see README section). Set the environment
variable CORR_PC_BIN to the compiled binary to (re)generate the reference
outputs; without it, the measureia side is compared against the committed
reference file.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_catalogues import radial_alignment_box_mock, responsivity

from measureia import MeasureIABox

# --- comparison configuration (must match the committed reference file) ---
R_LIMS = [0.5, 20.0]  # 3D separation r, Mpc/h
NUM_BINS_R = 10
NUM_BINS_MU = 20  # mu in [-1, 1]
DATASET = "box_multipoles_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"box_multipoles_corrpc.hdf5"
)


def run_measureia(mock, output_file):
	"""measureia multipoles on the mock; returns grids and multipoles."""
	data = {k: mock[k] for k in
			["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	ia = MeasureIABox(
		data, output_file,
		simulation=None, snapshot=None,
		separation_limits=R_LIMS,
		num_bins_r=NUM_BINS_R,
		num_bins_pi=NUM_BINS_MU,
		pi_max=R_LIMS[1],
		boxsize=mock["boxsize"],
		num_nodes=1,
	)
	ia.measure_xi_multipoles(DATASET, "both", 0, temp_file_path=False)
	out = {}
	with h5py.File(output_file, "r") as f:
		out["r"] = f[f"multipoles/xi_g_plus/{DATASET}_r"][:]
		out["mu"] = f[f"multipoles/xi_g_plus/{DATASET}_mu_r"][:]
		out["xi_gp"] = f[f"multipoles/xi_g_plus/{DATASET}"][:]
		out["xi_gg"] = f[f"multipoles/xi_gg/{DATASET}"][:]
		out["multipole_gp"] = f[f"multipoles_g_plus/{DATASET}"][:]
		out["multipole_gg"] = f[f"multipoles_gg/{DATASET}"][:]
	return out


def legendre_multipole(xi, mu_centers, l, sab):
	"""measureia's multipole integration (measure_IA_base._measure_multipoles)
	applied to an arbitrary xi(r, mu) grid: l=0, sab=0 for gg; l=2, sab=2
	for g+."""
	import math

	from scipy.special import lpmn

	L = np.array([lpmn(l, sab, m)[0][-1, -1] for m in mu_centers])
	dmu = 2.0 / len(mu_centers)
	weight = ((2 * l + 1) / 2.0
			  * math.factorial(l - sab) / math.factorial(l + sab))
	return weight * np.sum(xi * L[None, :] * dmu, axis=1)


def corrpc_write_inputs(mock, workdir):
	"""Write corr_pc input files for the mock (density = position sample,
	shape = shape sample), z-sorted as corr_pc's data_sorted=1 requires."""
	e = (1 - mock["q"] ** 2) / (1 + mock["q"] ** 2)
	ax = mock["Axis_Direction"]
	phi = np.arctan2(ax[:, 1], ax[:, 0])  # projected axis angle, LOS = z
	e1 = e * np.cos(2 * phi)
	e2 = -e * np.sin(2 * phi)  # corr_pc chirality (see module docstring)

	dens = mock["Position"][np.argsort(mock["Position"][:, 2])]
	order_s = np.argsort(mock["Position_shape_sample"][:, 2])
	shape = mock["Position_shape_sample"][order_s]

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
	lines = [
		("which_corr", 1), ("coordinates", 7), ("estimator", 0),
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
		("binR_min", R_LIMS[0]), ("binR_max", R_LIMS[1]),
		("n_bins", NUM_BINS_R), ("lin_bin", 0),
		("n_p_bin", NUM_BINS_MU), ("pmin", -1), ("pmax", 1),
		("z_min", 0), ("z_max", 2 * mock["boxsize"]), ("dz", 0.0001),
		("z_sep_min", -2 * mock["boxsize"]), ("z_sep_max", 2 * mock["boxsize"]),
		("periodic_box", 1), ("box_size", mock["boxsize"]),
	]
	inp = os.path.join(workdir, "mia.inp")
	with open(inp, "w") as f:
		for key, val in lines:
			f.write(f"{key}    {val}\n")
	return inp, out_pref


def run_corrpc(mock, workdir, binary):
	"""Run the corr_pc binary; return its xi(r, mu) grids."""
	inp, out_pref = corrpc_write_inputs(mock, workdir)
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	raw = np.loadtxt(out_pref + "bins2D_cross_final.dat")
	shape2d = (NUM_BINS_R, NUM_BINS_MU)
	return {
		"r": raw[:, 0].reshape(shape2d)[:, 0],
		"mu": raw[:, 1].reshape(shape2d)[0],
		"npairs": raw[:, 2].reshape(shape2d),
		"xi_gg": raw[:, 3].reshape(shape2d),
		"xi_gp": raw[:, 5].reshape(shape2d),
		"xi_gx": raw[:, 7].reshape(shape2d),
	}


def main():
	mock = radial_alignment_box_mock()
	R = responsivity(mock)
	# measureia's analytic RR uses (N_pos - 1) * N_shape; corr_pc N_pos * N_shape
	n_pos = len(mock["Position"])
	rr_norm = (n_pos - 1.0) / n_pos

	here = os.path.dirname(os.path.abspath(__file__))
	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	mia = run_measureia(mock, scratch)
	os.remove(scratch)
	print(f"responsivity 2R = {2 * R:.8f}, RR norm (N-1)/N = {rr_norm:.8f}")
	print(f"r = {mia['r']}")
	print(f"multipole_g+ (measureia) = {mia['multipole_gp']}")
	print(f"multipole_gg (measureia) = {mia['multipole_gg']}")

	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if binary and os.path.exists(binary):
		import tempfile
		with tempfile.TemporaryDirectory() as workdir:
			pc = run_corrpc(mock, workdir, binary)
		pc["multipole_gp"] = legendre_multipole(pc["xi_gp"], pc["mu"], 2, 2)
		pc["multipole_gg"] = legendre_multipole(pc["xi_gg"], pc["mu"], 0, 0)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["mock_seed"] = mock["seed"]
			f.attrs["responsivity_R"] = R
			f.attrs["rr_norm"] = rr_norm
			f.attrs["r_lims"] = R_LIMS
			f.attrs["source"] = "corr_pc github.com/sukhdeep2/corr_pc"
			for k in ["r", "mu", "npairs", "xi_gg", "xi_gp", "xi_gx",
					  "multipole_gp", "multipole_gg"]:
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
			for k in ["r", "mu", "npairs", "xi_gg", "xi_gp", "xi_gx",
					  "multipole_gp", "multipole_gg"]:
				pc[k] = f[k][:]

	print("\n--- xi(r, mu) grid comparison (measureia adjusted to corr_pc "
		  "conventions) ---")
	xi_gp_adj = mia["xi_gp"] * 2 * R * rr_norm
	xi_gg_adj = (mia["xi_gg"] + 1) * rr_norm - 1
	with np.errstate(invalid="ignore", divide="ignore"):
		print(f"xi_g+ ratio range: "
			  f"{np.nanmin(xi_gp_adj / pc['xi_gp']):.6f} .. "
			  f"{np.nanmax(xi_gp_adj / pc['xi_gp']):.6f}")
		print(f"xi_gg+1 ratio range: "
			  f"{np.nanmin((xi_gg_adj + 1) / (pc['xi_gg'] + 1)):.6f} .. "
			  f"{np.nanmax((xi_gg_adj + 1) / (pc['xi_gg'] + 1)):.6f}")

	print("\n--- multipole comparison ---")
	mp_gp_adj = mia["multipole_gp"] * 2 * R * rr_norm
	print(f"xi_g+,2 corr_pc   : {pc['multipole_gp']}")
	print(f"xi_g+,2 ratio     : {mp_gp_adj / pc['multipole_gp']}")
	# the (N-1)/N RR factor shifts xi_gg by a constant; l=0 integration of a
	# constant is the constant, so adjust the monopole accordingly
	mp_gg_adj = (mia["multipole_gg"] + 1) * rr_norm - 1
	print(f"xi_gg,0 corr_pc   : {pc['multipole_gg']}")
	print(f"xi_gg,0 ratio     : {mp_gg_adj / pc['multipole_gg']}")


if __name__ == "__main__":
	main()
