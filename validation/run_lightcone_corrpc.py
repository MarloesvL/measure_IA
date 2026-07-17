"""Cross-validation of MeasureIALightcone w_gg / w_g+ against corr_pc.

corr_pc (https://github.com/sukhdeep2/corr_pc, Singh 2021) in sky mode
(coordinates=0) measures xi(rp, pi) with the Landy-Szalay estimator and
explicit randoms; its g+ estimator S+D/RR - S+R/RR and its gg estimator
(SD - SR - DR)/RR + 1 are exactly measureia's lightcone 'galaxies'
estimator, so the same mock and configuration as the treecorr comparison
(run_lightcone_treecorr.py) is reused and the enforced comparison is w_g+
and w_gg.

Known convention/definition differences:

- Separation definitions: corr_pc uses rp = great-circle angle times the
  pair-mean comoving transverse distance and pi = the difference of radial
  comoving distances (plane-parallel), while measureia projects the 3D
  separation on the midpoint line of sight. Same class of curvature-term
  differences as treecorr's Rperp (bin-edge pair migration).
- Distance lookup: corr_pc reads distances from a tabulated z-grid with
  floor (no interpolation) lookup; the table is generated here from the
  same CCL cosmology on a dz=2e-6 grid so the quantisation (~0.01 Mpc) is
  negligible against the 10 Mpc pi bins.
- Ellipticity components: corr_pc rotates with
  e+ = cos(2 theta) e1 - sin(2 theta) e2. Applied to the raw
  survey-convention e1/e2 this reproduces measureia's radial-positive
  e+ directly (verified empirically: any single-component sign flip
  washes the w_g+ signal out to noise, the classic chirality failure
  mode; flipping both components only flips the overall sign). Note the
  corr_pc HSC example notebook flips e2 for its catalogue — do not apply
  that flip to survey-convention inputs.
- Signed-pi orientation: corr_pc internally reorders each pair-count run
  by sample size, so the SD and SR terms can carry mirrored signed-pi
  conventions; this cancels in w (sum over the symmetric pi range), which
  is why the comparison is enforced at the w level. The xi(rp, pi) grids
  are stored in the reference file for inspection only.
- No responsivity factor on either side (measureia lightcone default).

Set CORR_PC_BIN to the compiled binary (see the README for the no-MPI
build recipe) to (re)generate the reference outputs; without it the
measureia side is compared against the committed reference file.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pyccl as ccl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lightcone_treecorr as lc

DATASET = lc.DATASET
Z_MAX = 0.8
DZ_TABLE = 2e-6
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"lightcone_corrpc.hdf5"
)


def write_distance_table(path):
	"""Distance table in corr_pc's use_comoving format:
	z  H(z)[km/s/Mpc]  junk  D_C  D_M  on a uniform z-grid of step DZ_TABLE
	extending (as corr_pc's reader requires) to Z_MAX + 1."""
	z = np.arange(0.0, Z_MAX + 1.0 + 10 * DZ_TABLE, DZ_TABLE)
	a = 1.0 / (1.0 + z)
	H = ccl.h_over_h0(lc.COSMOLOGY, a) * lc.COSMOLOGY["h"] * 100.0
	chi = ccl.comoving_radial_distance(lc.COSMOLOGY, a)  # Mpc; = D_M (flat)
	np.savetxt(path, np.column_stack((z, H, np.zeros_like(z), chi, chi)))


def _write_sample(prefix, ra, dec, z, e1=None, e2=None):
	"""Write one corr_pc sample, z-sorted (corr_pc's data_sorted=1)."""
	order = np.argsort(z)
	np.savetxt(prefix + "_pos.dat", np.column_stack((ra[order], dec[order])))
	np.savetxt(prefix + "_z.dat", z[order])
	np.savetxt(prefix + "_jk.dat", np.zeros(len(z)), fmt="%i")
	if e1 is not None:
		# corr_pc's rotation e+ = cos(2t) e1 - sin(2t) e2 applied to the raw
		# survey-convention e1/e2 reproduces measureia's radial-positive e+
		# (verified empirically; no sign flip needed for this mock's inputs)
		np.savetxt(prefix + "_e.dat",
				   np.column_stack((e1[order], e2[order])))


def corrpc_write_inputs(data, randoms, workdir):
	pref = {k: os.path.join(workdir, k) for k in
			["shape", "density", "srand", "drand"]}
	_write_sample(pref["shape"], data["RA_shape_sample"],
				  data["DEC_shape_sample"], data["Redshift_shape_sample"],
				  data["e1"], data["e2"])
	_write_sample(pref["density"], data["RA"], data["DEC"], data["Redshift"])
	_write_sample(pref["srand"], randoms["RA_shape_sample"],
				  randoms["DEC_shape_sample"], randoms["Redshift_shape_sample"])
	_write_sample(pref["drand"], randoms["RA"], randoms["DEC"],
				  randoms["Redshift"])
	dist_file = os.path.join(workdir, "distances.dat")
	write_distance_table(dist_file)

	out_dir = os.path.join(workdir, "corr_data_out")
	os.makedirs(out_dir, exist_ok=True)
	out_pref = os.path.join(out_dir, "mia_")
	lines = [
		("which_corr", 1), ("coordinates", 0), ("estimator", 0),
		("data_sorted", 1), ("use_comoving", 1), ("do_jk", 0),
		("sig_crit", 0),
		("shape_pos", pref["shape"] + "_pos.dat"),
		("shape_z", pref["shape"] + "_z.dat"),
		("shape_e", pref["shape"] + "_e.dat"),
		("shape_wt", 0), ("shape_jk", pref["shape"] + "_jk.dat"),
		("density_patch", 0),
		("density_pos", pref["density"] + "_pos.dat"),
		("density_z", pref["density"] + "_z.dat"),
		("density_wt", 0), ("density_jk", pref["density"] + "_jk.dat"),
		("density_e", 0),
		("Srandom_pos", pref["srand"] + "_pos.dat"),
		("Srandom_z", pref["srand"] + "_z.dat"),
		("Srandoms_wt", 0), ("Srand_jk", pref["srand"] + "_jk.dat"),
		("Drandom_patch", 0),
		("Drandom_pos", pref["drand"] + "_pos.dat"),
		("Drandom_z", pref["drand"] + "_z.dat"),
		("Drandoms_wt", 0), ("drand_jk", pref["drand"] + "_jk.dat"),
		("distances", dist_file), ("patch_file", 0),
		("out_file", out_pref),
		("n_threads", 1),
		("n_shape", len(data["RA_shape_sample"])),
		("n_density", len(data["RA"])),
		("n_Srand", len(randoms["RA_shape_sample"])),
		("n_Drand", len(randoms["RA"])),
		("rand_subsample", 0),
		("n_jk", 0), ("n_patch", 0),
		("binR_min", lc.RP_LIMS[0]), ("binR_max", lc.RP_LIMS[1]),
		("n_bins", lc.NUM_BINS_RP), ("lin_bin", 0),
		("n_p_bin", lc.NUM_BINS_PI),
		("pmin", -lc.PI_MAX), ("pmax", lc.PI_MAX),
		("z_min", 0), ("z_max", Z_MAX), ("dz", DZ_TABLE),
		("z_sep_min", -0.1), ("z_sep_max", 0.1),
		("periodic_box", 0), ("box_size", 0),
	]
	inp = os.path.join(workdir, "mia.inp")
	with open(inp, "w") as f:
		for key, val in lines:
			f.write(f"{key}    {val}\n")
	return inp, out_pref


def run_corrpc(data, randoms, workdir, binary):
	"""Run corr_pc; return xi(rp, pi) grids and the pi-integrated w."""
	inp, out_pref = corrpc_write_inputs(data, randoms, workdir)
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	raw = np.loadtxt(out_pref + "bins2D_cross_final.dat")
	shape2d = (lc.NUM_BINS_RP, lc.NUM_BINS_PI)
	out = {
		"rp": raw[:, 0].reshape(shape2d)[:, 0],
		"pi": raw[:, 1].reshape(shape2d)[0],
		"npairs": raw[:, 2].reshape(shape2d),
		"xi_gg": raw[:, 3].reshape(shape2d),
		"xi_gp": raw[:, 5].reshape(shape2d),
	}
	dpi = 2.0 * lc.PI_MAX / lc.NUM_BINS_PI
	out["w_gg"] = np.sum(out["xi_gg"], axis=1) * dpi
	out["w_g_plus"] = np.sum(out["xi_gp"], axis=1) * dpi
	return out


def main():
	data, randoms, info, dist = lc.build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_corrpc_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	_, rp, wgp_mia, wgg_mia = lc.run_measureia(data, randoms, scratch, here + "/")
	os.remove(scratch)
	print(f"rp = {rp}")
	print(f"w_g+ (measureia) = {wgp_mia}")
	print(f"w_gg (measureia) = {wgg_mia}")

	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if binary and os.path.exists(binary):
		import tempfile
		with tempfile.TemporaryDirectory() as workdir:
			pc = run_corrpc(data, randoms, workdir, binary)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["source"] = "corr_pc github.com/sukhdeep2/corr_pc"
			f.attrs["dz_table"] = DZ_TABLE
			f.attrs["pi_max"] = lc.PI_MAX
			for k in ["rp", "pi", "npairs", "xi_gg", "xi_gp", "w_gg", "w_g_plus"]:
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
			for k in ["rp", "pi", "npairs", "xi_gg", "xi_gp", "w_gg", "w_g_plus"]:
				pc[k] = f[k][:]

	print("\n--- w comparison (measureia / corr_pc) ---")
	print(f"w_g+ corr_pc: {pc['w_g_plus']}")
	print(f"w_g+ ratio  : {wgp_mia / pc['w_g_plus']}")
	print(f"w_gg corr_pc: {pc['w_gg']}")
	print(f"w_gg ratio  : {wgg_mia / pc['w_gg']}")


if __name__ == "__main__":
	main()
