"""Cross-validation of MeasureIALightcone.measure_xi_multipoles vs corr_pc.

corr_pc (https://github.com/sukhdeep2/corr_pc, Singh 2021) in sky r-mu mode
(coordinates=1) measures xi_gg(r, mu) and xi_g+(r, mu) with the identical
'galaxies' estimator and explicit randoms measureia's lightcone multipole
pipeline uses (see run_lightcone_corrpc.py for the estimator mapping and
input conventions — same mock, same catalogues, same distance table, same
no-flip e1/e2 finding, same DRs patch requirement).

measureia's associated-Legendre integration is then applied to the corr_pc
grid and the multipoles are compared. Because P_0 and the l=2 associated
Legendre weight are even in mu, corr_pc's internal signed-pi reordering
(mirrored mu conventions between count terms; see run_lightcone_corrpc.py)
cancels exactly in the multipoles, so the enforced comparison is at the
multipole level; the (mu-mirrorable) grids are stored for inspection only.

Set CORR_PC_BIN to the compiled binary (README build recipe, DRs patch
required) to (re)generate the reference outputs; without it the measureia
side is compared against the committed reference file.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lightcone_treecorr as lc
from measureia.mocks import radial_alignment_lightcone_mock
from run_box_multipoles_corrpc import legendre_multipole
from run_lightcone_corrpc import corrpc_write_inputs

from measureia import MeasureIALightcone

# The (r, mu) grid dilutes the randoms over many more cells than the
# pi-integrated w, so this comparison uses a larger r_min and more randoms
# than the w-level scripts: with the defaults the smallest cells have empty
# empirical RR (xi undefined) on the measureia side.
R_LIMS = [2.0, 20.0]  # 3D separation r, Mpc
NUM_BINS_R = 6
NUM_BINS_MU = 10  # mu in [-1, 1]
N_RANDOMS_FACTOR = 25
DATASET = "lc_multipoles_corrpc_mock"
REFERENCE_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "reference_outputs",
	"lightcone_multipoles_corrpc.hdf5"
)


def build_catalogues():
	"""Same mock as the treecorr/corr_pc w comparisons but with denser
	randoms (see module constants)."""
	data, randoms, info = radial_alignment_lightcone_mock(
		n_randoms_factor=N_RANDOMS_FACTOR)
	for d in (data, randoms):
		d["Redshift"] = lc._redshift_of_chi(d.pop("r_com"))
		d["Redshift_shape_sample"] = lc._redshift_of_chi(d.pop("r_com_shape_sample"))
	return data, randoms, info


def run_measureia(data, randoms, output_file, temp_path):
	"""measureia lightcone multipoles; returns grids and multipoles."""
	ia = MeasureIALightcone(
		data={k: v for k, v in data.items()},
		randoms_data={k: v for k, v in randoms.items()},
		output_file_name=output_file,
		separation_limits=R_LIMS,
		num_bins_r=NUM_BINS_R,
		num_bins_pi=NUM_BINS_MU,
		pi_max=R_LIMS[1],
		num_nodes=1,
	)
	ia.measure_xi_multipoles("galaxies", DATASET, "both", 
							 tree=True, cosmology=lc.COSMOLOGY, over_h=False,
							 temp_file_path=temp_path)
	out = {}
	with h5py.File(output_file, "r") as f:
		out["r"] = f[f"multipoles/xi_g_plus/{DATASET}_r"][:]
		out["mu"] = f[f"multipoles/xi_g_plus/{DATASET}_mu_r"][:]
		out["xi_gp"] = f[f"multipoles/xi_g_plus/{DATASET}"][:]
		out["xi_gg"] = f[f"multipoles/xi_gg/{DATASET}"][:]
		out["multipole_gp"] = f[f"multipoles_g_plus/{DATASET}"][:]
		out["multipole_gg"] = f[f"multipoles_gg/{DATASET}"][:]
	return out


def run_corrpc(data, randoms, workdir, binary):
	"""Run corr_pc in sky r-mu mode; return xi(r, mu) grids and multipoles."""
	inp, out_pref = corrpc_write_inputs(data, randoms, workdir)
	# switch the shared input file to r-mu coordinates and this script's bins
	with open(inp) as f:
		text = f.read()
	for old, new in [
		("coordinates    0", "coordinates    1"),
		(f"binR_min    {lc.RP_LIMS[0]}", f"binR_min    {R_LIMS[0]}"),
		(f"binR_max    {lc.RP_LIMS[1]}", f"binR_max    {R_LIMS[1]}"),
		(f"n_bins    {lc.NUM_BINS_RP}", f"n_bins    {NUM_BINS_R}"),
		(f"n_p_bin    {lc.NUM_BINS_PI}", f"n_p_bin    {NUM_BINS_MU}"),
		(f"pmin    {-lc.PI_MAX}", "pmin    -1"),
		(f"pmax    {lc.PI_MAX}", "pmax    1"),
	]:
		assert old in text, old
		text = text.replace(old, new)
	with open(inp, "w") as f:
		f.write(text)
	subprocess.run([binary, inp], check=True, cwd=workdir,
				   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	raw = np.loadtxt(out_pref + "bins2D_cross_final.dat")
	shape2d = (NUM_BINS_R, NUM_BINS_MU)
	out = {
		"r": raw[:, 0].reshape(shape2d)[:, 0],
		"mu": raw[:, 1].reshape(shape2d)[0],
		"npairs": raw[:, 2].reshape(shape2d),
		"xi_gg": raw[:, 3].reshape(shape2d),
		"xi_gp": raw[:, 5].reshape(shape2d),
	}
	out["multipole_gp"] = legendre_multipole(out["xi_gp"], out["mu"], 2, 2)
	out["multipole_gg"] = legendre_multipole(out["xi_gg"], out["mu"], 0, 0)
	return out


def main():
	data, randoms, info = build_catalogues()
	here = os.path.dirname(os.path.abspath(__file__))

	scratch = os.path.join(here, f"{DATASET}_tmp.hdf5")
	if os.path.exists(scratch):
		os.remove(scratch)
	mia = run_measureia(data, randoms, scratch, here + "/")
	os.remove(scratch)
	print(f"r = {mia['r']}")
	print(f"multipole_g+ (measureia) = {mia['multipole_gp']}")
	print(f"multipole_gg (measureia) = {mia['multipole_gg']}")

	binary = os.environ.get("CORR_PC_BIN") or shutil.which("corr_pc")
	if binary and os.path.exists(binary):
		import tempfile
		with tempfile.TemporaryDirectory() as workdir:
			pc = run_corrpc(data, randoms, workdir, binary)
		os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
		with h5py.File(REFERENCE_FILE, "w") as f:
			f.attrs["source"] = "corr_pc github.com/sukhdeep2/corr_pc"
			f.attrs["r_lims"] = R_LIMS
			for k in ["r", "mu", "npairs", "xi_gg", "xi_gp",
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
			for k in ["r", "mu", "npairs", "xi_gg", "xi_gp",
					  "multipole_gp", "multipole_gg"]:
				pc[k] = f[k][:]

	print("\n--- multipole comparison (measureia / corr_pc) ---")
	print(f"xi_g+,2 corr_pc: {pc['multipole_gp']}")
	print(f"xi_g+,2 ratio  : {mia['multipole_gp'] / pc['multipole_gp']}")
	print(f"xi_gg,0 corr_pc: {pc['multipole_gg']}")
	print(f"xi_gg,0 ratio  : {mia['multipole_gg'] / pc['multipole_gg']}")


if __name__ == "__main__":
	main()
