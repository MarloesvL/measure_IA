"""Cost of the box jackknife edge correction (jk_edge_correction) on a dense mock.

One measurement per process (so peak memory is per configuration), single core, tree backend.
For each statistic (w, multipoles) and N_jk (8, 27, 64) it reports the wall time of the full
measure_xi_* call with the correction off and on, the time spent in the correction itself, and
the peak memory.

Usage:
	python validation/run_box_jk_benchmark.py                  # all configurations, one at a time
	python validation/run_box_jk_benchmark.py KIND N_JK EDGE   # one configuration (worker)
"""

import os
import resource
import subprocess
import sys
import tempfile
import time

from measureia import MeasureIABox
from measureia.mocks import radial_alignment_box_mock

BOXSIZE = 150.0
N_CENTRALS = 3750      # x 9 galaxies per halo: n = 1e-2 (h/Mpc)^3 in L = 150 Mpc/h
SEP = [1.0, 60.0]
NUM_BINS_R = 10
NUM_BINS_PI = 6        # pi bins (w) / mu bins are set separately below
PI_MAX = 30.0
NUM_BINS_MU = 10


def worker(kind, n_jk, edge):
	mock = radial_alignment_box_mock(n_centrals=N_CENTRALS, n_sat=8, sigma_sat=2.0, boxsize=BOXSIZE, seed=1000)
	data = {k: mock[k] for k in ["Position", "Position_shape_sample", "Axis_Direction", "q", "LOS"]}
	with tempfile.TemporaryDirectory() as tmp:
		ia = MeasureIABox(data, os.path.join(tmp, "b.hdf5"), separation_limits=SEP, num_bins_r=NUM_BINS_R,
						  num_bins_pi=NUM_BINS_PI if kind == "w" else NUM_BINS_MU, pi_max=PI_MAX,
						  boxsize=BOXSIZE, num_nodes=1)
		spent = [0.0]
		apply = ia._apply_box_jk_edge_correction

		def timed(*a, **k):
			t = time.time()
			apply(*a, **k)
			spent[0] += time.time() - t

		ia._apply_box_jk_edge_correction = timed
		fn = ia.measure_xi_w if kind == "w" else ia.measure_xi_multipoles
		t0 = time.time()
		fn("b", "both", num_jk=n_jk, temp_file_path=tmp + "/", jk_edge_correction=edge)
		total = time.time() - t0
	peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS reports bytes
	print(f"RESULT {kind} {n_jk} {edge} {len(data['Position'])} {len(data['Position_shape_sample'])} "
		  f"{total:.1f} {spent[0]:.1f} {peak:.2f}", flush=True)


def main():
	rows = []
	for kind in ("w", "multipoles"):
		for n_jk in (8, 27, 64):
			for edge in (False, True):
				out = subprocess.run([sys.executable, "-W", "ignore", __file__, kind, str(n_jk), str(edge)],
									 capture_output=True, text=True, check=True).stdout
				line = [ln for ln in out.splitlines() if ln.startswith("RESULT")][0].split()[1:]
				rows.append(line)
				print(" ".join(line), flush=True)
	print("\n| statistic | N_jk | off: total | on: total | of which correction | added | peak memory off / on |")
	print("|---|---|---|---|---|---|---|")
	for i in range(0, len(rows), 2):
		off, on = rows[i], rows[i + 1]
		t_off, t_on, t_corr = float(off[5]), float(on[5]), float(on[6])
		print(f"| {off[0]} | {off[1]} | {t_off:.0f} s | {t_on:.0f} s | {t_corr:.0f} s | "
			  f"{100 * (t_on / t_off - 1):.0f}% | {float(off[7]):.2f} / {float(on[7]):.2f} GB |")
	print(f"\nD = {rows[0][3]}, S = {rows[0][4]} galaxies; 1 core, tree backend")


if __name__ == "__main__":
	if len(sys.argv) > 1:
		worker(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "True")
	else:
		main()
