"""Read a MeasureIA output file and plot the result with jackknife error bars.

This continues from ``example_measure_IA_box.py``, which writes
``./example_IA_box.hdf5`` with dataset ``"mock"`` and ``num_jk=27``. Run that
script first, from this directory, then run this one. The ``ReadData`` class
loads the measured correlation functions (and their jackknife covariance) back
into convenient attributes.
"""
import matplotlib.pyplot as plt
from measureia import ReadData

# These must match the run that produced the output file.
simulation = None              # simulation identifier used at measurement time (None for the mock)
catalogue = "example_IA_box"   # output file name without the .hdf5 extension
data_path = "./"               # folder holding <catalogue>.hdf5  (reads data_path + catalogue + ".hdf5")
snapshot = None                # snapshot label; would select the "Snapshot_99" group if 99 was given
dataset_name = "mock"          # dataset name passed to measure_xi_w / measure_xi_multipoles
num_jk = 27                    # number of jackknife regions used (None if no covariance was measured)

# Load the output. read_MeasureIA_output fills whichever of w_gp, w_gg, multipoles_gp,
# multipoles_gg (and their rp/r bins, covariance and error attributes) are present in the file.
reader = ReadData(simulation=simulation, catalogue=catalogue, snapshot=snapshot, data_path=data_path)
reader.read_MeasureIA_output(dataset_name, num_jk)

# Plot w_g+ against r_p with jackknife error bars (errors_w_gp is the sqrt of the covariance diagonal).
fig, ax = plt.subplots()
ax.errorbar(reader.rp, reader.rp * reader.w_gp, yerr=reader.rp * reader.errors_w_gp,
			marker="o", linestyle="none", capsize=3, label=r"$r_p\, w_{g+}$")
ax.set_xscale("log")
ax.axhline(0.0, color="grey", linewidth=0.8)
ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
ax.set_ylabel(r"$r_p\, w_{g+}$ [Mpc/$h$]")
ax.legend()
fig.tight_layout()
fig.savefig("./w_g_plus.png", dpi=150)
print("saved ./w_g_plus.png")
