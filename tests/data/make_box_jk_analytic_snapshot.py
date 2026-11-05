"""
Regenerates tests/data/box_jk_analytic_snapshot.npz: the per-realisation DR, RD and S+R terms
of ``box_jk_analytic.analytic_dr_terms`` for a set of small configurations, as produced by the
module BEFORE its r-mu speed-up (dev at bbcd21b), with 16 Gauss-Legendre nodes per segment.

test_box_jk_analytic_geometry.py::TestSnapshot checks that the module (with _N_GL_RMU = 16)
still reproduces these numbers to rounding, so that memory-layout, chunking and
parallelisation changes cannot change the result. Do not regenerate it with a changed
module: the point is to pin the behaviour before the rewrite.

Usage (from the repository root, on a checkout of bbcd21b):
    python tests/data/make_box_jk_analytic_snapshot.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOXSIZE = 150.0
R_BINS = np.logspace(0, np.log10(60.0), 6)
PI_BINS = np.linspace(-30.0, 30.0, 7)
MU_BINS = np.linspace(-1.0, 1.0, 7)
MU_BINS_ASYM = np.array([-1.0, -0.3, 0.1, 0.6, 1.0])   # not symmetric: one pass per sample

# name: (binning, bins2, L_subboxes, weighted, rp_cut, samples, seed)
CASES = {
    "rmu_L2_w_same": ("rmu", MU_BINS, 2, True, 0.0, "same", 101),
    "rmu_L3_w_partial": ("rmu", MU_BINS, 3, True, 0.0, "partial", 102),
    "rmu_L4_rpcut_separate": ("rmu", MU_BINS, 4, False, 3.0, "separate", 103),
    "rmu_L3_asym": ("rmu", MU_BINS_ASYM, 3, True, 0.0, "partial", 104),
    "rppi_L3_w_partial": ("rppi", PI_BINS, 3, True, 0.0, "partial", 105),
    "rppi_L4_rpcut_separate": ("rppi", PI_BINS, 4, False, 3.0, "separate", 106),
}


def labels(points, L_subboxes):
    i = np.floor(points / (BOXSIZE / L_subboxes)).astype(int)
    return i[:, 0] * L_subboxes ** 2 + i[:, 1] * L_subboxes + i[:, 2]


def case_inputs(name):
    """Positional and keyword arguments of analytic_dr_terms for one case."""
    binning, bins2, L, weighted, rp_cut, samples, seed = CASES[name]
    rng = np.random.default_rng(seed)
    n_pos, n_shape = 150, 110
    pos = rng.uniform(0, BOXSIZE, (n_pos, 3))
    shape = rng.uniform(0, BOXSIZE, (n_shape, 3))
    if samples == "same":
        shape = pos.copy()
        n_shape = n_pos
    elif samples == "partial":
        shape[:50] = pos[20:70]
    theta = rng.uniform(0, np.pi, n_shape)
    e = rng.uniform(0, 0.8, n_shape)
    w = rng.uniform(0.5, 2.0, n_pos) if weighted else None
    w_s = rng.uniform(0.5, 2.0, n_shape) if weighted else None
    args = (pos, shape, theta, e, labels(pos, L), labels(shape, L), L)
    kwargs = dict(weight=w, weight_shape=w_s, boxsize=BOXSIZE, r_bins=R_BINS, bins2=bins2,
                  binning=binning, rp_cut=rp_cut)
    return args, kwargs


def main():
    from measureia import box_jk_analytic as bja
    assert bja._N_GL_RMU == 16
    out = {}
    for name in CASES:
        args, kwargs = case_inputs(name)
        for term, v in zip(["DR", "RD", "SpR"], bja.analytic_dr_terms(*args, **kwargs)):
            out[f"{name}/{term}"] = v
    np.savez_compressed(os.path.join(HERE, "box_jk_analytic_snapshot.npz"), **out)
    print(f"wrote {len(out)} arrays")


if __name__ == "__main__":
    main()
