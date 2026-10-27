"""
Regenerates tests/data/box_jk_before_edge_correction.npz: every dataset that MeasureIABox
writes for a small jackknife run, as produced by measureia BEFORE the analytic box-jackknife
edge correction was added (commit 636080d on dev).

test_box_jk_edge_correction.py::TestBackwardCompatibility::test_off_reproduces_previous_release
checks that ``jk_edge_correction=False`` still reproduces this snapshot. Do not regenerate it
with a version that includes the edge correction: the point is to pin the old behaviour.

Usage (from the repository root, on a checkout of 636080d):
    python tests/data/make_box_jk_snapshot.py
"""

import os
import sys
import tempfile
import pathlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tests/, for the test module's helpers
import test_box_jk_edge_correction as t  # noqa: E402


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for kind in ["w", "multipoles"]:
            for corr_type in ["both", "gg"]:
                obj = t._box(tmp, t._mock(6, n_shape=150, weights=True), f"{kind}_{corr_type}")
                fn = obj.measure_xi_w if kind == "w" else obj.measure_xi_multipoles
                fn("d", corr_type, num_jk=t.NUM_JK, temp_file_path=False)
                for path, v in t._all_datasets(obj).items():
                    out[f"{kind}/{corr_type}/{path}"] = v
    np.savez_compressed(os.path.join(HERE, "box_jk_before_edge_correction.npz"), **out)
    print(f"wrote {len(out)} datasets")


if __name__ == "__main__":
    main()
