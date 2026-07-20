"""A/B equivalence harness for the pair-kernel consolidation.

For each backend method migrated onto ``pair_kernel`` (see ``docs/REFACTOR_PLAN.md``),
the pre-kernel implementation is preserved under a ``_legacy_<name>`` copy. This harness
runs the legacy method and the kernel-backed method on the *same* fixture object, writing
under two different ``dataset_name`` values into the same HDF5 file, then compares **every**
dataset in the affected groups key-by-key.

Tree/mp paths are required to be bit-identical (the committed validation references were
generated from them), so comparisons use ``assert_array_equal``.

When a migrated path has been green in the full suite, its legacy copy and the corresponding
test entry here are deleted in the commit that migrates the next path.
"""

import numpy as np
import h5py
import pytest


def _collect_datasets(path):
    """Return {full_dataset_path: ndarray} for every dataset in the HDF5 file."""
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
    return out


def _index_by_prefix(datasets, name):
    """Group datasets written under ``dataset_name == name`` by (parent group, suffix).

    A leaf like ``legacyrun_SplusD`` under group ``.../xi_g_plus`` maps to
    key ``(".../xi_g_plus", "_SplusD")`` so the same physical quantity written under a
    different ``dataset_name`` lines up for comparison.
    """
    out = {}
    for full, arr in datasets.items():
        parent, _, leaf = full.rpartition("/")
        if leaf == name or leaf.startswith(name):
            # guard against a different name that merely shares a prefix
            suffix = leaf[len(name):]
            if leaf == name or suffix.startswith("_"):
                out[(parent, suffix)] = arr
    return out


def _assert_groups_bit_identical(output_file_name, name_legacy, name_kernel):
    datasets = _collect_datasets(output_file_name)
    legacy = _index_by_prefix(datasets, name_legacy)
    kernel = _index_by_prefix(datasets, name_kernel)
    assert legacy, "legacy method wrote no datasets — harness misconfigured"
    assert set(legacy) == set(kernel), (
        f"dataset key sets differ:\n only legacy: {set(legacy) - set(kernel)}\n"
        f" only kernel: {set(kernel) - set(legacy)}"
    )
    for key in legacy:
        np.testing.assert_array_equal(
            legacy[key], kernel[key],
            err_msg=f"mismatch in group '{key[0]}', suffix '{key[1]}'",
        )


def _non_contiguous_masks(n_pos, n_shape):
    """A weight-mask-exercising mask dict: non-contiguous selections so the
    weight-default-to-coordinate-mask path is actually tested."""
    pos_mask = np.zeros(n_pos, dtype=bool)
    pos_mask[::2] = True          # every other position
    shape_mask = np.zeros(n_shape, dtype=bool)
    shape_mask[1::3] = True       # every third shape, offset
    return pos_mask, shape_mask


# ---------------------------------------------------------------------------
# Step 1: _measure_xi_rp_pi_box_tree  (box, (rp, pi), tree, no jk, no mp)
# ---------------------------------------------------------------------------

class TestKernelEquivalenceBoxRpPiTree:
    NAME_LEGACY = "legacyrun"
    NAME_KERNEL = "kernelrun"

    def test_no_mask(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_measure_xi_rp_pi_box_tree(dataset_name=self.NAME_LEGACY, masks=None)
        obj._measure_xi_rp_pi_box_tree(dataset_name=self.NAME_KERNEL, masks=None)
        _assert_groups_bit_identical(obj.output_file_name, self.NAME_LEGACY, self.NAME_KERNEL)

    def test_ellipticity_definition(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_measure_xi_rp_pi_box_tree(dataset_name=self.NAME_LEGACY, masks=None,
                                              ellipticity='ellipticity')
        obj._measure_xi_rp_pi_box_tree(dataset_name=self.NAME_KERNEL, masks=None,
                                       ellipticity='ellipticity')
        _assert_groups_bit_identical(obj.output_file_name, self.NAME_LEGACY, self.NAME_KERNEL)

    def test_non_contiguous_mask(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        pos_mask, shape_mask = _non_contiguous_masks(obj.Num_position, obj.Num_shape)
        # fresh dict per call: both methods inject default weight masks in place
        masks_legacy = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        masks_kernel = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        obj._legacy_measure_xi_rp_pi_box_tree(dataset_name=self.NAME_LEGACY, masks=masks_legacy)
        obj._measure_xi_rp_pi_box_tree(dataset_name=self.NAME_KERNEL, masks=masks_kernel)
        _assert_groups_bit_identical(obj.output_file_name, self.NAME_LEGACY, self.NAME_KERNEL)

    def test_responsivity_off(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.responsivity_correction = False
        obj._legacy_measure_xi_rp_pi_box_tree(dataset_name=self.NAME_LEGACY, masks=None)
        obj._measure_xi_rp_pi_box_tree(dataset_name=self.NAME_KERNEL, masks=None)
        _assert_groups_bit_identical(obj.output_file_name, self.NAME_LEGACY, self.NAME_KERNEL)
