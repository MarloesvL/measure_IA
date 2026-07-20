"""A/B equivalence harness for the pair-kernel consolidation.

For each backend method migrated onto ``pair_kernel`` (see ``docs/REFACTOR_PLAN.md``),
the pre-kernel implementation is preserved under a ``_legacy_<name>`` copy. This harness
runs the legacy method and the kernel-backed method on the *same* fixture object, writing
under two different ``dataset_name`` values into the same HDF5 file, then compares **every**
dataset in the affected groups key-by-key.

Tree/mp paths are required to be bit-identical (the committed validation references were
generated from them), so those comparisons use ``assert_array_equal``. The brute path runs
on the shape-chunk order rather than the legacy position-outer order (a deliberate
consolidation choice, REFACTOR_PLAN.md section 4), so it is compared with
``assert_allclose(rtol=1e-10, atol=1e-13)``.

When a migrated path has been green in the full suite, its legacy copy and the corresponding
test entry here are deleted in the commit that migrates the next path.
"""

import os

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


def _assert_groups_equivalent(output_file_name, name_legacy, name_kernel, *, mode):
    """Compare every dataset written under two dataset_names across the affected groups.

    mode='exact' uses assert_array_equal (tree/mp paths are bit-identical); mode='allclose'
    uses assert_allclose(rtol=1e-10, atol=1e-13) for the brute path, which runs on the
    shape-chunk order rather than the legacy brute's position-outer order and so matches only
    to floating-point tolerance (REFACTOR_PLAN.md section 4).
    """
    datasets = _collect_datasets(output_file_name)
    legacy = _index_by_prefix(datasets, name_legacy)
    kernel = _index_by_prefix(datasets, name_kernel)
    assert legacy, "legacy method wrote no datasets — harness misconfigured"
    assert set(legacy) == set(kernel), (
        f"dataset key sets differ:\n only legacy: {set(legacy) - set(kernel)}\n"
        f" only kernel: {set(kernel) - set(legacy)}"
    )
    for key in legacy:
        msg = f"mismatch in group '{key[0]}', suffix '{key[1]}'"
        if mode == "exact":
            np.testing.assert_array_equal(legacy[key], kernel[key], err_msg=msg)
        else:
            np.testing.assert_allclose(legacy[key], kernel[key], rtol=1e-10, atol=1e-13,
                                       err_msg=msg)


def _assert_groups_bit_identical(output_file_name, name_legacy, name_kernel):
    _assert_groups_equivalent(output_file_name, name_legacy, name_kernel, mode="exact")


def _assert_groups_allclose(output_file_name, name_legacy, name_kernel):
    _assert_groups_equivalent(output_file_name, name_legacy, name_kernel, mode="allclose")


def _non_contiguous_masks(n_pos, n_shape):
    """A weight-mask-exercising mask dict: non-contiguous selections so the
    weight-default-to-coordinate-mask path is actually tested."""
    pos_mask = np.zeros(n_pos, dtype=bool)
    pos_mask[::2] = True          # every other position
    shape_mask = np.zeros(n_shape, dtype=bool)
    shape_mask[1::3] = True       # every third shape, offset
    return pos_mask, shape_mask


def _full_mask_dict(obj):
    """A full non-contiguous mask dict covering every data key the mp temp-write loop
    stores (all keys except LOS). Position/weight are position-aligned; the shape-sample,
    axis-direction, q and (unused-but-stored) Mass masks are shape-aligned."""
    pos_mask, shape_mask = _non_contiguous_masks(obj.Num_position, obj.Num_shape)
    return {
        "Position":              pos_mask,
        "weight":                pos_mask,
        "Position_shape_sample": shape_mask,
        "weight_shape_sample":   shape_mask,
        "Axis_Direction":        shape_mask,
        "q":                     shape_mask,
        "Mass":                  shape_mask,
    }


NAME_LEGACY = "legacyrun"
NAME_KERNEL = "kernelrun"


# ---------------------------------------------------------------------------
# Step 3: finish the box w (rp, pi) family
#   - _measure_xi_rp_pi_box_brute        (backend='brute', allclose vs legacy brute)
#   - _count_pairs_xi_rp_pi_box_brute    (DD-only, backend='brute', allclose)
#   - _count_pairs_xi_rp_pi_box_tree     (DD-only, backend='tree', bit-identical)
#   - _count_pairs_xi_rp_pi_box_batch/_multiprocessing (DD-only mp, bit-identical)
#
# The brute path runs the shape-chunk order rather than the legacy position-outer
# order (a deliberate consolidation choice), so it matches the legacy brute only to
# floating-point tolerance. The count mp path re-validates the shared DD-only kernel
# core (legacy = verbatim loop, kernel = accumulate(shapes=False)).
# ---------------------------------------------------------------------------

class TestKernelEquivalenceBoxRpPiBrute:
    def test_no_mask(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_measure_xi_rp_pi_box_brute(dataset_name=NAME_LEGACY, masks=None)
        obj._measure_xi_rp_pi_box_brute(dataset_name=NAME_KERNEL, masks=None)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_ellipticity_definition(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_measure_xi_rp_pi_box_brute(dataset_name=NAME_LEGACY, masks=None,
                                               ellipticity='ellipticity')
        obj._measure_xi_rp_pi_box_brute(dataset_name=NAME_KERNEL, masks=None,
                                        ellipticity='ellipticity')
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_responsivity_off(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj.responsivity_correction = False
        obj._legacy_measure_xi_rp_pi_box_brute(dataset_name=NAME_LEGACY, masks=None)
        obj._measure_xi_rp_pi_box_brute(dataset_name=NAME_KERNEL, masks=None)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_non_contiguous_mask(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        pos_mask, shape_mask = _non_contiguous_masks(obj.Num_position, obj.Num_shape)
        ml = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        mk = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        obj._legacy_measure_xi_rp_pi_box_brute(dataset_name=NAME_LEGACY, masks=ml)
        obj._measure_xi_rp_pi_box_brute(dataset_name=NAME_KERNEL, masks=mk)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)


class TestKernelEquivalenceBoxRpPiCountPairs:
    """DD-only count_pairs twins (corr_type='gg')."""

    def test_brute(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_count_pairs_xi_rp_pi_box_brute(dataset_name=NAME_LEGACY, masks=None)
        obj._count_pairs_xi_rp_pi_box_brute(dataset_name=NAME_KERNEL, masks=None)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_tree(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        obj._legacy_count_pairs_xi_rp_pi_box_tree(dataset_name=NAME_LEGACY, masks=None)
        obj._count_pairs_xi_rp_pi_box_tree(dataset_name=NAME_KERNEL, masks=None)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_tree_non_contiguous_mask(self, IA_mock_TNG300_n1):
        obj = IA_mock_TNG300_n1
        pos_mask, shape_mask = _non_contiguous_masks(obj.Num_position, obj.Num_shape)
        ml = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        mk = {"Position": pos_mask.copy(), "Position_shape_sample": shape_mask.copy()}
        obj._legacy_count_pairs_xi_rp_pi_box_tree(dataset_name=NAME_LEGACY, masks=ml)
        obj._count_pairs_xi_rp_pi_box_tree(dataset_name=NAME_KERNEL, masks=mk)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    # chunk_size 150 with 200 shapes gives two batches, the first spanning two inner
    # 100-chunks — exercising both the batch loop and the inner chunk loop.
    def test_multiprocessing(self, IA_mock_TNG300_n8):
        obj = IA_mock_TNG300_n8
        tmp = os.path.dirname(obj.output_file_name)
        obj._legacy_count_pairs_xi_rp_pi_box_multiprocessing(
            dataset_name=NAME_LEGACY, temp_file_path=tmp, masks=None, num_nodes=2, chunk_size=150)
        obj._count_pairs_xi_rp_pi_box_multiprocessing(
            dataset_name=NAME_KERNEL, temp_file_path=tmp, masks=None, num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_multiprocessing_full_mask(self, IA_mock_TNG300_n8):
        obj = IA_mock_TNG300_n8
        tmp = os.path.dirname(obj.output_file_name)
        obj._legacy_count_pairs_xi_rp_pi_box_multiprocessing(
            dataset_name=NAME_LEGACY, temp_file_path=tmp, masks=_full_mask_dict(obj),
            num_nodes=2, chunk_size=150)
        obj._count_pairs_xi_rp_pi_box_multiprocessing(
            dataset_name=NAME_KERNEL, temp_file_path=tmp, masks=_full_mask_dict(obj),
            num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)
