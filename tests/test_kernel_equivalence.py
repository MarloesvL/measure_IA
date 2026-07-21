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
test entry here are deleted in the commit that migrates the next path. The classes below
therefore cover only the most recently migrated family (currently the box (r, mu_r)
multipoles family); earlier families stay locked by the broader test suite.
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
        # _sigmasq was dropped from the kernel jk wrappers by user decision (only the
        # brute backend ever populated it); the legacy copies still write it, so skip it
        # on both sides — the accumulation is compared through every other dataset.
        if "sigmasq" in leaf:
            continue
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


def _lc_mask_dict(obj):
    """Non-contiguous lightcone mask dict (direct masks["RA"]-style indexing). Position-aligned
    (RA/DEC/Redshift) and shape-aligned (RA_shape_sample/.../e1/e2) selections differ so the
    weight-defaults-to-coordinate-mask path is exercised. weight/weight_shape_sample are left
    out on purpose — prepare_lightcone_samples injects them from the coordinate masks."""
    n_pos = len(obj.data["RA"])
    n_shape = len(obj.data["RA_shape_sample"])
    pos_mask = np.zeros(n_pos, dtype=bool)
    pos_mask[::2] = True
    shape_mask = np.zeros(n_shape, dtype=bool)
    shape_mask[1::3] = True
    return {
        "RA": pos_mask, "DEC": pos_mask, "Redshift": pos_mask,
        "RA_shape_sample": shape_mask, "DEC_shape_sample": shape_mask,
        "Redshift_shape_sample": shape_mask, "e1": shape_mask, "e2": shape_mask,
    }


def _lc_jk_indices(obj, num_jk=4):
    """Fresh 0-based jackknife patch-id arrays aligned to the (unmasked) position and shape
    samples. The assignment need only be a valid partition for the A/B comparison, not
    physical; a fresh copy is returned per call because the jk methods mutate them in place
    (``-= min``)."""
    n_pos = len(obj.data["RA"])
    n_shape = len(obj.data["RA_shape_sample"])
    jk_pos = (np.arange(n_pos) % num_jk).astype(int)
    jk_shape = (np.arange(n_shape) % num_jk).astype(int)
    return jk_pos, jk_shape


# ---------------------------------------------------------------------------
# Step 7: lightcone jackknife families (measure_w_lightcone_jk.py / measure_m_lightcone_jk.py)
# via accumulate(chunk_axis="position", jk=True).
#   - _{measure,count_pairs}_xi_{rp_pi,r_mur}_lightcone_jk_{brute,tree,batch,multiprocessing}
#
# Union-deletion mirrored vs the box path (chunked axis = position): each pair contributes to
# its position patch always and its shape patch where they differ. Pure delete-one reduction
# (Splus_D - Splus_D_jk[i]); responsivity is baked into e so there is NO per-realisation R_jk.
# tree/mp bit-identical, brute allclose. Fresh jk arrays per call (methods mutate in place).
# ---------------------------------------------------------------------------


class TestKernelEquivalenceSkyRpPiJkMeasure:
    def test_tree(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_rp_pi_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_rp_pi_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_tree_masked(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        m = _lc_mask_dict(obj)
        n_pos = int(np.count_nonzero(m["RA"]))
        n_shape = int(np.count_nonzero(m["RA_shape_sample"]))
        jp = (np.arange(n_pos) % 4).astype(int)
        js = (np.arange(n_shape) % 4).astype(int)
        obj._legacy_measure_xi_rp_pi_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), masks=_lc_mask_dict(obj), print_num=False)
        obj._measure_xi_rp_pi_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), masks=_lc_mask_dict(obj), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_tree_responsivity_on(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        obj.responsivity_correction = True
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_rp_pi_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_rp_pi_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_brute(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_rp_pi_lightcone_jk_brute(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_rp_pi_lightcone_jk_brute(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_multiprocessing(self, IA_mock_lc_n8):
        obj = IA_mock_lc_n8
        tmp = os.path.dirname(obj.output_file_name)
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_rp_pi_lightcone_jk_multiprocessing(NAME_LEGACY, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        obj._measure_xi_rp_pi_lightcone_jk_multiprocessing(NAME_KERNEL, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)


class TestKernelEquivalenceSkyRpPiJkCountPairs:
    def test_tree(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_rp_pi_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._count_pairs_xi_rp_pi_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_brute(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_rp_pi_lightcone_jk_brute(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._count_pairs_xi_rp_pi_lightcone_jk_brute(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_multiprocessing(self, IA_mock_lc_n8):
        obj = IA_mock_lc_n8
        tmp = os.path.dirname(obj.output_file_name)
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_rp_pi_lightcone_jk_multiprocessing(NAME_LEGACY, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        obj._count_pairs_xi_rp_pi_lightcone_jk_multiprocessing(NAME_KERNEL, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)


class TestKernelEquivalenceSkyRMuRJkMeasure:
    def test_tree(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_r_mur_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_r_mur_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_tree_responsivity_on(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        obj.responsivity_correction = True
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_r_mur_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_r_mur_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_brute(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_r_mur_lightcone_jk_brute(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._measure_xi_r_mur_lightcone_jk_brute(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_multiprocessing(self, IA_mock_lc_n8):
        obj = IA_mock_lc_n8
        tmp = os.path.dirname(obj.output_file_name)
        jp, js = _lc_jk_indices(obj)
        obj._legacy_measure_xi_r_mur_lightcone_jk_multiprocessing(NAME_LEGACY, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        obj._measure_xi_r_mur_lightcone_jk_multiprocessing(NAME_KERNEL, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)


class TestKernelEquivalenceSkyRMuRJkCountPairs:
    def test_tree(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_r_mur_lightcone_jk_tree(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._count_pairs_xi_r_mur_lightcone_jk_tree(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_brute(self, IA_mock_lc_n1):
        obj = IA_mock_lc_n1
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_r_mur_lightcone_jk_brute(NAME_LEGACY, jp.copy(), js.copy(), print_num=False)
        obj._count_pairs_xi_r_mur_lightcone_jk_brute(NAME_KERNEL, jp.copy(), js.copy(), print_num=False)
        _assert_groups_allclose(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)

    def test_multiprocessing(self, IA_mock_lc_n8):
        obj = IA_mock_lc_n8
        tmp = os.path.dirname(obj.output_file_name)
        jp, js = _lc_jk_indices(obj)
        obj._legacy_count_pairs_xi_r_mur_lightcone_jk_multiprocessing(NAME_LEGACY, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        obj._count_pairs_xi_r_mur_lightcone_jk_multiprocessing(NAME_KERNEL, jp.copy(), js.copy(), tmp, num_nodes=2, chunk_size=150)
        _assert_groups_bit_identical(obj.output_file_name, NAME_LEGACY, NAME_KERNEL)
