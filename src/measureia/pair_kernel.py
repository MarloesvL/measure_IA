r"""Consolidated pair-accumulation kernel for the measure_IA counting loops.

See ``docs/REFACTOR_PLAN.md`` for the full design and the migration order.
This module is built incrementally, one migration step at a time; each step
adds exactly the geometry/binning/backend combination it needs and leaves
everything else raising ``NotImplementedError`` rather than guessing at
behaviour that hasn't been ported (and equivalence-tested) yet.

Status (see REFACTOR_PLAN.md section 6 for the step numbering):
    Steps 1-4 DONE: box geometry, no jackknife, ``"tree"`` and ``"brute"`` backends,
    the DD-only (``shapes=False``) count_pairs path, and two binnings — ``BoxRpPi``
    (rp, pi) and ``BoxRMuR`` (r, mu_r, multipoles). Wired into the whole ``MeasureWBox``
    (rp, pi) family and the ``MeasureMultipolesBox`` (r, mu_r) family
    (``_measure_xi_r_mur_box_*`` / ``_count_pairs_xi_r_mur_box_*``). The mp
    orchestration (SharedMemory, temp-file offload, ``Pool``) stays in the backend
    wrapper; each worker calls ``accumulate`` single-process on its shape slice.
    (``_measure_xi_r_pi_box_brute`` is a dead, kernel-incompatible per-r-bin-signed-pi
    oddity, deliberately left un-migrated — see REFACTOR_PLAN.md / TASKS.md.)

Every public function here is pure with respect to its arguments except
where noted (``prepare_box_samples`` mutates the ``masks`` dict it is given,
matching the legacy in-place default-injection behaviour it replaces).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree


@dataclass
class SampleSet:
    """Position/weight/shape arrays for one measurement call, already masked.

    Fields mirror docs/REFACTOR_PLAN.md section 3.1. Box-only fields
    (``LOS_ind``, ``not_LOS``) are populated for the box geometry; the
    lightcone geometry (later migration steps) will add its own fields.
    """
    pos: np.ndarray
    pos_shape: np.ndarray
    weight: np.ndarray
    weight_shape: np.ndarray
    axis_direction: Optional[np.ndarray] = None
    e: Optional[np.ndarray] = None
    LOS_ind: Optional[int] = None
    not_LOS: Optional[np.ndarray] = None


@dataclass
class Grids:
    """Accumulated pair-count grids. ``Splus_D``/``Scross_D`` are None when
    the caller requested ``shapes=False`` (DD-only / count_pairs paths)."""
    DD: np.ndarray
    Splus_D: Optional[np.ndarray] = None
    Scross_D: Optional[np.ndarray] = None


def prepare_box_samples(data, masks, Num_position, Num_shape, *, shapes, ellipticity, base):
    """Apply masks and compute per-galaxy ellipticity size for a Box measurement.

    Reproduces, verbatim, the mask-application and ``e`` computation shared by
    every Box (rp, pi) / (r, mu_r) counting method: mask defaulting rules
    (``Position``/``Position_shape_sample`` masks default to "select all";
    ``weight``/``weight_shape_sample`` default to the coordinate mask and are
    written back into ``masks`` in place, matching the legacy fallback-default
    behaviour other code paths rely on), then ``e = f(q)`` for the requested
    ellipticity definition.

    Parameters
    ----------
    data : dict
        The object's ``self.data``.
    masks : dict or None
        Per-call mask dict; mutated in place to inject default ``weight``/
        ``weight_shape_sample`` masks when absent (legacy behaviour).
    Num_position, Num_shape : int
        Full (unmasked) sample sizes, used as the default "select all" mask length.
    shapes : bool
        If False, skip ``Axis_Direction``/``q``/``e`` (DD-only / count_pairs paths).
    ellipticity : str
        'distortion' or 'ellipticity'; see ``MeasureIABase.get_ellipticity``.
    base : object
        The calling instance (unused today; accepted for interface symmetry
        with future steps that need e.g. responsivity_correction here).

    Returns
    -------
    SampleSet
    """
    if masks is None:
        positions = data["Position"]
        positions_shape_sample = data["Position_shape_sample"]
        weight = data["weight"]
        weight_shape = data["weight_shape_sample"]
        axis_direction_v = data["Axis_Direction"] if shapes else None
        q = data["q"] if shapes else None
    else:
        pos_mask = masks.get("Position", np.ones(Num_position, dtype=bool))
        shape_mask = masks.get("Position_shape_sample", np.ones(Num_shape, dtype=bool))
        positions = data["Position"][pos_mask]
        positions_shape_sample = data["Position_shape_sample"][shape_mask]
        if "weight" not in masks:
            masks["weight"] = pos_mask
        if "weight_shape_sample" not in masks:
            masks["weight_shape_sample"] = shape_mask
        weight = data["weight"][masks["weight"]]
        weight_shape = data["weight_shape_sample"][masks["weight_shape_sample"]]
        if shapes:
            dir_mask = masks.get("Axis_Direction", shape_mask)
            q_mask = masks.get("q", shape_mask)
            axis_direction_v = data["Axis_Direction"][dir_mask]
            q = data["q"][q_mask]
        else:
            axis_direction_v = None
            q = None

    axis_direction = None
    e = None
    if shapes:
        axis_direction_len = np.sqrt(np.sum(axis_direction_v ** 2, axis=1))
        axis_direction = (axis_direction_v.transpose() / axis_direction_len).transpose()
        if ellipticity == 'distortion':
            e = (1 - q ** 2) / (1 + q ** 2)
        elif ellipticity == 'ellipticity':
            e = (1 - q) / (1 + q)
        else:
            raise ValueError("Invalid value for ellipticity. Choose 'distortion' or 'ellipticity'.")

    LOS_ind = data["LOS"]
    not_LOS = np.array([0, 1, 2])[np.isin([0, 1, 2], LOS_ind, invert=True)]

    return SampleSet(
        pos=positions, pos_shape=positions_shape_sample,
        weight=weight, weight_shape=weight_shape,
        axis_direction=axis_direction, e=e,
        LOS_ind=LOS_ind, not_LOS=not_LOS,
    )


class BoxRpPi:
    """(rp, pi) grid binning for a periodic Cartesian box.

    Clamping convention (box family, see REFACTOR_PLAN.md section 3.2): an
    index landing exactly on the upper edge (``== num_bins``) is folded back
    into the last bin.
    """

    def __init__(self, base):
        self.r_min = base.r_min
        self.r_max = base.r_max
        self.r_bins = base.r_bins
        self.pi_bins = base.pi_bins
        self.num_bins_r = base.num_bins_r
        self.num_bins_pi = base.num_bins_pi
        self.sub_box_len_logrp = (np.log10(base.r_max) - np.log10(base.r_min)) / base.num_bins_r
        self.sub_box_len_pi = (base.pi_bins[-1] - base.pi_bins[0]) / base.num_bins_pi

    def tree_coords(self, coords, not_LOS):
        """Coordinates the KDTree is built/queried on: the 2D projection, since the
        (rp, pi) r-window is the projected separation."""
        return coords[:, not_LOS]

    def bin_pairs(self, separation, not_LOS, LOS_ind):
        """Bin one shape galaxy's separations to its candidate position neighbours.

        Parameters
        ----------
        separation : (K, 3) ndarray
            ``position_shape[n] - position[candidates]``, already periodicity-wrapped.
        not_LOS : ndarray
            The two axis indices that are not the line-of-sight axis.
        LOS_ind : int
            The line-of-sight axis index.

        Returns
        -------
        mask : (K,) bool ndarray
            Which of the K candidates fall inside the (rp, pi) window.
        ind_r, ind_pi : (mask.sum(),) int ndarrays
            Bin indices for the surviving pairs, in the same order as
            ``separation[mask]``.
        projected_sep : (K, 2) ndarray
            ``separation[:, not_LOS]``, for reuse by the ellipticity-angle calc.
        separation_len : (K,) ndarray
            Projected separation length, for reuse by the ellipticity-angle calc.
        """
        projected_sep = separation[:, not_LOS]
        LOS = separation[:, LOS_ind]
        separation_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
        mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1]) * \
               (LOS >= self.pi_bins[0]) * (LOS < self.pi_bins[-1])
        ind_r = np.floor(
            np.log10(separation_len[mask]) / self.sub_box_len_logrp
            - np.log10(self.r_bins[0]) / self.sub_box_len_logrp
        )
        ind_r = np.array(ind_r, dtype=int)
        ind_pi = np.floor(
            LOS[mask] / self.sub_box_len_pi - self.pi_bins[0] / self.sub_box_len_pi
        )
        ind_pi = np.array(ind_pi, dtype=int)
        if np.any(ind_pi == self.num_bins_pi):
            ind_pi[ind_pi >= self.num_bins_pi] -= 1
        if np.any(ind_r == self.num_bins_r):
            ind_r[ind_r >= self.num_bins_r] -= 1
        return mask, ind_r, ind_pi, projected_sep, separation_len


class BoxRMuR:
    """(r, mu_r) grid binning for a periodic Cartesian box (multipoles).

    Unlike ``BoxRpPi``, the separation bin ``r`` is the *full 3D* separation length
    and ``mu_r = LOS / r``; the KDTree therefore operates on the full 3D coordinates
    (``tree_coords`` returns ``coords`` unchanged). An ``rp_cut`` on the projected
    (2D) separation length is applied inside the window mask. The ``mu_r`` sub-bin
    length is ``2.0 / num_bins_pi`` (mu_r runs over [-1, 1] with ``num_bins_pi`` bins).
    Same clamp convention as ``BoxRpPi`` (an index landing on ``num_bins`` folds into
    the last bin).
    """

    def __init__(self, base, rp_cut=0.0):
        self.r_min = base.r_min
        self.r_max = base.r_max
        self.r_bins = base.r_bins
        self.mu_r_bins = base.mu_r_bins
        self.num_bins_r = base.num_bins_r
        self.num_bins_pi = base.num_bins_pi
        self.rp_cut = rp_cut
        self.sub_box_len_logr = (np.log10(base.r_max) - np.log10(base.r_min)) / base.num_bins_r
        self.sub_box_len_mu_r = 2.0 / base.num_bins_pi

    def tree_coords(self, coords, not_LOS):
        """Coordinates the KDTree is built/queried on: the full 3D positions, since the
        (r, mu_r) r-window is the 3D separation."""
        return coords

    def bin_pairs(self, separation, not_LOS, LOS_ind):
        """Bin one shape galaxy's separations to its candidate position neighbours.

        Returns ``(mask, ind_r, ind_mu_r, projected_sep, projected_len)`` where
        ``projected_sep``/``projected_len`` are the 2D projection and its length, used
        by ``accumulate`` for the ellipticity-angle calc (identical to the (rp, pi)
        family). ``ind_r`` is binned on the 3D separation length; ``ind_mu_r`` on
        ``mu_r = LOS / r_3d``. The window keeps pairs with projected length > rp_cut and
        3D length in ``[r_bins[0], r_bins[-1])``.
        """
        projected_sep = separation[:, not_LOS]
        LOS = separation[:, LOS_ind]
        projected_len = np.sqrt(np.sum(projected_sep ** 2, axis=1))
        separation_len = np.sqrt(np.sum(separation ** 2, axis=1))
        with np.errstate(invalid='ignore'):
            mu_r = LOS / separation_len
        mask = (
            (projected_len > self.rp_cut)
            * (separation_len >= self.r_bins[0])
            * (separation_len < self.r_bins[-1])
        )
        ind_r = np.floor(
            np.log10(separation_len[mask]) / self.sub_box_len_logr
            - np.log10(self.r_bins[0]) / self.sub_box_len_logr
        )
        ind_r = np.array(ind_r, dtype=int)
        ind_mu_r = np.floor(
            mu_r[mask] / self.sub_box_len_mu_r - self.mu_r_bins[0] / self.sub_box_len_mu_r
        )
        ind_mu_r = np.array(ind_mu_r, dtype=int)
        if np.any(ind_mu_r == self.num_bins_pi):
            ind_mu_r[ind_mu_r >= self.num_bins_pi] -= 1
        if np.any(ind_r == self.num_bins_r):
            ind_r[ind_r >= self.num_bins_r] -= 1
        return mask, ind_r, ind_mu_r, projected_sep, projected_len


def accumulate(sample_set, binning, *, base, R=None, shapes=True,
               chunk_axis="shape", chunk_size_outer=100, jk=False, pool=None,
               pos_tree=None, backend="tree"):
    """Run the pair-accumulation loop and return the resulting grids.

    Implemented so far: box geometry, ``BoxRpPi`` / ``BoxRMuR`` binnings,
    ``chunk_axis="shape"``, no jackknife, backends ``"tree"`` and ``"brute"``. Later
    migration steps (REFACTOR_PLAN.md section 6, steps 5-7) extend this same function.

    Iteration order (outer loop over shape-sample chunks of ``chunk_size_outer``,
    inner loop over the chunk, vectorized ``np.add.at`` per shape galaxy) is fixed
    by the float-summation-order rule in REFACTOR_PLAN.md section 4 and must not
    change without re-deriving bit-identity against the legacy tree/mp paths.

    ``backend`` selects how each shape galaxy's candidate positions are chosen:
      - ``"tree"``: KDTree annulus query ``[r_min, r_max]`` against the position
        tree (the legacy tree/mp order — bit-identical).
      - ``"brute"``: every position is a candidate (full cross-join per chunk);
        the ``[r_min, r_max)`` window is applied by the binning mask. This runs on
        the *same* shape-chunk order as the tree backend rather than the legacy
        brute's position-outer order, so it matches the legacy brute only to
        floating-point tolerance (``allclose``), not bit-identically — a
        deliberate consolidation choice (REFACTOR_PLAN.md section 4). It counts
        exactly the same pairs the legacy brute did (same window mask), so integer
        (unit-weight) DD grids still match exactly.

    ``pos_tree`` may be a prebuilt ``KDTree`` over ``sample_set.pos[:, not_LOS]``
    (tree backend only). The multiprocessing path passes the tree it built once in
    the parent process (shared to every worker) rather than rebuilding it per
    batch; when None the tree is built here. ``sample_set.pos`` must be the same
    full position array the tree was built from either way.
    """
    if jk or pool is not None:
        raise NotImplementedError(
            "pair_kernel.accumulate: jackknife/multiprocessing accumulation is "
            "not migrated yet (see docs/REFACTOR_PLAN.md steps 5-7)."
        )
    if chunk_axis != "shape":
        raise NotImplementedError(
            "pair_kernel.accumulate: only chunk_axis='shape' (box tree/mp order) "
            "is implemented so far."
        )
    if not isinstance(binning, (BoxRpPi, BoxRMuR)):
        raise NotImplementedError(
            "pair_kernel.accumulate: only BoxRpPi / BoxRMuR binnings are implemented so far."
        )
    if backend not in ("tree", "brute"):
        raise NotImplementedError(
            f"pair_kernel.accumulate: unknown backend {backend!r} (expected 'tree' or 'brute')."
        )

    DD = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r)
    Splus_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None
    Scross_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None

    positions = sample_set.pos
    positions_shape_sample = sample_set.pos_shape
    weight = sample_set.weight
    weight_shape = sample_set.weight_shape
    not_LOS = sample_set.not_LOS
    LOS_ind = sample_set.LOS_ind

    if backend == "brute":
        all_positions = np.arange(len(positions))
    elif pos_tree is None:
        pos_tree = KDTree(binning.tree_coords(positions, not_LOS), boxsize=base.boxsize)
    for i in np.arange(0, len(positions_shape_sample), chunk_size_outer):
        i2 = min(len(positions_shape_sample), i + chunk_size_outer)
        positions_shape_sample_i = positions_shape_sample[i:i2]
        weight_shape_i = weight_shape[i:i2]
        if shapes:
            axis_direction_i = sample_set.axis_direction[i:i2]
            e_i = sample_set.e[i:i2]
        if backend == "brute":
            # every position is a candidate for every shape in the chunk
            ind_rbin_i = [all_positions] * len(positions_shape_sample_i)
        else:
            shape_tree = KDTree(binning.tree_coords(positions_shape_sample_i, not_LOS), boxsize=base.boxsize)
            ind_min_i = shape_tree.query_ball_tree(pos_tree, binning.r_min)
            ind_max_i = shape_tree.query_ball_tree(pos_tree, binning.r_max)
            ind_rbin_i = base.setdiff2D(ind_max_i, ind_min_i)

        for n in np.arange(0, len(positions_shape_sample_i)):
            if len(ind_rbin_i[n]) > 0:
                separation = positions_shape_sample_i[n] - positions[ind_rbin_i[n]]
                if base.periodicity:
                    separation[separation > base.L_0p5] -= base.boxsize
                    separation[separation < -base.L_0p5] += base.boxsize

                mask, ind_r, ind_pi, projected_sep, proj_len = binning.bin_pairs(
                    separation, not_LOS, LOS_ind
                )

                if shapes:
                    with np.errstate(invalid='ignore'):
                        separation_dir = (projected_sep.transpose() / proj_len).transpose()
                        phi = np.arccos(
                            separation_dir[:, 0] * axis_direction_i[n, 0]
                            + separation_dir[:, 1] * axis_direction_i[n, 1]
                        )
                    e_plus, e_cross = base.get_ellipticity(e_i[n], phi)
                    e_plus[np.isnan(e_plus)] = 0.0
                    e_cross[np.isnan(e_cross)] = 0.0
                    np.add.at(Splus_D, (ind_r, ind_pi),
                              (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_plus[mask]) / (2 * R))
                    np.add.at(Scross_D, (ind_r, ind_pi),
                              (weight[ind_rbin_i[n]][mask] * weight_shape_i[n] * e_cross[mask]) / (2 * R))
                np.add.at(DD, (ind_r, ind_pi), weight[ind_rbin_i[n]][mask] * weight_shape_i[n])

    return Grids(DD=DD, Splus_D=Splus_D, Scross_D=Scross_D)
