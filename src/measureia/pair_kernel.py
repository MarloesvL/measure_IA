r"""Consolidated pair-accumulation kernel for the measure_IA counting loops.

See ``docs/REFACTOR_PLAN.md`` for the full design and the migration order.
This module is built incrementally, one migration step at a time; each step
adds exactly the geometry/binning/backend combination it needs and leaves
everything else raising ``NotImplementedError`` rather than guessing at
behaviour that hasn't been ported (and equivalence-tested) yet.

Status (see REFACTOR_PLAN.md section 6 for the step numbering):
    Steps 1-5 DONE (box): box geometry, ``"tree"`` and ``"brute"`` backends, the DD-only
    (``shapes=False``) count_pairs path, jackknife (``jk=True``, union-deletion), and
    two binnings — ``BoxRpPi`` (rp, pi) and ``BoxRMuR`` (r, mu_r, multipoles). Wired
    into the ``MeasureWBox`` / ``MeasureMultipolesBox`` non-jk families and the
    ``MeasureWBoxJackknife`` / ``MeasureMBoxJackknife`` families. The mp orchestration
    (SharedMemory, temp-file offload, ``Pool``) stays in the backend wrapper; each
    worker calls ``accumulate`` single-process on its shape slice, and the parent sums
    the partial jk grids and computes ``R_jk`` via ``compute_R_jk``.
    (``_measure_xi_r_pi_box_brute`` is a dead, kernel-incompatible per-r-bin-signed-pi
    oddity, deliberately left un-migrated — see REFACTOR_PLAN.md / TASKS.md. The jk
    ``_sigmasq`` output was dropped by user decision — only the brute backend ever
    populated it, a pre-existing inconsistency.)

    Step 6 DONE (lightcone non-jk): sky geometry via ``prepare_lightcone_samples``
    (RA/DEC/z → 3D comoving with ``pyccl``; ``e = (e1, e2)`` pre-scaled by 1/(2R)) and
    the ``chunk_axis="position"`` accumulation path (``_accumulate_lightcone``) with the
    ``SkyRpPi`` / ``SkyRMuR`` binnings (midpoint LOS ``n_LOS = (s1+s2)/|s1+s2|``,
    east/north ellipticity-angle projection). Wired into the non-jk
    ``_{measure,count_pairs}_xi_{rp_pi,r_mur}_lightcone_{brute,tree}`` families. Unlike
    the box S+ grids, the lightcone S+ is not divided by ``2R`` (baked into ``e``).

    Step 7 DONE (lightcone jk): ``_accumulate_lightcone`` gained the ``jk=True``
    union-deletion path — **mirrored** vs the box (the chunked axis is the position, so a
    pair adds to its position patch always and its shape patch where they differ) — plus a
    prebuilt-``shape_tree`` argument so the mp workers reuse the parent's tree. The
    lightcone reduction is a pure delete-one (``Splus_D - Splus_D_jk[i]``); responsivity is
    baked into ``e`` globally, so there is no per-realisation ``R_jk`` (contrast
    ``compute_R_jk`` for the box). Wired into all 16
    ``_{measure,count_pairs}_xi_{rp_pi,r_mur}_lightcone_jk_{brute,tree,batch,multiprocessing}``
    methods; mp orchestration stays in the wrapper and each worker calls ``accumulate`` on
    its position slice. All counting loops now live in this module (step 8 = cleanup only:
    delete the dead ``*_old`` methods and the remaining ``_legacy_``/harness scaffolding).

Every public function here is pure with respect to its arguments except
where noted (``prepare_box_samples`` mutates the ``masks`` dict it is given,
matching the legacy in-place default-injection behaviour it replaces).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree
import pyccl as ccl


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
    # jackknife patch indices (int, per galaxy); jk_pos is position-aligned (full),
    # jk_shape is shape-aligned (chunked like axis_direction/e). Only used when jk=True.
    jk_pos: Optional[np.ndarray] = None
    jk_shape: Optional[np.ndarray] = None
    # Lightcone geometry only (chunk_axis="position"): local sky basis at each position
    # (east/north, N,3) for the ellipticity-angle projection, and n_pos (N,3, radial unit
    # vector). For the lightcone families ``e`` is (M,2) = (e1,e2) pre-scaled by 1/(2R).
    east: Optional[np.ndarray] = None
    north: Optional[np.ndarray] = None
    n_pos: Optional[np.ndarray] = None


@dataclass
class Grids:
    """Accumulated pair-count grids. ``Splus_D``/``Scross_D`` are None when
    the caller requested ``shapes=False`` (DD-only / count_pairs paths). The
    per-realisation jackknife grids ``DD_jk``/``Splus_D_jk`` are None unless
    ``jk=True`` (and ``Splus_D_jk`` also requires ``shapes=True``). ``Splus_D_jk``
    stores the *raw* (un-responsivity-divided) S+ contribution — responsivity is
    applied later in the reduction, matching the legacy jk grids."""
    DD: np.ndarray
    Splus_D: Optional[np.ndarray] = None
    Scross_D: Optional[np.ndarray] = None
    DD_jk: Optional[np.ndarray] = None
    Splus_D_jk: Optional[np.ndarray] = None


def prepare_box_samples(data, masks, Num_position, Num_shape, *, shapes, ellipticity, base,
                        require_full_masks=False):
    """Apply masks and compute per-galaxy ellipticity size for a Box measurement.

    Reproduces, verbatim, the mask-application and ``e`` computation shared by
    every Box (rp, pi) / (r, mu_r) counting method: mask defaulting rules
    (``Position``/``Position_shape_sample`` masks default to "select all";
    ``weight``/``weight_shape_sample`` default to the coordinate mask and are
    written back into ``masks`` in place, matching the legacy fallback-default
    behaviour other code paths rely on), then ``e = f(q)`` for the requested
    ellipticity definition.

    ``require_full_masks`` selects the mask-indexing convention: the non-jk methods
    default a missing ``Position``/``Position_shape_sample``/``Axis_Direction``/``q``
    mask (``.get`` with "select all" / coordinate-mask fallbacks), whereas the box
    jackknife methods index ``masks["Position"]`` etc. **directly** and raise KeyError
    on a partial dict — pass ``require_full_masks=True`` to reproduce that (see
    REFACTOR_PLAN.md section 3.1). ``weight``/``weight_shape_sample`` still default to
    the coordinate mask in both modes.

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
        if require_full_masks:
            pos_mask = masks["Position"]
            shape_mask = masks["Position_shape_sample"]
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
            if require_full_masks:
                dir_mask = masks["Axis_Direction"]
                q_mask = masks["q"]
            else:
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


def prepare_lightcone_samples(data, masks, *, shapes, cosmology, over_h,
                              responsivity_correction, base, print_num=True):
    """Apply masks and build the 3D comoving sky geometry for a Lightcone measurement.

    Reproduces, verbatim, the shared head of every lightcone (rp, pi) / (r, mu_r) counting
    method: RA/DEC/Redshift/e1/e2 mask-application (direct ``masks["RA"]``-style indexing —
    a partial dict raises KeyError, as in the legacy; ``weight``/``weight_shape_sample``
    default to the ``RA``/``RA_shape_sample`` coordinate masks and are written back into
    ``masks`` in place), redshift → comoving distance via ``pyccl`` (default cosmology when
    None; ``over_h`` scales by ``h``), the unit-sphere direction vectors, the position/shape
    comoving vectors ``s_pos``/``s_shape``, and — for ``shapes=True`` — the per-shape
    ellipticity ``e = (e1, e2)`` **pre-scaled by 1/(2R)** when ``responsivity_correction``
    (responsivity is baked into ``e`` here, not divided in the pair loop, unlike the box
    families — REFACTOR_PLAN.md section 3.2), plus the local ``east``/``north`` sky basis at
    each position.

    ``SampleSet`` layout for the lightcone geometry: ``pos = s_pos`` (N,3), ``pos_shape =
    s_shape`` (M,3), ``weight``/``weight_shape``; and when ``shapes``: ``e`` (M,2), ``east``
    (N,3), ``north`` (N,3), ``n_pos`` (N,3).

    Parameters
    ----------
    data : dict
        The object's ``self.data``.
    masks : dict or None
        Per-call mask dict; mutated in place to inject default ``weight``/
        ``weight_shape_sample`` masks when absent (legacy behaviour).
    shapes : bool
        If False, skip ``e1``/``e2``/responsivity and the ``east``/``north`` basis
        (DD-only / count_pairs paths).
    cosmology : pyccl.Cosmology or None
        Cosmology for redshift→comoving distance; a fixed default is built (and, when
        ``print_num``, announced) if None, matching the legacy.
    over_h : bool
        If True, multiply comoving distances by ``h`` (positions in cMpc/h).
    responsivity_correction : bool
        If True (and ``shapes``), pre-scale ``e1, e2`` by ``1/(2R)`` with ``R`` the
        weighted responsivity over the shape sample. Note the lightcone default is False
        (unlike the box families) — pass ``getattr(self, "responsivity_correction", False)``.
    base : object
        The calling instance (accepted for interface symmetry; unused here).
    print_num : bool
        Gate on the "No cosmology given" informational print, matching the legacy.

    Returns
    -------
    SampleSet
    """
    if masks is None:
        redshift = data["Redshift"]
        redshift_shape_sample = data["Redshift_shape_sample"]
        RA = data["RA"]
        RA_shape_sample = data["RA_shape_sample"]
        DEC = data["DEC"]
        DEC_shape_sample = data["DEC_shape_sample"]
        weight = data["weight"]
        weight_shape = data["weight_shape_sample"]
        if shapes:
            e1 = data["e1"]
            e2 = data["e2"]
    else:
        redshift = data["Redshift"][masks["Redshift"]]
        redshift_shape_sample = data["Redshift_shape_sample"][masks["Redshift_shape_sample"]]
        RA = data["RA"][masks["RA"]]
        RA_shape_sample = data["RA_shape_sample"][masks["RA_shape_sample"]]
        DEC = data["DEC"][masks["DEC"]]
        DEC_shape_sample = data["DEC_shape_sample"][masks["DEC_shape_sample"]]
        if shapes:
            e1 = data["e1"][masks["e1"]]
            e2 = data["e2"][masks["e2"]]
        if "weight" not in masks:
            masks["weight"] = masks["RA"]
        if "weight_shape_sample" not in masks:
            masks["weight_shape_sample"] = masks["RA_shape_sample"]
        weight = data["weight"][masks["weight"]]
        weight_shape = data["weight_shape_sample"][masks["weight_shape_sample"]]

    Num_position = len(RA)

    if cosmology is None:
        cosmology = ccl.Cosmology(Omega_c=0.225, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.0)
        if print_num:
            print("No cosmology given, using Omega_m=0.27, Omega_b=0.045, sigma8=0.8, h=0.7, n_s=1.")
    h = cosmology["h"]

    LOS_all = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift))
    LOS_all_shape_sample = ccl.comoving_radial_distance(cosmology, 1 / (1 + redshift_shape_sample))
    if over_h:
        LOS_all *= h
        LOS_all_shape_sample *= h

    e = None
    east = None
    north = None
    if shapes:
        if responsivity_correction:
            R = sum(weight_shape * (1 - (e1 ** 2 + e2 ** 2) / 2.0)) / sum(weight_shape)
            e1, e2 = e1 / (2 * R), e2 / (2 * R)
        e = np.array([e1, e2]).transpose()

    RA_rad = RA / 180 * np.pi
    RA_shape_sample_rad = RA_shape_sample / 180 * np.pi
    DEC_rad = DEC / 180 * np.pi
    DEC_shape_sample_rad = DEC_shape_sample / 180 * np.pi
    n_shape = np.array([np.cos(DEC_shape_sample_rad) * np.cos(RA_shape_sample_rad),
                        np.cos(DEC_shape_sample_rad) * np.sin(RA_shape_sample_rad),
                        np.sin(DEC_shape_sample_rad)]).transpose()
    s_shape = n_shape * np.array([LOS_all_shape_sample]).transpose()
    n_pos = np.array([np.cos(DEC_rad) * np.cos(RA_rad),
                      np.cos(DEC_rad) * np.sin(RA_rad),
                      np.sin(DEC_rad)]).transpose()
    if shapes:
        east = np.array([-np.sin(RA_rad), np.cos(RA_rad), np.zeros(Num_position)]).transpose()
        north = np.array([
            -np.sin(DEC_rad) * np.cos(RA_rad),
            -np.sin(DEC_rad) * np.sin(RA_rad),
            np.cos(DEC_rad)
        ]).transpose()
    s_pos = np.array([LOS_all]).transpose() * n_pos

    return SampleSet(
        pos=s_pos, pos_shape=s_shape,
        weight=weight, weight_shape=weight_shape,
        e=e, east=east, north=north, n_pos=n_pos,
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


class SkyRpPi:
    """(rp, pi) grid binning for a lightcone (RA, DEC, z → 3D comoving) sky.

    The line-of-sight direction is the pair *midpoint* radial direction
    ``n_LOS = (s_pos + s_shape) / |s_pos + s_shape|``; ``LOS = s . n_LOS`` is the signed
    line-of-sight separation (``pi`` runs over ``[-pi_max, pi_max]``) and the projected
    separation length is ``sqrt(|s|^2 - LOS^2)``. Query radius is
    ``sqrt(r_max^2 + pi_bins[-1]^2)`` (the position tree is queried against the shape tree
    over the annulus ``[r_min, query_r_max]``). Clamping convention (lightcone family, see
    REFACTOR_PLAN.md section 3.2): an index landing exactly on the upper edge
    (``== num_bins``) is set to the last bin (``num_bins - 1``).
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
        # KDTree query annulus for candidate selection (REFACTOR_PLAN.md section 3.2).
        self.query_r_min = base.r_min
        self.query_r_max = np.sqrt(base.r_max ** 2 + base.pi_bins[-1] ** 2)

    def bin_pairs(self, s, n_LOS, base):
        """Bin one position galaxy's separations ``s = s_shape[cand] - s_pos[n]`` to its
        candidate shape neighbours, given the per-pair midpoint LOS unit vectors ``n_LOS``.

        Returns ``(mask, ind_r, ind_pi, s_perp)`` where ``s_perp`` is the projected
        separation vector (``s`` minus its ``n_LOS`` component), used by ``accumulate`` for
        the ellipticity-angle calc.
        """
        LOS = base.calculate_dot_product_arrays(s, n_LOS)
        separation_len = np.sqrt(np.sum(s ** 2, axis=1) - LOS ** 2)
        s_perp = s - np.sum(s * n_LOS, axis=1, keepdims=True) * n_LOS
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
        if np.any(ind_r == self.num_bins_r):
            ind_r[np.where(ind_r == self.num_bins_r)] = self.num_bins_r - 1
        if np.any(ind_pi == self.num_bins_pi):
            ind_pi[np.where(ind_pi == self.num_bins_pi)] = self.num_bins_pi - 1
        return mask, ind_r, ind_pi, s_perp


class SkyRMuR:
    """(r, mu_r) grid binning for a lightcone sky (multipoles).

    Like ``SkyRpPi`` but the separation bin ``r`` is the full 3D separation length,
    ``mu_r = LOS / r`` (with the same midpoint ``n_LOS``), and there is no ``pi`` window.
    Query radius is the annulus ``[r_min, r_max]``. Same lightcone clamp convention as
    ``SkyRpPi``. ``mu_r`` sub-bin length is ``2.0 / num_bins_pi``.
    """

    def __init__(self, base):
        self.r_min = base.r_min
        self.r_max = base.r_max
        self.r_bins = base.r_bins
        self.mu_r_bins = base.mu_r_bins
        self.num_bins_r = base.num_bins_r
        self.num_bins_pi = base.num_bins_pi
        self.sub_box_len_logrp = (np.log10(base.r_max) - np.log10(base.r_min)) / base.num_bins_r
        self.sub_box_len_mu_r = 2.0 / base.num_bins_pi
        self.query_r_min = base.r_min
        self.query_r_max = base.r_max

    def bin_pairs(self, s, n_LOS, base):
        """Returns ``(mask, ind_r, ind_mu_r, s_perp)``; ``ind_r`` binned on the 3D
        separation length, ``ind_mu_r`` on ``mu_r = LOS / r_3d``. Window keeps 3D length
        in ``[r_bins[0], r_bins[-1])`` (no LOS window)."""
        LOS = base.calculate_dot_product_arrays(s, n_LOS)
        separation_len = np.sqrt(np.sum(s ** 2, axis=1))
        mu_r = LOS / separation_len
        s_perp = s - np.sum(s * n_LOS, axis=1, keepdims=True) * n_LOS
        mask = (separation_len >= self.r_bins[0]) * (separation_len < self.r_bins[-1])
        ind_r = np.floor(
            np.log10(separation_len[mask]) / self.sub_box_len_logrp
            - np.log10(self.r_bins[0]) / self.sub_box_len_logrp
        )
        ind_r = np.array(ind_r, dtype=int)
        ind_mu_r = np.floor(
            mu_r[mask] / self.sub_box_len_mu_r - self.mu_r_bins[0] / self.sub_box_len_mu_r
        )
        ind_mu_r = np.array(ind_mu_r, dtype=int)
        if np.any(ind_r == self.num_bins_r):
            ind_r[np.where(ind_r == self.num_bins_r)] = self.num_bins_r - 1
        if np.any(ind_mu_r == self.num_bins_pi):
            ind_mu_r[np.where(ind_mu_r == self.num_bins_pi)] = self.num_bins_pi - 1
        return mask, ind_r, ind_mu_r, s_perp


def _accumulate_lightcone(sample_set, binning, *, base, shapes, chunk_size_outer, backend,
                          jk=False, num_jk=None, shape_tree=None):
    """Lightcone (chunk_axis="position") pair-accumulation loop.

    Mirrors the legacy lightcone tree/brute counting order exactly (REFACTOR_PLAN.md
    section 4): outer loop over **position** chunks of ``chunk_size_outer``; per chunk build
    ``KDTree(s_pos_chunk)`` and query it against the single full ``KDTree(s_shape)`` over the
    binning's ``[query_r_min, query_r_max]`` annulus (tree backend) or take every shape as a
    candidate (brute backend); inner loop over the chunk, vectorised ``np.add.at`` per
    position. Ellipticity is per *shape* candidate here (``e`` is (M,2) = e1,e2 already
    scaled by 1/(2R)), and S+ grids are **not** divided by ``2R`` (baked into ``e``).

    ``jk=True`` (with ``num_jk`` = number of jackknife realisations) additionally accumulates
    the union-deletion per-realisation grids ``DD_jk`` (and, when ``shapes``, ``Splus_D_jk``).
    Note the axis-roles are **mirrored** versus the box path: the chunked (outer) axis is the
    *position*, so every pair contributes to the position's patch always and to the shape's
    patch only where the two differ. ``sample_set.jk_pos`` is position-aligned (chunked like
    ``s_pos``) and ``sample_set.jk_shape`` is shape-aligned (indexed by candidates), both
    already normalised to 0-based patch ids by the wrapper. Unlike the box jk, the lightcone
    reduction is a pure delete-one (``Splus_D - Splus_D_jk[i]``): responsivity is baked into
    ``e`` globally, so there is no per-realisation ``R_jk`` here.

    ``shape_tree`` may be a prebuilt ``KDTree(s_shape)`` (tree backend only) — the mp path
    builds it once in the parent process and shares it with every worker rather than
    rebuilding per batch; when None the tree is built here.
    """
    DD = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r)
    Splus_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None
    Scross_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None
    DD_jk = np.zeros((num_jk, binning.num_bins_r, binning.num_bins_pi)) if jk else None
    Splus_D_jk = np.zeros((num_jk, binning.num_bins_r, binning.num_bins_pi)) if (jk and shapes) else None

    s_pos = sample_set.pos
    s_shape = sample_set.pos_shape
    weight = sample_set.weight
    weight_shape = sample_set.weight_shape
    jk_shape = sample_set.jk_shape
    Num_position = len(s_pos)
    Num_shape = len(s_shape)

    if backend == "brute":
        all_shapes = np.arange(Num_shape)
    elif shape_tree is None:
        shape_tree = KDTree(s_shape)

    for i in np.arange(0, Num_position, chunk_size_outer):
        i2 = min(Num_position, i + chunk_size_outer)
        s_pos_i = s_pos[i:i2]
        weight_i = weight[i:i2]
        if shapes:
            east_i = sample_set.east[i:i2]
            north_i = sample_set.north[i:i2]
        if jk:
            jk_pos_i = sample_set.jk_pos[i:i2]
        if backend == "brute":
            ind_rbin_i = [all_shapes] * len(s_pos_i)
        else:
            pos_tree = KDTree(s_pos_i)
            ind_min_i = pos_tree.query_ball_tree(shape_tree, binning.query_r_min)
            ind_max_i = pos_tree.query_ball_tree(shape_tree, binning.query_r_max)
            ind_rbin_i = base.setdiff2D(ind_max_i, ind_min_i)

        for n in np.arange(0, len(s_pos_i)):
            cand = ind_rbin_i[n]
            if len(cand) > 0:
                L = s_pos_i[n] + s_shape[cand]
                n_LOS = L / np.sqrt(np.sum(L ** 2, axis=1))[:, None]
                s = s_shape[cand] - s_pos_i[n]
                mask, ind_r, ind_2nd, s_perp = binning.bin_pairs(s, n_LOS, base)

                if shapes:
                    x = np.sum(s_perp * east_i[n], axis=1)
                    y = np.sum(s_perp * north_i[n], axis=1)
                    phi = np.arctan2(y, x)
                    e_plus, e_cross = base.get_ellipticity(sample_set.e[cand], phi)
                    e_plus[np.isnan(e_plus)] = 0.0
                    e_cross[np.isnan(e_cross)] = 0.0
                    np.add.at(Splus_D, (ind_r, ind_2nd),
                              weight_i[n] * weight_shape[cand][mask] * e_plus[mask])
                    np.add.at(Scross_D, (ind_r, ind_2nd),
                              weight_i[n] * weight_shape[cand][mask] * e_cross[mask])
                np.add.at(DD, (ind_r, ind_2nd), weight_i[n] * weight_shape[cand][mask])

                if jk:
                    # union (two-sided) deletion, mirrored vs the box path: the chunked
                    # (outer) axis is the position, so every pair contributes to the
                    # position's patch always and to the shape's patch only where they differ.
                    # Order matches the legacy jk loop (Splus_D_jk before DD_jk).
                    chunked_patch = jk_pos_i[n]
                    other_patches = jk_shape[cand][mask]
                    other_diff = np.where(other_patches != chunked_patch)[0]
                    w_pairs = weight_i[n] * weight_shape[cand][mask]
                    if shapes:
                        np.add.at(Splus_D_jk, (chunked_patch, ind_r, ind_2nd), w_pairs * e_plus[mask])
                        np.add.at(Splus_D_jk,
                                  (other_patches[other_diff], ind_r[other_diff], ind_2nd[other_diff]),
                                  (w_pairs * e_plus[mask])[other_diff])
                    np.add.at(DD_jk, (chunked_patch, ind_r, ind_2nd), w_pairs)
                    np.add.at(DD_jk,
                              (other_patches[other_diff], ind_r[other_diff], ind_2nd[other_diff]),
                              w_pairs[other_diff])

    return Grids(DD=DD, Splus_D=Splus_D, Scross_D=Scross_D, DD_jk=DD_jk, Splus_D_jk=Splus_D_jk)


def accumulate(sample_set, binning, *, base, R=None, shapes=True,
               chunk_axis="shape", chunk_size_outer=100, jk=False, num_box=None,
               pos_tree=None, shape_tree=None, backend="tree"):
    """Run the pair-accumulation loop and return the resulting grids.

    Implemented so far: box geometry, ``BoxRpPi`` / ``BoxRMuR`` binnings,
    ``chunk_axis="shape"``, backends ``"tree"`` and ``"brute"``, optional jackknife.
    Later migration steps (REFACTOR_PLAN.md section 6, steps 6-7) extend this same
    function to the lightcone geometry.

    ``jk=True`` (with ``num_box`` = number of jackknife realisations) additionally
    accumulates the union-deletion per-realisation grids ``DD_jk`` (and, when
    ``shapes``, the raw ``Splus_D_jk``): every pair contributes to the shape's patch,
    and to the position's patch only where the two patches differ. ``sample_set``
    must then carry ``jk_pos`` (position-aligned) and ``jk_shape`` (shape-aligned)
    patch indices. This is a per-batch quantity in the mp path — the parent sums the
    partial jk grids and computes ``R_jk`` separately via ``compute_R_jk``.

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
    if backend not in ("tree", "brute"):
        raise NotImplementedError(
            f"pair_kernel.accumulate: unknown backend {backend!r} (expected 'tree' or 'brute')."
        )
    if chunk_axis == "position":
        if not isinstance(binning, (SkyRpPi, SkyRMuR)):
            raise NotImplementedError(
                "pair_kernel.accumulate: chunk_axis='position' (lightcone) requires a "
                "SkyRpPi / SkyRMuR binning."
            )
        if jk and num_box is None:
            raise ValueError("pair_kernel.accumulate: jk=True requires num_box (num_jk).")
        if backend == "brute" and shape_tree is not None:
            raise ValueError(
                "pair_kernel.accumulate: shape_tree is meaningless with backend='brute'."
            )
        return _accumulate_lightcone(
            sample_set, binning, base=base, shapes=shapes,
            chunk_size_outer=chunk_size_outer, backend=backend,
            jk=jk, num_jk=num_box, shape_tree=shape_tree,
        )
    if chunk_axis != "shape":
        raise NotImplementedError(
            "pair_kernel.accumulate: only chunk_axis='shape' (box) and "
            "chunk_axis='position' (lightcone) are implemented."
        )
    if not isinstance(binning, (BoxRpPi, BoxRMuR)):
        raise NotImplementedError(
            "pair_kernel.accumulate: only BoxRpPi / BoxRMuR binnings are implemented "
            "for the box (chunk_axis='shape') path."
        )
    if backend == "brute" and pos_tree is not None:
        raise ValueError(
            "pair_kernel.accumulate: pos_tree is meaningless with backend='brute' "
            "(no KDTree is built); pass pos_tree only with backend='tree'."
        )
    if jk and num_box is None:
        raise ValueError("pair_kernel.accumulate: jk=True requires num_box.")

    DD = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r)
    Splus_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None
    Scross_D = np.array([[0.0] * binning.num_bins_pi] * binning.num_bins_r) if shapes else None
    DD_jk = np.zeros((num_box, binning.num_bins_r, binning.num_bins_pi)) if jk else None
    Splus_D_jk = np.zeros((num_box, binning.num_bins_r, binning.num_bins_pi)) if (jk and shapes) else None

    positions = sample_set.pos
    positions_shape_sample = sample_set.pos_shape
    weight = sample_set.weight
    weight_shape = sample_set.weight_shape
    not_LOS = sample_set.not_LOS
    LOS_ind = sample_set.LOS_ind
    jk_pos = sample_set.jk_pos

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
        if jk:
            jk_shape_i = sample_set.jk_shape[i:i2]
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

                if jk:
                    # union (two-sided) deletion: every pair contributes to the shape's
                    # patch, and to the position's patch only where the two patches differ.
                    # Splus_D_jk stores the raw (un-/2R) S+ contribution; responsivity is
                    # applied later in the wrapper reduction. Order matches the legacy jk
                    # loop (Splus_D_jk before DD_jk).
                    shape_patch = jk_shape_i[n]
                    pos_patches = jk_pos[ind_rbin_i[n]][mask]
                    pos_diff = np.where(pos_patches != shape_patch)[0]
                    w_pairs = weight[ind_rbin_i[n]][mask] * weight_shape_i[n]
                    if shapes:
                        np.add.at(Splus_D_jk, (shape_patch, ind_r, ind_pi), w_pairs * e_plus[mask])
                        np.add.at(Splus_D_jk,
                                  (pos_patches[pos_diff], ind_r[pos_diff], ind_pi[pos_diff]),
                                  (w_pairs * e_plus[mask])[pos_diff])
                    np.add.at(DD_jk, (shape_patch, ind_r, ind_pi), w_pairs)
                    np.add.at(DD_jk,
                              (pos_patches[pos_diff], ind_r[pos_diff], ind_pi[pos_diff]),
                              w_pairs[pos_diff])

    return Grids(DD=DD, Splus_D=Splus_D, Scross_D=Scross_D, DD_jk=DD_jk, Splus_D_jk=Splus_D_jk)


def compute_R_jk(e, weight_shape, jk_shape, num_box, responsivity_correction):
    """Per-realisation (delete-one) responsivity: ``R_jk[i]`` is the responsivity over
    the shapes **not** in patch ``i``.

    This is a standalone reduction over the shape sample (not part of the pair loop),
    so the multiprocessing path calls it once in the parent from the *full* shape
    sample rather than per batch. Reproduces the legacy inline computation verbatim
    (including the ``responsivity_correction``/empty-patch fallback to ``0.5``).
    """
    R_jk = np.zeros(num_box)
    for i in np.arange(num_box):
        jk_mask = np.where(jk_shape != i)
        R_jk[i] = sum(weight_shape[jk_mask] * (1 - e[jk_mask] ** 2 / 2.0)) / sum(weight_shape[jk_mask]) \
            if responsivity_correction and sum(weight_shape[jk_mask]) > 0 else 0.5
    return R_jk
