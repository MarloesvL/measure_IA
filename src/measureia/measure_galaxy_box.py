r"""Per-galaxy alignment contributions to the box estimators.

Both box estimators are fixed linear maps on the pair sums. For the multipoles,

.. math::
    \tilde{\xi}_{g+,\ell}(r_b) = \sum_m K(b,m)\, S_+D(b,m), \qquad
    K(b,m) = \frac{2\ell+1}{2}\frac{(\ell-s)!}{(\ell+s)!}
             L^{\ell,s}(\mu_m)\,\Delta\mu_m \big/ RR_{g+}(b,m),

and for the projected correlation function,

.. math::
    w_{g+}(r_{p,b}) = \sum_m K(b,m)\, S_+D(b,m), \qquad
    K(b,m) = |\Delta\Pi_m| \big/ RR_{g+}(b,m).

In a periodic box :math:`RR_{g+}` is analytic, so :math:`K` is known before any pair
counting. This module resolves the same sum **per shape galaxy**:

.. math::
    Y_j(b) = \sum_m K(b,m) \sum_{i:(b,m)} \frac{e_+(j|i)}{2\mathcal{R}},
    \qquad P_j(b) = \sum_{i:\,\text{bin}\,b} 1

so that the estimator is :math:`\sum_j Y_j(b)` exactly (to floating-point summation
order), and :math:`Y_j(b)/P_j(b)` is the mean alignment contribution of galaxy ``j`` in
bin ``b``.

This is the quantity needed to regress the alignment signal on per-galaxy properties: a
per-bin least-squares fit of the pair contributions on standardised galaxy properties has
normal equations

.. math::
    (X^\top X)_{kl}(b) = \sum_j x_{k,j} x_{l,j} P_j(b), \qquad
    (X^\top y)_k(b)    = \sum_j x_{k,j} Y_j(b),

so *any* number of properties can be fitted from a single pair traversal, rather than
re-running the correlation function once per weighting.
"""
import math
import multiprocessing as mp

from . import worker_pool
import os
from multiprocessing import Pool, shared_memory

import h5py
import numpy as np
import sympy
from scipy.spatial import KDTree
from scipy.special import assoc_legendre_p

from . import pair_kernel
from .read_data import ReadData
from .write_data import create_group_hdf5, write_dataset_hdf5

_STATISTICS = ("multipoles", "w")


def jk_columns(out, n):
    r"""The delete-one column for realisation ``n``: what each galaxy contributes through
    position-sample partners lying in patch ``n``.

    Handles the sparse storage, where a galaxy only carries the patches it has pairs in.

    Parameters
    ----------
    out : dict
        Output of :meth:`MeasureGalaxyContributionsBox.measure_galaxy_contributions`.
    n : int
        Jackknife patch index.

    Returns
    -------
    (ndarray, ndarray)
        ``(Y_column, P_column)``, each (M, num_bins_r).
    """
    hit = out["jk_patches"] == n                             # (M, K), at most one per row
    Y = np.einsum("mk,mkb->mb", hit, out["Y_jk_values"])
    P = np.einsum("mk,mkb->mb", hit, out["P_jk_values"])
    return Y, P


def delete_one_estimator(out, n):
    r"""Rebuild delete-one jackknife realisation ``n`` of the estimator from the per-galaxy
    contributions.

    Drops every shape galaxy in patch ``n``, subtracts the pairs whose position partner is
    in patch ``n`` from the rest, and undoes the two normalisations that ``Y_jk_values``
    deliberately leaves off (the responsivity ``2R`` and the ``RR`` amplitude, both of
    which change under delete-one).

    Parameters
    ----------
    out : dict
        Output of :meth:`MeasureGalaxyContributionsBox.measure_galaxy_contributions`,
        measured with ``num_jk > 0``.
    n : int
        Jackknife patch index.

    Returns
    -------
    ndarray
        The estimator for realisation ``n``, shape (num_bins_r,).
    """
    keep = out["jk_shape"] != n
    Y_col, _ = jk_columns(out, n)
    total = out["Y_jk_values"].sum(axis=1)
    est = (total[keep] - Y_col[keep]).sum(axis=0)
    return est / (out["rr_ratio"][n] * 2 * out["R_jk"][n])


class MeasureGalaxyContributionsBox:
    """Mixin providing :meth:`measure_galaxy_contributions` for ``MeasureIABox``."""

    # ------------------------------------------------------------------
    # estimator geometry
    # ------------------------------------------------------------------

    def _galaxy_binning(self, statistic, rp_cut):
        """The binning object for the requested statistic (``rp_cut`` is multipoles-only)."""
        if statistic == "multipoles":
            return pair_kernel.BoxRMuR(self, rp_cut)
        return pair_kernel.BoxRpPi(self)

    def _galaxy_analytic_RR(self, statistic, volume, Num_position, Num_shape):
        r"""Analytic $RR_{g+}$ grid for the requested statistic.

        Delegates to the same ``get_random_pairs`` / ``get_random_pairs_r_mur`` the
        estimators use, so the normalisation conventions match exactly — including the
        fact that the two differ: the (r, mu_r) form carries a ``(Num_position - 1)``
        while the (rp, pi) form carries a plain ``Num_position``.
        """
        RR = np.zeros((self.num_bins_r, self.num_bins_pi))
        second_bins = self.mu_r_bins if statistic == "multipoles" else self.pi_bins
        get_pairs = (self.get_random_pairs_r_mur if statistic == "multipoles"
                     else self.get_random_pairs)
        for i in np.arange(0, self.num_bins_r):
            for p in np.arange(0, self.num_bins_pi):
                RR[i, p] = get_pairs(self.r_bins[i + 1], self.r_bins[i],
                                     second_bins[p + 1], second_bins[p],
                                     volume, "cross", Num_position, Num_shape)
        return RR

    def _galaxy_rr_ratio(self, statistic, volume_jk, n_pos_jk, n_shape_jk,
                         volume, Num_position, Num_shape):
        """``RR_jk / RR``, the scalar amplitude ratio between a delete-one realisation and
        the full sample.

        ``RR`` is separable — the bin geometry factors out and only the sample/volume
        amplitude changes — so the ratio taken from any single bin is the ratio for the
        whole grid. Deriving it by calling the estimator's own ``RR`` function, rather than
        restating the normalisation here, means this cannot drift from the convention the
        estimator actually uses (the two ``get_random_pairs*`` functions do not currently
        agree on whether the cross count carries ``Num_position - 1``).
        """
        get_pairs = (self.get_random_pairs_r_mur if statistic == "multipoles"
                     else self.get_random_pairs)
        second_bins = self.mu_r_bins if statistic == "multipoles" else self.pi_bins
        bin_args = (self.r_bins[1], self.r_bins[0], second_bins[1], second_bins[0])
        full = get_pairs(*bin_args, volume, "cross", Num_position, Num_shape)
        jk = get_pairs(*bin_args, volume_jk, "cross", n_pos_jk, n_shape_jk)
        return jk / full

    def _galaxy_projection_kernel(self, statistic, RR_g_plus, ell):
        r"""The matrix $K(b,m)$ that turns $S_+D(b,m)$ into the estimator.

        Mirrors the reductions in ``MeasureIABase._measure_multipoles`` and
        ``MeasureIABase._measure_w_g_i`` exactly — same Legendre call, same bin midpoints
        and widths — with the ``1/RR`` of the estimator folded in.
        """
        RR_denom = RR_g_plus.copy()
        RR_denom[RR_denom == 0] = 1  # same guard as the estimators; those bins are empty
        if statistic == "multipoles":
            dmur = self.mu_r_bins[1:] - self.mu_r_bins[:-1]
            mu_mid = self.mu_r_bins[:-1] + dmur / 2.0
            sab = ell
            L = assoc_legendre_p(sab, ell, mu_mid)[0]
            prefactor = (2 * ell + 1) / 2.0 * math.factorial(ell - sab) / math.factorial(ell + sab)
            weights = prefactor * L * dmur
        else:
            weights = np.abs(self.pi_bins[1:] - self.pi_bins[:-1])
        return weights[None, :] / RR_denom

    def _galaxy_separation_bins(self):
        dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
        return self.r_bins[:-1] + abs(dsep)

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    @worker_pool.pooled
    def measure_galaxy_contributions(self, dataset_name, num_jk=0, masks=None, rp_cut=None,
                                     ellipticity='distortion', responsivity=True, ell=2,
                                     statistic="multipoles", temp_file_path=None,
                                     chunk_size=1000, return_output=False):
        r"""Measure the per-shape-galaxy contributions to the box IA estimator.

        Runs one pair traversal and returns, per shape galaxy and radial bin, the
        projected alignment contribution ``Y`` and the pair count ``P`` described in the
        module docstring. With ``num_jk > 0`` it also returns the decomposition of both by
        the jackknife sub-box of the *position-sample* partner, which is enough to rebuild
        every delete-one realisation without re-counting any pairs.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset in the output file.
        num_jk : int, optional
            Number of jackknife regions; must be x^3 with x an int. Default 0 (none).
        masks : dict or NoneType, optional
            Mask dictionary in the same form as the data dictionary. Default None.
        rp_cut : float or NoneType, optional
            Minimum projected separation for a pair to be included. Multipoles only, as in
            ``measure_xi_multipoles``. Default None.
        ellipticity : str, optional
            'distortion' ((1-q^2)/(1+q^2)) or 'ellipticity' ((1-q)/(1+q)). Default
            'distortion'.
        responsivity : bool, optional
            If True (default), divide the shape signal by the responsivity 2R, as in the
            correlation-function measurement.
        ell : int, optional
            Multipole order; the spin is taken equal to it. Ignored when
            ``statistic="w"``. Default 2.
        statistic : str, optional
            Which estimator to project onto: ``"multipoles"`` for
            $\tilde{\xi}_{g+,\ell}(r)$ (r, mu_r binning) or ``"w"`` for $w_{g+}(r_p)$
            (rp, pi binning). Default "multipoles".
        temp_file_path : str or NoneType, optional
            Path where the data is temporarily offloaded during multiprocessing, so the
            parent does not hold a second copy in RAM. Required when ``num_nodes > 1``.
        chunk_size : int, optional
            Number of shape galaxies per multiprocessing task. Default 1000.
        return_output : bool, optional
            If True, return the results dict instead of writing it to the output file.
            Default False.

        Returns
        -------
        dict
            Only if ``return_output``. Keys: ``Y`` (M, num_bins_r), ``P`` (M, num_bins_r),
            ``r`` (num_bins_r,), and — when ``num_jk > 0`` — ``Y_jk``/``P_jk``
            (M, num_jk, num_bins_r) decomposed by the position partner's patch,
            ``jk_shape`` (M,) the patch of each shape galaxy, ``R_jk`` (num_jk,) the
            delete-one responsivities, ``rr_ratio`` (num_jk,) the scalar ``RR_jk / RR``
            amplitude ratios, and ``R`` the full-sample responsivity.

        Notes
        -----
        ``Y`` has the responsivity divided out and the full-sample ``RR`` folded in, so
        ``Y.sum(axis=0)`` is the ordinary estimator. ``Y_jk`` is stored **raw** — neither
        ``2R`` nor the per-realisation ``RR`` amplitude applied — matching the convention
        of the package's own ``Splus_D_jk`` grids. Delete-one realisation ``n`` is then

        .. code-block:: python

            keep = jk_shape != n
            est_n = (Y_jk[keep].sum(axis=1) - Y_jk[keep, n]).sum(axis=0)
            est_n /= rr_ratio[n] * 2 * R_jk[n]

        Runs on ``num_nodes`` cores (set at initialisation); ``num_nodes > 1`` requires
        ``temp_file_path`` and the usual ``if __name__ == "__main__":`` guard.
        """
        if self.data is not None and "RA" in self.data:
            raise TypeError("Given data is lightcone data (contains 'RA'). Use MeasureIALightcone instead.")
        if statistic not in _STATISTICS:
            raise ValueError(f"Unknown statistic {statistic!r}. Choose from {list(_STATISTICS)}.")
        if ellipticity not in ('distortion', 'ellipticity'):
            raise ValueError("Unknown value for ellipticity. Choose from ['distortion', 'ellipticity'].")
        if num_jk < 0:
            raise ValueError("num_jk must be >= 0.")
        num_nodes = getattr(self, "num_nodes", 1)
        if num_nodes > 1 and temp_file_path is None:
            raise ValueError(
                "measure_galaxy_contributions: num_nodes > 1 requires temp_file_path, where the "
                "catalogue is offloaded while the workers run."
            )
        L_subboxes = None
        if num_jk > 0:
            root, exact = sympy.integer_nthroot(num_jk, 3)
            if not exact:
                raise ValueError(f"Use x^3 as input for num_jk, with x an int. Got {num_jk}.")
            L_subboxes = root
            self.check_jackknife_max_separation(num_jk, self.boxsize, self.r_max, self.num_bins_r)

        self.responsivity_correction = responsivity
        masks = self.rename_input_keys(masks, self._input_name_map)

        sample_set = pair_kernel.prepare_box_samples(
            self.data, masks, self.Num_position, self.Num_shape,
            shapes=True, ellipticity=ellipticity, base=self,
            require_full_masks=num_nodes > 1,
        )
        Num_position = len(sample_set.pos)
        Num_shape = len(sample_set.pos_shape)
        e = sample_set.e
        weight_shape = sample_set.weight_shape
        R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
            if responsivity and sum(weight_shape) > 0 else 0.5
        self.rp_cut = 0.0 if rp_cut is None else rp_cut
        print(f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")

        L3 = self.boxsize ** 3
        RR_g_plus = self._galaxy_analytic_RR(statistic, L3, Num_position, Num_shape)
        K = self._galaxy_projection_kernel(statistic, RR_g_plus, ell)

        jk_pos = jk_shape = None
        if num_jk > 0:
            jk_pos, jk_shape = self._get_jackknife_region_indices(masks, L_subboxes)
            sample_set.jk_pos = jk_pos
            sample_set.jk_shape = jk_shape

        binning = self._galaxy_binning(statistic, self.rp_cut)
        pos_tree = KDTree(binning.tree_coords(sample_set.pos, sample_set.not_LOS), boxsize=self.boxsize)

        if num_nodes > 1:
            Y, P, Y_jk, P_jk, jk_patches = self._galaxy_contributions_multiprocessing(
                sample_set, statistic, K, R, num_jk, pos_tree, dataset_name,
                temp_file_path, chunk_size, num_nodes, masks,
            )
        else:
            grids = pair_kernel.accumulate(
                sample_set, binning, base=self, R=R, shapes=True,
                chunk_axis="shape", chunk_size_outer=100, backend="tree", pos_tree=pos_tree,
                per_galaxy=True, per_galaxy_proj=K, per_galaxy_jk=num_jk > 0,
                per_galaxy_jk_sparse=num_jk > 0,
                num_box=num_jk if num_jk > 0 else None,
            )
            Y, P = grids.Splus_D_gal, grids.DD_gal
            Y_jk, P_jk = grids.Splus_D_gal_jk_values, grids.DD_gal_jk_values
            jk_patches = grids.gal_jk_patches

        out = {"Y": Y, "P": P, "r": self._galaxy_separation_bins()}
        if num_jk > 0:
            R_jk = pair_kernel.compute_R_jk(e, weight_shape, jk_shape, num_jk, responsivity)
            # RR is analytic and separable: every (r, mu_r) or (rp, pi) bin carries the
            # same geometric factor, so RR_jk[n] / RR is a scalar and the projection
            # kernel K simply rescales by 1 / rr_ratio[n].
            volume_jk = L3 * (num_jk - 1) / num_jk
            rr_ratio = np.zeros(num_jk)
            for n in np.arange(num_jk):
                n_pos_jk = int(np.count_nonzero(jk_pos != n))
                n_shape_jk = int(np.count_nonzero(jk_shape != n))
                rr_ratio[n] = self._galaxy_rr_ratio(statistic, volume_jk, n_pos_jk, n_shape_jk,
                                                    L3, Num_position, Num_shape)
            out.update({"Y_jk_values": Y_jk, "P_jk_values": P_jk, "jk_patches": jk_patches,
                        "jk_shape": jk_shape, "R_jk": R_jk, "rr_ratio": rr_ratio, "R": R})

        if return_output:
            return out

        output_file = h5py.File(self.output_file_name, "a")
        group = create_group_hdf5(output_file, f"{self.snap_group}galaxy_contributions/{statistic}")
        write_dataset_hdf5(group, dataset_name + "_Y", data=out["Y"])
        write_dataset_hdf5(group, dataset_name + "_P", data=out["P"])
        write_dataset_hdf5(group, dataset_name + "_r", data=out["r"])
        if num_jk > 0:
            jk_group = create_group_hdf5(group, f"{dataset_name}_jk{num_jk}")
            for key in ("Y_jk_values", "P_jk_values", "jk_patches", "jk_shape",
                        "R_jk", "rr_ratio"):
                write_dataset_hdf5(jk_group, key, data=out[key])
            jk_group.attrs["R"] = out["R"]
        output_file.close()
        return None

    # ------------------------------------------------------------------
    # multiprocessing
    # ------------------------------------------------------------------

    def _galaxy_contributions_batch(self, i):
        """Per-galaxy shape-sample batch worker. Reads the shared-memory catalogue, runs
        ``pair_kernel.accumulate`` on the slice with the parent's tree and projection
        kernel, and returns that slice's per-galaxy arrays. Support function for
        :meth:`_galaxy_contributions_multiprocessing`."""
        i2 = min(self.Num_shape_masked, i + self.chunk_size)

        shms = []
        shared_data = {}
        for name, shape, dtype in self.shm_infos:
            shm = shared_memory.SharedMemory(name=name)
            shared_data[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            shms.append(shm)

        sample_set = pair_kernel.SampleSet(
            pos=shared_data[f"positions_{self.ID_shm}"],
            pos_shape=shared_data[f"positions_shape_sample_{self.ID_shm}"][i:i2],
            weight=shared_data[f"weight_{self.ID_shm}"],
            weight_shape=shared_data[f"weight_shape_{self.ID_shm}"][i:i2],
            axis_direction=shared_data[f"axis_direction_{self.ID_shm}"][i:i2],
            e=shared_data[f"e_{self.ID_shm}"][i:i2],
            LOS_ind=self.LOS_ind,
            not_LOS=self.not_LOS,
        )
        if self.num_box is not None:
            sample_set.jk_pos = shared_data[f"jk_region_indices_pos_{self.ID_shm}"]
            sample_set.jk_shape = shared_data[f"jk_region_indices_shape_{self.ID_shm}"][i:i2]

        binning = self._galaxy_binning(self._pg_statistic, self.rp_cut)
        grids = pair_kernel.accumulate(
            sample_set, binning, base=self, R=self.R, shapes=True,
            chunk_axis="shape", chunk_size_outer=100, backend="tree", pos_tree=self.pos_tree,
            per_galaxy=True, per_galaxy_proj=self._pg_proj,
            per_galaxy_jk=self.num_box is not None,
            per_galaxy_jk_sparse=self.num_box is not None, num_box=self.num_box,
        )
        for shm in shms:
            shm.close()
        return (grids.Splus_D_gal, grids.DD_gal, grids.Splus_D_gal_jk_values,
                grids.DD_gal_jk_values, grids.gal_jk_patches)

    def _galaxy_contributions_multiprocessing(self, sample_set, statistic, K, R, num_jk,
                                              pos_tree, dataset_name, temp_file_path,
                                              chunk_size, num_nodes, masks):
        """Run the per-galaxy accumulation across ``num_nodes`` processes.

        Follows the same pattern as the other box multiprocessing paths: the catalogue is
        copied into shared memory and offloaded to a temp file so the parent can drop its
        own copy, workers each handle a contiguous slice of the shape sample, and the
        per-galaxy arrays are concatenated back in slice order (``Pool.map`` preserves the
        order of ``indices``, and the galaxy axis is contiguous by construction).
        """
        positions = sample_set.pos
        positions_shape_sample = sample_set.pos_shape
        self.Num_position_masked = len(positions)
        self.Num_shape_masked = len(positions_shape_sample)
        self.R = R
        self.pos_tree = pos_tree
        self.chunk_size = chunk_size
        self.num_box = num_jk if num_jk > 0 else None
        self._pg_proj = K
        self._pg_statistic = statistic
        self.LOS_ind = sample_set.LOS_ind
        self.not_LOS = sample_set.not_LOS
        indices = np.arange(0, self.Num_shape_masked, chunk_size)

        figname = dataset_name.replace("/", "_").replace(".", "p")
        temp_file = f"{temp_file_path}/gal_{self.simname}_temp_data_{figname}.hdf5"
        file_temp = h5py.File(temp_file, "w")
        keys = []
        for k in self.data.keys():
            if k != "LOS":
                write_dataset_hdf5(file_temp, k, self.data[k])
                if masks is not None:
                    write_dataset_hdf5(file_temp, f"mask_{k}", masks[k])
                keys.append(k)
        file_temp.close()

        self.ID_shm = np.random.randint(100000)
        shared_data = {
            f"positions_{self.ID_shm}": positions,
            f"positions_shape_sample_{self.ID_shm}": positions_shape_sample,
            f"axis_direction_{self.ID_shm}": sample_set.axis_direction,
            f"e_{self.ID_shm}": sample_set.e,
            f"weight_{self.ID_shm}": sample_set.weight,
            f"weight_shape_{self.ID_shm}": sample_set.weight_shape,
        }
        if num_jk > 0:
            shared_data[f"jk_region_indices_pos_{self.ID_shm}"] = sample_set.jk_pos
            shared_data[f"jk_region_indices_shape_{self.ID_shm}"] = sample_set.jk_shape
        for k in shared_data.keys():
            try:
                shared_memory.SharedMemory(name=k).unlink()
            except FileNotFoundError:
                pass
        shm_blocks, self.shm_infos = [], []
        saved_data = self.data
        try:
            for k in shared_data.keys():
                shm = shared_memory.SharedMemory(name=k, create=True, size=shared_data[k].nbytes)
                shared_arr = np.ndarray(shared_data[k].shape, dtype=shared_data[k].dtype, buffer=shm.buf)
                np.copyto(shared_arr, shared_data[k])
                shm_blocks.append(shm)
                self.shm_infos.append([k, shared_data[k].shape, shared_data[k].dtype])
            self.data = {}
            if masks is not None:
                masks = {}
            del shared_data, shared_arr, saved_data
            del positions, positions_shape_sample
            sample_set.pos = sample_set.pos_shape = None
            sample_set.axis_direction = sample_set.e = None
            sample_set.jk_pos = sample_set.jk_shape = None
            with worker_pool.active_pool(num_nodes) as p:
                result = p.map(self._galaxy_contributions_batch, indices)
        finally:
            for shm in shm_blocks:
                shm.close()
                shm.unlink()
            # restore self.data from the temp file even if a worker failed
            if os.path.exists(temp_file):
                temp_obj = ReadData(self.simname, f"gal_{self.simname}_temp_data_{figname}", None,
                                    data_path=temp_file_path)
                for k in keys:
                    self.data[k] = temp_obj.read_cat(k)
                    if masks is not None:
                        masks[k] = temp_obj.read_cat(f"mask_{k}")
                self.data["LOS"] = self.LOS_ind
                os.remove(temp_file)

        Y = np.concatenate([r[0] for r in result], axis=0)
        P = np.concatenate([r[1] for r in result], axis=0)
        if num_jk > 0:
            # each worker packed its own slice, so K differs between them
            Y_jk = pair_kernel._pad_and_stack([r[2] for r in result], fill=0.0)
            P_jk = pair_kernel._pad_and_stack([r[3] for r in result], fill=0.0)
            jk_patches = pair_kernel._pad_and_stack([r[4] for r in result], fill=-1,
                                                    dtype=np.int32)
        else:
            Y_jk = P_jk = jk_patches = None
        return Y, P, Y_jk, P_jk, jk_patches
