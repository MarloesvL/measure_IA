r"""Per-galaxy alignment contributions to the multipole estimator, for a periodic box.

The multipole estimator is a fixed linear map on the pair sums,

.. math::
    \tilde{\xi}_{g+,\ell}(r_b) = \sum_m K(b,m)\, S_+D(b,m), \qquad
    K(b,m) = \frac{2\ell+1}{2}\frac{(\ell-s)!}{(\ell+s)!}
             L^{\ell,s}(\mu_m)\,\Delta\mu_m \big/ RR_{g+}(b,m),

and in a periodic box :math:`RR_{g+}` is analytic, so :math:`K` is known before any pair
counting. This module resolves that same sum **per shape galaxy**:

.. math::
    Y_j(b) = \sum_m K(b,m) \sum_{i:(r_{ij},\mu_{ij})\in(b,m)} \frac{e_+(j|i)}{2\mathcal{R}},
    \qquad P_j(b) = \sum_{i:r_{ij}\in b} 1

so that :math:`\tilde{\xi}_{g+,\ell}(r_b) = \sum_j Y_j(b)` exactly (to floating-point
summation order), and :math:`Y_j(b)/P_j(b)` is the mean alignment contribution of galaxy
``j`` in bin ``b``.

This is the quantity needed to regress the alignment signal on per-galaxy properties: a
per-radial-bin least-squares fit of the pair contributions on standardised galaxy
properties has normal equations

.. math::
    (X^\top X)_{kl}(b) = \sum_j x_{k,j} x_{l,j} P_j(b), \qquad
    (X^\top y)_k(b)    = \sum_j x_{k,j} Y_j(b),

so *any* number of properties can be fitted from a single pair traversal, rather than
re-running the correlation function once per weighting.
"""
import math

import h5py
import numpy as np
import sympy
from scipy.spatial import KDTree
from scipy.special import assoc_legendre_p

from . import pair_kernel
from .write_data import create_group_hdf5, write_dataset_hdf5


class MeasureGalaxyContributionsBox:
    """Mixin providing :meth:`measure_galaxy_contributions` for ``MeasureIABox``."""

    def _multipole_kernel(self, RR_g_plus, ell, sab):
        r"""The projection matrix :math:`K(b,m)` that turns :math:`S_+D(b,m)` into
        :math:`\tilde{\xi}_{g+,\ell}(r_b)`.

        Mirrors the reduction in ``MeasureIABase._measure_multipoles`` exactly: the same
        associated Legendre call, the same ``mu_r`` bin midpoints and the same full bin
        widths, with the ``1/RR`` of the estimator folded in.

        Parameters
        ----------
        RR_g_plus : ndarray
            Analytic random-pair grid in (r, mu_r), shape (num_bins_r, num_bins_pi).
        ell, sab : int
            Multipole order and spin. (2, 2) for xi_g+, (0, 0) for xi_gg.

        Returns
        -------
        ndarray
            K, shape (num_bins_r, num_bins_pi).
        """
        dmur = self.mu_r_bins[1:] - self.mu_r_bins[:-1]
        mu_mid = self.mu_r_bins[:-1] + dmur / 2.0
        L = assoc_legendre_p(sab, ell, mu_mid)[0]
        prefactor = (2 * ell + 1) / 2.0 * math.factorial(ell - sab) / math.factorial(ell + sab)
        RR_denom = RR_g_plus.copy()
        RR_denom[RR_denom == 0] = 1  # same guard as the estimator; those bins are empty anyway
        return prefactor * L[None, :] * dmur[None, :] / RR_denom

    def measure_galaxy_contributions(self, dataset_name, num_jk=0, masks=None, rp_cut=None,
                                     ellipticity='distortion', responsivity=True, ell=2,
                                     return_output=False):
        r"""Measure the per-shape-galaxy contributions to $\tilde{\xi}_{g+,\ell}(r)$.

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
            Minimum projected separation for a pair to be included. Default None.
        ellipticity : str, optional
            'distortion' ((1-q^2)/(1+q^2)) or 'ellipticity' ((1-q)/(1+q)). Default
            'distortion'.
        responsivity : bool, optional
            If True (default), divide the shape signal by the responsivity 2R, as in the
            correlation-function measurement.
        ell : int, optional
            Multipole order; the spin is taken equal to it (2 for xi_g+). Default 2.
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
            delete-one responsivities, and ``rr_ratio`` (num_jk,) the scalar
            ``RR_jk / RR`` amplitude ratios.

        Notes
        -----
        ``Y`` has the responsivity divided out and the full-sample ``RR`` folded in, so
        ``Y.sum(axis=0)`` is the ordinary multipole. ``Y_jk`` is stored **raw** — neither
        ``2R`` nor the per-realisation ``RR`` amplitude applied — matching the convention
        of the package's own ``Splus_D_jk`` grids. Delete-one realisation ``n`` is then

        .. code-block:: python

            keep = jk_shape != n
            xi_n = (Y_jk[keep].sum(axis=1) - Y_jk[keep, n]).sum(axis=0)
            xi_n /= rr_ratio[n] * 2 * R_jk[n]

        This path is single-process; ``num_nodes`` is not used.
        """
        if self.data is not None and "RA" in self.data:
            raise TypeError("Given data is lightcone data (contains 'RA'). Use MeasureIALightcone instead.")
        if ellipticity not in ('distortion', 'ellipticity'):
            raise ValueError("Unknown value for ellipticity. Choose from ['distortion', 'ellipticity'].")
        if num_jk < 0:
            raise ValueError("num_jk must be >= 0.")
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
        )
        Num_position = len(sample_set.pos)
        Num_shape = len(sample_set.pos_shape)
        e = sample_set.e
        weight_shape = sample_set.weight_shape
        R = sum(weight_shape * (1 - e ** 2 / 2.0)) / sum(weight_shape) \
            if responsivity and sum(weight_shape) > 0 else 0.5
        if rp_cut is None:
            rp_cut = 0.0

        L3 = self.boxsize ** 3
        RR_g_plus = np.zeros((self.num_bins_r, self.num_bins_pi))
        for i in np.arange(0, self.num_bins_r):
            for p in np.arange(0, self.num_bins_pi):
                RR_g_plus[i, p] = self.get_random_pairs_r_mur(
                    self.r_bins[i + 1], self.r_bins[i], self.mu_r_bins[p + 1], self.mu_r_bins[p],
                    L3, "cross", Num_position, Num_shape)
        K = self._multipole_kernel(RR_g_plus, ell, ell)

        if num_jk > 0:
            jk_pos, jk_shape = self._get_jackknife_region_indices(masks, L_subboxes)
            sample_set.jk_pos = jk_pos
            sample_set.jk_shape = jk_shape

        binning = pair_kernel.BoxRMuR(self, rp_cut)
        pos_tree = KDTree(binning.tree_coords(sample_set.pos, sample_set.not_LOS), boxsize=self.boxsize)
        print(f"There are {Num_shape} galaxies in the shape sample and {Num_position} galaxies in the position sample.")
        grids = pair_kernel.accumulate(
            sample_set, binning, base=self, R=R, shapes=True,
            chunk_axis="shape", chunk_size_outer=100, backend="tree", pos_tree=pos_tree,
            per_galaxy=True, per_galaxy_proj=K, per_galaxy_jk=num_jk > 0,
            num_box=num_jk if num_jk > 0 else None,
        )

        dsep = (self.r_bins[1:] - self.r_bins[:-1]) / 2.0
        separation = self.r_bins[:-1] + abs(dsep)
        out = {"Y": grids.Splus_D_gal, "P": grids.DD_gal, "r": separation}

        if num_jk > 0:
            R_jk = pair_kernel.compute_R_jk(e, weight_shape, jk_shape, num_jk, responsivity)
            # RR is analytic and separable in (r, mu_r): every bin carries the same
            # geometric factor, so RR_jk[n] / RR is a scalar and the projection kernel K
            # simply rescales by 1 / rr_ratio[n]. Note get_random_pairs_r_mur normalises
            # by (Num_position - 1) * Num_shape / volume, not Num_position * Num_shape —
            # dropping the -1 leaves a systematic ~1/N error in every realisation.
            volume_jk = L3 * (num_jk - 1) / num_jk
            rr_ratio = np.zeros(num_jk)
            denom = (Num_position - 1.0) * Num_shape / L3
            for n in np.arange(num_jk):
                n_pos_jk = int(np.count_nonzero(jk_pos != n))
                n_shape_jk = int(np.count_nonzero(jk_shape != n))
                rr_ratio[n] = ((n_pos_jk - 1.0) * n_shape_jk / volume_jk) / denom
            out.update({"Y_jk": grids.Splus_D_gal_jk, "P_jk": grids.DD_gal_jk,
                        "jk_shape": jk_shape, "R_jk": R_jk, "rr_ratio": rr_ratio, "R": R})

        if return_output:
            return out

        output_file = h5py.File(self.output_file_name, "a")
        group = create_group_hdf5(output_file, f"{self.snap_group}galaxy_contributions")
        write_dataset_hdf5(group, dataset_name + "_Y", data=out["Y"])
        write_dataset_hdf5(group, dataset_name + "_P", data=out["P"])
        write_dataset_hdf5(group, dataset_name + "_r", data=out["r"])
        if num_jk > 0:
            jk_group = create_group_hdf5(group, f"{dataset_name}_jk{num_jk}")
            for key in ("Y_jk", "P_jk", "jk_shape", "R_jk", "rr_ratio"):
                write_dataset_hdf5(jk_group, key, data=out[key])
            jk_group.attrs["R"] = R
        output_file.close()
        return None
