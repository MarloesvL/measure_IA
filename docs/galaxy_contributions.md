# Per-galaxy contributions

`MeasureIABox.measure_galaxy_contributions` resolves the box alignment estimator **per shape galaxy**: one
pair traversal returns, for every shape galaxy $j$ and radial bin $b$, its contribution $Y_j(b)$ to the
signal and the number of pairs $P_j(b)$ it contributed through, such that

$$\sum_j Y_j(b) = \tilde\xi_{g+,\ell}(r_b) \qquad\text{(or } w_{g+}(r_{p,b})\text{)}$$

is exactly the ordinary estimator. $Y_j(b)/P_j(b)$ is then the mean alignment contribution of that galaxy in
that bin.

This is the input needed to **regress the alignment signal on per-galaxy properties**. A per-bin
least-squares fit of the pair contributions on standardised galaxy properties has normal equations

$$(X^\top X)_{kl}(b) = \sum_j x_{k,j}\, x_{l,j}\, P_j(b), \qquad (X^\top y)_k(b) = \sum_j x_{k,j}\, Y_j(b),$$

so *any* number of properties can be fitted from a single pair traversal, instead of one correlation-function
run per weighting. It is available for the box only.

## Running it

```python
from measureia import MeasureIABox

mi = MeasureIABox(data_dict, output_file_name="./out.hdf5", boxsize=205.0)

out = mi.measure_galaxy_contributions(
	"ds1",
	num_jk=27,              # optional: also decompose by jackknife patch
	statistic="multipoles",  # or "w" for w_g+(r_p)
	ell=2,                   # multipole order (spin taken equal to it)
	temp_file_path=False,    # a path is required when num_nodes > 1
	return_output=True,      # omit to write to the output file instead
)

out["Y"].sum(axis=0)   # == the ordinary xi_g+,2(r) from measure_xi_multipoles
out["Y"] / out["P"]    # mean contribution per galaxy per bin
```

`statistic="multipoles"` projects onto $\tilde\xi_{g+,\ell}(r)$ in $(r, \mu_r)$ bins, `statistic="w"` onto
$w_{g+}(r_p)$ in $(r_p, \Pi)$ bins. The `ellipticity`, `responsivity`, `masks` and `rp_cut` arguments behave
exactly as in `measure_xi_multipoles`, so the per-galaxy decomposition matches the correlation function you
would otherwise measure. With `num_nodes > 1` the traversal runs on multiple cores and needs
`temp_file_path` plus the usual `if __name__ == "__main__":` guard.

## Jackknife realisations without re-counting pairs

With `num_jk > 0`, both $Y$ and $P$ are additionally decomposed by the jackknife sub-box of the
*position-sample* partner. That is enough to rebuild every delete-one realisation from the single traversal —
dropping the shape galaxies in the omitted patch, subtracting the pairs whose position partner lies in it,
and reapplying the two normalisations that change under delete-one (the responsivity $2\mathcal{R}$ and the
$RR$ amplitude). The `delete_one_estimator` helper does all of that:

```python
from measureia.measure_galaxy_box import delete_one_estimator, jk_columns

out = mi.measure_galaxy_contributions("ds1", num_jk=27, temp_file_path=False, return_output=True)

est_5 = delete_one_estimator(out, 5)   # realisation 5, identical to the stored one
Y_col, P_col = jk_columns(out, 5)      # what each galaxy contributes through patch 5
```

`delete_one_estimator(out, n)` reproduces `multipoles_g_plus/ds1_jk27/ds1_n` to floating-point summation
order. Use `jk_columns` when you need the per-galaxy columns themselves, for instance to jackknife a fitted
regression coefficient rather than the signal.

The jackknife arrays are stored **sparsely**: a galaxy's neighbours span a ball of $r_\mathrm{max}$, so it
reaches only a handful of sub-boxes however many there are, and the rest of the patch axis is structurally
zero. Only the patches a galaxy actually has pairs in are kept, with `jk_patches` recording which column is
which. Both helpers handle that indirection for you. The stored values are identical to the dense form — for
a COLIBRE-L400-sized run (301k shape galaxies, 125 patches, 12 bins) it is 0.78 GB per array instead of
3.6 GB.

!!! note "Y_jk_values is stored raw"
    `Y_jk_values` has neither the responsivity nor the per-realisation $RR$ amplitude applied, matching the
    convention of the package's own `Splus_D_jk` grids — `R_jk` and `rr_ratio` are stored alongside it so the
    normalisations can be applied per realisation. `Y` (full sample) *does* have both folded in, which is why
    it sums to the estimator directly. Use `delete_one_estimator` rather than combining the raw arrays
    by hand.

## Output

Written under a `galaxy_contributions` group; see
[Output file structure](output_structure.md#per-galaxy-contributions) for the datasets and their shapes. Pass
`return_output=True` to get the same arrays as a dictionary instead.

The method is documented in full on the [MeasureIABox API page](api/measureIABox.md).
