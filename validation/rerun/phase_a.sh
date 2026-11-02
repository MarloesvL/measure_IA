#!/bin/bash
# Review reruns, phase A: sparse mocks, w (rppi) and multipoles (rmu), 100 mocks each,
# 4 jobs at a time. Outputs in validation/rerun/.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}  # override with PY=/path/to/python
jobs=()
for b in rppi rmu; do
	for s in 0 25 50 75; do
		jobs+=("$PY -W ignore run_box_jk_cross_pairs.py terms --binning $b --n-jk 8 27 64 --n-mocks 25 --mock-start $s --out rerun/terms_sparse_${b}_${s}.npz > rerun/log_terms_sparse_${b}_${s}.txt 2>&1")
		jobs+=("$PY -W ignore run_box_jk_review_reruns.py analytic --binning $b --n-mocks 25 --mock-start $s --out rerun/analytic_sparse_${b}_${s}.npz > rerun/log_analytic_sparse_${b}_${s}.txt 2>&1")
	done
done
printf '%s\0' "${jobs[@]}" | xargs -0 -P 4 -n 1 bash -c
echo "phase A done"
