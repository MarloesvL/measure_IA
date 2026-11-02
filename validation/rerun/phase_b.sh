#!/bin/bash
# Review reruns, phase B: dense mocks (n = 1e-2, 3750 haloes), w (20 mocks) then multipoles
# (10 mocks), at most 2 jobs at a time. Outputs in validation/rerun/.
#
# Pausable and resumable:
#   soft pause : touch validation/rerun/PAUSE    (running jobs finish their mock, none start)
#   resume     : rm validation/rerun/PAUSE; bash validation/rerun/phase_b.sh   (skips finished work)
#   hard pause : pkill -STOP -f run_box_jk    /    continue: pkill -CONT -f run_box_jk
# Every job writes one file, under a temporary name renamed on completion, so an interrupted job
# leaves nothing behind and a finished one is never redone.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}  # override with PY=/path/to/python
NC=3750
NPARTS=4
WORKERS=2

# job "<output>" "<command without --out>": skipped if the output exists or PAUSE is set
job() {
	printf '%s\0' "out=$1; [ -f \"\$out\" ] && exit 0; [ -f rerun/PAUSE ] && exit 0; \
$2 --out \"\${out%.npz}.tmp.npz\" > \"\${out%.npz}.log\" 2>&1 && mv \"\${out%.npz}.tmp.npz\" \"\$out\""
}

run_stage() { xargs -0 -P "$WORKERS" -n 1 bash -c; }

for spec in "rppi 20" "rmu 10"; do
	set -- $spec
	b=$1; nm=$2
	# stage 1: counted RR of the random catalogue, in parts
	{ for p in $(seq 0 $((NPARTS - 1))); do
		job "rerun/rr_dense_${b}_${p}.npz" \
			"$PY -W ignore run_box_jk_cross_pairs.py rr_part --binning $b --n-centrals $NC --part $p --n-parts $NPARTS --n-jk 8 27 64"
	done; } | run_stage
	rr="rerun/rr_dense_${b}_0.npz rerun/rr_dense_${b}_1.npz rerun/rr_dense_${b}_2.npz rerun/rr_dense_${b}_3.npz"
	for f in $rr; do [ -f "$f" ] || { echo "RR part $f missing (paused?); stopping"; exit 0; }; done
	# stage 2: per-mock randoms terms and analytic terms
	{ for m in $(seq 0 $((nm - 1))); do
		job "rerun/terms_dense_${b}_${m}.npz" \
			"$PY -W ignore run_box_jk_cross_pairs.py terms --binning $b --n-centrals $NC --n-jk 8 27 64 --n-mocks 1 --mock-start $m --merge $rr"
		job "rerun/analytic_dense_${b}_${m}.npz" \
			"$PY -W ignore run_box_jk_review_reruns.py analytic --binning $b --n-centrals $NC --n-mocks 1 --mock-start $m"
	done; } | run_stage
done
if [ -f rerun/PAUSE ]; then echo "phase B paused"; else echo "phase B done"; fi
