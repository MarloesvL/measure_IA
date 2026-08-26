#!/bin/bash
# Cluster batch template for the measureia benchmark sweep.
#
# Deliberately scheduler-agnostic and deliberately incomplete: fill in the
# directives for your scheduler and the two scratch paths in machines.py
# before submitting. The paths are not guessed anywhere in this harness,
# because the difference between shared NFS and node-local disk is one of the
# things being measured.
#
# SLURM example directives (adjust or replace for PBS/LSF):
#   #SBATCH --job-name=measureia-bench
#   #SBATCH --nodes=1
#   #SBATCH --exclusive          # the machine must be quiet: this is a timing run
#   #SBATCH --cpus-per-task=32
#   #SBATCH --time=08:00:00
#   #SBATCH --output=bench-%j.log
#
# Submit one job per scratch filesystem so the two never mix:
#   sbatch benchmarks/job_template.sh nfs
#   sbatch benchmarks/job_template.sh local

set -euo pipefail

SCRATCH_NAME="${1:?usage: $0 <scratch-name from machines.py, e.g. nfs or local>}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

cd "$REPO"

# Threads are pinned per point by run_sweep.py; keep the launcher itself quiet
# so nothing inherits a stray thread count.
export OMP_NUM_THREADS=1

"$PYTHON" benchmarks/run_sweep.py \
    --machine cluster \
    --scratch "$SCRATCH_NAME" \
    --sweeps size threads internal \
    --timeout 7200

"$PYTHON" benchmarks/plot_results.py \
    "benchmarks/results/cluster_${SCRATCH_NAME}.jsonl"
