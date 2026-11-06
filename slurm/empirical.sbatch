#!/usr/bin/env bash
# CPU-only template; select partition, time, memory, CPUs and account at submission.
#SBATCH --job-name=transfer-experiments
#SBATCH --ntasks=1
#SBATCH --array=0-0%1
#SBATCH --output=.slurm/logs/empirical-%A_%a.out
#SBATCH --error=.slurm/logs/empirical-%A_%a.err

set -euo pipefail
: "${PYTHON_BIN:?Set PYTHON_BIN to the absolute Python path in your environment}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -x "$PYTHON_BIN" && -f empirical_exp2_new.py ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
read -r -a models <<< "${MODEL_COUNTS:-1 2 3 5 10 20}"
index=${SLURM_ARRAY_TASK_ID:?This template expects a Slurm array}
(( index < ${#models[@]} )) || { echo "Array index outside MODEL_COUNTS" >&2; exit 2; }
printf 'Job %s; model count %s; Python %s\n' "$SLURM_JOB_ID" "${models[$index]}" "$PYTHON_BIN"
git rev-parse HEAD 2>/dev/null || true
exec "$PYTHON_BIN" -u empirical_exp2_new.py \
    --number_of_models "${models[$index]}" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
