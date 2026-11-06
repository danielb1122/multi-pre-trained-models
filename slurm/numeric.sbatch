#!/usr/bin/env bash
# CPU-only template; provide site/resource options to sbatch, not in this file.
#SBATCH --job-name=synthetic-theory
#SBATCH --ntasks=1
#SBATCH --output=.slurm/logs/numeric-%j.out
#SBATCH --error=.slurm/logs/numeric-%j.err

set -euo pipefail
: "${PYTHON_BIN:?Set PYTHON_BIN to the absolute Python path in your environment}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -x "$PYTHON_BIN" && -f numeric.py ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
git rev-parse HEAD 2>/dev/null || true
exec "$PYTHON_BIN" -u numeric.py "$@"
