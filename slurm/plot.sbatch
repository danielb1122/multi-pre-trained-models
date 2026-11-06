#!/usr/bin/env bash
# Plotting can simulate CPU baselines; do not run it on a cluster login node.
#SBATCH --job-name=synthetic-plot
#SBATCH --ntasks=1
#SBATCH --output=.slurm/logs/plot-%j.out
#SBATCH --error=.slurm/logs/plot-%j.err

set -euo pipefail
: "${PYTHON_BIN:?Set PYTHON_BIN to the absolute Python path in your environment}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
plotter=${1:?First argument must be a plotter filename}
shift
case "$plotter" in
    combined_plot.py|decomposition_plot.py|de_be.py) ;;
    *) echo "Unknown paper plotter: $plotter" >&2; exit 2 ;;
esac
[[ -x "$PYTHON_BIN" && -f "$plotter" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
exec "$PYTHON_BIN" -u "$plotter" "$@"
