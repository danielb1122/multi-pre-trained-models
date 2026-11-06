# Slurm

Submit from the repository root with the Python environment active.
Add partition, time, and memory options for your cluster.

```bash
mkdir -p .slurm/logs
export PYTHON_BIN="$(command -v python)"
sbatch --cpus-per-task=4 --array=0-0 slurm/empirical.sbatch \
  --number_of_runs 2000 --alpha-points 30 --free_text example
```
