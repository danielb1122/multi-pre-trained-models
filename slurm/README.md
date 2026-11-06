# Optional Slurm templates

These templates are conveniences for readers with access to a Slurm cluster.
They are **not dependencies** of the experiments and are not specific to BGU.
For local execution, use the Python commands in the [main README](../README.md).

| Template | Runs |
| --- | --- |
| `empirical.sbatch` | `empirical_exp2_new.py`, with one source-model count per array task |
| `numeric.sbatch` | `numeric.py`, with the arguments supplied by the caller |
| `plot.sbatch` | A plotter filename followed by its arguments |

Submit from the repository root after activating your Python environment.
Set `PYTHON_BIN` to that environment's absolute Python path and create
`.slurm/logs` before submission. Partition, account/QoS, time, memory, and CPU
requests must match your own cluster's policy and the workload. No GPU is needed.
Run a small pilot before submitting the full experiment; do not compute on a
cluster login node.

```bash
mkdir -p .slurm/logs
export PYTHON_BIN="$(command -v python)"
# Define PARTITION, TIME_LIMIT and MEMORY for your CPU allocation first.
# Add --account/--qos if required by your cluster.
sbatch --partition="$PARTITION" --time="$TIME_LIMIT" --mem="$MEMORY" \
  --cpus-per-task=1 --array=0-0%1 slurm/empirical.sbatch \
  --number_of_runs 2 --source-ratios 0.5 2 --free_text pilot
```

The default array is a single-task pilot. For source-model counts
`1 2 3 5 10 20`, use `--array=0-5%1`. Increase concurrency only after checking
resource requirements. For T.6/T.7, export `MODEL_COUNTS=20`, retain
`--array=0-0%1`, and pass `--bad_models True` plus the relevant experiment settings.

The empirical worker count follows `SLURM_CPUS_PER_TASK`. The templates limit
BLAS threads to one per worker to avoid CPU oversubscription. Generated outputs
and scheduler logs are Git-ignored; scientific settings remain Python arguments.
