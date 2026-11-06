# Transfer Learning with Multiple Pretrained Models

Synthetic linear-regression experiments for **Transfer Learning of Linear
Regression with Multiple Pretrained Models: Benefiting from More Pretrained
Models via Overparameterization Debiasing** (NeurIPS 2026 manuscript).

This is a curated source-only copy, not an archive of published results.
It contains no neural-network training, real datasets, generated figures,
spreadsheets, or saved arrays. All experiments use CPU-based synthetic data.

## Code and figures

| Script | Role | Paper figures |
| --- | --- | --- |
| `empirical_exp2_new.py` | Monte Carlo transfer, debiasing, covariance tuning, bias/variance, heterogeneous sources | All empirical curves |
| `numeric.py` | Numerical evaluation of theoretical error expressions | 1-3, 5, O.2-O.3, P.2, T.1-T.4 |
| `combined_plot.py` | Theory/empirical overlays, factor curves, heterogeneous-source plots | 1-3, 6, O.3, T.1-T.3, T.5-T.7 |
| `decomposition_plot.py` | Squared bias, variance, and total error | 4, O.1, P.1 |
| `de_be.py` | Difference between two matched experiment folders | 5, O.2, P.2, T.4 |
| `slurm/*.sbatch` | Portable CPU templates for experiments and plotting | All families |

Panel settings, provenance, and **known paper/code discrepancies** are in
[FIGURES.md](FIGURES.md). Read that file before claiming exact reproduction.

## Setup and a small check

Use Python 3.10 or later. Requirements pin the environment used for local
verification, not the unavailable original publication environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python empirical_exp2_new.py --d 8 --n 4 --r 4 --number_of_models 1 \
  --number_of_runs 2 --workers 1 --source-ratios 0.5 2 --alpha-points 3 \
  --free_text smoke --output-dir artifacts/smoke
```

## Experiments and plotting

Example configuration for the identity-relation experiment in Figure 2b
(repeat for `m = 1, 2, 3, 5, 10, 20`). This is a figure-family recipe, not a
verified regeneration of the exact original sample grid.

```bash
python empirical_exp2_new.py --d 128 --n 96 --number_of_models 1 \
  --relation_description identity --sigma_eta 0.31623 --sigma_xi 0.31623 \
  --number_of_runs 2000 --alpha-points 30 --free_text fig2b
python numeric.py --d 128 --n 96 --number_of_models 1 2 3 5 10 20 \
  --relation_description identity --sigma_eta 0.31623 --sigma_xi 0.31623 \
  --free_text fig2b
```

Outputs go under `artifacts/` by default: `results/`, `numeric_result_icecream/`,
`decomposition_2/`, and `graphs/`. Experiment arguments are recorded in `configs/`.
Use matching result subdirectories below; these variables are placeholders.

```bash
python combined_plot.py --empirical "$EMPIRICAL_DIR" --numeric "$THEORY_DIR" --n 96
python combined_plot.py --empirical "$FACTOR_DIR" --emp_only True --factor True --n 32
python decomposition_plot.py --input "$DECOMPOSITION_DIR"
python de_be.py --first "$DEBIASED_DIR" --second "$NAIVE_DIR" \
  --numeric-first "$DEBIASED_THEORY_DIR" --numeric-second "$NAIVE_THEORY_DIR"
```

The recovered combined plotter also simulates **oracle-tuned** ridge and
minimum-norm reference lines. It is not a data-only plotting operation.
Pass the correct `--n`, `--d`, `--epsilon`, and `--cov_x_des`.

## BGU Slurm

Submit from this repository's root after activating a cluster environment and
creating `.slurm/logs`. Set `PYTHON_BIN` to that environment's absolute Python
path. The templates deliberately leave partition/account, time, memory, and CPU
requests to the caller; choose them after a pilot under current cluster policy.
No GPU is needed. Do not run the experiment or plotter on a login node.

```bash
mkdir -p .slurm/logs
export PYTHON_BIN="$(command -v python)"
# Define PARTITION, TIME_LIMIT and MEMORY for an approved CPU allocation first.
sbatch --partition="$PARTITION" --time="$TIME_LIMIT" --mem="$MEMORY" \
  --cpus-per-task=1 --array=0-0%1 slurm/empirical.sbatch \
  --number_of_runs 2 --source-ratios 0.5 2 --free_text pilot
```

For all six model counts, use `--array=0-5%1` (raise concurrency only after a
pilot). Pass the same experiment arguments as for Python. For T.6/T.7 set
`MODEL_COUNTS=20`, keep `--array=0-0%1`, and add `--bad_models True`.
`slurm/numeric.sbatch` takes numeric arguments; `slurm/plot.sbatch` takes a
plotter filename followed by its arguments. Generated files are Git-ignored.
