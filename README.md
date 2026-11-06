# Transfer Learning with Multiple Pretrained Models

Synthetic linear-regression experiments for **Transfer Learning of Linear
Regression with Multiple Pretrained Models: Benefiting from More Pretrained
Models via Overparameterization Debiasing** (NeurIPS 2026 manuscript).

This is a curated source-only copy, not an archive of published results.
It contains no neural-network training, real datasets, generated figures,
spreadsheets, or saved arrays. All experiments use CPU-based synthetic data.

**No BGU account, VPN, Slurm installation, GPU, or external dataset is required.**
Run the Python scripts on your own computer, a workstation, or a cloud CPU machine.
The optional cluster templates are separate from the experiment code.

## Code and figures

| Script | Role | Paper figures |
| --- | --- | --- |
| `empirical_exp2_new.py` | Monte Carlo transfer, debiasing, covariance tuning, bias/variance, heterogeneous sources | All empirical curves |
| `numeric.py` | Numerical evaluation of theoretical error expressions | 1-3, 5, O.2-O.3, P.2, T.1-T.4 |
| `combined_plot.py` | Theory/empirical overlays, factor curves, heterogeneous-source plots | 1-3, 6, O.3, T.1-T.3, T.5-T.7 |
| `decomposition_plot.py` | Squared bias, variance, and total error | 4, O.1, P.1 |
| `de_be.py` | Difference between two matched experiment folders | 5, O.2, P.2, T.4 |

Panel settings, provenance, and **known paper/code discrepancies** are in
[FIGURES.md](FIGURES.md). Read that file before claiming exact reproduction.

## Setup and a small check

Tested with Python 3.10. Requirements pin the environment used for local
verification, not the unavailable original publication environment. First clone
the repository and enter its folder:

```bash
git clone https://github.com/danielb1122/multi-pre-trained-models.git
cd multi-pre-trained-models
python3 -m venv .venv
source .venv/bin/activate
```

The environment commands above are for macOS/Linux. On Windows Command Prompt,
use `py -3.10 -m venv .venv` and `.venv\Scripts\activate.bat` instead.
After activating the environment, the following commands work without a cluster:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python empirical_exp2_new.py --d 8 --n 4 --r 4 --number_of_models 1 --number_of_runs 2 --workers 1 --source-ratios 0.5 2 --alpha-points 3 --free_text smoke --output-dir artifacts/smoke
```

This small example checks the pipeline; it does not reproduce a paper figure.
Full experiments are more computationally expensive. Use `--workers` to control
the number of CPU processes; the default outside a scheduler is one.

## Experiments and plotting

Example configuration for the identity-relation experiment in Figure 2b
(repeat for `m = 1, 2, 3, 5, 10, 20`). This is a figure-family recipe, not a
verified regeneration of the exact original sample grid.
Examples below use Bash line continuations; in other shells, put each command on
one line. The commands themselves do not depend on Bash or Slurm.

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

## Optional cluster execution

For readers who already use a Slurm-managed cluster, [slurm/README.md](slurm/README.md)
describes optional CPU submission templates. They only schedule the same Python
commands shown above, do not require BGU, and can be ignored for local execution.
