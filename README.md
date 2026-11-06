# Transfer Learning with Multiple Pretrained Models

Research code accompanying **Transfer Learning of Linear
Regression with Multiple Pretrained Models: Benefiting from More Pretrained
Models via Overparameterization Debiasing** (NeurIPS 2026 manuscript).

## Setup

Python 3.10, CPU.

```bash
git clone https://github.com/danielb1122/multi-pre-trained-models.git
cd multi-pre-trained-models
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.10` instead of `python3` and activate with
`.venv\Scripts\activate.bat`.

## Example

Run an identity-relation experiment and its theoretical counterpart:

```bash
python empirical_exp2_new.py --d 128 --n 96 --number_of_models 1 \
  --relation_description identity --sigma_eta 0.31623 --sigma_xi 0.31623 \
  --number_of_runs 2000 --alpha-points 30 --free_text example
python numeric.py --d 128 --n 96 --number_of_models 1 \
  --relation_description identity --sigma_eta 0.31623 --sigma_xi 0.31623 \
  --free_text example
```

Outputs are saved under `artifacts/`. Use `--workers` to set CPU parallelism
and `--help` for arguments. See [FIGURES.md](FIGURES.md) for plotting.

Tests: `python -m unittest discover -s tests -v`.
