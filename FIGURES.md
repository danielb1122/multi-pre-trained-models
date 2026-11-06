# Plotting

| Script | Output |
| --- | --- |
| `combined_plot.py` | Experiment and theory curves |
| `decomposition_plot.py` | Bias, variance, and total error |
| `de_be.py` | Differences between matched experiments |

Set `EMPIRICAL_DIR` and `THEORY_DIR` to the result folders from the README example:

```bash
python combined_plot.py --empirical "$EMPIRICAL_DIR" --numeric "$THEORY_DIR" --n 96
```

Plots are saved under `artifacts/graphs/`. Match `--n`, `--d`, `--epsilon`,
and `--cov_x_des` to the experiment.

## Protocol notes

Script defaults differ from the manuscript settings; the original source-ratio
grid is not included. Decomposition tuning uses true-parameter error, and the
ridge reference uses oracle test-set tuning rather than validation tuning.
Some random draws are unseeded; exact numerical reproduction is not guaranteed.
