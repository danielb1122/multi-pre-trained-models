# Figure map and provenance

## Scope

Mapping checked against the supplied 56-page NeurIPS 2026 manuscript, including
the main figures and appendices O, P, and T. The original directory was a working
research folder, not a versioned publication release. The mapping below identifies
the implementing code and the settings stated in captions, not an authenticated
record of which exact command produced each embedded image. No original result
arrays or figure images are distributed here.

Retained source files: `empirical_exp2_new.py`, `numeric.py`, `combined_plot.py`,
`decomposition_plot.py`, and `de_be.py`. Names are retained for traceability.
`paper_utils.py` and `tests/` are new portability/verification helpers.

The CPU templates in `slurm/` replace machine-specific launchers:
`foab_sbatch` launched covariance-factor runs; `sbatch_plot` actually launched
heterogeneous-source experiments; `new_sbatch` launched theory; `plot_sbatch`
and `plot_sbatch_de_be` launched plotting. These launchers had hard-coded paths,
an environment name, personal email, and exploratory settings. Their old resource
requests and exploratory grids are not presented as publication settings.

Excluded: CIFAR/UTKFace/ResNet transfer experiments, `real_dataset/`, old duplicate
engines, empty scripts, LMMSE exploration, heatmap exploration (not in this PDF),
spreadsheet export utilities, historical result-directory batch lists, and the
reference-code folder `code_from_yehudas_papper/`.

## Settings notation

Unless specified otherwise, use `--d 128`, `--b 1`, `--sigma_epsilon 0.31623`,
`--cov_x_des identity`, `--cov_z_q 0`, and model counts `1 2 3 5 10 20`.
The noise column below is the **variance** v = sigma_eta^2 = sigma_xi^2;
the CLI takes standard deviations: 0.05 -> 0.223607; 0.1 -> 0.31623;
0.25 -> 0.5; 0.5 -> 0.70711. These rounded values follow the recovered launchers.
`--n` is the target sample size, not the number of source models.

| Symbol | CLI |
| --- | --- |
| Identity relation | `--relation_description identity` |
| Projection onto r dimensions | `--relation_description subspace --r r` |
| Energy-preserving projection | `--relation_description eng_subspace --r r` |
| Ill-conditioned circulant | `--relation_description circulant --kappa 1000` |
| Assumed relation I (default) | `--assumption False` |
| Assumed relation equals true H | `--assumption True` |
| Scale assumed relation by min(1, n_tilde/d) | `--debias True` |
| Target exponential covariance, q=0.5 | `--cov_x_des exp_decay` |

## Main figures

| Panel | n | True relation | v | Additional settings / plot |
| --- | ---: | --- | ---: | --- |
| 1a | 112 | Energy-preserving, r=64 | 0.5 | Default assumed I; `combined_plot.py` |
| 1b | 32 | Energy-preserving, r=64 | 0.5 | Target exponential covariance; `combined_plot.py` |
| 2a | 32 | Identity | 0.5 | `combined_plot.py` |
| 2b | 96 | Identity | 0.1 | `combined_plot.py` |
| 3a | 32 | Identity | 0.1 | `--debias True`; `combined_plot.py` |
| 3b | 32 | **Ordinary subspace**, r=96 | 0.5 | `--debias True`; `combined_plot.py` |
| 4a/b | 32* | Circulant | 0.1 | `--decomposition True`, debias False/True; `decomposition_plot.py` |
| 5a/b | 32 | Identity | 0.05 / 0.5 | Run both debias states; `de_be.py` subtracts debiased minus naive |
| 6a/b | 32* | Energy-preserving, r=96 | 0.5* | `--cv True --debias_factor_val True --cov_z_q 0.5 --cov_x_des exp_decay --number_of_runs 100`; `combined_plot.py --emp_only True --factor True` |

`*` Values supplied by the recovered code/launcher or surrounding setup, rather
than explicitly restated in that figure's caption. In particular, the caption of
Figure 3b says ordinary subspace, not energy-preserving subspace.

## Appendix figures

| Figure | Configuration relative to the main table |
| --- | --- |
| O.1a/b | Identity, v=0.5, n=32; decomposition without/with debiasing |
| O.2a/b | Identity, v=0.5 / 0.1, n=32; debiased-minus-naive error |
| O.2c | Circulant, v=0.5, n=32, `--assumption True`; debiased-minus-naive |
| O.2d | Circulant, v=0.1, n=32, assumed I; debiased-minus-naive |
| O.3a-d | Circulant, v=0.1, n=32; (assumption, debias) = (False,False), (True,False), (False,True), (True,True) |
| P.1a-d | Same four O.3 configurations, `--decomposition True` |
| P.2a/b | O.3 matched-H error minus assumed-I error, without/with debiasing; `de_be.py` |
| T.1a | Energy-preserving r=96, n=64, v=0.25, assumed I |
| T.1b | Subspace r=96, n=32, v=0.1, assumed I |
| T.1c | Circulant, n=32, v=0.1, `--assumption True` |
| T.1d | Energy-preserving r=64, n=32, v=0.5, assumed I |
| T.2a-d | Identity; (n,v) = (96,0.1), (64,0.25), (32,0.5), (32,0.1) |
| T.3a/b | Subspace r=96; (n,v) = (64,0.5), (32,0.1); debiasing ambiguity below |
| T.3c | Energy-preserving r=64, n=64, v=0.1, target exponential covariance; same ambiguity |
| T.3d | Energy-preserving r=64, n=32, v=0.5; same ambiguity |
| T.4a-d | n=32; identity v=0.25, subspace r=64 v=0.1, energy-preserving r=64 v=0.5, matched circulant v=0.5; debiased-minus-naive |
| T.5 | Figure 6 family with source q = 0.5, 0.7, 0.9, 0.95, 0.98; both tuned (`--debias_factor_val True`) and naive (`False`) with `--cv True`; target q stays 0.5 |
| T.6 | `--bad_models True --number_of_models 20 --relation_description subspace --n 32`, v=0.5; debias False/True |
| T.7 | Same as T.6 with `--relation_description eng_subspace` |

T.6/T.7 use k = 0,2,4,6,8,10 weak sources with rank 32 and 20-k strong sources
with rank 96, fixed in the recovered engine. Plot their saved folders with
`combined_plot.py --emp_only True`. The paper's multi-panel layout is assembled
in the manuscript; these scripts produce constituent plots, not the LaTeX layout.

## Reproducibility limits to resolve before a publication rerun

1. Appendix H specifies 25 nonuniform source ratios, 2,000 repetitions, and 30
   alpha values in [1e-4,1e2]. The current empirical defaults are 36 requested
   ratios (35 evenly spaced plus 1.02), 200 repetitions, and 25 alpha values.
   Use `--number_of_runs 2000 --alpha-points 30`; the **exact original 25-point
   grid was not recovered**. An explicit grid can be passed with `--source-ratios`.
2. The decomposition branch defaults to 25 ratios, 40 outer draws and 15 inner
   draws per mean/covariance estimate, not the paper's 50/150/50. These counts
   are exposed as `--decomposition-points`, `--decomposition-main-runs`, and
   `--decomposition-sub-runs`. More importantly, this branch selects alpha by
   true-parameter error, whereas Appendix H describes validation tuning. Its
   two inner stages use different alpha grids (20 and 25 points). This scientific
   discrepancy is **preserved and disclosed**, not silently redefined.
3. Ordinary empirical runs use a shared independently generated 1,000-row
   validation feature matrix; labels and test sets are generated per run.
   The combined plotter's ridge reference selects alpha on simulated test data
   (an oracle reference), not 5-fold CV as stated for Figure 6/T.5 baselines.
4. Factor-tuning CV uses 10 alpha values and approximately 10 factor candidates
   including 1, with shuffled 5-fold splits (`random_state=42`). Source repetition
   seeds start from 12345, but several relation/decomposition draws use unseeded
   NumPy state. Historical seeds and a full original dependency lock are absent;
   bitwise reproduction is not promised.
5. T.3 is presented as debiasing but its caption says assumed I rather than rho I.
   Confirm the intended flag before rerunning it. O.3/P.1 also refer to T.7 for
   error differences; in the supplied PDF that comparison is actually P.2.

## Curation changes

Kept the experiment estimators and theoretical expressions. Removed commented-out
drafts, an unreachable older decomposition block, and a standalone sampling demo.
Made outputs portable (`--output-dir`), recorded CLI arguments, bounded workers
by the requested allocation, and fixed false/true parsing in theory/plot scripts.
Exposed small-run controls without changing their recovered defaults. Corrected
a double division in the saved mean-alpha diagnostic (not test error) and passed
the experiment's assumed matrices explicitly into worker-side debiasing.

The difference/decomposition plotters were cleaned up to accept input folders;
differences now verify model counts and x-grids instead of silently zipping files.
The difference plotter again supports theoretical overlays, which were commented
out in the recovered version. The combined plotter now honors target n/d/noise,
handles paths without trailing slashes, and does not interpret a two-row
heterogeneous-source error file as an alpha curve. Oracle baselines are labeled
accordingly. No full-scale scientific rerun was performed during curation.
