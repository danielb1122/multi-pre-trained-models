"""Difference curves, adapted from the original debiased-minus-naive plotter.

For Figure P.2, pass the matched-relation results first and identity results
second. The subtraction is always first minus second; no interpolation is done.
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_utils import add_output_argument, prepare_output


def load_curves(folder):
    curves = {}
    for path in sorted(Path(folder).glob("*.npy")):
        if path.name.startswith(("l2", "ridge")):
            continue
        match = re.search(r"num_models[=_](\d+)", path.name)
        if not match:
            continue
        m = int(match.group(1))
        if m in curves:
            raise ValueError(f"Multiple curves for m={m} in {folder}; select one experiment")
        data = np.load(path, allow_pickle=False)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError(f"Expected rows [x, error, ...]: {path}")
        curves[m] = data[:2]
    if not curves:
        raise ValueError(f"No model curves found in {folder}")
    return curves


def difference_curves(first, second):
    if first.keys() != second.keys():
        raise ValueError('Compared folders must have the same model counts')
    differences = {}
    for m, a in first.items():
        b = second[m]
        if a.shape != b.shape or not np.allclose(a[0], b[0], rtol=0, atol=1e-10):
            raise ValueError(f"Source grids differ for m={m}; do not subtract unmatched runs")
        differences[m] = np.stack((a[0], a[1] - b[1]))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--empirical_directory_de', '--first', dest='first', required=True)
    parser.add_argument('--empirical_directory_be', '--second', dest='second', required=True)
    parser.add_argument('--numeric-first')
    parser.add_argument('--numeric-second')
    parser.add_argument('--name', default='error_difference')
    add_output_argument(parser)
    args = parser.parse_args()
    if bool(args.numeric_first) != bool(args.numeric_second):
        parser.error('provide both numeric folders or neither')
    root = prepare_output(args)
    empirical = difference_curves(load_curves(args.first), load_curves(args.second))
    theory = (difference_curves(load_curves(args.numeric_first), load_curves(args.numeric_second))
              if args.numeric_first else {})
    if theory and theory.keys() != empirical.keys():
        parser.error('theory and empirical model counts differ')
    models = sorted(empirical, reverse=True)
    colors = dict(zip(models, plt.cm.magma(np.linspace(0, 0.8, len(models)))))
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in sorted(empirical):
        x, y = empirical[m]
        ax.plot(x, y, 'o', color=colors[m], markersize=4, label=f'{m} models')
        if m in theory:
            ax.plot(*theory[m], color=colors[m], linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set(xlabel=r'Source parameterization ($d/\tilde{n}$)',
           ylabel='Expected test error difference (first minus second)', xlim=(1.05, 5.1))
    ax.grid(alpha=0.3)
    ax.legend()
    path = root / 'graphs' / f'{args.name}.png'
    fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
