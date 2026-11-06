"""Plot saved bias/variance/total rows from empirical_exp2_new.py.

Preserves the original plot's line semantics, without machine-specific paths.
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_utils import add_output_argument, prepare_output


def plot_folder(folder, output_root):
    files = sorted(Path(folder).glob('*.npy'))
    if not files:
        raise ValueError(f'No decomposition arrays in {folder}')
    curves = {}
    for path in files:
        match = re.search(r'num_models=(\d+)', path.name)
        if not match:
            raise ValueError(f'Missing model count in {path.name}')
        m = int(match.group(1))
        if m in curves:
            raise ValueError(f'Multiple files for m={m}; select one experiment folder')
        data = np.load(path, allow_pickle=False)
        if data.ndim != 2 or data.shape[0] != 4:
            raise ValueError(f'Expected [x, bias2, variance, total] rows: {path}')
        curves[m] = data
    models = sorted(curves, reverse=True)
    colors = dict(zip(models, plt.cm.magma(np.linspace(0, 0.8, len(models)))))
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in sorted(curves):
        x, bias2, variance, total = curves[m]
        ax.plot(x, total, color=colors[m], alpha=0.4, linewidth=1)
        ax.plot(x, variance, color=colors[m], linestyle=':', linewidth=2.5)
        ax.plot(x, bias2, color=colors[m], linestyle='--', linewidth=2.5, label=f'm = {m}')
    model_legend = ax.legend(loc='upper left', title='Models')
    ax.add_artist(model_legend)
    ax.legend(handles=[Line2D([], [], color='black', linestyle=s, label=l)
                       for s, l in [('-', 'Total error'), (':', 'Variance'), ('--', 'Squared bias')]],
              loc='upper right')
    ax.set(xlabel=r'Source parameterization ($d/\tilde{n}$)',
           ylabel='Error (target task)', xlim=(0, 5.1), ylim=(0, 1.4))
    ax.grid(alpha=0.3)
    path = output_root / 'graphs' / f'{Path(folder).name}.png'
    fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='One experiment folder, or a parent containing experiment folders')
    add_output_argument(parser)
    args = parser.parse_args()
    root = prepare_output(args)
    if list(args.input.glob('*.npy')):
        folders = [args.input]
    else:
        folders = [p for p in sorted(args.input.iterdir()) if p.is_dir() and list(p.glob('*.npy'))]
    if not folders:
        parser.error('no experiment arrays found')
    for folder in folders:
        plot_folder(folder, root)


if __name__ == '__main__':
    main()
