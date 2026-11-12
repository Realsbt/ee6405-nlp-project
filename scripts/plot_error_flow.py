"""Figure for the error-flow and cue-dependence analysis.

Three panels:
  1. error flow between the four classes, with ai_generated as the hub
  2. cue separability vs error rate, per confused pair
  3. confusion rate when the discriminating cue is present vs absent
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/reports/comparison/error_flow.png'

SHORT = ['HW', 'POL', 'GEN', 'HUM']


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    flow = json.loads((ROOT / '.local/error_flow.json').read_text(encoding='utf-8'))
    dep = json.loads((ROOT / '.local/cue_dependence.json').read_text(encoding='utf-8'))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.05]})
    fig.suptitle('Where the 158 single-document errors go, and which cue causes them',
                 fontsize=13, y=0.975)

    # ---- panel 1: error flow ---------------------------------------------
    ax = axes[0]
    cm = np.asarray(flow['confusion']['matrix'])
    err = cm.copy()
    np.fill_diagonal(err, 0)
    im = ax.imshow(err, cmap='OrRd', vmin=0)
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        fontsize=9, color='#90a4ae')
            elif err[i, j]:
                ax.text(j, i, f'{err[i, j]}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if err[i, j] > 25 else '#3e2723')
    ax.set_xticks(range(4), SHORT)
    ax.set_yticks(range(4), SHORT)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f'Error flow · {int(err.sum())} errors\n'
                 f'GEN is involved in {100*dep["gen_hub"]["share"]:.1f}%', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- panel 2: cue separability vs error rate -------------------------
    ax = axes[1]
    loc = flow['cue_localisation']
    pairs = [('HW|POL', 'HW|POL'), ('POL|GEN', 'POL|GEN'), ('GEN|HUM', 'GEN|HUM')]
    cue_for = {'HW|POL': 'mean_word_len', 'POL|GEN': 'length', 'GEN|HUM': 'comma_per_100w'}
    xs, ys, labels = [], [], []
    for key, label in pairs:
        if key not in loc:
            continue
        xs.append(100 * loc[key][cue_for[key]]['consistency'])
        ys.append(100 * dep[key]['cue_absent']['confusion'])
        labels.append(label)
    ax.scatter(xs, ys, s=260, color='#1565c0', zorder=3, edgecolor='white', linewidth=1.4)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), xytext=(10, 8), textcoords='offset points', fontsize=10)
    ax.set_xlabel('cue separates the pair in-group (% of groups)')
    ax.set_ylabel('confusion when the cue is absent (%)')
    ax.set_title('The less separable a pair is,\nthe more the model fails', fontsize=10)
    ax.grid(alpha=.2)
    ax.set_xlim(60, 105)
    ax.set_ylim(0, 52)

    # ---- panel 3: present vs absent --------------------------------------
    ax = axes[2]
    keys = ['HW|POL', 'POL|GEN', 'GEN|HUM']
    x = np.arange(len(keys))
    present = [100 * dep[k]['cue_present']['confusion'] for k in keys]
    absent = [100 * dep[k]['cue_absent']['confusion'] for k in keys]
    ax.bar(x - 0.19, present, width=0.36, color='#2e7d32', label='cue present')
    ax.bar(x + 0.19, absent, width=0.36, color='#b71c1c', label='cue absent')
    for xi, (p, a) in enumerate(zip(present, absent)):
        ax.text(xi - 0.19, p + 0.9, f'{p:.1f}', ha='center', fontsize=9)
        ax.text(xi + 0.19, a + 0.9, f'{a:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(xi, max(p, a) + 4.2, f'+{a-p:.1f} pp', ha='center', fontsize=8.5,
                color='#b71c1c')
    ax.set_xticks(x, keys)
    ax.set_ylim(0, 52)
    ax.set_ylabel('group confusion rate (%)')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Removing the cue multiplies the error rate\n'
                 '(GEN|HUM: 12.6% -> 44.1%)', fontsize=10)
    ax.grid(axis='y', alpha=.2)

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
