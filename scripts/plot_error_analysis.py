"""Plot where the strongest model fails.

Four panels, all from .local/error_analysis.json (E30 on the published test set):
  1. confusion matrix with counts and row-normalised shading
  2. accuracy by domain x class, the slice that shows the real weakness
  3. accuracy by character length, showing the short-document penalty
  4. the ai_humanized class split by who did the humanising
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reports/comparison'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SHORT = ['HW', 'POL', 'GEN', 'HUM']


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads((ROOT / '.local/error_analysis.json').read_text(encoding='utf-8'))
    confusion = np.asarray(data['confusion'], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    # ---- 1. confusion matrix ---------------------------------------------
    ax = axes[0, 0]
    row = confusion / confusion.sum(axis=1, keepdims=True)
    im = ax.imshow(row, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{int(confusion[i, j])}\n{100*row[i, j]:.1f}%',
                    ha='center', va='center', fontsize=9,
                    color='white' if row[i, j] > 0.55 else '#222')
    ax.set_xticks(range(4), SHORT)
    ax.set_yticks(range(4), SHORT)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f'Confusion matrix · E30 · {100*data["accuracy"]:.2f}% accuracy\n'
                 f'{data["items"]} items, {int(confusion.sum() - np.trace(confusion))} errors', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='row share')

    # ---- 2. domain x class ------------------------------------------------
    ax = axes[0, 1]
    domains = data['domains']
    width = 0.2
    x = np.arange(len(domains))
    for c in range(4):
        vals, ns = [], []
        for d in domains:
            cell = data['domain_class'][d][LABELS[c]]
            vals.append(100 * cell['correct'] / cell['n'] if cell['n'] else np.nan)
            ns.append(cell['n'])
        ax.bar(x + (c - 1.5) * width, vals, width, label=f'{SHORT[c]} (n≈{int(np.mean(ns))})')
    ax.set_xticks(x, domains)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.axhline(100 * data['accuracy'], color='#444', linestyle='--', linewidth=1,
               label=f'overall {100*data["accuracy"]:.1f}%')
    ax.set_title('Accuracy by domain × class\nai_humanized is the weak class in every domain', fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc='lower left')
    ax.grid(axis='y', alpha=.15)

    # ---- 3. length --------------------------------------------------------
    ax = axes[1, 0]
    order = ['<1k', '1-2k', '2-3k', '3-5k', '>5k']
    accs, ns = [], []
    for b in order:
        cell = data['length_buckets'][b]
        accs.append(100 * cell['correct'] / cell['n'] if cell['n'] else np.nan)
        ns.append(cell['n'])
    bars = ax.bar(order, accs, color='#4f7f91')
    for bar, a, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.6, f'{a:.1f}%\nn={n}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(80, 101)
    ax.set_xlabel('document length (characters)')
    ax.set_title('Accuracy by document length\nshort documents are the hardest', fontsize=10)
    ax.grid(axis='y', alpha=.15)

    # ---- 4. humanizer split ----------------------------------------------
    ax = axes[1, 1]
    hz = data['humanizer']
    names = ['human', 'tool']
    accs = [100 * hz[k]['correct'] / hz[k]['n'] for k in names]
    ns = [hz[k]['n'] for k in names]
    bars = ax.bar(names, accs, color=['#b5651d', '#2e7d32'])
    for bar, a, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 1.5, f'{a:.1f}%\nn={n}',
                ha='center', va='bottom', fontsize=9)
    ax.axhline(100 * data['accuracy'], color='#444', linestyle='--', linewidth=1,
               label=f'overall {100*data["accuracy"]:.1f}%')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_xlabel('who produced the ai_humanized text')
    ax.set_title('ai_humanized split by humaniser\n"human" rewrites are near chance', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.15)

    fig.suptitle('Where the strongest model fails · E30 on the published 3,194-item test set', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'error_analysis.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT / "error_analysis.png"}')


if __name__ == '__main__':
    main()
