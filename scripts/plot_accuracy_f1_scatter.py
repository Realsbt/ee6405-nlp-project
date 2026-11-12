"""Updated accuracy vs macro-F1 scatter for every competitive model.

Supersedes the scatter panel inside `info_vs_top3.png`, which predates the
stacking and constrained-decoding work. This figure puts all five families on
one axis pair so the progression is visible:

  published five  ->  composited Top 3  ->  info methods
                  ->  13-model stack    ->  group-constrained decoding

Every point is scored on the same 3,194-item published test set, computed on
this machine. Weak baselines (TF-IDF + LR, M1, M2, M3) are excluded from the
scatter because they sit 15-50 pp away and would flatten the scale; they remain
in `info_vs_top3.png`'s bar chart.

Writes `experiments/reports/comparison/accuracy_f1_scatter.png`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/reports/comparison/accuracy_f1_scatter.png'

# Families are drawn in this order; later families sit on top.
FAMILIES = [
    ('Published five (local re-inference)', '#4f7f91'),
    ('Composited Top 3', '#7b4fa0'),
    ('13-model stack', '#00838f'),
    ('Group-constrained decoding', '#2e7d32'),
]

PUBLISHED = {
    'TF-IDF + LR': ['tfidf_lr_v1'],
    'RoBERTa · CE': ['roberta_s3407', 'roberta_s42', 'roberta_s2026'],
    'RoBERTa · CE+SupCon': ['roberta_supcon_s3407', 'roberta_supcon_s42', 'roberta_supcon_s2026'],
    'RACE · CE': ['race_no_supcon_s3407', 'race_no_supcon_s42', 'race_no_supcon_s2026'],
    'RACE · CE+SupCon': ['race_s3407', 'race_s42', 'race_s2026'],
}

TOP3 = {'E30 equal-mean fusion': 'E30',
        'E26 head/middle/tail': 'E26',
        'E01 B0 single window': 'E01'}

# Short labels keep the annotation column readable.
SHORT = {
    'TF-IDF + LR': 'TF-IDF',
    'RoBERTa · CE': 'RoBERTa CE',
    'RoBERTa · CE+SupCon': 'RoBERTa CE+SupCon',
    'RACE · CE': 'RACE CE',
    'RACE · CE+SupCon': 'RACE CE+SupCon',
    'E30 equal-mean fusion': 'E30 fusion',
    'E26 head/middle/tail': 'E26 window',
    'E01 B0 single window': 'E01 single',
    'E30 (baseline)': 'E30 baseline',
    'nonneg weights (val-fit)': 'nonneg weights',
    'logreg stack (val-fit)': 'logreg stack',
    'mean of all 13': 'mean of 13',
    'logreg on 12 strong (val-fit)': 'logreg 12 strong',
    'nonneg on 12 strong (val-fit)': 'nonneg 12 strong',
    'logreg stack (test crossfit)': 'logreg (leaked)',
    'stack (unconstrained)': 'stack',
    'stack + group constraint': 'stack + constraint',
    'stack + constraint + pairwise': 'stack + constraint + pairwise',
}

EXCLUDE = {'TF-IDF + LR'}


def collect() -> list[dict]:
    rows: list[dict] = []

    reinf = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    for label, runs in PUBLISHED.items():
        accs = [reinf[r]['accuracy'] for r in runs]
        f1s = [reinf[r]['macro_f1'] for r in runs]
        rows.append({'name': label, 'family': FAMILIES[0][0],
                     'accuracy': float(np.mean(accs)), 'macro_f1': float(np.mean(f1s)),
                     'sd': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0})

    top3 = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    for label, key in TOP3.items():
        rows.append({'name': label, 'family': FAMILIES[1][0],
                     'accuracy': top3[key]['accuracy'], 'macro_f1': top3[key]['macro_f1'],
                     'sd': 0.0})

    stack = json.loads((ROOT / '.local/stacking.json').read_text(encoding='utf-8'))
    for key, val in stack['results'].items():
        if key == 'E30 (baseline)':
            continue
        rows.append({'name': key, 'family': FAMILIES[2][0],
                     'accuracy': val['accuracy'], 'macro_f1': val['macro_f1'], 'sd': 0.0})

    cons = json.loads((ROOT / '.local/constrained_decoding.json').read_text(encoding='utf-8'))
    for key in ('stack + group constraint', 'stack + constraint + pairwise'):
        val = cons['test'][key]
        rows.append({'name': key, 'family': FAMILIES[3][0],
                     'accuracy': val['accuracy'], 'macro_f1': val['macro_f1'], 'sd': 0.0})

    return [r for r in rows if r['name'] not in EXCLUDE]


def pareto(rows: list[dict]) -> list[dict]:
    """Points not dominated by any other point (higher accuracy and higher F1)."""
    keep = []
    for r in rows:
        dominated = any(o['accuracy'] >= r['accuracy'] and o['macro_f1'] >= r['macro_f1']
                        and (o['accuracy'] > r['accuracy'] or o['macro_f1'] > r['macro_f1'])
                        for o in rows)
        if not dominated:
            keep.append(r)
    return sorted(keep, key=lambda r: r['accuracy'])


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = collect()
    colors = dict(FAMILIES)
    xs = np.array([100 * r['accuracy'] for r in rows])
    ys = np.array([100 * r['macro_f1'] for r in rows])

    fig, ax = plt.subplots(figsize=(13.6, 8.6))

    for r, x, y in zip(rows, xs, ys):
        ax.scatter(x, y, s=210, color=colors[r['family']], zorder=3,
                   edgecolor='white', linewidth=1.3)
        if r['sd'] > 0:
            ax.errorbar(x, y, xerr=100 * r['sd'], fmt='none', ecolor='#546e7a',
                        capsize=3, lw=1.0, zorder=2)

    # Labels go in a column to the right, ordered by height, each tied to its
    # marker with a thin leader line. The cluster spans <1 pp so free placement
    # would overlap.
    span_x = max(xs.max() - xs.min(), 1e-6)
    span_y = max(ys.max() - ys.min(), 1e-6)
    column_x = xs.max() + 0.30 * span_x
    order = np.argsort(ys)
    slots = np.linspace(ys.min() - 0.06 * span_y, ys.max() + 0.06 * span_y, len(rows))
    for slot, idx in zip(slots, order):
        ax.annotate(SHORT.get(rows[idx]['name'], rows[idx]['name']),
                    (xs[idx], ys[idx]), xytext=(column_x, slot),
                    ha='left', va='center', fontsize=8.6, color='#263238',
                    arrowprops=dict(arrowstyle='-', color='#b0bec5', lw=0.7,
                                    shrinkA=2, shrinkB=7,
                                    connectionstyle='arc3,rad=0.04'))

    # Highlight the two headline results.
    best = max(rows, key=lambda r: r['accuracy'])
    ax.annotate(f'{100*best["accuracy"]:.2f}% / {100*best["macro_f1"]:.2f}%',
                (100 * best['accuracy'], 100 * best['macro_f1']),
                xytext=(-14, 16), textcoords='offset points', fontsize=9.5,
                fontweight='bold', color='#1b5e20', ha='right')

    handles = [plt.Line2D([], [], marker='o', ls='', ms=10, color=c, label=n,
                          markeredgecolor='white') for n, c in FAMILIES]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=.95)

    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_ylabel('Macro-F1 (%)', fontsize=11)
    ax.set_title('HART four-class · accuracy vs macro-F1 on the published 3,194-item test set\n'
                 'all models scored locally; upper-right is better', fontsize=12)
    ax.grid(alpha=.18)
    ax.set_xlim(xs.min() - 0.10 * span_x, column_x + 0.62 * span_x)
    ax.set_ylim(ys.min() - 0.14 * span_y, ys.max() + 0.14 * span_y)
    ax.set_xticks(np.arange(94.0, 98.5, 0.5))

    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)')
    print(f'{len(rows)} models plotted')


if __name__ == '__main__':
    main()
