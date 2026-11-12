"""Figure for the two validation tests.

Three panels:
  1. placebo distribution for the group-structure claim
  2. human-rewrite recall under increasing model capacity (the label is the limit)
  3. pairwise ranking accuracy against its controls (the 87.7% is generic style)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/reports/comparison/validation_tests.png'


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    placebo = json.loads((ROOT / '.local/group_placebo.json').read_text(encoding='utf-8'))
    ident = json.loads((ROOT / '.local/human_rewrite_identifiability.json').read_text(encoding='utf-8'))
    controls = json.loads((ROOT / '.local/pairwise_controls.json').read_text(encoding='utf-8'))
    decisive = json.loads((ROOT / '.local/identifiability_decisive.json').read_text(encoding='utf-8'))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2),
                             gridspec_kw={'width_ratios': [1.05, 1.0, 1.0]})
    fig.suptitle('Two falsification tests: the group gain is real, the human-rewrite limit is the label',
                 fontsize=13, y=0.975)

    # ---- panel 1: placebo distribution -----------------------------------
    ax = axes[0]
    ref = placebo['reference']
    true_score = 100 * ref['constrained (true groups)']
    stack_score = 100 * ref['stack (unconstrained)']
    a = placebo['placebo_item_shuffle']
    b = placebo['placebo_pattern_shuffle']

    ax.axhline(stack_score, color='#90a4ae', ls='--', lw=1.4,
               label=f'unconstrained stack {stack_score:.2f}%')
    ax.axhline(true_score, color='#2e7d32', ls='-', lw=2.0,
               label=f'true groups {true_score:.2f}%')
    ax.errorbar([1], [100 * a['mean']], yerr=[[100 * a['mean'] - 100 * a['p05']],
                                              [100 * a['p95'] - 100 * a['mean']]],
                fmt='o', color='#b71c1c', ms=9, capsize=6, lw=1.8,
                label=f'item shuffle {100*a["mean"]:.2f}%')
    ax.errorbar([2], [100 * b['mean']], yerr=[[100 * b['mean'] - 100 * b['p05']],
                                              [100 * b['p95'] - 100 * b['mean']]],
                fmt='s', color='#ef6c00', ms=9, capsize=6, lw=1.8,
                label=f'pattern shuffle {100*b["mean"]:.2f}%')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['placebo A\nitems reassigned\nto random groups',
                        'placebo B\ntrue groups, random\nHUM-slot assignment'], fontsize=9)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(70, 100)
    ax.set_ylabel('test accuracy (%)')
    ax.legend(fontsize=8.2, loc='lower right')
    ax.set_title(f'Both placebos collapse  (p < {1/placebo["rounds"]:.3f}, {placebo["rounds"]} rounds)',
                 fontsize=10)

    # ---- panel 2: capacity sweep -----------------------------------------
    ax = axes[1]
    sep = ident['separability']
    dec = decisive['balanced_binary']
    rows = [
        ('E30 four-class\n(imbalanced)', 42.11, '#455a64'),
        ('13-model stack\n(imbalanced)', 43.86, '#90a4ae'),
        ('balanced binary\nword 1-2', 100 * dec['word 1-2']['recall'], '#1565c0'),
        ('balanced binary\nchar_wb 2-5', 100 * dec['char_wb 2-5']['recall'], '#1565c0'),
        ('stylometry\n+ GBM', 100 * sep['stylometry+gbm']['recall'], '#6a1b9a'),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.6, f'{v:.1f}%',
                ha='center', fontsize=9.5, fontweight='bold')
    ax.axhline(50, color='#b71c1c', ls=':', lw=1.3)
    ax.text(-0.42, 51.5, 'chance', fontsize=8, color='#b71c1c', ha='left')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel('human-rewrite recall (%)')
    ax.set_title('More capacity does not help\n(balancing the pool does not either)', fontsize=10)

    # ---- panel 3: pairwise controls --------------------------------------
    ax = axes[2]
    ctrl = [
        ('true siblings\n(same document)', 100 * controls['vectoriser']['train_only_accuracy'], '#2e7d32'),
        ('length-matched\nsubset', 100 * controls['length_matched']['accuracy'], '#2e7d32'),
        ('cross-document\npairs', 100 * controls['shuffled_pair_accuracy'], '#ef6c00'),
        ('GEN vs GEN\n(same class)', 100 * decisive['same_class_controls']['gen_vs_gen']['rate_b_is_human'], '#b71c1c'),
        ('HUM vs HUM\n(same class)', 100 * decisive['same_class_controls']['hum_vs_hum']['rate_b_is_human'], '#b71c1c'),
    ]
    labels = [c[0] for c in ctrl]
    vals = [c[1] for c in ctrl]
    colors = [c[2] for c in ctrl]
    bars = ax.bar(range(len(ctrl)), vals, color=colors, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.6, f'{v:.1f}%',
                ha='center', fontsize=9.5, fontweight='bold')
    ax.axhline(50, color='#b71c1c', ls=':', lw=1.3)
    ax.text(-0.42, 51.5, 'chance', fontsize=8, color='#b71c1c', ha='left')
    ax.set_xticks(range(len(ctrl)))
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel('P(ranker says b is human) (%)')
    ax.set_title('The pairwise signal is generic style,\nnot same-document discrimination', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
