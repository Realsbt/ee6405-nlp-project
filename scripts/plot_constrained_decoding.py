"""Figure for the constrained-decoding experiment.

Three panels:
  1. accuracy ladder from E30 to the constrained decoder
  2. the human-rewrite subgroup, which is the only class that moves
  3. why the pairwise model helps: P(HUM) of the human item vs its sibling GEN
"""
from __future__ import annotations

import collections
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/reports/comparison/constrained_decoding.png'

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from scripts.constrained_decoding import (
        ai_candidates, group_index, pair_features,
    )
    from scripts.stack_models import apply_logreg, fit_logreg, load_test, load_val

    payload = json.loads((ROOT / '.local/constrained_decoding.json').read_text(encoding='utf-8'))
    results = payload['test']

    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))
    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    val_stack = apply_logreg(w, np.stack([val_probs[n] for n in names]))
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))

    vrecs = [json.loads(line) for line in
             (ROOT / 'experiments/hart_grouped_v1/val.jsonl').read_text(encoding='utf-8').splitlines()
             if line.strip()]
    val_index = group_index(np.asarray([r['group_id'] for r in vrecs]))
    test_index = group_index(test_groups)
    val_cand = ai_candidates(val_stack, val_index)
    test_cand = ai_candidates(test_stack, test_index)

    x_val, y_val = [], []
    for g, c in val_cand.items():
        x_val.append(pair_features(val_stack, c))
        y_val.extend([1 if val_truth[i] == 3 else 0 for i in c])
    pairwise = LogisticRegression(max_iter=2000, class_weight='balanced').fit(
        np.vstack(x_val), np.asarray(y_val))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.4),
                             gridspec_kw={'width_ratios': [1.15, 0.85, 1.0]})
    fig.suptitle('Group-constrained decoding — the dataset structure is worth +3.3 pp',
                 fontsize=13, y=0.975)

    # ---- panel 1: accuracy ladder ----------------------------------------
    ax = axes[0]
    order = ['E30 (baseline)', 'stack (unconstrained)',
             'stack + group constraint', 'stack + constraint + pairwise']
    short = ['E30\nbaseline', 'stack\n(13 models)', 'stack +\ngroup constraint',
             'stack + constraint\n+ pairwise']
    accs = [100 * results[k]['accuracy'] for k in order]
    colors = ['#455a64', '#90a4ae', '#1565c0', '#2e7d32']
    bars = ax.bar(short, accs, color=colors, width=0.6)
    for bar, k in zip(bars, order):
        b = results[k]['bootstrap_vs_E30']
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * results[k]['accuracy'] + 0.12,
                f'{100*results[k]["accuracy"]:.2f}%', ha='center',
                fontsize=10.5, fontweight='bold')
        if k != order[0]:
            ax.text(bar.get_x() + bar.get_width() / 2, 93.6,
                    f'{b["mean_delta_pp"]:+.2f} pp\n[{b["ci95_pp"][0]:+.2f}, {b["ci95_pp"][1]:+.2f}]',
                    ha='center', fontsize=8, color='white')
    ax.set_ylim(93.2, 98.6)
    ax.set_ylabel('test accuracy (%)')
    ax.set_title('Each step is fit on validation only', fontsize=10)

    # ---- panel 2: human rewrite ------------------------------------------
    ax = axes[1]
    hum = [100 * results[k]['human_rewrite'] for k in order]
    tool = [100 * results[k]['tool_rewrite'] for k in order]
    x = np.arange(len(order))
    ax.bar(x - 0.19, hum, width=0.36, color='#b71c1c', label='human rewrite (n=57)')
    ax.bar(x + 0.19, tool, width=0.36, color='#6a1b9a', label='tool rewrite (n=170)')
    for xi, (h, t) in enumerate(zip(hum, tool)):
        ax.text(xi - 0.19, h + 1.2, f'{h:.1f}', ha='center', fontsize=8.5)
        ax.text(xi + 0.19, t + 1.2, f'{t:.1f}', ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['E30', 'stack', '+constraint', '+pairwise'], fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel('recall (%)')
    ax.legend(fontsize=8.5, loc='lower left')
    ax.set_title('Human rewrite finally moves\n43.9% → 52.6%', fontsize=10)

    # ---- panel 3: why the pairwise model works ---------------------------
    ax = axes[2]
    kinds = np.asarray([r.get('model_name', '') for r in
                        [json.loads(line) for line in
                         (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
                         if line.strip()]])
    rows = []
    for g, members in test_index.items():
        if len(members) != 4:
            continue
        human = [i for i in members if test_truth[i] == 3 and kinds[i] == 'human']
        gen = [i for i in members if test_truth[i] == 2]
        if not human or not gen:
            continue
        rows.append((test_stack[human[0], 3], test_stack[gen[0], 3]))
    rows = np.array(rows)
    ax.scatter(rows[:, 1], rows[:, 0], s=42, alpha=0.75, color='#b71c1c',
               edgecolor='white', linewidth=0.6)
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1.1)
    ax.set_xlabel('P(ai_humanized) of the sibling ai_generated item')
    ax.set_ylabel('P(ai_humanized) of the human-rewrite item')
    above = int((rows[:, 0] > rows[:, 1]).sum())
    ax.set_title(f'The signal is relative, not absolute\n'
                 f'human item ranks higher in {above}/{len(rows)} groups', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.62, 0.12, 'above the diagonal =\ncorrectly ranked', fontsize=8.5, color='#555')

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
