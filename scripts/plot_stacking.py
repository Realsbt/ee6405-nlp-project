"""Figure for the stacking experiment.

Two panels:
  1. every candidate vs E30 with a group-bootstrap interval
  2. where the stack's gain comes from: per-class recall and the humanized split
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reports/comparison/stacking.png'

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SHORT = ['HW', 'POL', 'GEN', 'HUM']


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    payload = json.loads((ROOT / '.local/stacking.json').read_text(encoding='utf-8'))
    results = payload['results']

    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    truth = np.asarray([LABELS.index(r['label']) for r in recs])
    kinds = np.asarray([r.get('model_name', '') for r in recs])

    local = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    campaign = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    probs = {n: np.asarray(e['probabilities'], dtype=float) for n, e in local.items()}
    probs.update({n: np.asarray(e['probabilities'], dtype=float) for n, e in campaign.items()})

    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.stack_models import apply_logreg, fit_logreg, load_val

    val_truth, val_probs = load_val()
    names = sorted(set(val_probs) & set(probs))
    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    stack_p = apply_logreg(w, np.stack([probs[n] for n in names]))
    base_p = probs['E30']

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2),
                             gridspec_kw={'width_ratios': [1.35, 1.0]})
    fig.suptitle('Stacking the 13 published models — fit on validation, test scored once',
                 fontsize=13, y=0.975)

    # ---- panel 1: candidates vs E30 --------------------------------------
    ax = axes[0]
    order = sorted(results, key=lambda k: results[k]['accuracy'])
    ys = np.arange(len(order))
    accs = [100 * results[k]['accuracy'] for k in order]
    colors = []
    for k in order:
        if k.startswith('E30'):
            colors.append('#455a64')
        elif 'LEAKED' in results[k]['note']:
            colors.append('#bdbdbd')
        elif results[k]['bootstrap_vs_E30']['ci95_pp'][0] > 0:
            colors.append('#2e7d32')
        else:
            colors.append('#90a4ae')
    ax.barh(ys, accs, color=colors, height=0.62)
    for y, k in zip(ys, order):
        b = results[k]['bootstrap_vs_E30']
        ax.text(100 * results[k]['accuracy'] + 0.06, y,
                f'{100*results[k]["accuracy"]:.2f}%  ({b["mean_delta_pp"]:+.2f} pp, '
                f'CI [{b["ci95_pp"][0]:+.2f}, {b["ci95_pp"][1]:+.2f}])',
                va='center', fontsize=8.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace(' (val-fit)', '\n(val-fit)').replace(' (test crossfit)', '\n(test crossfit)')
                        for k in order], fontsize=8.5)
    ax.set_xlim(93.6, 96.6)
    ax.set_xlabel('test accuracy (%)')
    ax.axvline(100 * results['E30 (baseline)']['accuracy'], color='#455a64',
               linestyle='--', linewidth=1.1)
    ax.set_title('Only the logistic-regression stack clears zero\n'
                 'green = 95% CI excludes zero; grey = leaked reference', fontsize=10)

    # ---- panel 2: per-class recall + humanized split ---------------------
    ax = axes[1]
    base_rec = [100 * (base_p.argmax(1)[truth == c] == c).mean() for c in range(4)]
    stack_rec = [100 * (stack_p.argmax(1)[truth == c] == c).mean() for c in range(4)]
    x = np.arange(4)
    ax.bar(x - 0.19, base_rec, width=0.36, label='E30', color='#90a4ae')
    ax.bar(x + 0.19, stack_rec, width=0.36, label='stack', color='#2e7d32')
    for xi, (a, b) in enumerate(zip(base_rec, stack_rec)):
        ax.text(xi - 0.19, a + 0.8, f'{a:.1f}', ha='center', fontsize=8.5)
        ax.text(xi + 0.19, b + 0.8, f'{b:.1f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n(n={int((truth==c).sum())})' for c, s in enumerate(SHORT)], fontsize=9)
    ax.set_ylim(60, 104)
    ax.set_ylabel('recall (%)')
    ax.legend(loc='lower left', fontsize=9)

    # annotate the humanized split inside the HUM group
    tool = (truth == 3) & (kinds == 'tool')
    human = (truth == 3) & (kinds == 'human')
    tool_b = 100 * (base_p.argmax(1)[tool] == 3).mean()
    tool_s = 100 * (stack_p.argmax(1)[tool] == 3).mean()
    hum_b = 100 * (base_p.argmax(1)[human] == 3).mean()
    hum_s = 100 * (stack_p.argmax(1)[human] == 3).mean()
    ax.text(2.42, 64.5, f'tool  {tool_b:.1f} → {tool_s:.1f}\nhuman {hum_b:.1f} → {hum_s:.1f}',
            ha='center', fontsize=8.5, color='#4a148c',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f3e5f5', edgecolor='#ce93d8'))
    ax.set_title('Every class improves, but human rewrite barely moves\n'
                 'the stack fixes GEN↔POL, not the humanized boundary', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
