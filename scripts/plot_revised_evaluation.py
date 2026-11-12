"""Figure for the audited evaluation protocol.

Four panels, each showing uncertainty rather than a bare point estimate:
  1. humanized sub-populations with Wilson 95% intervals
  2. ceiling sensitivity: the "ceiling" moves with the recall assumption
  3. short-document routing: leakage-free protocol, CI touching zero
  4. two-stage generator gap, stratified and length-matched
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reports/comparison/revised_evaluation.png'


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audit = json.loads((ROOT / '.local/audit.json').read_text(encoding='utf-8'))
    short_model = json.loads((ROOT / '.local/short_document_model.json').read_text(encoding='utf-8'))

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.8))
    fig.suptitle('Audited evaluation protocol — E30 on the published 3,194-item test set\n'
                 'every panel shows uncertainty, not just a point estimate',
                 fontsize=12.5, y=0.985)

    # ---- panel 1: subgroups with Wilson intervals ------------------------
    ax = axes[0, 0]
    sub = audit['subgroups']
    kinds = ['tool', 'human']
    labels = [f'tool rewrite\n(n={sub["tool"]["n"]})', f'human rewrite\n(n={sub["human"]["n"]})']
    accs = [100 * sub[k]['accuracy'] for k in kinds]
    lo = [100 * (sub[k]['accuracy'] - sub[k]['wilson95'][0]) for k in kinds]
    hi = [100 * (sub[k]['wilson95'][1] - sub[k]['accuracy']) for k in kinds]
    bars = ax.bar(labels, accs, color=['#6a1b9a', '#b71c1c'], width=0.5)
    ax.errorbar(labels, accs, yerr=[lo, hi], fmt='none', ecolor='#222',
                capsize=8, linewidth=1.8)
    for bar, acc, up in zip(bars, accs, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + up + 2.5, f'{acc:.1f}%',
                ha='center', fontsize=11, fontweight='bold')
    ax.axhline(25, color='#888', linestyle='--', linewidth=1)
    ax.text(1.32, 26.5, 'four-class\nchance', fontsize=8, color='#666', ha='left')
    ax.set_xlim(-0.6, 1.95)
    ax.set_ylim(0, 105)
    ax.set_ylabel('accuracy (%)')
    diff = sub['difference']
    ax.set_title(f'1. `ai_humanized` is two populations\n'
                 f'difference {100*diff["delta"]:+.1f} pp, 95% CI '
                 f'[{100*diff["newcombe95"][0]:+.1f}, {100*diff["newcombe95"][1]:+.1f}]',
                 fontsize=10)

    # ---- panel 2: ceiling sensitivity ------------------------------------
    ax = axes[0, 1]
    scen = audit['ceiling']['scenarios']
    names = list(scen)
    recalls = [100 * scen[n]['recall'] for n in names]
    accs = [100 * scen[n]['accuracy'] for n in names]
    ax.plot(recalls, accs, 'o-', color='#1565c0', linewidth=2, markersize=8)
    for x, y in zip(recalls, accs):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax.axhline(100 * 0.9424, color='#c62828', linestyle='--', linewidth=1.2)
    ax.text(recalls[0], 100 * 0.9424 - 0.12, 'current 94.24%', fontsize=8.5,
            color='#c62828', va='top')
    ax.set_xlabel('assumed recall on human-rewrite items (%)')
    ax.set_ylabel('implied "ceiling" (%)')
    ax.set_ylim(93.8, 96.4)
    ax.set_title('2. The "ceiling" is an assumption, not a bound\n'
                 '28% → 100% recall moves it from 94.74% to 96.02%', fontsize=10)

    # ---- panel 3: short documents, leakage-free --------------------------
    ax = axes[1, 0]
    test = short_model['test']
    boot = short_model['bootstrap']
    names = ['E30\nbaseline', 'dedicated\nshort model', 'blended\n(val-selected)']
    values = [100 * test['baseline'], 100 * test['standalone'], 100 * test['blended']]
    bars = ax.bar(names, values, color=['#90a4ae', '#1565c0', '#2e7d32'], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f'{value:.2f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(2, values[2]), xytext=(2, values[0]),
                arrowprops=dict(arrowstyle='<->', color='#2e7d32', linewidth=1.6))
    ax.text(2.28, (values[0] + values[2]) / 2,
            f'{test["delta_pp"]:+.2f} pp\n95% CI [{boot["ci95_pp"][0]:+.2f}, '
            f'{boot["ci95_pp"][1]:+.2f}]\nP(better)={boot["p_better"]:.3f}',
            fontsize=8.5, va='center', color='#1b5e20')
    ax.set_ylim(min(values) - 3.0, max(values) + 2.0)
    ax.set_xlim(-0.6, 3.1)
    ax.set_ylabel('accuracy on the 375 short items (%)')
    ax.set_title('3. Short documents: directional, not confirmed\n'
                 'variant and weight chosen on validation; CI lower bound touches zero',
                 fontsize=10)

    # ---- panel 4: two-stage, length-matched ------------------------------
    ax = axes[1, 1]
    matched = audit['two_stage']['length_matched']
    bucket_names = {'0-1000': '<1k', '1000-2000': '1–2k',
                    '2000-3000': '2–3k', '3000-1000000000': '>3k'}
    buckets = [bucket_names.get(m['bucket'], m['bucket']) for m in matched]
    gaps = [m['gap_pp'] for m in matched]
    ns = [m['n_gemini'] for m in matched]
    colors = ['#b71c1c' if g > 20 else '#f9a825' if g > 5 else '#2e7d32' for g in gaps]
    bars = ax.bar(buckets, gaps, color=colors, width=0.55)
    for bar, gap, n in zip(bars, gaps, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, gap + 1.2,
                f'{gap:+.1f} pp\n(n={n})', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='#333', linewidth=1)
    ax.set_ylim(-6, max(gaps) + 12)
    ax.set_xlabel('document length (characters)')
    ax.set_ylabel('accuracy gap: other second stage − gemini (%)')
    perm = audit['two_stage']['permutation']
    ax.set_title(f'4. Two-stage gap is real but length-dependent\n'
                 f'overall {perm["observed_gap_pp"]:+.1f} pp, permutation '
                 f'p<0.0002; vanishes above 3k chars', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
