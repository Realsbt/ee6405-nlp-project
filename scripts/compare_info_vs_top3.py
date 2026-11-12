"""Compare the three information-theoretic methods against the composited Top 3.

Everything here is computed on this machine on the published 3,194-item test
set:

  * the published five models are re-inferred from their checkpoints by
    scripts/reinfer_published_local.py (parity vs the archived files is
    reported in .local/local_reinference.json);
  * the campaign's composited Top 3 (E30, E26, E01) are rebuilt by
    scripts/reconstruct_campaign_top3.py with the campaign's own recipes;
  * M1/M2/M3 are trained by scripts/eval_info_on_published_test.py on a pool
    that excludes every published test group.

Fusion weights are chosen by group-disjoint cross-fitting: the test groups are
split in two halves, the weight is fitted on one half and scored on the other,
then the halves swap. The reported fused score is therefore out-of-fold and
does not benefit from tuning on the items it is scored on.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reports/comparison'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']

PUBLISHED = {
    'TF-IDF + LR': ['tfidf_lr_v1'],
    'RoBERTa · CE': ['roberta_s3407', 'roberta_s42', 'roberta_s2026'],
    'RoBERTa · CE+SupCon': ['roberta_supcon_s3407', 'roberta_supcon_s42', 'roberta_supcon_s2026'],
    'RACE · CE': ['race_no_supcon_s3407', 'race_no_supcon_s42', 'race_no_supcon_s2026'],
    'RACE · CE+SupCon': ['race_s3407', 'race_s42', 'race_s2026'],
}

TOP3 = {
    'E30 equal-mean fusion': 'E30',
    'E26 head/middle/tail': 'E26',
    'E01 B0 single window': 'E01',
}

NEW = {
    'M1 entropy + compressibility': 'M1_entropy_compression',
    'M2 stylometric features': 'M2_stylometric',
    'M3 char n-gram cross-entropy': 'M3_char_ngram_cross_entropy',
}

# Standalone baselines that are far below the competitive cluster. They stay in
# the bar chart but are dropped from the scatter so the neural models are not
# compressed into a single pixel.
SCATTER_EXCLUDE = {
    'TF-IDF + LR',
    'M1 entropy + compressibility',
    'M2 stylometric features',
    'M3 char n-gram cross-entropy',
}


def test_labels() -> np.ndarray:
    labels = []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            labels.append(LABELS.index(json.loads(line)['true_label']))
    return np.asarray(labels)


def test_groups() -> np.ndarray:
    groups = []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            groups.append(json.loads(line)['group_id'])
    return np.asarray(groups)


def score(y: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    pred = probabilities.argmax(1)
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro'))
    return {'accuracy': acc, 'macro_f1': f1, 'J': (acc + f1) / 2}


def load_published_local() -> dict[str, dict]:
    data = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    out = {}
    for label, runs in PUBLISHED.items():
        probas = [np.asarray(data[run]['probabilities']) for run in runs]
        accs = [data[run]['accuracy'] for run in runs]
        f1s = [data[run]['macro_f1'] for run in runs]
        out[label] = {
            'accuracy': float(np.mean(accs)),
            'accuracy_sd': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            'macro_f1': float(np.mean(f1s)),
            'macro_f1_sd': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            'J': float((np.mean(accs) + np.mean(f1s)) / 2),
            'runs': len(runs),
            'max_abs_diff_vs_archived': max(data[run]['max_abs_diff_vs_archived'] for run in runs),
            'proba': np.mean(probas, axis=0),
        }
    return out


def load_top3() -> dict[str, dict]:
    data = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    return {label: {'accuracy': data[key]['accuracy'], 'macro_f1': data[key]['macro_f1'],
                    'J': data[key]['J'], 'runs': 1, 'proba': np.asarray(data[key]['probabilities'])}
            for label, key in TOP3.items()}


def load_new() -> dict[str, dict]:
    data = json.loads((ROOT / '.local/info_methods_published_test.json').read_text(encoding='utf-8'))
    return {label: {'accuracy': data[key]['accuracy'], 'macro_f1': data[key]['macro_f1'],
                    'J': data[key]['J'], 'runs': 1, 'proba': np.asarray(data[key]['proba'])}
            for label, key in NEW.items()}


def best_weight(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> float:
    """Weight on `extra` that maximises J on the given rows."""
    grid = np.linspace(0.0, 1.0, 21)
    best, best_j = 0.0, -1.0
    for w in grid:
        row = score(y, (1 - w) * base + w * extra)
        if row['J'] > best_j:
            best, best_j = float(w), row['J']
    return best


def crossfit_fusion(y: np.ndarray, groups: np.ndarray, base: np.ndarray, extra: np.ndarray) -> dict:
    """Group-disjoint two-fold cross-fitting of the fusion weight."""
    unique = np.asarray(sorted(set(groups)))
    rng = np.random.default_rng(20261107)
    rng.shuffle(unique)
    half = set(unique[:len(unique) // 2].tolist())
    fold_a = np.asarray([g in half for g in groups])
    fused = np.zeros_like(base)
    weights = []
    for fit_mask, apply_mask in ((fold_a, ~fold_a), (~fold_a, fold_a)):
        w = best_weight(y[fit_mask], base[fit_mask], extra[fit_mask])
        weights.append(w)
        fused[apply_mask] = (1 - w) * base[apply_mask] + w * extra[apply_mask]
    row = score(y, fused)
    row['weights'] = weights
    row['proba'] = fused
    return row


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, rounds: int = 2000) -> dict:
    """P(accuracy(b) > accuracy(a)) over item resamples."""
    rng = np.random.default_rng(20261107)
    pred_a, pred_b = a.argmax(1), b.argmax(1)
    deltas = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        deltas[i] = (pred_b[idx] == y[idx]).mean() - (pred_a[idx] == y[idx]).mean()
    return {'mean_delta_pp': float(100 * deltas.mean()),
            'ci95_pp': [float(100 * np.percentile(deltas, 2.5)), float(100 * np.percentile(deltas, 97.5))],
            'p_b_better': float((deltas > 0).mean())}


def plot(published, top3, new, fused) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = [('Published five (local re-inference)', published, '#4f7f91'),
              ('Composited Top 3 (campaign recipes)', top3, '#7b4fa0'),
              ('Information-theoretic methods', new, '#b5651d'),
              ('Fusion: info method + composited Top 3', fused, '#2e7d32')]

    names, accs, f1s, colors, sds = [], [], [], [], []
    for _, block, color in groups:
        for name, row in block.items():
            names.append(name)
            accs.append(100 * row['accuracy'])
            f1s.append(100 * row['macro_f1'])
            colors.append(color)
            sds.append(100 * row.get('accuracy_sd', 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.4), gridspec_kw={'width_ratios': [1.25, 1]})
    y = np.arange(len(names))

    ax = axes[0]
    ax.barh(y - 0.2, accs, height=0.38, color=colors, alpha=.92, label='Accuracy')
    ax.barh(y + 0.2, f1s, height=0.38, color=colors, alpha=.45, label='Macro-F1')
    ax.errorbar(accs, y - 0.2, xerr=sds, fmt='none', ecolor='#333', capsize=3, linewidth=1)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(a + 0.6, i - 0.2, f'{a:.1f}', va='center', fontsize=7.5)
        ax.text(f + 0.6, i + 0.2, f'{f:.1f}', va='center', fontsize=7.5)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xlabel('Percent')
    ax.set_title('All models scored locally on the same 3,194-item published test set\n'
                 'bars are mean, whiskers are seed SD', fontsize=10)
    offset = 0
    for _, block, _ in groups[:-1]:
        offset += len(block)
        ax.axhline(offset - 0.5, color='#999', linestyle='--', linewidth=1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=.15)

    # The weak baselines sit far from the competitive cluster and would flatten
    # the scale, so the scatter shows only the models that are actually in
    # contention. The bar chart above still reports every model.
    ax = axes[1]
    shown = [i for i, n in enumerate(names) if n not in SCATTER_EXCLUDE]
    xs = np.asarray([accs[i] for i in shown])
    ys = np.asarray([f1s[i] for i in shown])
    for i in shown:
        ax.scatter(accs[i], f1s[i], s=190, color=colors[i], zorder=3,
                   edgecolor='white', linewidth=1.2)

    # The neural models differ by a few tenths of a point, so their markers
    # overlap. Put the labels in a column to the right of the cluster, ordered
    # by height, and connect each one to its marker.
    span_x = max(xs.max() - xs.min(), 1e-6)
    span_y = max(ys.max() - ys.min(), 1e-6)
    column_x = xs.max() + 0.42 * span_x
    by_height = np.argsort(ys)
    slots = np.linspace(ys.min() - 0.10 * span_y, ys.max() + 0.10 * span_y, len(shown))
    lx = np.empty(len(shown))
    ly = np.empty(len(shown))
    for slot, idx in enumerate(by_height):
        lx[idx] = column_x
        ly[idx] = slots[slot]

    for rank, i in enumerate(shown, start=1):
        ax.annotate(str(rank), (accs[i], f1s[i]), xytext=(lx[rank - 1], ly[rank - 1]),
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold',
                    zorder=5, bbox=dict(boxstyle='circle,pad=0.28', facecolor=colors[i],
                                        edgecolor='white', linewidth=0.8),
                    arrowprops=dict(arrowstyle='-', color='#bbb', linewidth=0.6,
                                    shrinkA=0, shrinkB=6,
                                    connectionstyle='arc3,rad=0.05'))

    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Macro-F1 (%)')
    ax.set_title('Accuracy vs Macro-F1 · upper-right is better\n'
                 'competitive models only', fontsize=10)
    ax.grid(alpha=.15)
    pad_x = max(0.25, 0.10 * span_x)
    pad_y = max(0.25, 0.10 * span_y)
    ax.set_xlim(xs.min() - pad_x, column_x + 0.30 * span_x)
    ax.set_ylim(min(ys.min(), ly.min()) - pad_y, max(ys.max(), ly.max()) + pad_y)
    legend = '\n'.join(f'{rank:2}. {names[i]}' for rank, i in enumerate(shown, start=1))
    ax.text(1.03, 0.5, legend, transform=ax.transAxes, fontsize=6.8, va='center',
            family='monospace', linespacing=1.5)

    fig.suptitle('HART four-class · information-theoretic methods vs the composited Top 3', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'info_vs_top3.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT / "info_vs_top3.png"}')


def main() -> None:
    y = test_labels()
    groups = test_groups()
    published = load_published_local()
    top3 = load_top3()
    new = load_new()

    fused = {}
    for new_name, new_row in new.items():
        for top_name, top_row in top3.items():
            row = crossfit_fusion(y, groups, top_row['proba'], new_row['proba'])
            row['base'] = top_name
            row['extra'] = new_name
            fused[f'{new_name} + {top_name}'] = row

    def table(title, block):
        print(f'\n{title}')
        print(f'{"Model":52} {"Acc":>8} {"MacroF1":>9} {"J":>8} {"Runs":>5}')
        print('-' * 86)
        for name, row in block.items():
            print(f'{name:52} {100*row["accuracy"]:7.2f}% {100*row["macro_f1"]:8.2f}% '
                  f'{100*row["J"]:7.2f}% {row.get("runs", 1):5}')

    table('Published five (re-inferred locally)', published)
    table('Composited Top 3 (campaign recipes, published test set)', top3)
    table('Information-theoretic methods', new)
    table('Fusion (group-disjoint cross-fitted weight)', fused)

    best_top3 = max(top3, key=lambda n: top3[n]['J'])
    best_fused = max(fused, key=lambda n: fused[n]['J'])
    print(f'\nBest composited Top 3: {best_top3}  J={100*top3[best_top3]["J"]:.2f}%')
    print(f'Best fusion:           {best_fused}  J={100*fused[best_fused]["J"]:.2f}%')
    print(f'Delta J:               {100*(fused[best_fused]["J"] - top3[best_top3]["J"]):+.2f} pp')

    boot = paired_bootstrap(y, top3[best_top3]['proba'], fused[best_fused]['proba'])
    print(f'Paired bootstrap vs {best_top3}: {boot["mean_delta_pp"]:+.2f} pp '
          f'95% CI [{boot["ci95_pp"][0]:+.2f}, {boot["ci95_pp"][1]:+.2f}] '
          f'P(better)={boot["p_b_better"]:.2f}')

    payload = {
        'test_items': int(len(y)),
        'published': {k: {kk: vv for kk, vv in v.items() if kk != 'proba'} for k, v in published.items()},
        'top3': {k: {kk: vv for kk, vv in v.items() if kk != 'proba'} for k, v in top3.items()},
        'new': {k: {kk: vv for kk, vv in v.items() if kk != 'proba'} for k, v in new.items()},
        'fused': {k: {kk: vv for kk, vv in v.items() if kk != 'proba'} for k, v in fused.items()},
        'best_top3': best_top3,
        'best_fused': best_fused,
        'bootstrap': boot,
    }
    (ROOT / '.local/info_vs_top3.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    plot(published, top3, new, fused)


if __name__ == '__main__':
    main()
