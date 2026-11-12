"""Verify the M1+E26 fusion: per-class detail, weight sensitivity, significance.

The 0.5 weight in the first run was a fixed prior, not tuned. This script checks
whether the gain survives a weight sweep and a paired bootstrap, so the report
can state what is actually supported.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'experiments/campaign12h/ee6405_scorefirst_36_12h'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def load():
    manifest = json.loads((CAMPAIGN / 'protocol/split_manifest.json').read_text(encoding='utf-8'))
    s_records = [r for r in manifest['records'] if r['partition'] == 'S']
    y = np.asarray([int(r['label_id']) for r in s_records])
    e26 = json.loads((CAMPAIGN / 'top5_revision_01/E26_prediction_with_ids.json').read_text(encoding='utf-8'))
    e26_proba = np.asarray([r['probabilities'] for r in e26['prediction_rows']])
    info = json.loads((ROOT / '.local/info_methods.json').read_text(encoding='utf-8'))
    m1 = np.asarray(info['M1_entropy_compression']['proba'])
    return y, e26_proba, m1


def scores(y, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average='macro'))


def main() -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    y, e26, m1 = load()
    base_acc, base_f1 = scores(y, e26.argmax(1))
    print(f'E26 baseline      acc={base_acc:.4f} f1={base_f1:.4f}\n')

    print('Weight sweep (w = weight on M1):')
    best = None
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        fused = (1 - w) * e26 + w * m1
        acc, f1 = scores(y, fused.argmax(1))
        j = (acc + f1) / 2
        marker = ''
        if best is None or j > best[0]:
            best = (j, w, acc, f1)
            marker = '  <- best'
        print(f'  w={w:.1f}  acc={acc:.4f}  f1={f1:.4f}  J={j:.4f}{marker}')

    print(f'\nBest J at w={best[1]:.1f}: acc={best[2]:.4f} f1={best[3]:.4f}')
    print('NOTE: the best weight is selected on S, so it is optimistic. '
          'The fixed w=0.5 result is the honest one.\n')

    fused = 0.5 * e26 + 0.5 * m1
    pred = fused.argmax(1)
    acc, f1 = scores(y, pred)
    print(f'Fixed w=0.5       acc={acc:.4f} f1={f1:.4f}')
    print(f'Delta vs E26      acc={100*(acc-base_acc):+.2f} pp  f1={100*(f1-base_f1):+.2f} pp\n')

    base_pred = e26.argmax(1)
    w2r = int(((base_pred != y) & (pred == y)).sum())
    r2w = int(((base_pred == y) & (pred != y)).sum())
    print(f'Wrong->right {w2r}   Right->wrong {r2w}   net {w2r - r2w:+d}\n')

    print('Per-class F1 (E26 -> fused):')
    for i, name in enumerate(LABELS):
        from sklearn.metrics import f1_score
        a = f1_score(y == i, base_pred == i, zero_division=0)
        b = f1_score(y == i, pred == i, zero_division=0)
        print(f'  {name:20} {a:.4f} -> {b:.4f}  ({100*(b-a):+.2f} pp)')

    print('\nConfusion matrix (fused):')
    print(confusion_matrix(y, pred))

    # Paired bootstrap over items: is the accuracy gain distinguishable from noise?
    rng = np.random.default_rng(3407)
    n = len(y)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        a = (base_pred[idx] == y[idx]).mean()
        b = (pred[idx] == y[idx]).mean()
        deltas.append(b - a)
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f'\nPaired bootstrap accuracy delta: mean={deltas.mean():+.4f} '
          f'95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0)={float((deltas > 0).mean()):.3f}')

    out = {
        'baseline': {'accuracy': base_acc, 'macro_f1': base_f1},
        'fused_w05': {'accuracy': acc, 'macro_f1': f1},
        'delta_pp': {'accuracy': 100 * (acc - base_acc), 'macro_f1': 100 * (f1 - base_f1)},
        'wrong_to_right': w2r,
        'right_to_wrong': r2w,
        'bootstrap': {'mean': float(deltas.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
                      'p_positive': float((deltas > 0).mean())},
        'best_weight_on_S': {'w': best[1], 'accuracy': best[2], 'macro_f1': best[3],
                             'note': 'selected on S, optimistic'},
    }
    (ROOT / '.local/fusion_verification.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\nwrote .local/fusion_verification.json')


if __name__ == '__main__':
    main()
