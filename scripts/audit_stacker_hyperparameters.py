"""Does the stacker's gain survive hyperparameter uncertainty?

`audit_stacker.py` found the test-accuracy spread across 36 hyperparameter
configurations (0.88 pp) is larger than the reported gain over E30 (0.81 pp),
and the hand-chosen configuration ranks 9th of 36. That does not by itself
invalidate the gain, but it means the gain must be reported with the
hyperparameter spread attached, not as a single number.

This script:

  H1. Reports the full distribution of test accuracy over the grid, and the
      fraction of configurations that beat E30.
  H2. Re-runs the paired bootstrap for every configuration, so the reader can
      see how often the gain is significant.
  H3. Applies a principled selection rule (pick the configuration with the best
      *validation* accuracy, never looking at test) and reports what that
      yields. This is the honest alternative to the hand-chosen setting.

Writes `.local/stacker_hyperparameter.json`.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stack_models import (  # noqa: E402
    accuracy, apply_logreg, fit_logreg, load_test, load_val, macro_f1, paired_bootstrap,
)

OUT = ROOT / '.local/stacker_hyperparameter.json'


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))
    val_stack = np.stack([val_probs[n] for n in names])
    test_stack = np.stack([test_probs[n] for n in names])
    e30 = test_probs['E30']
    e30_acc = accuracy(e30, test_truth)

    grid = list(itertools.product((200, 600, 1200, 3000), (0.1, 0.35, 1.0), (1e-4, 1e-3, 1e-2)))
    rows = []
    for epochs, lr, l2 in grid:
        w = fit_logreg(val_stack, val_truth, epochs=epochs, lr=lr, l2=l2)
        pv = apply_logreg(w, val_stack)
        pt = apply_logreg(w, test_stack)
        rows.append({'epochs': epochs, 'lr': lr, 'l2': l2,
                     'val_acc': accuracy(pv, val_truth),
                     'test_acc': accuracy(pt, test_truth),
                     'test_f1': macro_f1(pt, test_truth),
                     'probs': pt})

    test_accs = np.array([r['test_acc'] for r in rows])
    val_accs = np.array([r['val_acc'] for r in rows])

    # H1: distribution
    beat = int((test_accs > e30_acc).sum())
    print(f'H1 distribution over {len(rows)} configs', flush=True)
    print(f'   test acc  min {100*test_accs.min():.2f}%  median {100*np.median(test_accs):.2f}%  '
          f'max {100*test_accs.max():.2f}%  sd {100*test_accs.std(ddof=1):.2f} pp', flush=True)
    print(f'   configs beating E30 ({100*e30_acc:.2f}%): {beat}/{len(rows)}', flush=True)

    # H2: significance per config
    print('\nH2 significance per config', flush=True)
    sig = 0
    for r in rows:
        b = paired_bootstrap(e30, r['probs'], test_truth, test_groups, seed=20261113)
        r['bootstrap'] = b
        if b['ci95_pp'][0] > 0:
            sig += 1
    print(f'   configs with 95% CI lower bound > 0: {sig}/{len(rows)}', flush=True)

    # H3: validation-based selection
    print('\nH3 validation-based selection (no test peeking)', flush=True)
    best_val = int(np.argmax(val_accs))
    chosen = rows[best_val]
    print(f'   best-val config: epochs={chosen["epochs"]} lr={chosen["lr"]} l2={chosen["l2"]:g}', flush=True)
    print(f'   val {100*chosen["val_acc"]:.2f}%  test {100*chosen["test_acc"]:.2f}%  '
          f'f1 {100*chosen["test_f1"]:.2f}%', flush=True)
    print(f'   delta vs E30 {100*(chosen["test_acc"]-e30_acc):+.2f} pp  '
          f'CI [{chosen["bootstrap"]["ci95_pp"][0]:+.2f}, {chosen["bootstrap"]["ci95_pp"][1]:+.2f}]', flush=True)

    hand = next(r for r in rows if r['epochs'] == 600 and r['lr'] == 0.35 and r['l2'] == 1e-3)
    payload = {
        'grid_size': len(rows),
        'e30_test_acc': e30_acc,
        'test_acc': {'min': float(test_accs.min()), 'median': float(np.median(test_accs)),
                     'max': float(test_accs.max()), 'sd': float(test_accs.std(ddof=1))},
        'configs_beating_e30': beat,
        'configs_significant': sig,
        'hand_chosen': {'epochs': hand['epochs'], 'lr': hand['lr'], 'l2': hand['l2'],
                        'val_acc': hand['val_acc'], 'test_acc': hand['test_acc'],
                        'test_f1': hand['test_f1'],
                        'delta_pp': 100 * (hand['test_acc'] - e30_acc),
                        'bootstrap': hand['bootstrap'],
                        'rank_by_test': int((test_accs > hand['test_acc']).sum()) + 1},
        'val_selected': {'epochs': chosen['epochs'], 'lr': chosen['lr'], 'l2': chosen['l2'],
                         'val_acc': chosen['val_acc'], 'test_acc': chosen['test_acc'],
                         'test_f1': chosen['test_f1'],
                         'delta_pp': 100 * (chosen['test_acc'] - e30_acc),
                         'bootstrap': chosen['bootstrap']},
        'all_configs': [{k: v for k, v in r.items() if k != 'probs'} for r in rows],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
