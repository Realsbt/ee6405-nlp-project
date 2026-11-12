"""How was the best single-document model obtained, and what is wrong with it?

The deployable model is a multinomial logistic-regression stacker over the 13
published models' probabilities, fit on the campaign validation split. This
script audits that construction.

  A1. Hyperparameter sensitivity. The stacker's epochs/lr/l2 were fixed by hand,
      never tuned. Sweep them and report the spread of test accuracy. If the
      spread is comparable to the reported gain over E30, the gain is not
      distinguishable from an arbitrary choice.

  A2. Selection circularity. The 13 checkpoints were selected on the same
      validation split the stacker is fit on. Measure the optimism by fitting
      the stacker on half of validation and scoring the other half.

  A3. Seed redundancy. Three seeds per configuration means the stacker sees
      near-duplicate inputs. Check the correlation between same-config seeds
      and whether dropping to one seed per config changes the result.

  A4. Class-prior shift. Validation has 110 humanized items, test has 227.
      Check whether the stacker's behaviour depends on that prior.

Writes `.local/stacker_audit.json`.
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

from scripts.stack_models import (  # noqa: E402
    accuracy, apply_logreg, fit_logreg, load_test, load_val, macro_f1,
)

OUT = ROOT / '.local/stacker_audit.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SEED = 20261113


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))
    val_stack = np.stack([val_probs[n] for n in names])
    test_stack = np.stack([test_probs[n] for n in names])
    e30 = test_probs['E30']
    payload: dict = {'models': names}

    # ---- A1: hyperparameter sensitivity ----------------------------------
    print('A1 hyperparameter sensitivity', flush=True)
    grid = list(itertools.product((200, 600, 1200, 3000), (0.1, 0.35, 1.0), (1e-4, 1e-3, 1e-2)))
    rows = []
    for epochs, lr, l2 in grid:
        w = fit_logreg(val_stack, val_truth, epochs=epochs, lr=lr, l2=l2)
        pv = apply_logreg(w, val_stack)
        pt = apply_logreg(w, test_stack)
        rows.append({'epochs': epochs, 'lr': lr, 'l2': l2,
                     'val_acc': accuracy(pv, val_truth),
                     'test_acc': accuracy(pt, test_truth),
                     'test_f1': macro_f1(pt, test_truth)})
    test_accs = np.array([r['test_acc'] for r in rows])
    chosen = next(r for r in rows if r['epochs'] == 600 and r['lr'] == 0.35 and r['l2'] == 1e-3)
    payload['A1_hyperparameters'] = {
        'grid_size': len(rows),
        'test_acc_min': float(test_accs.min()),
        'test_acc_max': float(test_accs.max()),
        'test_acc_mean': float(test_accs.mean()),
        'test_acc_sd': float(test_accs.std(ddof=1)),
        'chosen': chosen,
        'chosen_rank': int((test_accs > chosen['test_acc']).sum()) + 1,
        'e30_test_acc': accuracy(e30, test_truth),
        'gain_over_e30_pp': 100 * (chosen['test_acc'] - accuracy(e30, test_truth)),
        'spread_pp': 100 * (test_accs.max() - test_accs.min()),
    }
    print(f'   {len(rows)} configs, test acc {100*test_accs.min():.2f}%–{100*test_accs.max():.2f}% '
          f'(sd {100*test_accs.std(ddof=1):.2f} pp)', flush=True)
    print(f'   chosen config ranks {payload["A1_hyperparameters"]["chosen_rank"]}/{len(rows)}', flush=True)
    print(f'   gain over E30 {payload["A1_hyperparameters"]["gain_over_e30_pp"]:+.2f} pp, '
          f'spread {payload["A1_hyperparameters"]["spread_pp"]:.2f} pp', flush=True)

    # ---- A2: selection circularity ---------------------------------------
    print('\nA2 selection circularity (half-val fit, half-val score)', flush=True)
    rng = np.random.default_rng(SEED)
    vgroups = np.asarray([json.loads(l)['group_id'] for l in
                          (ROOT / 'experiments/hart_grouped_v1/val.jsonl')
                          .read_text(encoding='utf-8').splitlines() if l.strip()])
    uniq = np.array(sorted(set(vgroups)))
    rng.shuffle(uniq)
    half = set(uniq[:len(uniq) // 2])
    mask_a = np.array([g in half for g in vgroups])
    mask_b = ~mask_a
    w_ab = fit_logreg(val_stack[:, mask_a], val_truth[mask_a])
    w_ba = fit_logreg(val_stack[:, mask_b], val_truth[mask_b])
    in_a = accuracy(apply_logreg(w_ab, val_stack[:, mask_a]), val_truth[mask_a])
    out_a = accuracy(apply_logreg(w_ab, val_stack[:, mask_b]), val_truth[mask_b])
    in_b = accuracy(apply_logreg(w_ba, val_stack[:, mask_b]), val_truth[mask_b])
    out_b = accuracy(apply_logreg(w_ba, val_stack[:, mask_a]), val_truth[mask_a])
    payload['A2_circularity'] = {
        'fit_half_in_sample': float((in_a + in_b) / 2),
        'fit_half_out_of_sample': float((out_a + out_b) / 2),
        'optimism_pp': 100 * ((in_a + in_b) / 2 - (out_a + out_b) / 2),
    }
    print(f'   in-sample {100*(in_a+in_b)/2:.2f}%  out-of-sample {100*(out_a+out_b)/2:.2f}%  '
          f'optimism {100*((in_a+in_b)/2-(out_a+out_b)/2):+.2f} pp', flush=True)

    # ---- A3: seed redundancy ---------------------------------------------
    print('\nA3 seed redundancy', flush=True)
    configs = collections.defaultdict(list)
    for n in names:
        if n == 'tfidf_lr_v1':
            continue
        base = n.rsplit('_s', 1)[0]
        configs[base].append(n)
    corrs = []
    for base, members in configs.items():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members, 2):
            pa = val_probs[a].ravel()
            pb = val_probs[b].ravel()
            corrs.append(float(np.corrcoef(pa, pb)[0, 1]))
    one_per = [sorted(m)[0] for m in configs.values()] + ['tfidf_lr_v1']
    w_one = fit_logreg(np.stack([val_probs[n] for n in one_per]), val_truth)
    p_one = apply_logreg(w_one, np.stack([test_probs[n] for n in one_per]))
    payload['A3_seed_redundancy'] = {
        'mean_corr_same_config_seeds': float(np.mean(corrs)),
        'min_corr': float(np.min(corrs)),
        'max_corr': float(np.max(corrs)),
        'one_seed_per_config_test_acc': accuracy(p_one, test_truth),
        'one_seed_per_config_test_f1': macro_f1(p_one, test_truth),
        'n_models_one_per_config': len(one_per),
    }
    print(f'   same-config seed correlation {np.mean(corrs):.4f} '
          f'[{np.min(corrs):.4f}, {np.max(corrs):.4f}]', flush=True)
    print(f'   one seed per config ({len(one_per)} models): '
          f'{100*accuracy(p_one, test_truth):.2f}% / {100*macro_f1(p_one, test_truth):.2f}%', flush=True)

    # ---- A4: class prior ------------------------------------------------
    print('\nA4 class prior', flush=True)
    val_dist = collections.Counter(val_truth.tolist())
    test_dist = collections.Counter(test_truth.tolist())
    payload['A4_class_prior'] = {
        'val': {LABELS[k]: v for k, v in sorted(val_dist.items())},
        'test': {LABELS[k]: v for k, v in sorted(test_dist.items())},
        'val_hum_rate': val_dist[3] / len(val_truth),
        'test_hum_rate': test_dist[3] / len(test_truth),
    }
    for k in range(4):
        print(f'   {LABELS[k]:20s} val {val_dist[k]:5d} ({100*val_dist[k]/len(val_truth):5.2f}%)  '
              f'test {test_dist[k]:5d} ({100*test_dist[k]/len(test_truth):5.2f}%)', flush=True)

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
