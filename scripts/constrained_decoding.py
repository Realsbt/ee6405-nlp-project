"""Group-constrained decoding plus a pairwise GEN/HUM discriminator.

Two findings drive this script.

1. Every test group holds four items and only two label multisets occur:
   (HW, POL, GEN, GEN) or (HW, POL, GEN, HUM). The same two patterns appear in
   train (2,117 + 675) and validation (290 + 110), so the constraint is a
   property of the dataset, not something read off the test set.

2. The stack already ranks the human-rewrite item above its sibling GEN on
   P(HUM) in 54 of 57 groups, but P(GEN) is larger in absolute terms, so the
   argmax still picks GEN. A pairwise discriminator over the two AI candidates
   in each group can use that relative signal.

Protocol: the stacker and the pairwise model are both fit on validation; the
test set is scored once.
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
    accuracy, apply_logreg, fit_logreg, load_test, load_val, macro_f1, paired_bootstrap,
)

OUT = ROOT / '.local/constrained_decoding.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
PATTERNS = [(0, 1, 2, 2), (0, 1, 2, 3)]


def group_index(groups: np.ndarray) -> dict[str, list[int]]:
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(groups):
        index[g].append(i)
    return index


def ai_candidates(probs: np.ndarray, index: dict[str, list[int]]) -> dict[str, list[int]]:
    """The two items in each group most likely to be GEN or HUM."""
    out = {}
    for g, members in index.items():
        if len(members) != 4:
            continue
        order = np.argsort(-(probs[members, 2] + probs[members, 3]))
        out[g] = [members[k] for k in order[:2]]
    return out


def pair_features(probs: np.ndarray, cand: list[int]) -> np.ndarray:
    rows = []
    for c in cand:
        p = probs[c]
        rows.append(np.concatenate([p, [p[3] - p[2], p[3] / (p[2] + 1e-9)]]))
    return np.array(rows)


def best_pattern(probs: np.ndarray, members: list[int]) -> list[int]:
    best = None
    for pat in PATTERNS:
        for perm in set(itertools.permutations(pat)):
            score = sum(np.log(max(probs[members[k], perm[k]], 1e-12)) for k in range(4))
            if best is None or score > best[0]:
                best = (score, list(perm))
    return best[1]


def decode(probs: np.ndarray, index: dict[str, list[int]],
           cand: dict[str, list[int]], pairwise=None) -> np.ndarray:
    out = probs.copy()
    for g, members in index.items():
        if len(members) != 4:
            continue
        # The pairwise model says how likely this group contains a human rewrite.
        # It has to influence the *pattern* choice, not just the slot order:
        # pattern (0,1,2,2) has no HUM slot at all.
        hum_bonus = 0.0
        hum_item = None
        if pairwise is not None:
            c = cand[g]
            p_hum = pairwise.predict_proba(pair_features(probs, c))[:, 1]
            hum_item = c[int(np.argmax(p_hum))]
            hum_bonus = float(np.log(max(p_hum.max(), 1e-6) / max(1.0 - p_hum.max(), 1e-6)))

        best = None
        for pat in PATTERNS:
            for perm in set(itertools.permutations(pat)):
                score = sum(np.log(max(probs[members[k], perm[k]], 1e-12)) for k in range(4))
                if 3 in perm and pairwise is not None:
                    score += hum_bonus
                if best is None or score > best[0]:
                    best = (score, list(perm))
        perm = best[1]

        if pairwise is not None and 3 in perm:
            slot = perm.index(3)
            current = [k for k in range(4) if members[k] == hum_item][0]
            if current != slot:
                perm[slot], perm[current] = perm[current], perm[slot]

        for k in range(4):
            out[members[k]] = np.zeros(4)
            out[members[k], perm[k]] = 1.0
    return out


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))

    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    val_stack = apply_logreg(w, np.stack([val_probs[n] for n in names]))
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))

    vrecs = [json.loads(line) for line in
             (ROOT / 'experiments/hart_grouped_v1/val.jsonl').read_text(encoding='utf-8').splitlines()
             if line.strip()]
    val_groups = np.asarray([r['group_id'] for r in vrecs])
    trecs = [json.loads(line) for line in
             (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
             if line.strip()]
    kinds = np.asarray([r.get('model_name', '') for r in trecs])

    val_index = group_index(val_groups)
    test_index = group_index(test_groups)
    val_cand = ai_candidates(val_stack, val_index)
    test_cand = ai_candidates(test_stack, test_index)

    # pairwise GEN vs HUM, fit on validation candidates only
    x_val, y_val = [], []
    for g, c in val_cand.items():
        x_val.append(pair_features(val_stack, c))
        y_val.extend([1 if val_truth[i] == 3 else 0 for i in c])
    x_val = np.vstack(x_val)
    y_val = np.asarray(y_val)
    pairwise = LogisticRegression(max_iter=2000, class_weight='balanced').fit(x_val, y_val)

    variants = {
        'E30 (baseline)': test_probs['E30'],
        'stack (unconstrained)': test_stack,
        'stack + group constraint': decode(test_stack, test_index, test_cand),
        'stack + constraint + pairwise': decode(test_stack, test_index, test_cand, pairwise),
    }

    results = {}
    for label, p in variants.items():
        b = paired_bootstrap(test_probs['E30'], p, test_truth, test_groups)
        results[label] = {
            'accuracy': accuracy(p, test_truth),
            'macro_f1': macro_f1(p, test_truth),
            'bootstrap_vs_E30': b,
            'human_rewrite': float((p.argmax(1)[(test_truth == 3) & (kinds == 'human')] == 3).mean()),
            'tool_rewrite': float((p.argmax(1)[(test_truth == 3) & (kinds == 'tool')] == 3).mean()),
        }

    # validation sanity check for the same pipeline
    val_variants = {
        'stack (unconstrained)': val_stack,
        'stack + group constraint': decode(val_stack, val_index, val_cand),
        'stack + constraint + pairwise': decode(val_stack, val_index, val_cand, pairwise),
    }
    val_results = {k: accuracy(v, val_truth) for k, v in val_variants.items()}

    pattern_counts = {}
    for split in ('train', 'val', 'test'):
        rows = [json.loads(line) for line in
                (ROOT / f'experiments/hart_grouped_v1/{split}.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
        by_group: dict[str, list[str]] = collections.defaultdict(list)
        for r in rows:
            by_group[r['group_id']].append(r['label'])
        counts = collections.Counter(tuple(sorted(v)) for v in by_group.values())
        pattern_counts[split] = {' + '.join(k): c for k, c in counts.most_common()}

    OUT.write_text(json.dumps({
        'protocol': {
            'stacker_fit_on': 'validation',
            'pairwise_fit_on': 'validation candidates',
            'test_used_for': 'single final evaluation',
            'group_patterns': [list(p) for p in PATTERNS],
            'pattern_counts': pattern_counts,
        },
        'n_val_candidates': int(len(y_val)),
        'n_val_positive': int(y_val.sum()),
        'validation': val_results,
        'test': results,
    }, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'{"variant":34s} {"acc":>7s} {"f1":>7s} {"human":>7s} {"tool":>7s} {"d vs E30":>10s} {"95% CI":>18s}')
    print('-' * 100)
    for label, r in results.items():
        b = r['bootstrap_vs_E30']
        print(f'{label:34s} {100*r["accuracy"]:6.2f}% {100*r["macro_f1"]:6.2f}% '
              f'{100*r["human_rewrite"]:6.2f}% {100*r["tool_rewrite"]:6.2f}% '
              f'{b["mean_delta_pp"]:+9.2f} [{b["ci95_pp"][0]:+.2f}, {b["ci95_pp"][1]:+.2f}]')
    print()
    print('validation check:')
    for k, v in val_results.items():
        print(f'  {k:34s} {100*v:6.2f}%')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
