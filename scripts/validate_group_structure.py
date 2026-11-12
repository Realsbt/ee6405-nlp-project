"""Placebo test: does the group-constraint gain survive when the structure is destroyed?

The claim under test is that the +2.5 pp from constrained decoding comes from the
*real* group structure (each group holds HW + POL + GEN + one of GEN/HUM), not
from the constraint machinery itself.

Two placebos, each repeated many times to build a null distribution:

  A. item-level shuffle — randomly reassign items to groups, preserving group
     sizes. The decoder still imposes the same two patterns, but on random
     collections of four items.
  B. pattern-level shuffle — keep the true groups, but randomly decide which
     groups are allowed a HUM slot, preserving the 571/227 ratio.

If the gain is real, both placebos collapse to roughly the unconstrained stack.
The whole pipeline (stacker + pairwise model) is refit on shuffled validation
data inside each iteration, so the placebo is not handicapped.

Writes `.local/group_placebo.json`.
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

from scripts.constrained_decoding import (  # noqa: E402
    PATTERNS, ai_candidates, group_index, pair_features,
)
from scripts.stack_models import (  # noqa: E402
    accuracy, apply_logreg, fit_logreg, load_test, load_val,
)

OUT = ROOT / '.local/group_placebo.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
ROUNDS = 200
SEED = 20261113


def decode_with_patterns(probs: np.ndarray, index: dict[str, list[int]],
                         cand: dict[str, list[int]], pairwise,
                         allowed: dict[str, bool] | None = None) -> np.ndarray:
    """Constrained decode; `allowed` overrides which groups may take a HUM slot."""
    out = probs.copy()
    for g, members in index.items():
        if len(members) != 4:
            continue
        c = cand[g]
        p_hum = pairwise.predict_proba(pair_features(probs, c))[:, 1]
        bonus = float(np.log(max(p_hum.max(), 1e-6) / max(1.0 - p_hum.max(), 1e-6)))
        hum_item = c[int(np.argmax(p_hum))]

        patterns = PATTERNS
        if allowed is not None and not allowed[g]:
            patterns = [(0, 1, 2, 2)]

        best = None
        for pat in patterns:
            for perm in set(itertools.permutations(pat)):
                score = sum(np.log(max(probs[members[k], perm[k]], 1e-12)) for k in range(4))
                if 3 in perm:
                    score += bonus
                if best is None or score > best[0]:
                    best = (score, list(perm))
        perm = best[1]
        if 3 in perm:
            slot = perm.index(3)
            current = [k for k in range(4) if members[k] == hum_item][0]
            if current != slot:
                perm[slot], perm[current] = perm[current], perm[slot]
        for k in range(4):
            out[members[k]] = np.zeros(4)
            out[members[k], perm[k]] = 1.0
    return out


def fit_pairwise(probs: np.ndarray, index: dict[str, list[int]],
                 cand: dict[str, list[int]], truth: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    x, y = [], []
    for g, c in cand.items():
        x.append(pair_features(probs, c))
        y.extend([1 if truth[i] == 3 else 0 for i in c])
    return LogisticRegression(max_iter=2000, class_weight='balanced').fit(
        np.vstack(x), np.asarray(y))


def shuffle_items(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reassign items to groups, preserving the multiset of group sizes."""
    sizes = collections.Counter(groups)
    slots = []
    for g, n in sizes.items():
        slots.extend([g] * n)
    slots = np.asarray(slots)
    rng.shuffle(slots)
    return slots


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))

    vrecs = [json.loads(line) for line in
             (ROOT / 'experiments/hart_grouped_v1/val.jsonl').read_text(encoding='utf-8').splitlines()
             if line.strip()]
    val_groups = np.asarray([r['group_id'] for r in vrecs])

    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    val_stack = apply_logreg(w, np.stack([val_probs[n] for n in names]))
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))

    # ---- reference points -------------------------------------------------
    val_index = group_index(val_groups)
    test_index = group_index(test_groups)
    val_cand = ai_candidates(val_stack, val_index)
    test_cand = ai_candidates(test_stack, test_index)
    pairwise = fit_pairwise(val_stack, val_index, val_cand, val_truth)

    reference = {
        'stack (unconstrained)': accuracy(test_stack, test_truth),
        'constrained (true groups)': accuracy(
            decode_with_patterns(test_stack, test_index, test_cand, pairwise), test_truth),
    }

    # ---- placebo A: item-level shuffle ------------------------------------
    rng = np.random.default_rng(SEED)
    placebo_a = []
    for _ in range(ROUNDS):
        vg = shuffle_items(val_groups, rng)
        tg = shuffle_items(test_groups, rng)
        vi, ti = group_index(vg), group_index(tg)
        vc, tc = ai_candidates(val_stack, vi), ai_candidates(test_stack, ti)
        pw = fit_pairwise(val_stack, vi, vc, val_truth)
        placebo_a.append(accuracy(decode_with_patterns(test_stack, ti, tc, pw), test_truth))

    # ---- placebo B: pattern-level shuffle ---------------------------------
    true_has_hum = {g: any(test_truth[i] == 3 for i in m) for g, m in test_index.items()}
    n_hum = sum(true_has_hum.values())
    keys = list(test_index)
    placebo_b = []
    for _ in range(ROUNDS):
        chosen = set(rng.choice(keys, size=n_hum, replace=False))
        allowed = {g: (g in chosen) for g in keys}
        placebo_b.append(accuracy(
            decode_with_patterns(test_stack, test_index, test_cand, pairwise, allowed), test_truth))

    placebo_a = np.asarray(placebo_a)
    placebo_b = np.asarray(placebo_b)
    true_score = reference['constrained (true groups)']

    payload = {
        'rounds': ROUNDS,
        'reference': reference,
        'placebo_item_shuffle': {
            'mean': float(placebo_a.mean()),
            'std': float(placebo_a.std()),
            'p05': float(np.percentile(placebo_a, 5)),
            'p95': float(np.percentile(placebo_a, 95)),
            'max': float(placebo_a.max()),
            'p_value': float((placebo_a >= true_score).mean()),
        },
        'placebo_pattern_shuffle': {
            'mean': float(placebo_b.mean()),
            'std': float(placebo_b.std()),
            'p05': float(np.percentile(placebo_b, 5)),
            'p95': float(np.percentile(placebo_b, 95)),
            'max': float(placebo_b.max()),
            'p_value': float((placebo_b >= true_score).mean()),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'{"condition":34s} {"accuracy":>10s}')
    print('-' * 46)
    print(f'{"stack (unconstrained)":34s} {100*reference["stack (unconstrained)"]:9.2f}%')
    print(f'{"constrained (true groups)":34s} {100*true_score:9.2f}%')
    print()
    for label, arr in (('placebo A: item shuffle', placebo_a),
                       ('placebo B: pattern shuffle', placebo_b)):
        print(f'{label:34s} {100*arr.mean():9.2f}%  sd {100*arr.std():.2f}  '
              f'[{100*np.percentile(arr,5):.2f}, {100*np.percentile(arr,95):.2f}]  '
              f'p={(arr >= true_score).mean():.3f}')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
