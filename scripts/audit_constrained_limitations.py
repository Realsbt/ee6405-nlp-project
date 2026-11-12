"""Limitations audit for group-constrained decoding.

Six probes, each targeting a way the reported +3.53 pp could be overstated or
the method could fail outside the benchmark:

  L1. Unfair baseline. The reported delta compares a *constrained* pipeline
      against an *unconstrained* E30. Apply the same constraint to E30 and to
      every published model, then re-measure the delta. This is the honest
      comparison.

  L2. Deployment reality. The constraint needs the other three items in the
      group. Measure accuracy when only 1, 2, or 3 siblings are available, and
      when the group is a singleton (the realistic single-document case).

  L3. Pattern assumption. The decoder only allows two label multisets. Count
      how often the true group violates them, and measure the damage when the
      pattern is wrong.

  L4. Prior stability. The 571/227 split is a dataset prior. Check whether it
      holds per domain; if it drifts, the constraint is domain-dependent.

  L5. Small-group handling. The test set has one group of 2 items. Check
      whether it is silently skipped.

  L6. Cost. The constraint is O(patterns x permutations) per group; report the
      wall-clock overhead.

Writes `.local/constrained_limitations.json`.
"""
from __future__ import annotations

import collections
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.constrained_decoding import (  # noqa: E402
    PATTERNS, ai_candidates, group_index, pair_features,
)
from scripts.stack_models import (  # noqa: E402
    accuracy, apply_logreg, fit_logreg, load_test, load_val,
)

OUT = ROOT / '.local/constrained_limitations.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def decode(probs, index, cand, pairwise, allowed=None):
    out = probs.copy()
    for g, members in index.items():
        if len(members) != 4:
            continue
        c = cand[g]
        p_hum = pairwise.predict_proba(pair_features(probs, c))[:, 1]
        bonus = float(np.log(max(p_hum.max(), 1e-6) / max(1.0 - p_hum.max(), 1e-6)))
        hum_item = c[int(np.argmax(p_hum))]
        patterns = PATTERNS if allowed is None or allowed[g] else [(0, 1, 2, 2)]
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


def fit_pairwise(probs, index, cand, truth):
    from sklearn.linear_model import LogisticRegression
    x, y = [], []
    for g, c in cand.items():
        x.append(pair_features(probs, c))
        y.extend([1 if truth[i] == 3 else 0 for i in c])
    return LogisticRegression(max_iter=2000, class_weight='balanced').fit(
        np.vstack(x), np.asarray(y))


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))

    vrecs = [json.loads(l) for l in
             (ROOT / 'experiments/hart_grouped_v1/val.jsonl').read_text(encoding='utf-8').splitlines()
             if l.strip()]
    trecs = [json.loads(l) for l in
             (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
             if l.strip()]
    val_groups = np.asarray([r['group_id'] for r in vrecs])
    val_domains = np.asarray([r['domain'] for r in vrecs])
    test_domains = np.asarray([r['domain'] for r in trecs])

    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    val_stack = apply_logreg(w, np.stack([val_probs[n] for n in names]))
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))

    val_index = group_index(val_groups)
    test_index = group_index(test_groups)
    val_cand = ai_candidates(val_stack, val_index)
    test_cand = ai_candidates(test_stack, test_index)
    pairwise = fit_pairwise(val_stack, val_index, val_cand, val_truth)

    payload: dict = {}

    # ---- L1: fair baseline ------------------------------------------------
    print('L1 fair baseline', flush=True)
    camp = json.loads((ROOT / '.local/campaign_top3_val.json').read_text(encoding='utf-8'))
    e30_val = np.asarray(camp['E30'], dtype=float)
    e30_test = test_probs['E30']
    l1 = {}
    e30_val_cand = ai_candidates(e30_val, val_index)
    e30_test_cand = ai_candidates(e30_test, test_index)
    pw_e30 = fit_pairwise(e30_val, val_index, e30_val_cand, val_truth)
    l1['E30 unconstrained'] = accuracy(e30_test, test_truth)
    l1['E30 + constraint + pairwise'] = accuracy(
        decode(e30_test, test_index, e30_test_cand, pw_e30), test_truth)
    l1['stack unconstrained'] = accuracy(test_stack, test_truth)
    l1['stack + constraint + pairwise'] = accuracy(
        decode(test_stack, test_index, test_cand, pairwise), test_truth)
    for k, v in l1.items():
        print(f'   {k:34s} {100*v:6.2f}%', flush=True)
    l1['delta_E30_fair_pp'] = 100 * (l1['E30 + constraint + pairwise'] - l1['E30 unconstrained'])
    l1['delta_stack_pp'] = 100 * (l1['stack + constraint + pairwise'] - l1['stack unconstrained'])
    print(f'   fair delta on E30   {l1["delta_E30_fair_pp"]:+.2f} pp', flush=True)
    print(f'   fair delta on stack {l1["delta_stack_pp"]:+.2f} pp', flush=True)
    payload['L1_fair_baseline'] = {k: float(v) for k, v in l1.items()}

    # ---- L2: partial group availability -----------------------------------
    # With k of 4 siblings the 4-slot pattern cannot be applied. The honest
    # simulation is to decode the k available items against every length-k
    # sub-multiset of the two allowed patterns.
    print('\nL2 partial group availability', flush=True)
    l2 = {}
    rng = np.random.default_rng(20261113)
    for keep in (1, 2, 3, 4):
        allowed_k = sorted({tuple(sorted(c)) for pat in PATTERNS
                            for c in itertools.combinations(pat, keep)})
        accs = []
        for _ in range(20):
            out = test_stack.copy()
            for g, members in test_index.items():
                if len(members) != 4:
                    continue
                chosen = list(members) if keep == 4 else list(
                    rng.choice(members, size=keep, replace=False))
                best = None
                for pat in allowed_k:
                    for perm in set(itertools.permutations(pat)):
                        score = sum(np.log(max(test_stack[chosen[k], perm[k]], 1e-12))
                                    for k in range(keep))
                        if best is None or score > best[0]:
                            best = (score, list(perm))
                for k in range(keep):
                    out[chosen[k]] = np.zeros(4)
                    out[chosen[k], best[1][k]] = 1.0
            accs.append(accuracy(out, test_truth))
        l2[f'keep {keep} of 4'] = float(np.mean(accs))
        print(f'   keep {keep}/4  {100*np.mean(accs):6.2f}%  '
              f'(patterns: {len(allowed_k)})', flush=True)
    payload['L2_partial_group'] = l2

    # ---- L3: pattern violations -------------------------------------------
    print('\nL3 pattern assumption', flush=True)
    l3 = {}
    for split, groups, truth in (('val', val_groups, val_truth), ('test', test_groups, test_truth)):
        idx = group_index(groups)
        viol = 0
        total = 0
        for g, members in idx.items():
            if len(members) != 4:
                continue
            total += 1
            ms = tuple(sorted(int(truth[i]) for i in members))
            if ms not in {(0, 1, 2, 2), (0, 1, 2, 3)}:
                viol += 1
        l3[split] = {'groups': total, 'violations': viol,
                     'violation_rate': viol / total if total else 0.0}
        print(f'   {split}: {viol}/{total} groups violate the two-pattern assumption '
              f'({100*viol/total:.2f}%)', flush=True)
    payload['L3_pattern_violations'] = l3

    # ---- L4: prior stability per domain -----------------------------------
    print('\nL4 prior stability per domain', flush=True)
    l4 = {}
    for split, groups, domains, truth in (('val', val_groups, val_domains, val_truth),
                                          ('test', test_groups, test_domains, test_truth)):
        idx = group_index(groups)
        per = collections.defaultdict(lambda: [0, 0])
        for g, members in idx.items():
            if len(members) != 4:
                continue
            d = domains[members[0]]
            per[d][0] += 1
            if any(truth[i] == 3 for i in members):
                per[d][1] += 1
        l4[split] = {d: {'groups': n, 'with_hum': h, 'hum_rate': h / n}
                     for d, (n, h) in per.items()}
        for d, v in l4[split].items():
            print(f'   {split} {d:10s} {v["with_hum"]:4d}/{v["groups"]:4d} '
                  f'= {100*v["hum_rate"]:5.2f}%', flush=True)
    payload['L4_prior_by_domain'] = l4

    # ---- L5: small groups -------------------------------------------------
    print('\nL5 small groups', flush=True)
    sizes = collections.Counter(len(m) for m in test_index.values())
    l5 = {'size_histogram': {str(k): v for k, v in sorted(sizes.items())}}
    small = [g for g, m in test_index.items() if len(m) != 4]
    l5['non_four_groups'] = len(small)
    l5['items_in_non_four_groups'] = sum(len(test_index[g]) for g in small)
    print(f'   group sizes: {dict(sorted(sizes.items()))}', flush=True)
    print(f'   {len(small)} group(s) not of size 4, '
          f'{l5["items_in_non_four_groups"]} items silently skipped', flush=True)
    payload['L5_small_groups'] = l5

    # ---- L6: cost ---------------------------------------------------------
    print('\nL6 cost', flush=True)
    t0 = time.perf_counter()
    for _ in range(5):
        decode(test_stack, test_index, test_cand, pairwise)
    decode_s = (time.perf_counter() - t0) / 5
    l6 = {'decode_seconds_per_3194_items': decode_s,
          'groups': len(test_index),
          'patterns': len(PATTERNS),
          'permutations_per_pattern': 12}
    print(f'   decode {decode_s*1000:.1f} ms for {len(test_index)} groups', flush=True)
    payload['L6_cost'] = l6

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
