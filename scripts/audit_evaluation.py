"""Credibility audit for the revised evaluation protocol.

Addresses four questions the earlier report answered too confidently:

  2. Confidence intervals and sample sizes for the human-rewrite subgroup, so a
     57-item slice is not over-interpreted.
  3. Whether the "irreducible error" and "ceiling" framing survives scrutiny.
  4. Whether the two-stage generator gap is a real second-stage effect or an
     artefact of sample composition (domain, length, first stage).

Writes `.local/audit.json`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SHORT = ['HW', 'POL', 'GEN', 'HUM']
HART = ROOT / 'assets/hart_official'
SEED = 20261112


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly for small n and extreme p."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval, for comparison with Wilson on tiny slices."""
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return (lo, hi)


def load_frame():
    ids, truth = [], []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            ids.append(record['item_id'])
            truth.append(LABELS.index(record['true_label']))
    wanted = set(ids)
    meta = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    meta[record['item_id']] = record
    probabilities = np.asarray(json.loads((ROOT / '.local/campaign_top3_published_test.json')
                                          .read_text(encoding='utf-8'))['E30']['probabilities'])
    return ids, np.asarray(truth), probabilities, meta


# --------------------------------------------------------------------------
# 2. small-sample honesty for the humanized subgroups
# --------------------------------------------------------------------------
def subgroup_intervals(ids, truth, probabilities, meta) -> dict:
    pred = probabilities.argmax(1)
    kinds = np.asarray([meta[i].get('model_name', 'N/A') for i in ids])

    out = {}
    for kind in ('tool', 'human'):
        mask = (truth == 3) & (kinds == kind)
        n = int(mask.sum())
        correct = int((pred[mask] == 3).sum())
        w_lo, w_hi = wilson(correct, n)
        c_lo, c_hi = clopper_pearson(correct, n)
        confusion = {LABELS[c]: int((pred[mask] == c).sum()) for c in range(4)}
        out[kind] = {
            'n': n, 'correct': correct, 'accuracy': correct / n if n else 0.0,
            'wilson95': [w_lo, w_hi], 'clopper_pearson95': [c_lo, c_hi],
            'interval_width_pp': 100 * (w_hi - w_lo),
            'confusion': confusion,
        }

    # does the tool/human difference survive its own uncertainty?
    a, b = out['tool'], out['human']
    diff = a['accuracy'] - b['accuracy']
    # Newcombe hybrid score interval for the difference of two proportions
    p1, n1 = a['accuracy'], a['n']
    p2, n2 = b['accuracy'], b['n']
    l1, u1 = wilson(a['correct'], n1)
    l2, u2 = wilson(b['correct'], n2)
    lo = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    out['difference'] = {'delta': diff, 'newcombe95': [float(lo), float(hi)],
                         'excludes_zero': bool(lo > 0 or hi < 0)}

    # how many human-rewrite items would be needed to halve the interval?
    width = out['human']['interval_width_pp']
    out['human']['n_for_half_width'] = int(np.ceil(a['n'] * 0 + 57 * (width / (width / 2)) ** 2))
    return out


# --------------------------------------------------------------------------
# 3. is the "ceiling" claim defensible?
# --------------------------------------------------------------------------
def ceiling_sensitivity(ids, truth, probabilities, meta, separability) -> dict:
    """Recompute the ceiling under a range of assumptions about recoverability."""
    pred = probabilities.argmax(1)
    kinds = np.asarray([meta[i].get('model_name', 'N/A') for i in ids])
    total_errors = int((pred != truth).sum())
    human_mask = (truth == 3) & (kinds == 'human')
    human_errors = int(((pred != truth) & human_mask).sum())
    n_human = int(human_mask.sum())

    recalls = {
        'observed binary classifier (28%)': 0.28,
        'optimistic 50%': 0.50,
        'optimistic 75%': 0.75,
        'perfect (100%)': 1.00,
    }
    rows = {}
    for name, recall in recalls.items():
        recovered = recall * n_human
        remaining = total_errors - recovered
        rows[name] = {
            'recall': recall,
            'recovered_errors': recovered,
            'remaining_errors': remaining,
            'accuracy': (len(truth) - remaining) / len(truth),
        }

    # the ceiling also assumes every non-human-rewrite error is fixable, which
    # is itself unproven; report the ceiling with that caveat made explicit
    return {
        'total_errors': total_errors,
        'human_rewrite_items': n_human,
        'human_rewrite_errors': human_errors,
        'scenarios': rows,
        'caveat': ('the ceiling assumes all non-human-rewrite errors are fixable '
                   'and that human-rewrite recall can be raised to the stated level; '
                   'neither is demonstrated'),
    }


# --------------------------------------------------------------------------
# 4. two-stage generator: is the second-stage effect confounded?
# --------------------------------------------------------------------------
def two_stage_control(ids, truth, probabilities, meta) -> dict:
    pred = probabilities.argmax(1)
    names = [meta[i].get('model_name', 'N/A') for i in ids]
    domains = [meta[i]['group_id'].split('-', 1)[0] for i in ids]
    lengths = np.asarray([len(meta[i]['article']) for i in ids])

    rows = []
    for k, name in enumerate(names):
        if '_THEN_' not in name:
            continue
        first, second = name.split('_THEN_', 1)
        rows.append({'k': k, 'first': first, 'second': second,
                     'domain': domains[k], 'length': lengths[k],
                     'correct': int(pred[k] == truth[k])})

    def summarise(subset):
        if not subset:
            return {'n': 0, 'accuracy': None}
        return {'n': len(subset),
                'accuracy': float(np.mean([r['correct'] for r in subset]))}

    by_second = collections.defaultdict(list)
    for r in rows:
        by_second[r['second']].append(r)
    overall = {k: summarise(v) for k, v in by_second.items()}

    # composition check: are the gemini-second items different in domain/length?
    gem = [r for r in rows if r['second'].startswith('gemini')]
    other = [r for r in rows if not r['second'].startswith('gemini')]
    composition = {
        'gemini_second': {
            'n': len(gem),
            'domain_mix': dict(collections.Counter(r['domain'] for r in gem)),
            'median_length': float(np.median([r['length'] for r in gem])) if gem else None,
            'first_stage_mix': dict(collections.Counter(r['first'] for r in gem)),
        },
        'other_second': {
            'n': len(other),
            'domain_mix': dict(collections.Counter(r['domain'] for r in other)),
            'median_length': float(np.median([r['length'] for r in other])) if other else None,
            'first_stage_mix': dict(collections.Counter(r['first'] for r in other)),
        },
    }

    # stratified comparison: hold domain and first stage fixed
    strata = collections.defaultdict(lambda: {'gemini': [], 'other': []})
    for r in rows:
        key = (r['domain'], r['first'])
        bucket = 'gemini' if r['second'].startswith('gemini') else 'other'
        strata[key][bucket].append(r['correct'])

    stratified = []
    for key, bucket in sorted(strata.items()):
        if len(bucket['gemini']) < 3 or len(bucket['other']) < 3:
            continue
        stratified.append({
            'domain': key[0], 'first_stage': key[1],
            'n_gemini': len(bucket['gemini']),
            'acc_gemini': float(np.mean(bucket['gemini'])),
            'n_other': len(bucket['other']),
            'acc_other': float(np.mean(bucket['other'])),
            'gap_pp': 100 * (float(np.mean(bucket['other'])) - float(np.mean(bucket['gemini']))),
        })

    # length-matched comparison
    gem_len = np.asarray([r['length'] for r in gem])
    other_len = np.asarray([r['length'] for r in other])
    matched = []
    if len(gem) and len(other):
        for lo, hi in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 10 ** 9)):
            g = [r['correct'] for r in gem if lo <= r['length'] < hi]
            o = [r['correct'] for r in other if lo <= r['length'] < hi]
            if len(g) >= 5 and len(o) >= 5:
                matched.append({'bucket': f'{lo}-{hi}', 'n_gemini': len(g),
                                'acc_gemini': float(np.mean(g)), 'n_other': len(o),
                                'acc_other': float(np.mean(o)),
                                'gap_pp': 100 * (float(np.mean(o)) - float(np.mean(g)))})

    # permutation test on the raw gap
    rng = np.random.default_rng(SEED)
    gem_correct = np.asarray([r['correct'] for r in gem], dtype=float)
    other_correct = np.asarray([r['correct'] for r in other], dtype=float)
    observed = other_correct.mean() - gem_correct.mean()
    pooled = np.concatenate([gem_correct, other_correct])
    n_gem = len(gem_correct)
    null = np.empty(5000)
    for i in range(5000):
        rng.shuffle(pooled)
        null[i] = pooled[n_gem:].mean() - pooled[:n_gem].mean()
    p_value = float((np.abs(null) >= abs(observed)).mean())

    return {
        'overall_by_second': overall,
        'composition': composition,
        'stratified_by_domain_and_first': stratified,
        'length_matched': matched,
        'permutation': {'observed_gap_pp': 100 * float(observed), 'p_value': p_value,
                        'rounds': 5000},
    }


# --------------------------------------------------------------------------
# 5. four-class remains the primary task
# --------------------------------------------------------------------------
def primary_task_note() -> dict:
    return {
        'primary': 'four-class (human_written / human_ai_polished / ai_generated / ai_humanized)',
        'auxiliary': ['binary (human vs machine-involved)',
                      'three-class (humanized excluded)',
                      'five-class (humanized split into tool / human)'],
        'rationale': ('the four-class label set is the task the dataset defines and the '
                      'one the published baselines report; the coarser views are '
                      'diagnostics that explain where the four-class loss comes from, '
                      'not replacement headline numbers'),
    }


def main() -> None:
    ids, truth, probabilities, meta = load_frame()
    separability = json.loads((ROOT / '.local/separability.json').read_text(encoding='utf-8'))

    print('[2] humanized subgroup intervals')
    subgroups = subgroup_intervals(ids, truth, probabilities, meta)
    for kind in ('tool', 'human'):
        row = subgroups[kind]
        print(f'  {kind:6} n={row["n"]:3}  acc {100*row["accuracy"]:6.2f}%  '
              f'Wilson95 [{100*row["wilson95"][0]:.1f}, {100*row["wilson95"][1]:.1f}]  '
              f'width {row["interval_width_pp"]:.1f} pp')
        print(f'         confusion {row["confusion"]}')
    diff = subgroups['difference']
    print(f'  difference {100*diff["delta"]:+.2f} pp  '
          f'Newcombe95 [{100*diff["newcombe95"][0]:+.1f}, {100*diff["newcombe95"][1]:+.1f}]  '
          f'excludes zero: {diff["excludes_zero"]}')

    print('\n[3] ceiling sensitivity')
    ceiling = ceiling_sensitivity(ids, truth, probabilities, meta, separability)
    for name, row in ceiling['scenarios'].items():
        print(f'  {name:34} recovered {row["recovered_errors"]:5.1f}  '
              f'accuracy {100*row["accuracy"]:6.2f}%')

    print('\n[4] two-stage control')
    control = two_stage_control(ids, truth, probabilities, meta)
    for name, row in sorted(control['overall_by_second'].items(), key=lambda kv: kv[1]['accuracy']):
        print(f'  second={name:34} n={row["n"]:4}  acc {100*row["accuracy"]:6.2f}%')
    print(f'  permutation p={control["permutation"]["p_value"]:.4f}  '
          f'gap {control["permutation"]["observed_gap_pp"]:+.2f} pp')
    print('  stratified (domain x first stage):')
    for row in control['stratified_by_domain_and_first']:
        print(f'    {row["domain"]:8} first={row["first_stage"][:22]:22} '
              f'gemini n={row["n_gemini"]:3} {100*row["acc_gemini"]:6.2f}%  '
              f'other n={row["n_other"]:3} {100*row["acc_other"]:6.2f}%  '
              f'gap {row["gap_pp"]:+6.2f} pp')
    print('  length-matched:')
    for row in control['length_matched']:
        print(f'    {row["bucket"]:12} gemini n={row["n_gemini"]:3} {100*row["acc_gemini"]:6.2f}%  '
              f'other n={row["n_other"]:3} {100*row["acc_other"]:6.2f}%  gap {row["gap_pp"]:+6.2f} pp')

    payload = {
        'subgroups': subgroups,
        'ceiling': ceiling,
        'two_stage': control,
        'primary_task': primary_task_note(),
    }
    (ROOT / '.local/audit.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print('\nwrote .local/audit.json')


if __name__ == '__main__':
    main()
