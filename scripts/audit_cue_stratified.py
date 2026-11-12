"""Stage A: does the comma association survive stratification?

The earlier analysis reported that GEN|HUM confusion rises from 12.58% to
44.12% when the comma cue is atypical. That is a marginal association. It could
be a proxy for something else: human vs tool rewrite, domain, or generator.

This script re-runs the comparison inside every stratum and reports whether the
gap survives. If the gap collapses inside strata, the comma cue is a proxy and
must not be treated as a repair target.

Also reports the exact group-size distribution, because 798 x 4 = 3192, not
3194, and the earlier report did not state the exception.

Writes `.local/cue_stratified.json`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stack_models import (  # noqa: E402
    apply_logreg, fit_logreg, load_test, load_val,
)

OUT = ROOT / '.local/cue_stratified.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def comma_density(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text)
    return text.count(',') / max(len(words), 1) * 100


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))
    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))
    pred = test_stack.argmax(1)

    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]

    payload: dict = {}

    # ---- group size distribution (the 3194 vs 3192 correction) -----------
    sizes = collections.Counter(collections.Counter(r['group_id'] for r in recs).values())
    payload['group_sizes'] = {str(k): v for k, v in sorted(sizes.items())}
    payload['n_items'] = len(recs)
    payload['n_groups'] = sum(sizes.values())
    print(f'group sizes: {dict(sorted(sizes.items()))}  '
          f'({sum(sizes.values())} groups, {len(recs)} items)', flush=True)

    # ---- build GEN|HUM group table ---------------------------------------
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(test_groups):
        by_group[g].append(i)

    rows = []
    for g, idx in by_group.items():
        if len(idx) != 4:
            continue
        by_label = {test_truth[i]: i for i in idx}
        if 2 not in by_label or 3 not in by_label:
            continue
        ig, ih = by_label[2], by_label[3]
        cg = comma_density(recs[ig]['article'])
        ch = comma_density(recs[ih]['article'])
        # population direction: GEN has more commas than HUM on average
        present = cg > ch
        confused = (pred[ig] == 3) or (pred[ih] == 2)
        hum_rec = recs[ih]
        rows.append({
            'group': g,
            'present': present,
            'confused': confused,
            'domain': hum_rec['domain'],
            'hum_kind': hum_rec.get('model_name'),
            'gen_model': recs[ig].get('model_name'),
        })
    print(f'{len(rows)} groups contain both GEN and HUM', flush=True)

    def rate(subset):
        n = len(subset)
        k = sum(1 for r in subset if r['confused'])
        lo, hi = wilson(k, n)
        return {'n': n, 'confused': k, 'rate': k / n if n else 0.0,
                'wilson95': [lo, hi]}

    overall_present = rate([r for r in rows if r['present']])
    overall_absent = rate([r for r in rows if not r['present']])
    payload['overall'] = {'present': overall_present, 'absent': overall_absent,
                          'gap_pp': 100 * (overall_absent['rate'] - overall_present['rate'])}
    print(f'\noverall  present {100*overall_present["rate"]:5.2f}% (n={overall_present["n"]})  '
          f'absent {100*overall_absent["rate"]:5.2f}% (n={overall_absent["n"]})  '
          f'gap {payload["overall"]["gap_pp"]:+.2f} pp', flush=True)

    # ---- stratify --------------------------------------------------------
    strata = {
        'hum_kind': lambda r: r['hum_kind'],
        'domain': lambda r: r['domain'],
        'gen_family': lambda r: (r['gen_model'] or '').split('_THEN_')[0],
        'two_stage': lambda r: 'two-stage' if '_THEN_' in (r['gen_model'] or '') else 'single',
    }
    payload['strata'] = {}
    for name, key in strata.items():
        print(f'\n--- stratified by {name} ---', flush=True)
        groups = collections.defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        table = {}
        for k, sub in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            p = rate([r for r in sub if r['present']])
            a = rate([r for r in sub if not r['present']])
            gap = 100 * (a['rate'] - p['rate']) if p['n'] and a['n'] else None
            table[str(k)] = {'present': p, 'absent': a, 'gap_pp': gap}
            gs = f'{gap:+6.2f}' if gap is not None else '   n/a'
            print(f'   {str(k)[:34]:34s} present {100*p["rate"]:5.2f}% (n={p["n"]:3d})  '
                  f'absent {100*a["rate"]:5.2f}% (n={a["n"]:3d})  gap {gs} pp', flush=True)
        payload['strata'][name] = table

    # ---- how many strata keep the direction ------------------------------
    summary = {}
    for name, table in payload['strata'].items():
        valid = [v for v in table.values() if v['gap_pp'] is not None]
        keep = sum(1 for v in valid if v['gap_pp'] > 0)
        summary[name] = {'strata_with_both_cells': len(valid),
                         'strata_keeping_direction': keep}
        print(f'\n{name}: {keep}/{len(valid)} strata keep the direction', flush=True)
    payload['direction_summary'] = summary

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
