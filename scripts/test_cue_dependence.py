"""Does the classifier fail exactly when the discriminating cue is absent?

`analyze_error_flow.py` showed that cue separability tracks error rate: pairs
that are easy to separate in-group (HW|GEN, 97%) have few errors, pairs that
are hard (GEN|HUM, 70%) have many. That is a correlation across pairs. This
script tests the mechanism at the item level.

For each confused pair, take the cue that best separates the two versions
in-group, then split the groups into "cue present" and "cue absent". If the
classifier relies on that cue, its error rate should be much higher in the
"absent" half.

Also tests whether the model's probability actually tracks the cue, by
correlating the cue difference with the model's log-odds difference.

Writes `.local/cue_dependence.json`.
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

OUT = ROOT / '.local/cue_dependence.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SHORT = ['HW', 'POL', 'GEN', 'HUM']


def cue_value(text: str, name: str) -> float:
    words = re.findall(r"[A-Za-z']+", text)
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n = max(len(words), 1)
    if name == 'comma_per_100w':
        return text.count(',') / n * 100
    if name == 'long_word_rate':
        return sum(1 for w in words if len(w) > 8) / n
    if name == 'mean_word_len':
        return float(np.mean([len(w) for w in words])) if words else 0.0
    if name == 'length':
        return float(len(text))
    if name == 'mean_sent_len':
        lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0]
        return float(np.mean(lens))
    if name == 'ttr':
        return len(set(words)) / n
    raise KeyError(name)


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

    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(test_groups):
        by_group[g].append(i)

    # The cue that best separates each pair, taken from the earlier analysis.
    PAIRS = {
        'GEN|HUM': ('comma_per_100w', 2, 3),
        'POL|GEN': ('length', 1, 2),
        'HW|POL': ('mean_word_len', 0, 1),
    }

    payload: dict = {}

    for pair, (cue, a, b) in PAIRS.items():
        print(f'\n{pair}  cue = {cue}', flush=True)
        rows = []
        for g, idx in by_group.items():
            if len(idx) != 4:
                continue
            by_label = {test_truth[i]: i for i in idx}
            if a not in by_label or b not in by_label:
                continue
            ia, ib = by_label[a], by_label[b]
            va = cue_value(recs[ia]['article'], cue)
            vb = cue_value(recs[ib]['article'], cue)
            # "cue present" = the cue orders the two versions the way the
            # population mean does (a above b, or b above a).
            rows.append({'group': g, 'ia': ia, 'ib': ib, 'va': va, 'vb': vb})

        if not rows:
            continue
        diffs = np.array([r['va'] - r['vb'] for r in rows])
        # Direction: which side is larger on average.
        sign = 1.0 if diffs.mean() > 0 else -1.0
        present = [r for r, d in zip(rows, diffs) if sign * d > 0]
        absent = [r for r, d in zip(rows, diffs) if sign * d <= 0]

        def pair_error(subset):
            """Fraction of these groups where the classifier confuses a and b."""
            if not subset:
                return None, 0
            wrong = 0
            for r in subset:
                pa, pb = pred[r['ia']], pred[r['ib']]
                # count a group as confused if either item is misassigned to
                # the other's class
                if pa == b or pb == a:
                    wrong += 1
            return wrong / len(subset), len(subset)

        e_present, n_present = pair_error(present)
        e_absent, n_absent = pair_error(absent)
        print(f'   cue present {n_present:4d} groups  confusion {100*e_present:5.2f}%', flush=True)
        print(f'   cue absent  {n_absent:4d} groups  confusion {100*e_absent:5.2f}%', flush=True)
        if e_absent is not None and e_present is not None:
            print(f'   gap {100*(e_absent - e_present):+.2f} pp', flush=True)

        # Does the model's log-odds track the cue?
        lo = []
        for r in rows:
            p_a = test_stack[r['ia'], a]
            p_b = test_stack[r['ib'], b]
            lo.append(np.log(max(p_a, 1e-9)) - np.log(max(p_b, 1e-9)))
        lo = np.asarray(lo)
        corr = float(np.corrcoef(sign * diffs, lo)[0, 1]) if len(rows) > 2 else 0.0
        print(f'   corr(cue diff, model log-odds) = {corr:+.3f}', flush=True)

        payload[pair] = {
            'cue': cue,
            'n_groups': len(rows),
            'cue_present': {'n': n_present, 'confusion': e_present},
            'cue_absent': {'n': n_absent, 'confusion': e_absent},
            'gap_pp': 100 * (e_absent - e_present) if (e_absent is not None and e_present is not None) else None,
            'corr_cue_vs_logodds': corr,
        }

    # ---- how many errors are attributable to the GEN hub ------------------
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(test_truth, pred):
        cm[t, p] += 1
    total_err = int(len(test_truth) - cm.trace())
    gen_involved = int(cm[:, 2].sum() - cm[2, 2] + cm[2, :].sum() - cm[2, 2])
    payload['gen_hub'] = {
        'total_errors': total_err,
        'errors_involving_generated': gen_involved,
        'share': gen_involved / total_err,
    }
    print(f'\nGEN hub: {gen_involved}/{total_err} errors involve ai_generated '
          f'({100*gen_involved/total_err:.1f}%)', flush=True)

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
