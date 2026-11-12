"""Where do the 158 single-document errors go, and which cues cause them?

The group structure is used here as a *controlled comparison*, not as extra
information. Every group holds four versions of the same source document:

    human_written  ->  human_ai_polished  ->  ai_generated  ->  ai_humanized

so a confusion between two of them is a confusion between two texts that share
the same topic, argument and vocabulary. That makes the error attributable to
the transformation rather than to the content.

Three analyses:

  E1. Error flow. Full 4x4 confusion matrix for the deployable single-document
      classifier, plus the per-group error pattern (which pairs co-occur).

  E2. Cue localisation. For each confused pair, measure which surface cues
      separate the two versions *within* the same group. If a cue separates
      them in-group but the classifier still fails, the cue is present but
      unused. If a cue does not separate them in-group, the pair is genuinely
      hard.

  E3. Cue survival after rewriting. Track each cue from human_written through
      the three transformations. A cue that survives is usable; one that is
      destroyed by rewriting cannot be relied on.

Writes `.local/error_flow.json`.
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
    accuracy, apply_logreg, fit_logreg, load_test, load_val,
)

OUT = ROOT / '.local/error_flow.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SHORT = ['HW', 'POL', 'GEN', 'HUM']


def cues(text: str) -> dict[str, float]:
    """Surface cues that a provenance classifier could plausibly rely on."""
    words = re.findall(r"[A-Za-z']+", text)
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n = max(len(words), 1)
    ns = max(len(sents), 1)
    lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0]
    return {
        'length': float(len(text)),
        'mean_sent_len': float(np.mean(lens)),
        'sd_sent_len': float(np.std(lens)),
        'mean_word_len': float(np.mean([len(w) for w in words])) if words else 0.0,
        'ttr': len(set(words)) / n,
        'comma_per_100w': text.count(',') / n * 100,
        'semicolon_per_100w': text.count(';') / n * 100,
        'colon_per_100w': text.count(':') / n * 100,
        'emdash_per_100w': text.count('—') / n * 100,
        'paren_per_100w': text.count('(') / n * 100,
        'long_word_rate': sum(1 for w in words if len(w) > 8) / n,
        'the_per_100w': text.count(' the ') / n * 100,
        'of_per_100w': text.count(' of ') / n * 100,
        'and_per_100w': text.count(' and ') / n * 100,
        'sent_per_1000w': ns / n * 1000,
    }


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

    payload: dict = {'n': len(test_truth), 'accuracy': float(accuracy(test_stack, test_truth))}

    # ---- E1: error flow ---------------------------------------------------
    print('E1 error flow', flush=True)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(test_truth, pred):
        cm[t, p] += 1
    payload['confusion'] = {'labels': LABELS, 'matrix': cm.tolist()}

    flows = []
    for t in range(4):
        for p in range(4):
            if t != p and cm[t, p]:
                flows.append({'true': LABELS[t], 'pred': LABELS[p], 'n': int(cm[t, p]),
                              'share_of_errors': float(cm[t, p] / (len(test_truth) - cm.trace()))})
    flows.sort(key=lambda r: -r['n'])
    payload['flows'] = flows
    print(f'   total errors {len(test_truth) - cm.trace()}', flush=True)
    for f in flows:
        print(f'   {SHORT[LABELS.index(f["true"])]:>3} -> {SHORT[LABELS.index(f["pred"])]:<3} '
              f'{f["n"]:4d}  {100*f["share_of_errors"]:5.1f}%', flush=True)

    # ---- E1b: per-group error pattern -------------------------------------
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(test_groups):
        by_group[g].append(i)
    patterns = collections.Counter()
    for g, idx in by_group.items():
        if len(idx) != 4:
            continue
        wrong = tuple(sorted(SHORT[test_truth[i]] + '>' + SHORT[pred[i]]
                             for i in idx if pred[i] != test_truth[i]))
        patterns[wrong] += 1
    payload['group_error_patterns'] = [
        {'pattern': list(k), 'groups': v} for k, v in patterns.most_common(12)]
    print('\n   most common per-group error patterns:', flush=True)
    for k, v in patterns.most_common(6):
        print(f'     {v:4d} groups  {list(k)}', flush=True)

    # ---- E2: cue localisation within groups -------------------------------
    print('\nE2 cue localisation (within-group separation)', flush=True)
    cue_names = list(cues('x').keys())
    # For each ordered pair (A, B), how often does cue c rank A above B in-group?
    pair_stats: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for g, idx in by_group.items():
        if len(idx) != 4:
            continue
        by_label = {test_truth[i]: i for i in idx}
        if len(by_label) < 4:
            continue
        vals = {lab: cues(recs[i]['article']) for lab, i in by_label.items()}
        for a in range(4):
            for b in range(4):
                if a >= b:
                    continue
                key = f'{SHORT[a]}|{SHORT[b]}'
                for c in cue_names:
                    pair_stats[key][c].append(vals[a][c] - vals[b][c])

    cue_table = {}
    for key, per_cue in pair_stats.items():
        row = {}
        for c, diffs in per_cue.items():
            d = np.asarray(diffs)
            # effect size: how consistently the cue orders the two versions
            row[c] = {
                'mean_diff': float(d.mean()),
                'consistency': float(max((d > 0).mean(), (d < 0).mean())),
                'n': int(len(d)),
            }
        cue_table[key] = row
    payload['cue_localisation'] = cue_table

    for key in ('HW|POL', 'POL|GEN', 'GEN|HUM', 'HW|GEN', 'HW|HUM', 'POL|HUM'):
        if key not in cue_table:
            continue
        best = sorted(cue_table[key].items(), key=lambda kv: -kv[1]['consistency'])[:3]
        print(f'   {key}: ' + ', '.join(f'{c} {100*v["consistency"]:.0f}%' for c, v in best),
              flush=True)

    # ---- E3: cue survival across transformations --------------------------
    print('\nE3 cue survival (mean value per label)', flush=True)
    survival = {}
    for lab in range(4):
        idx = [i for i in range(len(test_truth)) if test_truth[i] == lab]
        vals = [cues(recs[i]['article']) for i in idx]
        survival[LABELS[lab]] = {c: float(np.mean([v[c] for v in vals])) for c in cue_names}
    payload['cue_survival'] = survival
    for c in ('length', 'mean_sent_len', 'ttr', 'comma_per_100w', 'emdash_per_100w'):
        row = '  '.join(f'{SHORT[i]} {survival[LABELS[i]][c]:8.2f}' for i in range(4))
        print(f'   {c:20s} {row}', flush=True)

    # ---- E2b: does the classifier use the cue it could? -------------------
    # For the GEN->HUM confusion, check whether the cue that best separates
    # them in-group is also the cue the model's probability tracks.
    print('\nE2b cue availability vs model failure', flush=True)
    gen_hum = cue_table.get('GEN|HUM', {})
    usable = {c: v for c, v in gen_hum.items() if v['consistency'] >= 0.60}
    payload['gen_hum_usable_cues'] = usable
    print(f'   cues separating GEN from HUM in >=60% of groups: {len(usable)}', flush=True)
    for c, v in sorted(usable.items(), key=lambda kv: -kv[1]['consistency'])[:5]:
        print(f'     {c:20s} consistency {100*v["consistency"]:.1f}%  '
              f'mean diff {v["mean_diff"]:+.3f}', flush=True)

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
