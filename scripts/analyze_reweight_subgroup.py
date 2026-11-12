"""Did the reweighting help the subgroup it targeted, even if the total is flat?

The controlled experiment gave control 93.33% / 90.58% and treatment
93.43% / 90.58%. That is a null result on the total. Three questions remain:

  S1. Did the treatment help the *target* subgroup (GEN/HUM groups whose comma
      cue is atypical), even if it hurt elsewhere?
  S2. Did continued training itself degrade the base model? The base
      roberta_s3407 scored 93.45% / 90.72%; both arms scored lower.
  S3. Rescue/damage ledger: how many previously-wrong items became right, and
      how many previously-right items became wrong.

Writes `.local/reweight_subgroup.json`.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / '.local/reweight_subgroup.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def comma(t: str) -> float:
    w = re.findall(r"[A-Za-z']+", t)
    return t.count(',') / max(len(w), 1) * 100


def main() -> None:
    exp = json.loads((ROOT / '.local/reweight_experiment.json').read_text(encoding='utf-8'))
    local = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    truth = np.asarray([LABELS.index(r['label']) for r in recs])
    base = np.asarray(local['roberta_s3407']['probabilities'])

    arms = {}
    for name, v in exp['arms'].items():
        probs = np.asarray(v['probabilities'])
        order = {iid: k for k, iid in enumerate(v['item_ids'])}
        arms[name] = probs[[order[r['item_id']] for r in recs]]

    payload: dict = {}

    # ---- S2: degradation --------------------------------------------------
    print('S2 continued-training degradation', flush=True)
    base_acc = local['roberta_s3407']['accuracy']
    print(f'   base (published)          {100*base_acc:.2f}% / '
          f'{100*local["roberta_s3407"]["macro_f1"]:.2f}%', flush=True)
    payload['base'] = {'accuracy': base_acc, 'macro_f1': local['roberta_s3407']['macro_f1']}
    payload['arms'] = {}
    for name, p in arms.items():
        acc = float((p.argmax(1) == truth).mean())
        print(f'   {name:26s} {100*acc:.2f}%  delta vs base {100*(acc-base_acc):+.2f} pp',
              flush=True)
        payload['arms'][name] = {'accuracy': acc, 'delta_vs_base_pp': 100 * (acc - base_acc)}

    # ---- S3: rescue / damage ledger --------------------------------------
    print('\nS3 rescue / damage ledger', flush=True)
    ctrl = arms['control_uniform'].argmax(1)
    trt = arms['treatment_cue_weighted'].argmax(1)
    base_pred = base.argmax(1)
    ledger = {}
    for tag, a, b in (('treatment vs control', ctrl, trt),
                      ('treatment vs base', base_pred, trt),
                      ('control vs base', base_pred, ctrl)):
        rescued = int(((a != truth) & (b == truth)).sum())
        damaged = int(((a == truth) & (b != truth)).sum())
        ledger[tag] = {'rescued': rescued, 'damaged': damaged, 'net': rescued - damaged}
        print(f'   {tag:22s} rescued {rescued:3d}  damaged {damaged:3d}  net {rescued-damaged:+3d}',
              flush=True)
    payload['ledger'] = ledger

    # ---- S1: subgroup ----------------------------------------------------
    print('\nS1 subgroup: GEN/HUM groups by comma atypicality', flush=True)
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        by_group.setdefault(r['group_id'], []).append(i)

    rows = []
    for g, idx in by_group.items():
        if len(idx) != 4:
            continue
        by_label = {truth[i]: i for i in idx}
        if 2 not in by_label or 3 not in by_label:
            continue
        ig, ih = by_label[2], by_label[3]
        rows.append({'ig': ig, 'ih': ih,
                     'present': comma(recs[ig]['article']) > comma(recs[ih]['article'])})

    def pair_acc(pred, subset):
        if not subset:
            return None
        ok = sum(1 for r in subset if pred[r['ig']] == 2 and pred[r['ih']] == 3)
        return ok / len(subset)

    subgroup = {}
    for tag, subset in (('cue_present', [r for r in rows if r['present']]),
                        ('cue_absent', [r for r in rows if not r['present']])):
        entry = {'n': len(subset)}
        for name, p in arms.items():
            entry[name] = pair_acc(p.argmax(1), subset)
        entry['base'] = pair_acc(base_pred, subset)
        subgroup[tag] = entry
        print(f'   {tag:12s} n={len(subset):3d}  base {100*entry["base"]:5.2f}%  '
              f'control {100*entry["control_uniform"]:5.2f}%  '
              f'treatment {100*entry["treatment_cue_weighted"]:5.2f}%', flush=True)
    payload['subgroup'] = subgroup

    # ---- humanized recall ------------------------------------------------
    print('\nHumanized recall', flush=True)
    hum = {}
    for name, p in list(arms.items()) + [('base', base)]:
        pred = p.argmax(1)
        m = truth == 3
        hum[name] = float((pred[m] == 3).mean())
        print(f'   {name:26s} {100*hum[name]:5.2f}%', flush=True)
    payload['humanized_recall'] = hum

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
