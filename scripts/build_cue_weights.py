"""Stage A/B bridge: build surface-cue difficulty weights for training.

The main model only sees the first 512 tokens, so the cue model must use the
same visible window. Otherwise the weights would be computed from text the
classifier never sees.

Protocol:
  * features are computed on the 512-token-truncated text, matching TextDataset
  * the cue model is a multinomial logistic regression on those features
  * out-of-fold predictions are produced with **group-disjoint** folds, so a
    training item's difficulty is never estimated from its own source document
  * weights are w_i = 1 + (1 - b_i), then normalised per class so the four
    class totals are unchanged

Writes `.local/cue_weights.json` with per-item weights for the train split.
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

OUT = ROOT / '.local/cue_weights.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
MAX_TOKENS = 512
N_FOLDS = 5
SEED = 20261113


def visible_window(text: str) -> str:
    """Approximate the 512-token window the classifier actually sees."""
    tokens = re.findall(r"\S+", text)
    return ' '.join(tokens[:MAX_TOKENS])


def features(text: str) -> list[float]:
    words = re.findall(r"[A-Za-z']+", text)
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n = max(len(words), 1)
    lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0]
    return [
        len(text) / 1000.0,
        len(words) / 100.0,
        float(np.mean([len(w) for w in words])) if words else 0.0,
        float(np.std([len(w) for w in words])) if words else 0.0,
        sum(1 for w in words if len(w) > 8) / n,
        len(set(words)) / n,
        text.count(',') / n * 100,
        text.count(';') / n * 100,
        text.count(':') / n * 100,
        text.count('—') / n * 100,
        text.count('(') / n * 100,
        float(np.mean(lens)),
        float(np.std(lens)),
        len(sents) / 10.0,
        text.count(' the ') / n * 100,
        text.count(' of ') / n * 100,
        text.count(' and ') / n * 100,
        sum(1 for c in text if c.isupper()) / max(len(text), 1) * 10,
    ]


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/train.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    print(f'train items {len(recs)}', flush=True)

    x = np.asarray([features(visible_window(r['article'])) for r in recs])
    y = np.asarray([LABELS.index(r['label']) for r in recs])
    groups = np.asarray([r['group_id'] for r in recs])

    # ---- group-disjoint out-of-fold difficulty ---------------------------
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, N_FOLDS)
    oof = np.zeros((len(recs), 4))
    for k, held in enumerate(folds):
        held_set = set(held.tolist())
        mask = np.array([g in held_set for g in groups])
        scaler = StandardScaler().fit(x[~mask])
        clf = LogisticRegression(max_iter=3000, class_weight='balanced', C=1.0)
        clf.fit(scaler.transform(x[~mask]), y[~mask])
        oof[mask] = clf.predict_proba(scaler.transform(x[mask]))
        print(f'  fold {k+1}/{N_FOLDS}: held {mask.sum()} items', flush=True)

    b = oof[np.arange(len(y)), y]
    print(f'\ncue-model OOF accuracy {100*(oof.argmax(1) == y).mean():.2f}%', flush=True)
    print(f'  b (prob of true class)  mean {b.mean():.3f}  '
          f'p10 {np.percentile(b,10):.3f}  p90 {np.percentile(b,90):.3f}', flush=True)

    # ---- weights ---------------------------------------------------------
    raw = 1.0 + (1.0 - b)
    weights = raw.copy()
    for c in range(4):
        m = y == c
        weights[m] = raw[m] / raw[m].mean()
    print(f'\nweights  mean {weights.mean():.3f}  min {weights.min():.3f}  '
          f'max {weights.max():.3f}', flush=True)
    for c in range(4):
        m = y == c
        print(f'  {LABELS[c]:20s} n={m.sum():5d}  mean weight {weights[m].mean():.3f}', flush=True)

    payload = {
        'protocol': {
            'features_on': f'first {MAX_TOKENS} whitespace tokens (matches classifier window)',
            'folds': N_FOLDS,
            'fold_unit': 'source document group (disjoint)',
            'weight_rule': 'w = 1 + (1 - b), normalised per class to mean 1',
            'cue_model_oof_accuracy': float((oof.argmax(1) == y).mean()),
        },
        'items': [{'item_id': r['item_id'], 'label': r['label'],
                   'b': float(bi), 'weight': float(wi)}
                  for r, bi, wi in zip(recs, b, weights)],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
