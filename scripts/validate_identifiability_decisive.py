"""Decisive control: is human-rewrite hard because of the label, or the setup?

`validate_pairwise_controls.py` showed the pairwise ranker scores 87.72% on true
siblings but still 80.70% on cross-document pairs. That means most of the
pairwise signal is a generic "this text reads as human-rewritten" detector, not
same-document discrimination. Two questions follow:

  Q1. Is the absolute classifier's 1.75-29.82% recall a *data* limit or a
      *class-imbalance* limit? Test: train a binary GEN-vs-HUM classifier on a
      balanced pool (only those two classes), then evaluate on the test set's
      GEN and HUM items. If recall jumps, the four-class failure is an
      imbalance artefact, not label noise.

  Q2. Does the pairwise ranker actually use document identity? Test: pair
      GEN_i with GEN_j (both generated). A document-aware ranker should be at
      chance; a style-only ranker will also be at chance. Then pair HUM_i with
      HUM_j. If the ranker is at chance on both, it is a pure style detector.

Writes `.local/identifiability_decisive.json`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / '.local/identifiability_decisive.json'
SEED = 20261113


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in
            (ROOT / f'experiments/hart_grouped_v1/{split}.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]


def is_human_rewrite(r: dict) -> bool:
    return r['label'] == 'ai_humanized' and r.get('model_name') == 'human'


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = load_split('train')
    test = load_split('test')

    # ---- Q1: balanced binary GEN vs HUM ----------------------------------
    tr = [r for r in train if r['label'] == 'ai_generated' or is_human_rewrite(r)]
    te = [r for r in test if r['label'] == 'ai_generated' or is_human_rewrite(r)]
    y_tr = np.array([1 if is_human_rewrite(r) else 0 for r in tr])
    y_te = np.array([1 if is_human_rewrite(r) else 0 for r in te])
    print(f'Q1 balanced pool: train {len(tr)} (pos {y_tr.sum()})  '
          f'test {len(te)} (pos {y_te.sum()})', flush=True)

    results = {}
    for tag, kw in (('word 1-2', dict(ngram_range=(1, 2))),
                    ('char_wb 2-5', dict(analyzer='char_wb', ngram_range=(2, 5)))):
        vec = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=200000, **kw)
        x_tr = vec.fit_transform([r['article'] for r in tr])
        x_te = vec.transform([r['article'] for r in te])
        clf = LogisticRegression(max_iter=3000, class_weight='balanced', C=1.0).fit(x_tr, y_tr)
        proba = clf.predict_proba(x_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        fn = int(((pred == 0) & (y_te == 1)).sum())
        tn = int(((pred == 0) & (y_te == 0)).sum())
        results[tag] = {
            'accuracy': float((pred == y_te).mean()),
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'specificity': tn / (tn + fp),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        print(f'   {tag:14s} acc {100*results[tag]["accuracy"]:5.2f}%  '
              f'recall {100*results[tag]["recall"]:5.2f}%  '
              f'prec {100*results[tag]["precision"]:5.2f}%  '
              f'spec {100*results[tag]["specificity"]:5.2f}%', flush=True)

    # ---- Q2: same-class pair controls ------------------------------------
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in test:
        by_group[r['group_id']].append(r)

    def pairs_of(kind: str) -> list[tuple[str, str]]:
        out = []
        for g, items in by_group.items():
            gen = [i for i in items if i['label'] == 'ai_generated']
            hum = [i for i in items if is_human_rewrite(i)]
            if kind == 'gen_hum' and gen and hum:
                out.append((gen[0]['article'], hum[0]['article']))
        return out

    train_pairs = []
    by_group_tr: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train:
        by_group_tr[r['group_id']].append(r)
    for g, items in by_group_tr.items():
        gen = [i for i in items if i['label'] == 'ai_generated']
        hum = [i for i in items if is_human_rewrite(i)]
        if gen and hum:
            train_pairs.append((gen[0]['article'], hum[0]['article']))

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)
    vec.fit([t for p in train_pairs for t in p])

    def diff(a, b):
        va, vb = vec.transform([a, b])
        return np.asarray((vb - va).todense()).ravel()

    x = np.array([diff(a, b) for a, b in train_pairs])
    x = np.vstack([x, -x])
    y = np.concatenate([np.ones(len(train_pairs)), np.zeros(len(train_pairs))])
    ranker = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit(x, y)

    rng = np.random.default_rng(SEED)
    gen_texts = [i['article'] for items in by_group.values()
                 for i in items if i['label'] == 'ai_generated']
    hum_texts = [i['article'] for items in by_group.values()
                 for i in items if is_human_rewrite(i)]

    controls = {}
    for tag, pool in (('gen_vs_gen', gen_texts), ('hum_vs_hum', hum_texts)):
        idx = rng.integers(0, len(pool), size=(400, 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        xs = np.array([diff(pool[i], pool[j]) for i, j in idx])
        rate = float((ranker.decision_function(xs) > 0).mean())
        controls[tag] = {'n': int(len(idx)), 'rate_b_is_human': rate}
        print(f'Q2 {tag:12s} n={len(idx):4d}  P(ranker says b is human) = {100*rate:5.2f}%',
              flush=True)

    payload = {
        'balanced_binary': results,
        'same_class_controls': controls,
        'true_pair_accuracy': 0.8772,
        'cross_document_accuracy': 0.8070,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
