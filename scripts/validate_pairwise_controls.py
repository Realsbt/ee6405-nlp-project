"""Controls for the pairwise GEN-vs-HUM ranking result.

`validate_human_rewrite.py` found 87.72% pairwise ranking accuracy, which
contradicts the 1.75-29.82% recall of the absolute classifiers. Before that
number can be believed, three confounds must be ruled out:

  C1. Length. Human rewrites may be systematically shorter or longer than their
      GEN sibling. A length-only rule would then score well without reading
      any content. Control: rank by length alone, and by length + punctuation.

  C2. Transductive vectoriser. The TF-IDF vocabulary was fit on train + test
      pairs. Control: refit on train pairs only.

  C3. Pair identity. If the model is memorising document topics rather than
      rewrite direction, it should fail on pairs whose two texts come from
      *different* groups. Control: shuffled-pair negative control.

Also reports the length statistics so the confound can be judged directly.

Writes `.local/pairwise_controls.json`.
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

OUT = ROOT / '.local/pairwise_controls.json'
SEED = 20261113


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in
            (ROOT / f'experiments/hart_grouped_v1/{split}.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]


def collect_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_group[r['group_id']].append(r)
    pairs = []
    for g, items in by_group.items():
        gen = [i for i in items if i['label'] == 'ai_generated']
        hum = [i for i in items if i['label'] == 'ai_humanized' and i.get('model_name') == 'human']
        if gen and hum:
            pairs.append((gen[0]['article'], hum[0]['article']))
    return pairs


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_pairs = collect_pairs(load_split('train'))
    test_pairs = collect_pairs(load_split('test'))
    print(f'train pairs {len(train_pairs)}  test pairs {len(test_pairs)}', flush=True)

    # ---- C1: length statistics -------------------------------------------
    len_gen = np.array([len(a) for a, _ in test_pairs], dtype=float)
    len_hum = np.array([len(b) for _, b in test_pairs], dtype=float)
    ratio = len_hum / len_gen
    print(f'\nC1 length: GEN median {np.median(len_gen):.0f}  HUM median {np.median(len_hum):.0f}',
          flush=True)
    print(f'   HUM/GEN ratio  median {np.median(ratio):.3f}  '
          f'p05 {np.percentile(ratio, 5):.3f}  p95 {np.percentile(ratio, 95):.3f}', flush=True)
    print(f'   HUM shorter in {int((ratio < 1).sum())}/{len(ratio)} pairs', flush=True)

    # length-only rule: predict the shorter text is the human rewrite
    shorter_is_hum = int((len_hum < len_gen).sum())
    longer_is_hum = int((len_hum > len_gen).sum())
    length_rule = max(shorter_is_hum, longer_is_hum) / len(test_pairs)
    print(f'   length-only rule: {100*length_rule:.2f}% '
          f'(shorter={shorter_is_hum}, longer={longer_is_hum})', flush=True)

    # ---- C2: train-only vectoriser ---------------------------------------
    def fit_eval(fit_corpus, tag):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                              max_features=200000)
        vec.fit(fit_corpus)

        def diff(a, b):
            va, vb = vec.transform([a, b])
            return np.asarray((vb - va).todense()).ravel()

        x = np.array([diff(a, b) for a, b in train_pairs])
        x = np.vstack([x, -x])
        y = np.concatenate([np.ones(len(train_pairs)), np.zeros(len(train_pairs))])
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit(x, y)
        xt = np.array([diff(a, b) for a, b in test_pairs])
        acc = float((clf.decision_function(xt) > 0).mean())
        print(f'   {tag:34s} {100*acc:5.2f}%', flush=True)
        return acc, clf, vec, diff

    print('\nC2 vectoriser scope', flush=True)
    acc_trans, _, _, _ = fit_eval(
        [t for p in train_pairs for t in p] + [t for p in test_pairs for t in p],
        'transductive (train+test vocab)')
    acc_train, clf, vec, diff = fit_eval(
        [t for p in train_pairs for t in p], 'train-only vocab')

    # ---- C3: shuffled-pair negative control ------------------------------
    rng = np.random.default_rng(SEED)
    shuffled = []
    for a, _ in test_pairs:
        b = test_pairs[int(rng.integers(len(test_pairs)))][1]
        shuffled.append((a, b))
    xs = np.array([diff(a, b) for a, b in shuffled])
    acc_shuf = float((clf.decision_function(xs) > 0).mean())
    print(f'\nC3 shuffled-pair control (different groups)  {100*acc_shuf:5.2f}%', flush=True)

    # ---- C1b: length-matched subset --------------------------------------
    matched = [(a, b) for a, b in test_pairs if 0.9 <= len(b) / len(a) <= 1.1]
    if matched:
        xm = np.array([diff(a, b) for a, b in matched])
        acc_match = float((clf.decision_function(xm) > 0).mean())
        print(f'C1b length-matched subset (n={len(matched)})            '
              f'{100*acc_match:5.2f}%', flush=True)
    else:
        acc_match = None

    payload = {
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs),
        'length': {
            'gen_median': float(np.median(len_gen)),
            'hum_median': float(np.median(len_hum)),
            'ratio_median': float(np.median(ratio)),
            'hum_shorter_count': shorter_is_hum,
            'length_only_rule_accuracy': length_rule,
        },
        'vectoriser': {
            'transductive_accuracy': acc_trans,
            'train_only_accuracy': acc_train,
        },
        'shuffled_pair_accuracy': acc_shuf,
        'length_matched': {
            'n': len(matched),
            'accuracy': acc_match,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
