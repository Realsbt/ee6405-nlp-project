"""Evaluate the three information-theoretic methods on the published test set.

The campaign's S partition is contaminated: 657 of its 960 items appear in the
published models' training data. Comparing against the published five models
therefore requires the published test set, which those models never saw.

This script trains M1/M2/M3 on a pool that excludes every published test group,
then scores them on the same 3,194-item test set used by the published results.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from info_stylometry_methods import (  # noqa: E402
    LABEL_ID, LABELS, CharNgramModel, m1_features, m2_features,
)

HART = ROOT / 'assets/hart_official'


def published_test_groups() -> set[str]:
    groups = set()
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            groups.add(json.loads(line)['group_id'])
    return groups


def load_train_pool(exclude_groups: set[str]) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    for split in ('train_graph.jsonl', 'val_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['group_id'] in exclude_groups:
                    continue
                texts.append(record['article'])
                labels.append(LABEL_ID[record['label']])
    return texts, np.asarray(labels)


def load_published_test() -> tuple[list[str], np.ndarray]:
    """Test texts in the published prediction order.

    The published split is a different StratifiedGroupKFold, so its test items
    are scattered across the official HART files. Search all three.
    """
    order = []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            order.append((record['item_id'], LABEL_ID[record['true_label']]))
    wanted = {item for item, _ in order}
    by_id = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    by_id[record['item_id']] = record['article']
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f'{len(missing)} published test items have no text, e.g. {sorted(missing)[:2]}')
    texts = [by_id[item] for item, _ in order]
    labels = np.asarray([label for _, label in order])
    return texts, labels


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro'))
    return {'accuracy': acc, 'macro_f1': f1, 'J': (acc + f1) / 2}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    exclude = published_test_groups()
    print(f'excluding {len(exclude)} published test groups', flush=True)
    train_texts, train_labels = load_train_pool(exclude)
    print(f'train {len(train_texts)}  labels {np.bincount(train_labels)}', flush=True)

    test_texts, test_labels = load_published_test()
    print(f'test {len(test_texts)}  labels {np.bincount(test_labels)}', flush=True)

    results = {}

    print('M1…', flush=True)
    x_tr = np.asarray([m1_features(t) for t in train_texts])
    x_te = np.asarray([m1_features(t) for t in test_texts])
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(scaler.transform(x_tr), train_labels)
    proba = clf.predict_proba(scaler.transform(x_te))
    results['M1_entropy_compression'] = {**score(test_labels, proba.argmax(1)), 'proba': proba.tolist()}
    print('  ', results['M1_entropy_compression']['accuracy'], flush=True)

    print('M2…', flush=True)
    x_tr = np.asarray([m2_features(t) for t in train_texts])
    x_te = np.asarray([m2_features(t) for t in test_texts])
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(scaler.transform(x_tr), train_labels)
    proba = clf.predict_proba(scaler.transform(x_te))
    results['M2_stylometric'] = {**score(test_labels, proba.argmax(1)), 'proba': proba.tolist()}
    print('  ', results['M2_stylometric']['accuracy'], flush=True)

    print('M3…', flush=True)
    model = CharNgramModel(n=4, alpha=0.1).fit(train_texts, train_labels.tolist())
    proba = np.asarray([model.predict_proba(t) for t in test_texts])
    results['M3_char_ngram_cross_entropy'] = {**score(test_labels, proba.argmax(1)), 'proba': proba.tolist()}
    print('  ', results['M3_char_ngram_cross_entropy']['accuracy'], flush=True)

    out = ROOT / '.local/info_methods_published_test.json'
    out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nwrote {out}')
    for name, row in results.items():
        print(f"{name:34} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
