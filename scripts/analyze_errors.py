"""Where does the best model actually fail?

Loads the locally recomputed E30 probabilities (the strongest composited model)
and joins them with the HART metadata, then breaks the error rate down by every
factor the release exposes: domain, true label, generator model, text length,
and whether the 512-token window truncated the document.

Also dumps the worst confusion pairs with real text so the failure mode can be
read rather than guessed.
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
SHORT = {'human_written': 'HW', 'human_ai_polished': 'POL',
         'ai_generated': 'GEN', 'ai_humanized': 'HUM'}
HART = ROOT / 'assets/hart_official'


def load_predictions() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Item ids, true label ids, and E30 probabilities in published order."""
    data = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    probabilities = np.asarray(data['E30']['probabilities'])
    ids, truth = [], []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            ids.append(record['item_id'])
            truth.append(LABELS.index(record['true_label']))
    return ids, np.asarray(truth), probabilities


def load_metadata(ids: list[str]) -> dict[str, dict]:
    wanted = set(ids)
    meta = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    meta[record['item_id']] = record
    missing = wanted - set(meta)
    if missing:
        raise SystemExit(f'{len(missing)} items have no metadata')
    return meta


def rate_table(title: str, keys: list[str], truth: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for key in sorted(set(keys)):
        mask = np.asarray([k == key for k in keys])
        n = int(mask.sum())
        correct = int((pred[mask] == truth[mask]).sum())
        rows.append({'key': key, 'n': n, 'correct': correct,
                     'errors': n - correct, 'accuracy': correct / n if n else 0.0})
    rows.sort(key=lambda r: r['accuracy'])
    print(f'\n{title}')
    print(f'{"":34} {"n":>6} {"errors":>7} {"acc":>8}')
    print('-' * 60)
    for row in rows:
        print(f'{str(row["key"])[:34]:34} {row["n"]:6} {row["errors"]:7} {100*row["accuracy"]:7.2f}%')
    return rows


def main() -> None:
    ids, truth, probabilities = load_predictions()
    pred = probabilities.argmax(1)
    meta = load_metadata(ids)

    domains = [meta[i]['group_id'].split('-', 1)[0] for i in ids]
    labels = [LABELS[t] for t in truth]
    models = [meta[i].get('model_name', 'N/A') for i in ids]
    lengths = [len(meta[i]['article']) for i in ids]
    truncated = [bool(meta[i].get('input_truncated', False)) for i in ids]

    overall = float((pred == truth).mean())
    print(f'items {len(ids)}  accuracy {100*overall:.2f}%  errors {int((pred != truth).sum())}')

    # ---- confusion matrix -------------------------------------------------
    print('\nConfusion matrix (rows = truth, cols = predicted)')
    header = 'truth \\ pred'.ljust(14) + ''.join(f'{SHORT[l]:>8}' for l in LABELS) + f'{"recall":>9}'
    print(header)
    print('-' * len(header))
    for t, label in enumerate(LABELS):
        row = [int(((truth == t) & (pred == p)).sum()) for p in range(4)]
        recall = row[t] / max(1, sum(row))
        print(f'{SHORT[label]:<14}' + ''.join(f'{v:8}' for v in row) + f'{100*recall:8.1f}%')

    # ---- breakdowns -------------------------------------------------------
    rate_table('By domain', domains, truth, pred)
    rate_table('By true label', labels, truth, pred)
    rate_table('By domain x label', [f'{d} / {SHORT[l]}' for d, l in zip(domains, labels)], truth, pred)
    rate_table('By generator model (top 12 by count)', models, truth, pred)
    rate_table('By truncation', ['truncated' if t else 'full text' for t in truncated], truth, pred)

    buckets = ['<1k', '1-2k', '2-3k', '3-5k', '>5k']
    def bucket(n):
        if n < 1000: return '<1k'
        if n < 2000: return '1-2k'
        if n < 3000: return '2-3k'
        if n < 5000: return '3-5k'
        return '>5k'
    rate_table('By character length', [bucket(n) for n in lengths], truth, pred)

    # ---- worst confusion pairs -------------------------------------------
    pairs = collections.Counter()
    for t, p in zip(truth, pred):
        if t != p:
            pairs[(LABELS[t], LABELS[p])] += 1
    print('\nTop confusion pairs')
    print(f'{"truth -> predicted":46} {"count":>6}')
    print('-' * 54)
    for (t, p), count in pairs.most_common(10):
        print(f'{t + " -> " + p:46} {count:6}')

    # ---- hardest slices ---------------------------------------------------
    print('\nHardest slices (n >= 40, sorted by accuracy)')
    slices = collections.defaultdict(list)
    for i, item in enumerate(ids):
        slices[f'{domains[i]} / {SHORT[labels[i]]}'].append(i)
        slices[f'{SHORT[labels[i]]} / {models[i]}'].append(i)
    rows = []
    for key, idx in slices.items():
        if len(idx) < 40:
            continue
        idx = np.asarray(idx)
        rows.append((key, len(idx), float((pred[idx] == truth[idx]).mean())))
    rows.sort(key=lambda r: r[2])
    print(f'{"slice":52} {"n":>6} {"acc":>8}')
    print('-' * 68)
    for key, n, acc in rows[:15]:
        print(f'{key[:52]:52} {n:6} {100*acc:7.2f}%')

    # ---- concrete failures ------------------------------------------------
    print('\nConcrete failures (highest confidence wrong predictions)')
    wrong = np.flatnonzero(pred != truth)
    confidence = probabilities[wrong, pred[wrong]]
    for i in wrong[np.argsort(-confidence)][:8]:
        text = meta[ids[i]]['article']
        print(f'\n  {ids[i]}')
        print(f'  truth={LABELS[truth[i]]}  pred={LABELS[pred[i]]}  '
              f'p(pred)={probabilities[i, pred[i]]:.3f}  chars={len(text)}  '
              f'model={meta[ids[i]].get("model_name")}')
        print(f'  {text[:260].replace(chr(10), " ")}…')

    payload = {
        'items': len(ids),
        'accuracy': overall,
        'confusion': [[int(((truth == t) & (pred == p)).sum()) for p in range(4)] for t in range(4)],
        'confusion_pairs': [{'truth': t, 'predicted': p, 'count': c} for (t, p), c in pairs.most_common()],
        'hardest_slices': [{'slice': k, 'n': n, 'accuracy': a} for k, n, a in rows[:20]],
        'domains': sorted(set(domains)),
        'domain_class': {
            d: {LABELS[c]: {'n': int(((np.asarray(domains) == d) & (truth == c)).sum()),
                            'correct': int(((np.asarray(domains) == d) & (truth == c) & (pred == c)).sum())}
                for c in range(4)}
            for d in sorted(set(domains))
        },
        'length_buckets': {
            b: {'n': int(sum(1 for n in lengths if bucket(n) == b)),
                'correct': int(sum(1 for n, t, p in zip(lengths, truth, pred) if bucket(n) == b and t == p))}
            for b in buckets
        },
        'humanizer': {
            hz: {'n': int(sum(1 for m, l in zip(models, labels) if l == 'ai_humanized' and m == hz)),
                 'correct': int(sum(1 for m, l, t, p in zip(models, labels, truth, pred)
                                  if l == 'ai_humanized' and m == hz and t == p))}
            for hz in ('human', 'tool')
        },
        'per_class': {
            LABELS[c]: {
                'support': int((truth == c).sum()),
                'precision': float(((pred == c) & (truth == c)).sum() / max(1, (pred == c).sum())),
                'recall': float(((pred == c) & (truth == c)).sum() / max(1, (truth == c).sum())),
            }
            for c in range(4)
        },
    }
    (ROOT / '.local/error_analysis.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print('\nwrote .local/error_analysis.json')


if __name__ == '__main__':
    main()
