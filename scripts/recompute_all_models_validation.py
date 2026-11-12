"""Recompute all 13 published models on the campaign validation split.

The stacking experiment needs a selection set that is disjoint from the
published test set. This produces the same 13 probability matrices on the
grouped validation split so a stacker can be fit there and evaluated on the
test set exactly once.

Writes `.local/all_models_val.json`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.neural import load_records  # noqa: E402
from scripts.reconstruct_campaign_top3 import DATA_ROOT  # noqa: E402
from scripts.reinfer_published_local import (  # noqa: E402
    LABELS, RUNS, predict_neural, predict_tfidf,
)

HART = ROOT / 'assets/hart_official'


def load_val_records() -> list[dict]:
    """Grouped validation items joined with their RST tree from the official files."""
    val = load_records('val', DATA_ROOT)
    wanted = {r['item_id'] for r in val}
    by_id: dict[str, dict] = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    by_id[record['item_id']] = record
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f'{len(missing)} validation items have no text')
    return [{'item_id': r['item_id'], 'label': r['label'], 'label_id': r['label_id'],
             'article': by_id[r['item_id']]['article'],
             'rst_structure': by_id[r['item_id']].get('rst_structure')} for r in val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='*', default=RUNS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=str(ROOT / '.local/all_models_val.json'))
    args = parser.parse_args()

    records = load_val_records()
    truth = np.asarray([r['label_id'] for r in records])
    print(f'validation items {len(records)}  labels {np.bincount(truth)}', flush=True)

    payload: dict[str, dict] = {
        'item_ids': [r['item_id'] for r in records],
        'labels': [LABELS[int(v)] for v in truth],
        'models': {},
    }
    for run in args.runs:
        started = time.perf_counter()
        print(f'{run}…', flush=True)
        if run == 'tfidf_lr_v1':
            probabilities = predict_tfidf(run, records)
        else:
            probabilities = predict_neural(run, records, args.device)
        accuracy = float((probabilities.argmax(1) == truth).mean())
        payload['models'][run] = {
            'accuracy': accuracy,
            'probabilities': np.asarray(probabilities).tolist(),
            'seconds': time.perf_counter() - started,
        }
        print(f'    acc {100*accuracy:.2f}%  {time.perf_counter() - started:.1f}s', flush=True)

    pathlib.Path(args.out).write_text(json.dumps(payload), encoding='utf-8')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
