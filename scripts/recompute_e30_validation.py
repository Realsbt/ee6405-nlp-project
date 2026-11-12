"""Recompute E30 on the validation split so model selection can avoid the test set.

The short-document audit needs a selection set that is disjoint from the
published test set. The campaign's own validation split is the natural choice:
it was used to pick checkpoints, so it is not pristine, but it is strictly
disjoint from the test items and is the only labelled set available besides
train.

Writes `.local/campaign_top3_val.json`.
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
from scripts.reconstruct_campaign_top3 import (  # noqa: E402
    DATA_ROOT, load_b0, predict_b0, predict_mc_dropout, predict_tfidf_word_char,
)

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=str(ROOT / '.local/campaign_top3_val.json'))
    args = parser.parse_args()

    val = load_records('val', DATA_ROOT)
    train = load_records('train', DATA_ROOT)
    y = np.asarray([r['label_id'] for r in val])
    print(f'val {len(val)}  labels {np.bincount(y)}', flush=True)

    model = load_b0(args.device)

    started = time.perf_counter()
    print('E01 B0 single window…', flush=True)
    e01 = torch.softmax(torch.tensor(predict_b0(val, model, args.device)), dim=1).numpy()
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    started = time.perf_counter()
    print('E05 TF-IDF word+char LR…', flush=True)
    e05 = predict_tfidf_word_char(train, val)
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    started = time.perf_counter()
    print('E25 MC-Dropout x5…', flush=True)
    e25 = predict_mc_dropout(val, model, args.device)
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    e30 = (e01 + e05 + e25) / 3
    accuracy = float((e30.argmax(1) == y).mean())
    print(f'E30 val accuracy {100*accuracy:.2f}%', flush=True)

    payload = {
        'item_ids': [r['item_id'] for r in val],
        'labels': [LABELS[int(v)] for v in y],
        'E01': e01.tolist(),
        'E05': e05.tolist(),
        'E25': e25.tolist(),
        'E30': e30.tolist(),
        'accuracy': accuracy,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload), encoding='utf-8')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
