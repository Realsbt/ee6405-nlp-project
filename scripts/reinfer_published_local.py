"""Re-infer all 13 published checkpoints locally on the published test set.

Fair comparison requires every model to be scored by the same code path on the
same items. The published prediction files were produced on the original host;
this script reloads each checkpoint here and recomputes the probabilities, then
checks them against the archived files.

RACE needs RST trees. The HART release ships them, so no parser run is needed:
the trees are read from assets/hart_official and passed through the same
RawGraphDataset preprocessing the published runs used.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
HART = ROOT / 'assets/hart_official'

RUNS = [
    'tfidf_lr_v1',
    'roberta_s3407', 'roberta_s42', 'roberta_s2026',
    'roberta_supcon_s3407', 'roberta_supcon_s42', 'roberta_supcon_s2026',
    'race_no_supcon_s3407', 'race_no_supcon_s42', 'race_no_supcon_s2026',
    'race_s3407', 'race_s42', 'race_s2026',
]


def checkpoint_path(run: str) -> pathlib.Path:
    """Prefer the local resource store, fall back to the E: download."""
    from experiments.paths import resolve_recorded_path

    report = json.loads((ROOT / f'experiments/reports/{run}/result.json').read_text(encoding='utf-8'))
    recorded = report.get('checkpoint_path') or report.get('model_path')
    local = resolve_recorded_path(recorded)
    if local.is_file():
        return local
    alt = pathlib.Path(r'E:\ee6405-ckpt') / f'{run}-best.pt'
    if alt.is_file():
        return alt
    raise SystemExit(f'No checkpoint for {run}')


def load_test_records() -> list[dict]:
    """Published test items with their text and RST tree, in prediction order."""
    order = []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            order.append((record['item_id'], record['true_label']))
    wanted = {item for item, _ in order}
    by_id = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        with (HART / split).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    by_id[record['item_id']] = record
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f'{len(missing)} test items have no text')
    return [{'item_id': item, 'label': label, 'label_id': LABELS.index(label),
             'article': by_id[item]['article'],
             'rst_structure': by_id[item].get('rst_structure')} for item, label in order]


def predict_tfidf(run: str, records: list[dict]) -> np.ndarray:
    import joblib
    from experiments.paths import resolve_recorded_path

    report = json.loads((ROOT / f'experiments/reports/{run}/result.json').read_text(encoding='utf-8'))
    model = joblib.load(resolve_recorded_path(report['model_path']))
    # The published TF-IDF input is the character prefix covered by 512 tokens.
    from experiments.neural import tokenizer_local
    tokenizer = tokenizer_local()
    texts = []
    for record in records:
        encoded = tokenizer(record['article'], max_length=512, truncation=True, return_offsets_mapping=True)
        end = max((b for a, b in encoded['offset_mapping'] if b > a), default=0)
        texts.append(record['article'][:end])
    return model.predict_proba(texts)


def predict_neural(run: str, records: list[dict], device: str) -> np.ndarray:
    from experiments.neural import TextDataset, build_model, collate_batch, tokenizer_local
    from experiments.train_neural import evaluate

    report = json.loads((ROOT / f'experiments/reports/{run}/result.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(checkpoint_path(run), map_location='cpu', weights_only=True)
    # The archived config records the original host's backbone path; map it.
    from experiments.paths import resolve_recorded_path
    config = json.loads(json.dumps(report['config']))
    config['architecture']['backbone_model_path'] = str(
        resolve_recorded_path(config['architecture']['backbone_model_path']))
    tokenizer = tokenizer_local()
    if config['model_type'] == 'roberta':
        dataset = TextDataset(records, tokenizer)
    else:
        from experiments.raw_graph import RawGraphDataset
        missing = [r['item_id'] for r in records if not r.get('rst_structure')]
        if missing:
            raise SystemExit(f'{len(missing)} RACE items have no RST tree')
        dataset = RawGraphDataset(records, tokenizer)
    model = build_model(config, checkpoint.get('metadata')).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    _, probabilities, ids, seconds = evaluate(model, dataset, config['batch_size'])
    if ids != [r['item_id'] for r in records]:
        raise SystemExit(f'{run}: evaluation order does not match the published order')
    print(f'    forward {seconds:.1f}s', flush=True)
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='*', default=RUNS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=str(ROOT / '.local/local_reinference.json'))
    args = parser.parse_args()

    records = load_test_records()
    print(f'test items {len(records)}', flush=True)
    y = np.asarray([r['label_id'] for r in records])

    results = {}
    for run in args.runs:
        print(f'{run}…', flush=True)
        started = time.perf_counter()
        if run.startswith('tfidf'):
            probabilities = predict_tfidf(run, records)
        else:
            probabilities = predict_neural(run, records, args.device)
        elapsed = time.perf_counter() - started

        archived = []
        with (ROOT / f'experiments/reports/{run}/test_predictions.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                archived.append([record['probabilities'][k] for k in LABELS])
        archived = np.asarray(archived)
        diff = float(np.abs(probabilities - archived).max())

        from sklearn.metrics import accuracy_score, f1_score
        pred = probabilities.argmax(1)
        results[run] = {
            'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'max_abs_diff_vs_archived': diff,
            'seconds': elapsed,
            'probabilities': probabilities.tolist(),
        }
        print(f'    acc={results[run]["accuracy"]:.4f} f1={results[run]["macro_f1"]:.4f} '
              f'maxdiff={diff:.2e} {elapsed:.1f}s', flush=True)

    pathlib.Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
