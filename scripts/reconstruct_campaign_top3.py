"""Rebuild the campaign's composited Top 3 on the published test set.

The campaign scored E01/E05/E25/E26/E30/E31 on its own S partition, which is
contaminated: 657 of its 960 items appear in the published models' training
data. A fair comparison against the published five models therefore has to
re-run the same recipes on the published 3,194-item test set.

Every recipe here is copied from experiments/campaign12h.py so the numbers are
produced by the same code path, only with a different evaluation set:

  E01  B0 (RoBERTa + CE) single 512-token window
  E05  TF-IDF word(1,2) + char_wb(3,5) logistic regression
  E25  MC-Dropout, 5 stochastic passes over B0
  E26  head / middle / tail 512-token window probability mean
  E30  equal probability mean of E01 + E05 + E25
  E31  nonnegative probability fusion of E01 + E05 + E25 (recorded weights)
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

from experiments.common import classification_metrics  # noqa: E402
from experiments.glpc_model import GLPCModel  # noqa: E402
from experiments.neural import load_records, tokenizer_local  # noqa: E402

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
B0_PATH = ROOT / 'experiments/glpc_runs/glpc_b0_s3407/best.pt'
CAMPAIGN = ROOT / 'experiments/campaign12h/ee6405_scorefirst_36_12h'
DATA_ROOT = ROOT / 'experiments/hart_grouped_v1'
HART = ROOT / 'assets/hart_official'


def load_b0(device: str):
    checkpoint = torch.load(B0_PATH, map_location='cpu', weights_only=False)
    config = dict(checkpoint['config'])
    config['model_type'] = 'roberta'
    config['classifier_type'] = 'mlp'
    model = GLPCModel(config, mode='baseline')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval().to(device)
    return model


def predict_b0(records, model, device, batch_size=16):
    tokenizer = tokenizer_local()
    rows = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            part = records[start:start + batch_size]
            encoded = tokenizer([r['article'] for r in part], max_length=512, truncation=True,
                                padding=True, return_tensors='pt')
            output = model({'input_ids': encoded['input_ids'].to(device),
                            'attention_mask': encoded['attention_mask'].to(device)})
            rows.extend(output['logits'].float().cpu().numpy())
    return np.asarray(rows, dtype=np.float64)


def predict_mc_dropout(records, model, device, passes=5, seed=20261107):
    model.train()
    torch.manual_seed(seed)
    collected = []
    for _ in range(passes):
        collected.append(torch.softmax(torch.tensor(predict_b0(records, model, device)), dim=1).numpy())
    model.eval()
    return np.mean(collected, axis=0)


def predict_head_middle_tail(records, model, device):
    tokenizer = tokenizer_local()
    rows = []
    with torch.no_grad():
        for record in records:
            encoded = tokenizer(record['article'], max_length=512, truncation=False,
                                padding=False, return_tensors='pt')
            ids = encoded['input_ids'][0]
            windows = [ids[:512]]
            if len(ids) > 512:
                start = max(0, (len(ids) - 512) // 2)
                windows = [ids[:512], ids[start:start + 512], ids[-512:]]
            probs = []
            for window in windows:
                batch_ids = window.unsqueeze(0).to(device)
                mask = torch.ones_like(batch_ids)
                logits = model({'input_ids': batch_ids, 'attention_mask': mask})['logits']
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
            rows.append(np.mean(probs, axis=0))
    return np.asarray(rows)


def predict_tfidf_word_char(train, records):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion

    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=100000)),
    ])
    x_train = vectorizer.fit_transform([r['article'][:12000] for r in train])
    x_eval = vectorizer.transform([r['article'][:12000] for r in records])
    classifier = LogisticRegression(C=4, max_iter=500, random_state=20261107)
    classifier.fit(x_train, [r['label_id'] for r in train])
    return classifier.predict_proba(x_eval)


def load_published_test() -> list[dict]:
    """Published test items with text, in the published prediction order."""
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
                    by_id[record['item_id']] = record['article']
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f'{len(missing)} published test items have no text')
    return [{'item_id': item, 'label': label, 'label_id': LABELS.index(label),
             'article': by_id[item]} for item, label in order]


def score(y: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = classification_metrics(y.tolist(), probabilities)
    return {'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
            'J': (metrics['accuracy'] + metrics['macro_f1']) / 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=str(ROOT / '.local/campaign_top3_published_test.json'))
    args = parser.parse_args()

    records = load_published_test()
    y = np.asarray([r['label_id'] for r in records])
    print(f'test {len(records)}  labels {np.bincount(y)}', flush=True)

    train = load_records('train', DATA_ROOT)
    print(f'train {len(train)}', flush=True)

    model = load_b0(args.device)
    results = {}

    def record(task_id, probabilities, note):
        row = score(y, probabilities)
        results[task_id] = {**row, 'note': note, 'probabilities': probabilities.tolist()}
        print(f'  {task_id:4} acc={row["accuracy"]:.4f} f1={row["macro_f1"]:.4f} J={row["J"]:.4f}', flush=True)

    started = time.perf_counter()
    print('E01 B0 single window…', flush=True)
    e01 = torch.softmax(torch.tensor(predict_b0(records, model, args.device)), dim=1).numpy()
    record('E01', e01, 'B0 RoBERTa+CE, first 512 tokens')
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    started = time.perf_counter()
    print('E05 TF-IDF word+char LR…', flush=True)
    e05 = predict_tfidf_word_char(train, records)
    record('E05', e05, 'TF-IDF word(1,2)+char_wb(3,5) logistic regression')
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    started = time.perf_counter()
    print('E25 MC-Dropout x5…', flush=True)
    e25 = predict_mc_dropout(records, model, args.device)
    record('E25', e25, 'MC-Dropout, 5 stochastic passes over B0')
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    started = time.perf_counter()
    print('E26 head/middle/tail…', flush=True)
    e26 = predict_head_middle_tail(records, model, args.device)
    record('E26', e26, 'head/middle/tail 512-token window probability mean')
    print(f'    {time.perf_counter() - started:.1f}s', flush=True)

    e30 = (e01 + e05 + e25) / 3
    record('E30', e30, 'equal probability mean of E01+E05+E25')

    weights = json.loads((CAMPAIGN / 'tasks/E31/selection.json').read_text(encoding='utf-8'))['config']['weights']
    e31 = weights[0] * e01 + weights[1] * e05 + weights[2] * e25
    record('E31', e31, f'nonnegative fusion of E01+E05+E25, weights {weights}')

    pathlib.Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
