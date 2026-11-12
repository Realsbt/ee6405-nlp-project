"""Build the final comparison table: 3 information-theoretic methods x recent Top 3.

Every number is recomputed from ID-bound prediction files on the campaign's S
partition. No test or HC3 data is read.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'experiments/campaign12h/ee6405_scorefirst_36_12h'
REV = CAMPAIGN / 'top5_revision_01'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']

TOP3 = ['E26', 'E01', 'E02']
METHODS = {
    'M1_entropy_compression': 'Entropy + compressibility profile',
    'M2_stylometric': 'Stylometric feature vector',
    'M3_char_ngram_cross_entropy': 'Character n-gram cross-entropy',
}


def load_labels() -> np.ndarray:
    manifest = json.loads((CAMPAIGN / 'protocol/split_manifest.json').read_text(encoding='utf-8'))
    return np.asarray([int(r['label_id']) for r in manifest['records'] if r['partition'] == 'S'])


def load_proba(task: str) -> np.ndarray:
    data = json.loads((REV / f'{task}_prediction_with_ids.json').read_text(encoding='utf-8'))
    return np.asarray([r['probabilities'] for r in data['prediction_rows']])


def score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import accuracy_score, f1_score
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro'))
    return acc, f1, (acc + f1) / 2


def main() -> None:
    y = load_labels()
    info = json.loads((ROOT / '.local/info_methods.json').read_text(encoding='utf-8'))
    baselines = {t: load_proba(t) for t in TOP3}
    methods = {k: np.asarray(info[k]['proba']) for k in METHODS}

    rows = []

    for key, label in METHODS.items():
        acc, f1, j = score(y, methods[key].argmax(1))
        rows.append({'method': label, 'kind': 'standalone', 'partner': '—',
                     'accuracy': acc, 'macro_f1': f1, 'J': j})

    for task in TOP3:
        acc, f1, j = score(y, baselines[task].argmax(1))
        rows.append({'method': f'{task} (campaign baseline)', 'kind': 'baseline', 'partner': '—',
                     'accuracy': acc, 'macro_f1': f1, 'J': j})

    for key, label in METHODS.items():
        for task in TOP3:
            fused = 0.5 * baselines[task] + 0.5 * methods[key]
            acc, f1, j = score(y, fused.argmax(1))
            base_acc, base_f1, _ = score(y, baselines[task].argmax(1))
            rows.append({'method': label, 'kind': 'fused', 'partner': task,
                         'accuracy': acc, 'macro_f1': f1, 'J': j,
                         'delta_acc_pp': 100 * (acc - base_acc),
                         'delta_f1_pp': 100 * (f1 - base_f1)})

    rows.sort(key=lambda r: -r['J'])

    print(f'{"Method":38} {"Partner":8} {"Acc":>8} {"MacroF1":>8} {"J":>8} {"dAcc":>7} {"dF1":>7}')
    print('-' * 92)
    for r in rows:
        d_acc = f"{r['delta_acc_pp']:+.2f}" if 'delta_acc_pp' in r else ''
        d_f1 = f"{r['delta_f1_pp']:+.2f}" if 'delta_f1_pp' in r else ''
        print(f"{r['method']:38} {r['partner']:8} {100*r['accuracy']:7.2f}% "
              f"{100*r['macro_f1']:7.2f}% {100*r['J']:7.2f}% {d_acc:>7} {d_f1:>7}")

    out = {'partition': 'S', 'samples': int(len(y)), 'label_order': LABELS,
           'fusion_weight': 0.5, 'rows': rows}
    (ROOT / '.local/final_comparison.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\nwrote .local/final_comparison.json')


if __name__ == '__main__':
    main()
