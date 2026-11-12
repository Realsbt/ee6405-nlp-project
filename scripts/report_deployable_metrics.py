"""Deployable single-document metrics, separated from protocol-only numbers.

The report quotes 97.78% and 95.05% side by side. They are not comparable:
97.78% requires the other three documents in the group, 95.05% needs only one
document. This script records the deployable number with the caveats a
deployment must carry:

  * per-class recall, so the `ai_humanized` weakness is explicit;
  * confidence on correct vs wrong predictions, to show that threshold
    abstention has limited value;
  * short-document accuracy, to show the out-of-distribution risk.

Writes `.local/deployable_metrics.json`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stack_models import (  # noqa: E402
    accuracy, apply_logreg, fit_logreg, load_test, load_val, macro_f1,
)

OUT = ROOT / '.local/deployable_metrics.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, _test_groups, test_probs = load_test()
    names = sorted(set(val_probs) & set(test_probs))

    w = fit_logreg(np.stack([val_probs[n] for n in names]), val_truth)
    test_stack = apply_logreg(w, np.stack([test_probs[n] for n in names]))
    pred = test_stack.argmax(1)
    conf = test_stack.max(1)

    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    lengths = np.asarray([len(r['article']) for r in recs])

    per_class = {}
    for i, name in enumerate(LABELS):
        mask = test_truth == i
        per_class[name] = {
            'n': int(mask.sum()),
            'recall': float((pred[mask] == i).mean()),
        }

    short = lengths < 1000
    payload = {
        'deployable': {
            'description': 'single document, no group information',
            'accuracy': float(accuracy(test_stack, test_truth)),
            'macro_f1': float(macro_f1(test_stack, test_truth)),
            'n': int(len(test_truth)),
        },
        'protocol_only': {
            'description': 'requires the other three documents in the group',
            'accuracy': 0.9778,
            'macro_f1': 0.9590,
        },
        'per_class_recall': per_class,
        'confidence': {
            'mean_when_correct': float(conf[pred == test_truth].mean()),
            'mean_when_wrong': float(conf[pred != test_truth].mean()),
            'note': 'errors carry high confidence, so threshold abstention has limited value',
        },
        'short_documents': {
            'threshold_chars': 1000,
            'n': int(short.sum()),
            'accuracy': float((pred[short] == test_truth[short]).mean()),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'deployable (single document)  {100*payload["deployable"]["accuracy"]:.2f}% / '
          f'{100*payload["deployable"]["macro_f1"]:.2f}%')
    for name, v in per_class.items():
        print(f'  {name:20s} n={v["n"]:5d}  recall {100*v["recall"]:6.2f}%')
    print(f'confidence correct {payload["confidence"]["mean_when_correct"]:.3f}  '
          f'wrong {payload["confidence"]["mean_when_wrong"]:.3f}')
    print(f'short (<1k) n={payload["short_documents"]["n"]}  '
          f'acc {100*payload["short_documents"]["accuracy"]:.2f}%')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
