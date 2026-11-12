"""Short-document routing, with every selection decision made off the test set.

Audit note: the first version of this experiment chose the vectoriser variant by
test accuracy and tuned the blend weight on test labels. Both are leakage. This
version fixes the protocol:

  * vectoriser variant  -> chosen on the validation split
  * blend weight        -> chosen on the validation split
  * test set            -> touched exactly once, to report the final number

The validation split is not pristine (it selected the campaign checkpoints), so
the test number is still an estimate rather than a guarantee. It is, however,
free of the specific leakage that inflated the earlier +3.79 pp.

Writes `.local/short_document_model.json`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
HART = ROOT / 'assets/hart_official'
SEED = 20261112
SHORT_CHARS = 1000

VARIANTS = {
    'char_wb 2-5': dict(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                        sublinear_tf=True, max_features=200000),
    'word 1-2': dict(analyzer='word', ngram_range=(1, 2), min_df=2,
                     sublinear_tf=True, max_features=200000),
}


def load(split):
    with (HART / f'{split}_graph.jsonl').open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def load_test_frame():
    ids, truth = [], []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            ids.append(record['item_id'])
            truth.append(LABELS.index(record['true_label']))
    wanted = set(ids)
    meta = {}
    for split in ('train', 'val', 'test'):
        for record in load(split):
            if record['item_id'] in wanted:
                meta[record['item_id']] = record
    return ids, np.asarray(truth), meta


def fit_variant(name, texts, y):
    model = make_pipeline(TfidfVectorizer(**VARIANTS[name]),
                          LogisticRegression(max_iter=3000, C=4.0, class_weight='balanced'))
    model.fit(texts, y)
    return model


def best_weight(base, extra, truth, grid):
    """Weight on `extra` that maximises accuracy on the given rows."""
    best_w, best_acc = 0.0, -1.0
    for w in grid:
        acc = float((((1 - w) * base + w * extra).argmax(1) == truth).mean())
        if acc > best_acc:
            best_w, best_acc = float(w), acc
    return best_w, best_acc


def main():
    train, val = load('train'), load('val')
    ids, truth, meta = load_test_frame()

    lengths = np.asarray([len(meta[i]['article']) for i in ids])
    short = lengths < SHORT_CHARS

    e30_test = np.asarray(json.loads((ROOT / '.local/campaign_top3_published_test.json')
                                     .read_text(encoding='utf-8'))['E30']['probabilities'])
    val_frame = json.loads((ROOT / '.local/campaign_top3_val.json').read_text(encoding='utf-8'))
    e30_val = np.asarray(val_frame['E30'])
    val_truth = np.asarray([LABELS.index(x) for x in val_frame['labels']])
    # the campaign validation split is the grouped one; the official HART val
    # file uses different item ids, so read metadata from the grouped split
    with (ROOT / 'experiments/hart_grouped_v1/val.jsonl').open(encoding='utf-8') as handle:
        val_meta = {r['item_id']: r for r in (json.loads(line) for line in handle)}
    val_short = np.asarray([len(val_meta[i]['article']) < SHORT_CHARS for i in val_frame['item_ids']])

    print(f'val short items {int(val_short.sum())}  test short items {int(short.sum())}')
    print(f'E30 val short acc {100*(e30_val[val_short].argmax(1)==val_truth[val_short]).mean():.2f}%')
    print(f'E30 test short acc {100*(e30_test[short].argmax(1)==truth[short]).mean():.2f}%')

    # ---- train the dedicated models on short TRAIN documents only --------
    short_train = [r for r in train if len(r['article']) < SHORT_CHARS]
    y_tr = np.asarray([LABELS.index(r['label']) for r in short_train])
    texts_tr = [r['article'] for r in short_train]
    print(f'short training documents {len(short_train)}')

    val_short_records = [val_meta[i] for i in val_frame['item_ids'] if len(val_meta[i]['article']) < SHORT_CHARS]
    val_short_text = [r['article'] for r in val_short_records]
    val_short_truth = val_truth[val_short]

    test_short_records = [meta[i] for i in ids if len(meta[i]['article']) < SHORT_CHARS]
    test_short_text = [r['article'] for r in test_short_records]
    test_short_truth = truth[short]

    # ---- selection on VALIDATION ----------------------------------------
    grid = np.round(np.arange(0.0, 0.85, 0.05), 2)
    selection = {}
    for name in VARIANTS:
        model = fit_variant(name, texts_tr, y_tr)
        proba_val = model.predict_proba(val_short_text)
        standalone = float((proba_val.argmax(1) == val_short_truth).mean())
        w, blended = best_weight(e30_val[val_short], proba_val, val_short_truth, grid)
        selection[name] = {'standalone_val': standalone, 'weight': w, 'blended_val': blended}
        print(f'  [val] {name:12} standalone {100*standalone:6.2f}%  '
              f'w={w:.2f} -> {100*blended:6.2f}%')

    chosen = max(selection, key=lambda k: selection[k]['blended_val'])
    weight = selection[chosen]['weight']
    print(f'  selected on val: {chosen}  w={weight:.2f}')

    # ---- single evaluation on TEST --------------------------------------
    model = fit_variant(chosen, texts_tr, y_tr)
    proba_test = model.predict_proba(test_short_text)
    standalone_test = float((proba_test.argmax(1) == test_short_truth).mean())
    fused = (1 - weight) * e30_test[short] + weight * proba_test
    blended_test = float((fused.argmax(1) == test_short_truth).mean())
    baseline_test = float((e30_test[short].argmax(1) == test_short_truth).mean())

    print(f'\n[test] baseline {100*baseline_test:.2f}%  standalone {100*standalone_test:.2f}%  '
          f'blended {100*blended_test:.2f}%  ({100*(blended_test-baseline_test):+.2f} pp)')

    # ---- bootstrap over GROUPS, not items -------------------------------
    groups = np.asarray([meta[i]['group_id'] for i in ids if len(meta[i]['article']) < SHORT_CHARS])
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(SEED)
    base_pred = e30_test[short].argmax(1)
    blend_pred = fused.argmax(1)
    deltas = np.empty(4000)
    for i in range(4000):
        pick = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([np.flatnonzero(groups == unique[g]) for g in pick])
        deltas[i] = ((blend_pred[idx] == test_short_truth[idx]).mean()
                     - (base_pred[idx] == test_short_truth[idx]).mean())
    ci = [float(100 * np.percentile(deltas, 2.5)), float(100 * np.percentile(deltas, 97.5))]
    print(f'  group bootstrap delta {100*deltas.mean():+.2f} pp  '
          f'95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  P(better)={float((deltas > 0).mean()):.3f}')

    # ---- full test set effect -------------------------------------------
    full = e30_test.copy()
    full[short] = fused
    full_acc = float((full.argmax(1) == truth).mean())
    base_full = float((e30_test.argmax(1) == truth).mean())
    print(f'  full test set {100*base_full:.2f}% -> {100*full_acc:.2f}% ({100*(full_acc-base_full):+.2f} pp)')

    payload = {
        'protocol': {
            'variant_selected_on': 'validation split',
            'weight_selected_on': 'validation split',
            'test_used_for': 'single final evaluation',
            'bootstrap_unit': 'group',
        },
        'n_short_val': int(val_short.sum()),
        'n_short_test': int(short.sum()),
        'selection': selection,
        'chosen_variant': chosen,
        'weight': weight,
        'test': {'baseline': baseline_test, 'standalone': standalone_test,
                 'blended': blended_test, 'delta_pp': 100 * (blended_test - baseline_test)},
        'bootstrap': {'mean_delta_pp': float(100 * deltas.mean()), 'ci95_pp': ci,
                      'p_better': float((deltas > 0).mean())},
        'full_test': {'baseline': base_full, 'routed': full_acc,
                      'delta_pp': 100 * (full_acc - base_full)},
    }
    (ROOT / '.local/short_document_model.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print('\nwrote .local/short_document_model.json')


if __name__ == '__main__':
    main()
