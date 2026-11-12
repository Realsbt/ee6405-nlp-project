"""Revised evaluation protocol for the four-class provenance task.

Implements the four changes recommended by the error analysis:

1. Report `ai_humanized` as two sub-populations (`tool` / `human`) instead of
   one class, and add the coarser three-class and binary views.
2. Quantify the irreducible error floor instead of chasing the four-class score.
3. Handle short documents separately (length-aware blending + abstention).
4. Report grouped by generator family rather than as a single aggregate.

Every number is computed locally on the published 3,194-item test set from the
already-recomputed E30 probabilities, so nothing here depends on the original
host. Writes `.local/revised_evaluation.json`.
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
SEED = 20261112


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_predictions() -> tuple[list[str], np.ndarray, np.ndarray]:
    data = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    probabilities = np.asarray(data['E30']['probabilities'], dtype=float)
    ids, truth = [], []
    with (ROOT / 'experiments/reports/tfidf_lr_v1/test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            ids.append(record['item_id'])
            truth.append(LABELS.index(record['true_label']))
    return ids, np.asarray(truth), probabilities


def load_metadata(ids: list[str]) -> dict[str, dict]:
    wanted = set(ids)
    meta: dict[str, dict] = {}
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


def load_auxiliary() -> dict[str, np.ndarray]:
    """Auxiliary scorers used by the short-document routing experiment."""
    rein = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    info = json.loads((ROOT / '.local/info_methods_published_test.json').read_text(encoding='utf-8'))
    return {
        'TF-IDF + LR': np.asarray(rein['tfidf_lr_v1']['probabilities'], dtype=float),
        'M2 stylometric': np.asarray(info['M2_stylometric']['proba'], dtype=float),
        'M3 char n-gram CE': np.asarray(info['M3_char_ngram_cross_entropy']['proba'], dtype=float),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(truth: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict:
    pred = probabilities.argmax(1)
    accuracy = float((pred == truth).mean())
    f1s, recalls = [], []
    for c in range(n_classes):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
    macro_f1 = float(np.mean(f1s))
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'J': accuracy + macro_f1,
            'per_class_f1': f1s, 'per_class_recall': recalls, 'n': int(len(truth))}


def collapse(probabilities: np.ndarray, mapping: list[int], n_out: int) -> np.ndarray:
    """Sum the source-class probabilities into the target classes."""
    out = np.zeros((len(probabilities), n_out))
    for source, target in enumerate(mapping):
        out[:, target] += probabilities[:, source]
    return out


def group_folds(groups: np.ndarray, k: int = 2) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    assignment = {g: i % k for i, g in enumerate(unique)}
    return [np.asarray([assignment[g] == f for g in groups]) for f in range(k)]


def crossfit_blend(truth: np.ndarray, groups: np.ndarray, base: np.ndarray,
                   extra: np.ndarray, mask: np.ndarray, grid: np.ndarray) -> dict:
    """Pick the blend weight on one group-disjoint fold, apply it to the other."""
    folds = group_folds(groups)
    fused = base.copy()
    weights = []
    for f, apply_mask in enumerate(folds):
        fit_mask = mask & ~apply_mask
        best_w, best_acc = 0.0, -1.0
        for w in grid:
            blended = (1 - w) * base + w * extra
            acc = float((blended[fit_mask].argmax(1) == truth[fit_mask]).mean())
            if acc > best_acc:
                best_w, best_acc = float(w), acc
        weights.append(best_w)
        target = mask & apply_mask
        fused[target] = (1 - best_w) * base[target] + best_w * extra[target]
    return {'probabilities': fused, 'weights': weights}


# --------------------------------------------------------------------------
# 1. humanized sub-populations + coarser views
# --------------------------------------------------------------------------
def humanized_split(ids, truth, probabilities, meta) -> dict:
    pred = probabilities.argmax(1)
    kinds = np.asarray([meta[i].get('model_name', 'N/A') for i in ids])

    rows = {}
    for kind in ('tool', 'human'):
        mask = (truth == 3) & (kinds == kind)
        n = int(mask.sum())
        correct = int((pred[mask] == 3).sum())
        rows[kind] = {
            'n': n,
            'correct': correct,
            'accuracy': correct / n if n else 0.0,
            'predicted_as': {LABELS[c]: int((pred[mask] == c).sum()) for c in range(4)},
        }

    # five-class view: HUM is split into HUM-tool / HUM-human
    five = ['HW', 'POL', 'GEN', 'HUM-tool', 'HUM-human']
    five_truth = np.asarray([3 if (t == 3 and k == 'tool') else 4 if (t == 3 and k == 'human') else t
                             for t, k in zip(truth, kinds)])
    five_pred = np.asarray([3 if (p == 3 and k == 'tool') else 4 if (p == 3 and k == 'human') else p
                            for p, k in zip(pred, kinds)])
    confusion = [[int(((five_truth == t) & (five_pred == p)).sum()) for p in range(5)] for t in range(5)]
    five_metrics = metrics(five_truth, np.eye(5)[five_pred], 5)

    # three-class view: drop the humanized items entirely
    keep = truth != 3
    three = metrics(truth[keep], probabilities[keep][:, :3], 3)

    # binary view: human-written vs machine-involved
    binary_truth = (truth != 0).astype(int)
    binary_proba = collapse(probabilities, [0, 1, 1, 1], 2)
    binary = metrics(binary_truth, binary_proba, 2)

    return {'subclasses': rows, 'five_class': {'labels': five, 'confusion': confusion, **five_metrics},
            'three_class': three, 'binary': binary}


# --------------------------------------------------------------------------
# 2. irreducible error floor
# --------------------------------------------------------------------------
def error_floor(ids, truth, probabilities, meta, separability) -> dict:
    pred = probabilities.argmax(1)
    kinds = np.asarray([meta[i].get('model_name', 'N/A') for i in ids])
    wrong = pred != truth
    total_errors = int(wrong.sum())

    human_rewrite = (truth == 3) & (kinds == 'human')
    human_errors = int((wrong & human_rewrite).sum())

    # If every human-rewritten item were scored at the dedicated binary
    # classifier's recall, how many of those errors would remain?
    recall = next((r['combined_recall'] for r in separability if r['kind'] == 'human'), 0.0)
    n_human = int(human_rewrite.sum())
    recoverable = recall * n_human
    remaining = human_errors - recoverable

    ceiling_accuracy = (len(truth) - (total_errors - recoverable)) / len(truth)

    return {
        'total_errors': total_errors,
        'human_rewrite_items': n_human,
        'human_rewrite_errors': human_errors,
        'human_rewrite_error_share': human_errors / total_errors if total_errors else 0.0,
        'dedicated_binary_recall': recall,
        'recoverable_errors': recoverable,
        'irreducible_errors': remaining,
        'irreducible_share': remaining / total_errors if total_errors else 0.0,
        'ceiling_accuracy': ceiling_accuracy,
        'current_accuracy': float((pred == truth).mean()),
    }


# --------------------------------------------------------------------------
# 3. short-document handling
# --------------------------------------------------------------------------
def short_documents(ids, truth, probabilities, meta, auxiliary) -> dict:
    lengths = np.asarray([len(meta[i]['article']) for i in ids])
    groups = np.asarray([meta[i]['group_id'] for i in ids])
    short = lengths < 1000
    pred = probabilities.argmax(1)

    baseline = float((pred[short] == truth[short]).mean())

    grid = np.round(np.arange(0.0, 0.65, 0.05), 2)
    blends = {}
    for name, extra in auxiliary.items():
        result = crossfit_blend(truth, groups, probabilities, extra, short, grid)
        fused = result['probabilities']
        blends[name] = {
            'accuracy': float((fused[short].argmax(1) == truth[short]).mean()),
            'weights': result['weights'],
            'standalone_accuracy': float((extra[short].argmax(1) == truth[short]).mean()),
        }

    best_name = max(blends, key=lambda k: blends[k]['accuracy'])

    # abstention curve on the short slice
    confidence = probabilities[short].max(1)
    abstention = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = confidence >= tau
        abstention.append({
            'tau': tau,
            'coverage': float(keep.mean()),
            'accuracy_at_coverage': float((pred[short][keep] == truth[short][keep]).mean()) if keep.any() else 0.0,
        })

    # same curve on the full test set, for reference
    full_confidence = probabilities.max(1)
    full_abstention = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = full_confidence >= tau
        full_abstention.append({
            'tau': tau,
            'coverage': float(keep.mean()),
            'accuracy_at_coverage': float((pred[keep] == truth[keep]).mean()) if keep.any() else 0.0,
        })

    return {
        'n_short': int(short.sum()),
        'baseline_accuracy': baseline,
        'blends': blends,
        'best_blend': best_name,
        'best_blend_accuracy': blends[best_name]['accuracy'],
        'abstention_short': abstention,
        'abstention_full': full_abstention,
    }


# --------------------------------------------------------------------------
# 4. generator-grouped reporting
# --------------------------------------------------------------------------
def generator_families(ids, truth, probabilities, meta) -> dict:
    pred = probabilities.argmax(1)
    names = [meta[i].get('model_name', 'N/A') for i in ids]

    def family(name: str) -> str:
        if name == 'N/A':
            return 'human_written (no generator)'
        if name == 'human':
            return 'human rewrite'
        if name == 'tool':
            return 'tool rewrite'
        if '_THEN_' in name:
            return 'two-stage (A_THEN_B)'
        return 'single-stage LLM'

    families = [family(n) for n in names]
    rows = {}
    for fam in sorted(set(families)):
        mask = np.asarray([f == fam for f in families])
        n = int(mask.sum())
        correct = int((pred[mask] == truth[mask]).sum())
        rows[fam] = {'n': n, 'correct': correct, 'errors': n - correct,
                     'accuracy': correct / n if n else 0.0,
                     'share_of_errors': 0.0}

    total_errors = int((pred != truth).sum())
    for row in rows.values():
        row['share_of_errors'] = row['errors'] / total_errors if total_errors else 0.0

    # per-generator detail, restricted to machine-involved items
    detail = {}
    for name in sorted(set(names)):
        if name in ('N/A', 'human', 'tool'):
            continue
        mask = np.asarray([n == name for n in names])
        n = int(mask.sum())
        if n < 10:
            continue
        detail[name] = {'n': n, 'accuracy': float((pred[mask] == truth[mask]).mean()),
                        'two_stage': '_THEN_' in name}

    # accuracy by label inside each family
    by_label = {}
    for fam in sorted(set(families)):
        fam_mask = np.asarray([f == fam for f in families])
        by_label[fam] = {}
        for c, label in enumerate(LABELS):
            mask = fam_mask & (truth == c)
            if mask.sum() == 0:
                continue
            by_label[fam][label] = {'n': int(mask.sum()),
                                    'accuracy': float((pred[mask] == c).mean())}

    # two-stage items: which stage actually drives the difficulty?
    stages = {'first': collections.defaultdict(list), 'second': collections.defaultdict(list)}
    for k, name in enumerate(names):
        if '_THEN_' not in name:
            continue
        first, second = name.split('_THEN_', 1)
        stages['first'][first].append(k)
        stages['second'][second].append(k)
    stage_rows = {}
    for stage, buckets in stages.items():
        stage_rows[stage] = {}
        for name, idx in buckets.items():
            idx = np.asarray(idx)
            stage_rows[stage][name] = {'n': int(len(idx)),
                                       'accuracy': float((pred[idx] == truth[idx]).mean())}

    return {'families': rows, 'per_generator': detail, 'family_by_label': by_label,
            'two_stage_by_stage': stage_rows}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def main() -> None:
    ids, truth, probabilities = load_predictions()
    meta = load_metadata(ids)
    auxiliary = load_auxiliary()
    separability_path = ROOT / '.local/separability.json'
    if not separability_path.exists():
        raise SystemExit('missing .local/separability.json — run '
                         '`python scripts/test_humanized_separability.py` first')
    separability = json.loads(separability_path.read_text(encoding='utf-8'))

    baseline = metrics(truth, probabilities, 4)
    print(f'baseline E30: acc {100*baseline["accuracy"]:.2f}%  macro-F1 {100*baseline["macro_f1"]:.2f}%')

    split = humanized_split(ids, truth, probabilities, meta)
    print('\n[1] humanized sub-populations')
    for kind, row in split['subclasses'].items():
        print(f'  {kind:6} n={row["n"]:4}  acc {100*row["accuracy"]:6.2f}%  '
              f'predicted {row["predicted_as"]}')
    print(f'  five-class  acc {100*split["five_class"]["accuracy"]:.2f}%  '
          f'macro-F1 {100*split["five_class"]["macro_f1"]:.2f}%')
    print(f'  three-class acc {100*split["three_class"]["accuracy"]:.2f}%  '
          f'macro-F1 {100*split["three_class"]["macro_f1"]:.2f}%  (n={split["three_class"]["n"]})')
    print(f'  binary      acc {100*split["binary"]["accuracy"]:.2f}%  '
          f'macro-F1 {100*split["binary"]["macro_f1"]:.2f}%')

    floor = error_floor(ids, truth, probabilities, meta, separability)
    print('\n[2] irreducible error floor')
    print(f'  total errors {floor["total_errors"]}  human-rewrite errors {floor["human_rewrite_errors"]} '
          f'({100*floor["human_rewrite_error_share"]:.1f}%)')
    print(f'  dedicated binary recall {100*floor["dedicated_binary_recall"]:.1f}%  '
          f'irreducible {floor["irreducible_errors"]:.1f} ({100*floor["irreducible_share"]:.1f}%)')
    print(f'  ceiling {100*floor["ceiling_accuracy"]:.2f}% vs current {100*floor["current_accuracy"]:.2f}%')

    short = short_documents(ids, truth, probabilities, meta, auxiliary)
    print('\n[3] short documents (<1k chars)')
    print(f'  n={short["n_short"]}  baseline {100*short["baseline_accuracy"]:.2f}%')
    for name, row in short['blends'].items():
        print(f'  blend with {name:20} {100*row["accuracy"]:6.2f}%  '
              f'(standalone {100*row["standalone_accuracy"]:5.2f}%, weights {row["weights"]})')
    print(f'  best: {short["best_blend"]} -> {100*short["best_blend_accuracy"]:.2f}%')

    families = generator_families(ids, truth, probabilities, meta)
    print('\n[4] generator families')
    for fam, row in sorted(families['families'].items(), key=lambda kv: kv[1]['accuracy']):
        print(f'  {fam:32} n={row["n"]:5}  acc {100*row["accuracy"]:6.2f}%  '
              f'errors {row["errors"]:4} ({100*row["share_of_errors"]:5.1f}% of all)')
    for stage, buckets in families['two_stage_by_stage'].items():
        print(f'  two-stage by {stage} stage:')
        for name, row in sorted(buckets.items(), key=lambda kv: -kv[1]['accuracy']):
            print(f'    {name:34} n={row["n"]:4}  acc {100*row["accuracy"]:6.2f}%')

    payload = {
        'baseline': baseline,
        'humanized_split': split,
        'error_floor': floor,
        'short_documents': short,
        'generator_families': families,
        'separability': separability,
    }
    (ROOT / '.local/revised_evaluation.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print('\nwrote .local/revised_evaluation.json')


if __name__ == '__main__':
    main()
