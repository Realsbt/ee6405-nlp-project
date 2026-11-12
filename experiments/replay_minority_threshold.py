"""Replay AI-humanized decision thresholds on frozen HART predictions."""

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.common import LABELS, classification_metrics, read_jsonl, write_json
from experiments.paths import PROJECT_ROOT
from experiments.train_stacking import aligned_probabilities, labels

TARGET = LABELS.index("ai_humanized")
GENERATED = LABELS.index("ai_generated")
WEIGHTS = (1, 1.25, 1.5, 2, 3, 4)
EVIDENCE = PROJECT_ROOT / "docs/project-archive/latest"
FUSION = EVIDENCE / "fusion/evaluation"
SNAPSHOT = PROJECT_ROOT / "experiments/reports/yingqi_snapshot"
DATA_ROOT = EVIDENCE / "splits"


def weighted(probabilities, weight, pair_only):
    scores = probabilities.copy()
    eligible = probabilities.argmax(axis=1) == GENERATED if pair_only else slice(None)
    scores[eligible, TARGET] *= weight
    return scores / scores.sum(axis=1, keepdims=True)


def select(y, probabilities, pair_only):
    base = probabilities.argmax(axis=1)
    macro_floor = f1_score(y, base, average="macro")
    precision_floor = precision_score(y == TARGET, base == TARGET, zero_division=0) - .05
    candidates = []
    for weight in WEIGHTS:
        pred = weighted(probabilities, weight, pair_only).argmax(axis=1)
        macro = f1_score(y, pred, average="macro")
        precision = precision_score(y == TARGET, pred == TARGET, zero_division=0)
        if macro >= macro_floor and precision >= precision_floor:
            candidates.append((f1_score(y == TARGET, pred == TARGET), macro, -weight))
    return -max(candidates)[2]


def compare(y, probabilities, folds):
    result = {"base": classification_metrics(y, probabilities)}
    for method, pair_only in (("all_competitors", False), ("generated_only", True)):
        oof = np.empty_like(probabilities)
        fold_results = []
        for fit, held in folds:
            weight = select(y[fit], probabilities[fit], pair_only)
            oof[held] = weighted(probabilities[held], weight, pair_only)
            target = y[held] == TARGET
            base = probabilities[held].argmax(axis=1)
            adjusted = oof[held].argmax(axis=1)
            fold_results.append({
                "weight": weight,
                "target_support": int(target.sum()),
                "base_target_hits": int((target & (base == TARGET)).sum()),
                "adjusted_target_hits": int((target & (adjusted == TARGET)).sum()),
                "macro_f1_delta": float(f1_score(y[held], adjusted, average="macro")
                                        - f1_score(y[held], base, average="macro")),
            })
        result[method] = {
            "folds": fold_results,
            "oof": classification_metrics(y, oof),
            "final_weight": select(y, probabilities, pair_only),
        }
    return result


def main():
    dev = read_jsonl(DATA_ROOT / "val.jsonl")
    y = labels(dev)
    groups = np.array([row["split_block_id"] for row in dev])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=3407).split(y, y, groups))
    assert all(not (set(groups[fit]) & set(groups[held])) for fit, held in folds)
    assert np.array_equal(np.sort(np.concatenate([held for _, held in folds])), np.arange(len(y)))

    sources = {
        f"gated_{seed}": (folder / "val_predictions.jsonl", folder / "test_predictions.jsonl")
        for seed in (3407, 42, 2026)
        for folder in (FUSION / f"fusion_gated_s{seed}",)
    }
    sources["stacking"] = tuple(
        SNAPSHOT / "stacking" / f"{part}_predictions.jsonl" for part in ("development_oof", "test")
    )
    frozen_dev = {name: aligned_probabilities(dev, path) for name, (path, _) in sources.items()}
    path = EVIDENCE / "encoder-pilot/formal/deberta_full/validation_predictions.jsonl"
    frozen_dev["gated_3407_deberta"] = (frozen_dev["gated_3407"] + aligned_probabilities(dev, path)) / 2

    assert np.array_equal(weighted(frozen_dev["gated_3407"], 1, False).argmax(1),
                          frozen_dev["gated_3407"].argmax(1))
    results = {name: compare(y, p, folds) for name, p in frozen_dev.items()}
    output = Path("outputs/minority_threshold_replay")
    write_json(output / "validation.json", results)

    test = read_jsonl(DATA_ROOT / "test.jsonl")
    assert set(groups).isdisjoint(row["split_block_id"] for row in test)
    test_y = labels(test)
    transfer = {}
    for name, (_, path) in sources.items():
        probabilities = aligned_probabilities(test, path)
        transfer[name] = {"base": classification_metrics(test_y, probabilities)}
        for method, pair_only in (("all_competitors", False), ("generated_only", True)):
            weight = results[name][method]["final_weight"]
            transfer[name][method] = classification_metrics(test_y, weighted(probabilities, weight, pair_only))
    write_json(output / "historical_test.json", transfer)


if __name__ == "__main__":
    main()
