"""Verify the Yingqi snapshot using only Python's standard library."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("human_written", "human_ai_polished", "ai_generated", "ai_humanized")


def read(path):
    return json.loads(path.read_text())


def macro_f1(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected or len({row["item_id"] for row in rows}) != expected:
        raise ValueError(f"Invalid prediction count/IDs: {path}")
    cm = [[0] * 4 for _ in range(4)]
    for row in rows:
        if not set(row).issubset({
            "item_id", "group_id", "domain", "true_label", "prediction", "probabilities",
        }):
            raise ValueError(f"Unreviewed prediction fields: {path}")
        p = [row["probabilities"][label] for label in LABELS]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p):
            raise ValueError("Invalid probabilities")
        if not math.isclose(sum(p), 1, abs_tol=1e-5):
            raise ValueError("Probabilities do not sum to one")
        pred = max(range(4), key=p.__getitem__)
        cm[LABELS.index(row["true_label"])][pred] += 1
    scores = []
    for i in range(4):
        denominator = sum(cm[i]) + sum(row[i] for row in cm)
        scores.append(2 * cm[i][i] / denominator if denominator else 0)
    return sum(scores) / 4


def check_score(path, expected_count, score):
    actual = macro_f1(path, expected_count)
    if not math.isclose(actual, score, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch: {path}: {actual} != {score}")
    return actual


def main():
    snapshot = ROOT / "experiments/reports/yingqi_snapshot"
    manifest = read(snapshot / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        path = snapshot / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    same = ROOT / "experiments/reports/same_window/seed_3407"
    for record in read(same / "download_manifest.json")["files"]:
        if hashlib.sha256((same / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Same-window hash mismatch: {record['path']}")
    for condition in ("text", "graph", "shuffled", "edu"):
        run = same / condition
        result = read(run / "result.json")
        if result["status"] != "completed" or result["seed"] != 3407:
            raise ValueError("Incomplete/mislabeled same-window run")
        for split, count in (("validation", 1600), ("test", 3194)):
            check_score(run / f"{split}_predictions.jsonl", count, result[split]["macro_f1"])
        print(f"same-window {condition}: test {result['test']['macro_f1']:.4%}")
    count = 0
    for run in sorted((snapshot / "fusion").glob("*/*_s*")):
        if not (run / "result.json").exists():
            continue
        result = read(run / "result.json")
        if result["test_evaluated"] or result["test"] is not None:
            raise ValueError("Screening unexpectedly evaluated test")
        check_score(run / "validation_predictions.jsonl", 1600,
                    result["validation"]["macro_f1"])
        count += 1
    if count != 5:
        raise ValueError(f"Expected five completed fusion trainings, found {count}")
    selected = read(snapshot / "fusion/selection.json")
    scores = selected["all_validation_macro_f1"]
    candidates = [name for name in scores if name != "fusion_cls"]
    if selected["selected"] != max(candidates, key=scores.get):
        raise ValueError("Frozen fusion selection mismatch")
    results = read(snapshot / "stacking/results.json")
    score = check_score(snapshot / "stacking/test_predictions.jsonl", 3194,
                        results["hart"]["stacking"]["macro_f1"])
    external_result = read(snapshot / "external/results.json")
    external = check_score(
        snapshot / "external/stack_all_original_predictions.jsonl", 359,
        external_result["metrics"]["stack_all_original"]["four_class"]["macro_f1"],
    )
    print(f"fusion: {count} completed validation-only trainings, selected {selected['selected']}")
    print(f"stacking: HART {score:.4%}; MixSet subset {external:.4%}")
    latest = ROOT / "docs/project-archive/latest"
    summary = read(latest / "fusion/summary.json")
    if summary["status"] != "completed" or summary["formal_training_runs"] != 8:
        raise ValueError("Fusion continuation incomplete")
    scores = {}
    for condition in ("fusion_cls", "fusion_gated"):
        scores[condition] = [
            check_score(latest / f"fusion/evaluation/{condition}_s{seed}/test_predictions.jsonl",
                        3194, summary["test"][condition][str(seed)]["macro_f1"])
            for seed in (3407, 42, 2026)
        ]
        mean = statistics.mean(scores[condition])
        if not math.isclose(mean, summary["macro_f1"][condition]["mean"], abs_tol=1e-12):
            raise ValueError(f"Fusion mean mismatch: {condition}")
        print(f"fusion continuation {condition}: three-seed test mean {mean:.4%}")
    delta = statistics.mean(g - c for g, c in zip(scores["fusion_gated"], scores["fusion_cls"]))
    if not math.isclose(delta, summary["paired_candidate_minus_baseline"]["mean"], abs_tol=1e-12):
        raise ValueError("Fusion paired difference mismatch")
    for condition in ("deberta_full", "roberta_full"):
        run = latest / "encoder-pilot/formal" / condition
        result = read(run / "result.json")
        if (result["status"] != "validation_completed" or result["completed_epochs"] != 4
                or result["test_evaluated"] or result["test"] is not None):
            raise ValueError(f"Incomplete or mislabeled encoder pilot: {condition}")
        score = check_score(run / "validation_predictions.jsonl", 1600,
                            result["validation"]["macro_f1"])
        print(f"encoder pilot {condition}: validation {score:.4%}")
    print("PASS: hashes, counts, labels, probabilities and saved Macro-F1 verified")


if __name__ == "__main__":
    main()
