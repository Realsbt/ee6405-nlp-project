#!/usr/bin/env python3
"""Check the committed prediction tables using only the Python standard library.

This recomputes reported metrics from saved predictions. It does not reload
checkpoints, obtain original labels, audit data splits, or repeat inference.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


LABELS = ("human_written", "human_ai_polished", "ai_generated", "ai_humanized")
FAMILIES = {
    "tfidf_lr_v1": ("tfidf_lr_v1",),
    "roberta": tuple(f"roberta_s{seed}" for seed in (3407, 42, 2026)),
    "roberta_supcon": tuple(f"roberta_supcon_s{seed}" for seed in (3407, 42, 2026)),
    "race": tuple(f"race_s{seed}" for seed in (3407, 42, 2026)),
    "race_no_supcon": tuple(f"race_no_supcon_s{seed}" for seed in (3407, 42, 2026)),
}
PRIMARY = ("tfidf_lr_v1", "roberta_s3407", "race_s3407", "race_no_supcon_s3407")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path, count):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(len(rows) == count, f"{path}: expected {count} predictions")
    ids = [row["item_id"] for row in rows]
    require(all(isinstance(item, str) and item for item in ids), f"{path}: invalid IDs")
    require(len(set(ids)) == count, f"{path}: duplicate prediction IDs")
    return rows


def close(actual, expected, context, tolerance=1e-9):
    require(isinstance(expected, (int, float)) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f"{context}: recomputed {actual!r}, saved {expected!r}")


def probabilities(row, key, context):
    values = row[key]
    require(set(values) == set(LABELS), f"{context}: unexpected score labels")
    scores = [values[label] for label in LABELS]
    require(all(isinstance(value, (int, float)) and math.isfinite(value)
                and 0 <= value <= 1 for value in scores), f"{context}: invalid probability")
    require(math.isclose(sum(scores), 1, rel_tol=0, abs_tol=1e-6),
            f"{context}: probabilities do not sum to one")
    return scores


def auroc(targets, scores):
    """Mann–Whitney AUROC, assigning average ranks to tied scores."""
    ordered = sorted(zip(scores, targets))
    positives = sum(targets)
    negatives = len(targets) - positives
    require(positives > 0 and negatives > 0, "AUROC requires both classes")
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(target for _, target in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification(targets, predictions, classes):
    matrix = [[0] * classes for _ in range(classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    per_class = []
    for index in range(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(dict(support=support, precision=precision, recall=recall, f1=f1))
    metrics = {
        "samples": len(targets),
        "accuracy": sum(matrix[i][i] for i in range(classes)) / len(targets),
        "macro_f1": statistics.mean(row["f1"] for row in per_class),
        "macro_precision": statistics.mean(row["precision"] for row in per_class),
        "macro_recall": statistics.mean(row["recall"] for row in per_class),
    }
    return metrics, matrix, per_class


def binary_metrics(targets, scores):
    predictions = [int(score >= 0.5) for score in scores]
    metrics, matrix, per_class = classification(targets, predictions, 2)
    return {
        "samples": metrics["samples"], "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["macro_recall"],
        "auroc": auroc(targets, scores), "ai_precision": per_class[1]["precision"],
        "ai_recall": per_class[1]["recall"],
        "human_false_positive_rate_at_0_5": matrix[0][1] / sum(matrix[0]),
        "confusion_matrix_human_ai": matrix,
    }


def compare_metrics(actual, saved, context):
    for key, value in actual.items():
        if isinstance(value, list):
            require(value == saved[key], f"{context}/{key}: confusion matrix mismatch")
        else:
            close(value, saved[key], f"{context}/{key}")


def check_hash(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == expected, f"{path}: stored summary hash mismatch")


def check(reports):
    summary = read_json(reports / "comparison/summary.json")
    auxiliary = read_json(reports / "auxiliary/summary.json")
    require(summary["status"] == auxiliary["status"] == "completed", "Summary is incomplete")
    require(not summary["missing_verified_runs"] and not summary["missing_auxiliary_runs"]
            and not auxiliary["missing_runs"], "Summary lists missing runs")
    runs = tuple(run for family in FAMILIES.values() for run in family)
    require(set(summary["analyses"]) == set(runs), "Unexpected main experiment set")
    require(set(summary["hc3_binary_auxiliary"]) == set(auxiliary["analyses"]) == set(PRIMARY),
            "Unexpected auxiliary experiment set")
    require(summary["test_samples"] == 3194, "Unexpected HART test size")
    recomputed, saved_rows, identity = {}, {}, None
    for run in runs:
        folder = reports / run
        result = read_json(folder / "result.json")
        verification = read_json(folder / "verification.json")
        require(result["status"] == "completed" and verification["status"] == "passed",
                f"{run}: saved result or verification is incomplete")
        require(result["data_protocol_sha256"] == summary["data_protocol_sha256"],
                f"{run}: protocol hash differs from summary")
        rows = read_rows(folder / "test_predictions.jsonl", 3194)
        current = [(r["item_id"], r["true_label"], r["group_id"], r["domain"]) for r in rows]
        if identity is None:
            identity = current
        require(current == identity, f"{run}: IDs, labels, groups, domains, or order differ")
        targets, predictions, all_scores = [], [], []
        for row in rows:
            scores = probabilities(row, "probabilities", f"{run}/{row['item_id']}")
            prediction = max(range(4), key=scores.__getitem__)
            require(row["prediction"] == LABELS[prediction], f"{run}: prediction is not argmax")
            targets.append(LABELS.index(row["true_label"]))
            predictions.append(prediction)
            all_scores.append(scores)
        metrics, matrix, per_class = classification(targets, predictions, 4)
        aucs = [auroc([target == i for target in targets], [scores[i] for scores in all_scores])
                for i in range(4)]
        metrics["macro_auroc_ovr"] = statistics.mean(aucs)
        metrics["confusion_matrix"] = matrix
        compare_metrics(metrics, result["test"], f"{run}/result")
        compare_metrics(metrics, summary["analyses"][run]["test"], f"{run}/summary")
        for i, label in enumerate(LABELS):
            compare_metrics(dict(per_class[i], auroc_ovr=aucs[i]),
                            result["test"]["per_class"][label], f"{run}/{label}")
        check_hash(folder / "result.json", summary["analyses"][run]["result_sha256"])
        check_hash(folder / "test_predictions.jsonl", summary["analyses"][run]["predictions_sha256"])
        recomputed[run], saved_rows[run] = metrics, (targets, all_scores)
        print(f"HART {run:24s} accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}")

    require(set(summary["training_seed_summaries"]) == set(FAMILIES), "Unexpected model families")
    for family, family_runs in FAMILIES.items():
        recorded = summary["training_seed_summaries"][family]
        require(recorded["runs"] == list(family_runs), f"{family}: incorrect run order")
        for metric in ("accuracy", "macro_f1", "macro_auroc_ovr"):
            values = [recomputed[run][metric] for run in family_runs]
            saved = recorded[metric]
            require(saved["n"] == len(values) and len(saved["values"]) == len(values),
                    f"{family}/{metric}: incorrect run count")
            for value, expected in zip(values, saved["values"]):
                close(value, expected, f"{family}/{metric}/values")
            close(statistics.mean(values), saved["mean"], f"{family}/{metric}/mean")
            if len(values) > 1:
                close(statistics.stdev(values), saved["sample_sd"], f"{family}/{metric}/SD")
            else:
                require(saved["sample_sd"] is None, f"{family}: single run must have null SD")

    factorial = summary["factorial_ablation"]
    require(factorial["status"] == "completed" and not factorial["missing_runs"], "Incomplete factorial analysis")
    contrasts = {
        "supcon_with_roberta": {"roberta_supcon": 1, "roberta": -1},
        "supcon_with_race": {"race": 1, "race_no_supcon": -1},
        "race_minus_roberta_with_ce": {"race_no_supcon": 1, "roberta": -1},
        "race_minus_roberta_with_supcon": {"race": 1, "roberta_supcon": -1},
        "interaction": {"race": 1, "race_no_supcon": -1, "roberta_supcon": -1, "roberta": 1},
    }
    require(set(factorial["effects"]) == set(contrasts), "Incorrect factorial contrasts")
    for name, coefficients in contrasts.items():
        effect = factorial["effects"][name]
        require(effect["seeds"] == [3407, 42, 2026] and effect["coefficients"] == coefficients,
                f"{name}: wrong seed pairing or contrast")
        for metric in ("accuracy", "macro_f1", "macro_auroc_ovr"):
            values = [sum(weight * recomputed[f"{kind}_s{seed}"][metric] for kind, weight in coefficients.items())
                      for seed in (3407, 42, 2026)]
            recorded = effect["metrics"][metric]
            require(recorded["n"] == 3 and len(recorded["values"]) == 3, f"{name}: wrong count")
            for actual, saved in zip(values, recorded["values"]):
                close(actual, saved, f"{name}/{metric}/paired value")
            close(statistics.mean(values), recorded["mean"], f"{name}/{metric}/mean")
            close(statistics.stdev(values), recorded["sample_sd"], f"{name}/{metric}/SD")

    identity = None
    for run in PRIMARY:
        folder = reports / run
        result = read_json(folder / "hc3_aux_v1_1_result.json")
        require(result["status"] == "completed" and result["protocol_id"] == "hc3_aux_v1_1"
                and result["threshold"] == 0.5 and result["fit_or_tuning_on_hc3"] is False,
                f"{run}: auxiliary protocol changed")
        require(result["protocol_sha256"] == auxiliary["protocol_sha256"],
                f"{run}: auxiliary protocol hash differs from summary")
        rows = read_rows(folder / "hc3_aux_v1_1_predictions.jsonl", 1000)
        current = [(r["item_id"], r["true_binary_label"], r["question_group"], r["source"]) for r in rows]
        if identity is None:
            identity = current
        require(current == identity, f"{run}: auxiliary IDs, labels, groups, sources, or order differ")
        targets, scores = [], []
        for row in rows:
            values = probabilities(row, "four_class_scores", f"{run}/{row['item_id']}")
            score = values[2] + values[3]
            require(row["true_binary_label"] in (0, 1), f"{run}: invalid binary label")
            # Neural inference originally added the two probabilities in FP32;
            # JSON reload sums them in Python's FP64. Allow only that rounding.
            close(score, row["ai_creator_score"], f"{run}: AI creator score", tolerance=1e-7)
            require(row["binary_prediction"] == int(score >= 0.5), f"{run}: threshold mismatch")
            targets.append(row["true_binary_label"])
            scores.append(score)
        require(sum(targets) == 500, f"{run}: expected 500 human and 500 AI answers")
        metrics = binary_metrics(targets, scores)
        compare_metrics(metrics, result["metrics"], f"{run}/HC3 result")
        compare_metrics(metrics, summary["hc3_binary_auxiliary"][run]["metrics"], f"{run}/HC3 comparison")
        compare_metrics(metrics, auxiliary["analyses"][run]["binary_metrics"], f"{run}/HC3 auxiliary")
        check_hash(folder / "hc3_aux_v1_1_predictions.jsonl", auxiliary["analyses"][run]["predictions_sha256"])
        hart_targets, hart_scores = saved_rows[run]
        aligned = binary_metrics([int(target >= 2) for target in hart_targets],
                                 [values[2] + values[3] for values in hart_scores])
        compare_metrics(aligned, auxiliary["analyses"][run]["hart_aligned_binary_reference"],
                        f"{run}/HART binary reference")
        print(f"HC3  {run:24s} accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f} auroc={metrics['auroc']:.6f}")
    print("PASS: 13 HART runs, 4 HC3 runs, seed aggregates and five paired factorial contrasts agree.")
    print("Scope: saved-prediction consistency only; no original data or checkpoint replay.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/reports",
                        help="directory containing the committed reports")
    args = parser.parse_args()
    try:
        check(args.reports)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
