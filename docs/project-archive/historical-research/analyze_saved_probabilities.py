"""Describe saved predictions without training, calibration, or threshold tuning."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


LABELS = ("human_written", "human_ai_polished", "ai_generated", "ai_humanized")


def probability_vector(values):
    if set(values) != set(LABELS):
        raise ValueError("Expected exactly four named probabilities")
    probabilities = [values[label] for label in LABELS]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probability")
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
        raise ValueError("Probabilities do not sum to one")
    # Correct only the accepted serialization/softmax rounding, not model scores.
    total = sum(probabilities)
    return [p / total for p in probabilities]


def binary_metrics(pairs):
    """Brier and equal-width reliability bins for the AI-creator event."""
    if not pairs or any(y not in (0, 1) or not math.isfinite(p)
                        or not 0 <= p <= 1 for y, p in pairs):
        raise ValueError("Need nonempty binary labels and finite probabilities")
    bins = [[] for _ in range(10)]
    for y, p in pairs:
        bins[min(int(p * 10), 9)].append((y, p))
    details = []
    ece = 0.0
    for index, rows in enumerate(bins):
        mean_score = sum(p for y, p in rows) / len(rows) if rows else None
        frequency = sum(y for y, p in rows) / len(rows) if rows else None
        if rows:
            ece += len(rows) / len(pairs) * abs(mean_score - frequency)
        details.append({"lower": index / 10, "upper": (index + 1) / 10,
                        "count": len(rows), "mean_ai_score": mean_score,
                        "observed_ai_fraction": frequency})
    negatives = sum(y == 0 for y, p in pairs)
    positives = len(pairs) - negatives
    return {
        "n": len(pairs),
        "binary_brier": sum((p - y) ** 2 for y, p in pairs) / len(pairs),
        "positive_class_ece_10_bins": ece,
        "fpr_at_fixed_0_5": (sum(y == 0 and p >= 0.5 for y, p in pairs)
                             / negatives if negatives else None),
        "tpr_at_fixed_0_5": (sum(y == 1 and p >= 0.5 for y, p in pairs)
                             / positives if positives else None),
        "reliability_bins": details,
    }


def analyze_file(path, binary=False):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs: {path}")
    pairs, multiclass_errors = [], []
    for row in rows:
        p = probability_vector(row["four_class_scores"] if binary else row["probabilities"])
        score = p[2] + p[3]
        if binary:
            y = row["true_binary_label"]
            if not math.isclose(score, row["ai_creator_score"], abs_tol=1e-6):
                raise ValueError("Stored AI score disagrees with four-class probabilities")
        else:
            target = LABELS.index(row["true_label"])
            y = int(target >= 2)
            multiclass_errors.append(sum((value - int(i == target)) ** 2
                                         for i, value in enumerate(p)))
        pairs.append((y, score))
    result = binary_metrics(pairs)
    result["multiclass_brier_sum_over_classes"] = (
        sum(multiclass_errors) / len(rows) if multiclass_errors else None)
    result["prediction_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    reports = root / "experiments/reports"
    hart = sorted(reports.glob("*/test_predictions.jsonl"))
    hc3 = sorted(reports.glob("*/hc3_aux_v1_1_predictions.jsonl"))
    if not hart:
        raise FileNotFoundError("No saved HART predictions")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "scope": "Post-hoc descriptive saved-prediction analysis; no training, "
                 "calibration fitting, threshold selection or independent validation.",
        "repository_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definitions": {
            "ai_creator_score": "P(ai_generated) + P(ai_humanized)",
            "human_ai_polished": "Human creator for this binary task",
            "brier": "Mean squared binary probability error; lower is better",
            "multiclass_brier": "Mean sum of four squared probability errors; range 0-2",
            "ece": "Positive-class reliability, 10 equal-width bins; descriptive, "
                   "bin-sensitive, not top-label confidence ECE",
            "rounding": "Four-class vectors within 1e-5 of unit sum are normalized "
                        "for analysis only; saved predictions remain unchanged",
            "cross_dataset_limit": "Same binary event definition, different populations",
        },
        "HART": {p.parent.name: analyze_file(p) for p in hart},
        "HC3": {p.parent.name: analyze_file(p, binary=True) for p in hc3},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("dataset run n binary_Brier AI_score_ECE FPR@0.5 TPR@0.5")
    for dataset in ("HART", "HC3"):
        for run, r in result[dataset].items():
            print(dataset, run, r["n"], *(f"{r[key]:.6f}" for key in
                  ("binary_brier", "positive_class_ece_10_bins",
                   "fpr_at_fixed_0_5", "tpr_at_fixed_0_5")))


if __name__ == "__main__":
    main()
