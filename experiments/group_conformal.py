"""Group-max split conformal sets for frozen four-class classifiers (stdlib only).

Calibration units are independent source groups, not correlated text variants.
The API only needs model scores at prediction time; group metadata is not used
to decode labels or impose any composition constraint on a document batch.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

LABELS = ("human_written", "human_ai_polished", "ai_generated", "ai_humanized")


def validate_probabilities(probabilities):
    if set(probabilities) != set(LABELS):
        raise ValueError("Expected exactly the four HART probability keys")
    values = list(probabilities.values())
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Probabilities must be finite numbers in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
        raise ValueError("Probabilities must sum to one")


def validate_rows(rows):
    if not rows:
        raise ValueError("Prediction file is empty")
    seen = set()
    for row in rows:
        for key in ("item_id", "group_id", "domain", "true_label", "probabilities"):
            if key not in row:
                raise ValueError("Missing field: " + key)
        for key in ("item_id", "group_id", "domain"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("Expected nonempty string: " + key)
        if row["item_id"] in seen:
            raise ValueError("Duplicate item_id: " + row["item_id"])
        seen.add(row["item_id"])
        if row["true_label"] not in LABELS:
            raise ValueError("Unknown true label")
        validate_probabilities(row["probabilities"])
    return rows


def load_rows(path):
    return validate_rows([json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()])


def split_groups(rows, seed="6405-conformal-v1", fraction=0.5):
    """Label/score-independent deterministic source-group partition."""
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    groups = sorted({r["group_id"] for r in rows}, key=lambda g: (
        hashlib.sha256((seed + "\0" + g).encode()).hexdigest(), g))
    count = int(len(groups) * fraction)
    if count == 0 or count == len(groups):
        raise ValueError("Need at least one calibration and one evaluation group")
    calibration = set(groups[:count])
    return ([r for r in rows if r["group_id"] in calibration],
            [r for r in rows if r["group_id"] not in calibration])


def assert_disjoint(calibration, evaluation):
    for key in ("item_id", "group_id"):
        if {r[key] for r in calibration} & {r[key] for r in evaluation}:
            raise ValueError("Calibration/evaluation overlap in " + key)


def fit(calibration, alpha=0.1):
    """One nonconformity score per group: max(1 - p(true)) over its rows.

    With exchangeable groups and a fixed model, the finite-sample quantile
    targets simultaneous label coverage within a new group. This is neither
    class-conditional coverage nor a guarantee on singleton-only accuracy.
    """
    validate_rows(calibration)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    scores = defaultdict(list)
    for r in calibration:
        scores[r["group_id"]].append(1 - r["probabilities"][r["true_label"]])
    ordered = sorted(max(values) for values in scores.values())
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    # Bounded nonconformity: q=1 includes all labels if the rank exceeds n.
    threshold = ordered[rank - 1] if rank <= len(ordered) else 1.0
    return {"schema": "ee6405-group-conformal/v1", "alpha": alpha,
            "threshold": threshold, "quantile_rank": rank,
            "calibration_groups": len(ordered), "calibration_rows": len(calibration),
            "labels": list(LABELS), "score": "group_max_one_minus_true_probability"}


def predict_set(probabilities, artifact):
    """Reusable inference API: no true labels, item IDs or source groups needed."""
    validate_probabilities(probabilities)
    if artifact.get("schema") != "ee6405-group-conformal/v1" or artifact.get("labels") != list(LABELS):
        raise ValueError("Incompatible conformal artifact")
    threshold = artifact["threshold"]
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid conformal threshold")
    candidates = [label for label in LABELS if 1 - probabilities[label] <= threshold]
    return {"prediction_set": candidates,
            "needs_review": len(candidates) != 1,
            "review_reason": ("empty_set" if not candidates else "ambiguous_set") if len(candidates) != 1 else None,
            "singleton_label": candidates[0] if len(candidates) == 1 else None,
            "original_argmax": max(LABELS, key=probabilities.get)}


def evaluate(rows, artifact):
    outputs = []
    groups = defaultdict(list)
    singleton_correct = singleton_count = covered = total_size = empty = top1_correct = 0
    for row in rows:
        result = predict_set(row["probabilities"], artifact)
        hit = row["true_label"] in result["prediction_set"]
        size = len(result["prediction_set"])
        covered += hit
        total_size += size
        empty += size == 0
        top1_correct += result["original_argmax"] == row["true_label"]
        if size == 1:
            singleton_count += 1
            singleton_correct += hit
        groups[row["group_id"]].append(hit)
        outputs.append({k: row[k] for k in ("item_id", "group_id", "domain", "true_label")})
        outputs[-1].update(result, covered=hit)
    n = len(rows)
    return {"rows": n, "groups": len(groups), "top1_accuracy": top1_correct / n,
            "set_coverage": covered / n, "group_all_covered": sum(all(v) for v in groups.values()) / len(groups),
            "average_set_size": total_size / n, "singleton_rate": singleton_count / n,
            "singleton_accuracy": singleton_correct / singleton_count if singleton_count else None,
            "review_rate": 1 - singleton_count / n, "empty_set_rate": empty / n}, outputs


def run(calibration, evaluation, alpha):
    validate_rows(calibration)
    validate_rows(evaluation)
    assert_disjoint(calibration, evaluation)
    artifact = fit(calibration, alpha)
    metrics, predictions = evaluate(evaluation, artifact)
    strata = {}
    for field in ("domain", "true_label"):
        strata[field] = {value: evaluate([r for r in evaluation if r[field] == value], artifact)[0]
                         for value in sorted({r[field] for r in evaluation})}
    return artifact, {"overall": metrics, "strata": strata}, predictions


def write_text(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    artifact, metrics, predictions = run(load_rows(args.calibration), load_rows(args.evaluation), args.alpha)
    artifact["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in (("calibration", args.calibration), ("evaluation", args.evaluation))}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "calibrator.json", artifact)
    write_json(args.output / "metrics.json", metrics)
    write_text(args.output / "prediction_sets.jsonl", "".join(json.dumps(r) + "\n" for r in predictions))


if __name__ == "__main__":
    main()
