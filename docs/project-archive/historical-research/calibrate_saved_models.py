"""Temperature calibration from stored probabilities, fitted on HART val only."""

import json
from pathlib import Path
from race_cpu_replay import (
    ROOT, LABELS, np, read_jsonl, file_hash, write, fit_temperature,
    temperature_probabilities, evaluate, negative_only_threshold,
)


def probability_logits(rows, key="probabilities"):
    p = np.array([[r[key][label] for label in LABELS] for r in rows], dtype=float)
    if (not np.isfinite(p).all() or (p <= 0).any()
            or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError("Stored probabilities must be positive, finite and sum to one")
    p /= p.sum(1, keepdims=True)
    return np.log(p)


def main():
    root = ROOT / "saved_temperature_v1"
    root.mkdir()
    for kind, run in (("tfidf", "tfidf_lr_v1"), ("roberta", "roberta_s3407")):
        source = ROOT / f"transfer_{kind}_v1"
        val_path = source / "validation_predictions.jsonl"
        val = read_jsonl(val_path)
        z = probability_logits(val)
        y = np.array([LABELS.index(r["true_label"]) for r in val])
        calibration = fit_temperature(z, y)
        for name, t in (("uncalibrated", 1), ("temperature_scaled", calibration["temperature"])):
            p = temperature_probabilities(z, t)
            calibration[name + "_threshold"] = negative_only_threshold(p[y < 2, 2:].sum(1))
        calibration.update({"run": run, "validation_predictions_sha256": file_hash(val_path),
                            "scope": "HART val-only fit from log stored probabilities, not raw logits; "
                                     "no zero probabilities/clipping. Val also used for checkpoint selection.",
                            "script_sha256": file_hash(__file__)})
        write(root / f"{kind}_calibration.json", calibration)
        archive = ROOT / "repository/experiments/reports" / run
        result = {}
        for name, path, key, label_key, four_class in (
            ("hart", archive / "test_predictions.jsonl", "probabilities", "true_label", True),
            ("hc3", archive / "hc3_aux_v1_1_predictions.jsonl", "four_class_scores", "true_binary_label", False),
            ("mixset", source / "mixset_predictions.jsonl", "probabilities", "true_label", True),
        ):
            rows = read_jsonl(path)
            labels = [LABELS.index(r[label_key]) if four_class else r[label_key] for r in rows]
            result[name] = evaluate(probability_logits(rows, key), labels, calibration, four_class)
            result[name]["prediction_sha256"] = file_hash(path)
        write(root / f"{kind}_report.json", result)
        print(kind, "temperature", calibration["temperature"], flush=True)


if __name__ == "__main__":
    main()
