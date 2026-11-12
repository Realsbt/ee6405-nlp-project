"""Diagnose frozen validation predictions without opening test or exporting text."""

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from experiments.common import LABELS, classification_metrics, file_hash, read_jsonl, write_json
from experiments.paths import DATA_ROOT
from experiments.train_stacking import aligned_probabilities


def diagnose(records, predictions):
    y = np.array([LABELS.index(row["label"]) for row in records])
    arrays = {}
    metrics = {}
    for name, probability in predictions.items():
        p = np.asarray(probability, dtype=float)
        metrics[name] = {
            "overall": classification_metrics(y, p),
            "domains": {},
        }
        for domain in sorted({row["domain"] for row in records}):
            indices = np.array([i for i, row in enumerate(records) if row["domain"] == domain])
            metrics[name]["domains"][domain] = classification_metrics(y[indices], p[indices])
        arrays[name] = p.argmax(1)
    pairs = {}
    for a, b in itertools.combinations(arrays, 2):
        wrong_a, wrong_b = arrays[a] != y, arrays[b] != y
        both = wrong_a & wrong_b
        union = wrong_a | wrong_b
        pairs[f"{a}__{b}"] = {
            "both_wrong": int(both.sum()),
            "a_wrong_b_correct": int((wrong_a & ~wrong_b).sum()),
            "b_wrong_a_correct": int((wrong_b & ~wrong_a).sum()),
            "wrong_set_jaccard": float(both.sum() / union.sum()) if union.any() else 0.0,
        }
    all_wrong = np.logical_and.reduce([prediction != y for prediction in arrays.values()])
    hard = [{
        "item_id": row["item_id"], "group_id": row["group_id"],
        "domain": row["domain"], "true_label": row["label"],
        "predictions": {name: LABELS[int(values[i])] for name, values in arrays.items()},
    } for i, row in enumerate(records) if all_wrong[i]]
    return {
        "split": "validation", "samples": len(records), "test_opened": False,
        "models": metrics, "pairwise_errors": pairs, "all_models_wrong": hard,
        "limitations": [
            "Validation has already been used for checkpoint/candidate selection.",
            "Error overlap is diagnostic; it is not an achievable ensemble score.",
            "No test scores, source text or authorship explanations are exported.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing diagnostics")
    source = args.data_root / "val.jsonl"
    protocol = json.loads((args.data_root / "protocol.json").read_text())
    if file_hash(source) != protocol["partitions"]["val"]["sha256"]:
        raise ValueError("Validation partition changed")
    records = read_jsonl(source)
    probabilities, hashes = {}, {"validation": file_hash(source)}
    for entry in args.prediction:
        name, path = entry.split("=", 1)
        if name in probabilities:
            raise ValueError("Duplicate model name")
        probabilities[name] = aligned_probabilities(records, Path(path))
        hashes[name] = file_hash(path)
    result = diagnose(records, probabilities)
    result["input_sha256"] = hashes
    args.output.mkdir(parents=True)
    write_json(args.output / "diagnostics.json", result)
    lines = ["# Validation error diagnosis", "", "No test partition was opened.", "",
             "| Model | Macro-F1 | AI-humanized F1 | AI-humanized recall |",
             "|---|---:|---:|---:|"]
    for name, model in result["models"].items():
        m = model["overall"]
        h = m["per_class"]["ai_humanized"]
        lines.append(f"| {name} | {m['macro_f1']:.4%} | {h['f1']:.4%} | {h['recall']:.4%} |")
    lines.extend(["", f"All supplied models wrong: {len(result['all_models_wrong'])}.",
                  "Pairwise overlaps are in diagnostics.json; they are not ensemble performance."])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
