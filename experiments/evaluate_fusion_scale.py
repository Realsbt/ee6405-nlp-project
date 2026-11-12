"""Strictly reload frozen scale controls, with explicit opt-in for test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common import LABELS, classification_metrics, file_hash, write_json
from experiments.evaluate_edu_fusion import verify_run
from experiments.fusion_scale_controls import ScaleControlModel, scale_dataset
from experiments.train_same_window import evaluate_partitions


def reload_scale_run(run, cache, output, device="cpu", batch_size=16, evaluate_test=False):
    if output.exists():
        raise FileExistsError("Preserve existing evidence and choose a new output directory")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    checkpoint, training_result = verify_run(run, cache)
    config = checkpoint["config"]
    if config["recipe_id"] != "edu_residual_scale_controls_v1":
        raise ValueError("Not a scale-control checkpoint")
    model = ScaleControlModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation_data = scale_dataset(cache, "val", config["condition"])
    output.mkdir(parents=True)
    validation, test = evaluate_partitions(
        model, validation_data, cache=cache, condition=config["condition"],
        batch_size=batch_size, output=output, dataset_factory=scale_dataset,
        validation_only=not evaluate_test,
    )
    splits = {}
    for split, metrics in (("validation", validation), ("test", test)):
        if metrics is None:
            continue
        filename = f"{split}_predictions.jsonl"
        rows = [json.loads(line) for line in (output / filename).read_text().splitlines()]
        probabilities = np.array([[row["probabilities"][label] for label in LABELS] for row in rows])
        difference = None
        if (run / filename).exists():
            saved = [json.loads(line) for line in (run / filename).read_text().splitlines()]
            if [row["item_id"] for row in saved] != [row["item_id"] for row in rows]:
                raise ValueError("Saved and reloaded prediction IDs differ")
            expected = np.array([[row["probabilities"][label] for label in LABELS] for row in saved])
            difference = float(np.max(np.abs(probabilities - expected)))
            if difference > 1e-4:
                raise ValueError("Saved and reloaded probabilities differ by more than 1e-4")
        labels = np.array([LABELS.index(row["true_label"]) for row in rows])
        by_domain = {}
        for domain in sorted({row["domain"] for row in rows}):
            mask = np.array([row["domain"] == domain for row in rows])
            by_domain[domain] = classification_metrics(labels[mask], probabilities[mask])
        splits[split] = {
            "metrics": metrics, "by_domain": by_domain,
            "max_saved_probability_difference": difference,
        }
    result = {
        "status": "completed", "condition": config["condition"], "seed": config["seed"],
        "test_evaluated": evaluate_test, "splits": splits,
        "checkpoint_sha256": training_result["best_checkpoint_sha256"],
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "selection": "Frozen reload only; no candidate, threshold or hyperparameter selection",
    }
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    reload_scale_run(args.run, args.cache, args.output, args.device,
                     args.batch_size, args.evaluate_test)


if __name__ == "__main__":
    main()
