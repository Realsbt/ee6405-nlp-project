"""Strict frozen-checkpoint evaluation of segmentation/consistency variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common import LABELS, file_hash, write_json
from experiments.evaluate_edu_fusion import verify_run
from experiments.fusion_extensions import FusionExtensionModel, extension_dataset
from experiments.train_same_window import evaluate_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous evidence")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    checkpoint, training = verify_run(args.run, args.cache)
    config = checkpoint["config"]
    if config["recipe_id"] != "fusion_segmentation_consistency_v1":
        raise ValueError("Not a segmentation/consistency checkpoint")
    if file_hash(args.run / "validation_partitions.json") != training["validation_partitions_sha256"]:
        raise ValueError("Validation selection/calibration partition changed")
    model = FusionExtensionModel(config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation = extension_dataset(args.cache, "val", config["condition"])
    args.output.mkdir(parents=True)
    val_metrics, test_metrics = evaluate_partitions(
        model, validation, cache=args.cache, condition=config["condition"],
        batch_size=16, output=args.output, dataset_factory=extension_dataset,
        validation_only=not args.evaluate_test,
    )
    # Recompute saved validation predictions exactly before accepting the reload.
    saved = [json.loads(line) for line in (args.run / "validation_predictions.jsonl").read_text().splitlines()]
    reloaded = [json.loads(line) for line in (args.output / "validation_predictions.jsonl").read_text().splitlines()]
    if [row["item_id"] for row in saved] != [row["item_id"] for row in reloaded]:
        raise ValueError("Validation IDs differ after reload")
    left = np.array([[row["probabilities"][label] for label in LABELS] for row in saved])
    right = np.array([[row["probabilities"][label] for label in LABELS] for row in reloaded])
    difference = float(np.max(np.abs(left - right)))
    if difference > 1e-4:
        raise ValueError("Reload probabilities differ by more than 1e-4")
    write_json(args.output / "result.json", {
        "status": "completed", "condition": config["condition"], "seed": config["seed"],
        "test_evaluated": args.evaluate_test, "validation": val_metrics, "test": test_metrics,
        "max_validation_probability_difference": difference,
        "checkpoint_sha256": training["best_checkpoint_sha256"],
        "cache_manifest_sha256": file_hash(args.cache / "manifest.json"),
    })


if __name__ == "__main__":
    main()
