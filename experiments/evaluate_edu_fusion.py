"""Reload a frozen fusion run, exporting validation evidence and optional test metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common import LABELS, classification_metrics, file_hash, write_json, write_jsonl
from experiments.edu_fusion import EDUFusionModel, fusion_dataset
from experiments.neural import collate_batch, tokenizer_local
from experiments.train_same_window import evaluate, prediction_rows


def verify_run(run, cache):
    recorded = json.loads((run / "config.json").read_text())
    if recorded["training_only_smoke"]:
        raise ValueError("Smoke checkpoints are engineering checks, not trained performance evidence")
    result = json.loads((run / "result.json").read_text())
    if result["status"] not in ("validation_completed", "completed"):
        raise ValueError("Training must finish before checkpoint evaluation")
    if file_hash(cache / "manifest.json") != recorded["cache_manifest_sha256"]:
        raise ValueError("Cache manifest differs from training")
    if file_hash(run / "best.pt") != result["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash differs from training")
    sources = json.loads((run / "source_hashes.json").read_text())
    for name, expected in sources.items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError(f"Training source changed: {name}")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != recorded["config"]:
        raise ValueError("Checkpoint configuration differs from training")
    return checkpoint, result


def explanation_rows(model, dataset, tokenizer, examples_per_class):
    """Choose the first validation items per class, never the best-looking errors."""
    selected = [int(index) for label in range(4)
                for index in np.flatnonzero(dataset.labels == label)[:examples_per_class]]
    model.eval()
    rows = []
    with torch.inference_mode():
        for index in selected:
            item = dataset[index]
            output = model(collate_batch([item]))
            probabilities = output["logits"].softmax(-1)[0].cpu().tolist()
            weights = output["edu_weights"][0].cpu().tolist() if output["edu_weights"] else []
            token_ids = item["full_text_input_ids"].tolist()
            record = dataset.records[index]
            rows.append({
                "item_id": record["item_id"],
                "true_label": record["label"],
                "prediction": LABELS[int(np.argmax(probabilities))],
                "probabilities": dict(zip(LABELS, probabilities)),
                "residual_gate": float(output["edu_gate"][0, 0]),
                "edus": [{
                    "token_span": [start, end],
                    "decoded_excerpt": tokenizer.decode(token_ids[start:end]),
                    "attention_weight": weights[edu_index] if weights else None,
                } for edu_index, (start, end) in enumerate(item["edu_token_spans"])],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--examples-per-class", type=int, default=2)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Open test only after the candidate and recipe have been frozen")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve prior evidence")
    if args.batch_size <= 0 or args.examples_per_class < 0:
        parser.error("Batch size must be positive and example count nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    checkpoint, training_result = verify_run(args.run, args.cache)
    config = checkpoint["config"]
    model = EDUFusionModel(config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    args.output.mkdir(parents=True)
    results = {}
    for split in (("val", "test") if args.evaluate_test else ("val",)):
        dataset = fusion_dataset(args.cache, split, config["condition"])
        metrics, probabilities, ids = evaluate(model, dataset, args.batch_size)
        saved_path = args.run / ("validation_predictions.jsonl" if split == "val"
                                 else "test_predictions.jsonl")
        reload_difference = None
        if saved_path.exists():
            saved = [json.loads(line) for line in saved_path.read_text().splitlines()]
            if [row["item_id"] for row in saved] != ids:
                raise ValueError("Saved and reloaded prediction IDs differ")
            saved_probabilities = np.array([[row["probabilities"][label] for label in LABELS]
                                            for row in saved])
            reload_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
            if reload_difference > 1e-4:
                raise ValueError(f"Reload probabilities differ by {reload_difference}, exceeding 1e-4")
        write_jsonl(args.output / f"{split}_predictions.jsonl",
                    prediction_rows(dataset, probabilities, ids))
        by_domain = {}
        for domain in sorted({row["domain"] for row in dataset.records}):
            mask = np.array([row["domain"] == domain for row in dataset.records])
            by_domain[domain] = classification_metrics(dataset.labels[mask], probabilities[mask])
        results[split] = {
            "metrics": metrics, "by_domain": by_domain,
            "max_saved_probability_difference": reload_difference,
        }
        if split == "val" and args.examples_per_class:
            write_json(args.output / "validation_explanations.local.json", {
                "selection": "First items per class in frozen validation order",
                "warning": "Contains decoded text. Attention weights are not causal attributions.",
                "boundary_tokens_may_overlap": True,
                "items": explanation_rows(model, dataset, tokenizer_local(), args.examples_per_class),
            })
    write_json(args.output / "result.json", {
        "status": "completed", "condition": config["condition"], "seed": config["seed"],
        "test_evaluated": args.evaluate_test, "splits": results,
        "checkpoint_sha256": training_result["best_checkpoint_sha256"],
        "cache_manifest_sha256": file_hash(args.cache / "manifest.json"),
        "selection": "Reload only; no training or threshold fitting",
    })


if __name__ == "__main__":
    main()
