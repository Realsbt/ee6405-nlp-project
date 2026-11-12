"""Separate matched two-forward trainer; does not change running fusion v1."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from experiments.common import file_hash, write_json, write_jsonl
from experiments.edu_fusion import FUSION_CONDITIONS
from experiments.fusion_extensions import (
    EXTENSION_CONDITIONS, FusionExtensionModel, extension_config, extension_dataset,
    paired_dropout_loss, validation_partition,
)
from experiments.neural import BalancedBatchSampler, collate_batch
from experiments.paths import UPSTREAM, resolve_recorded_path
from experiments.train_neural import save_checkpoint
from experiments.train_same_window import evaluate, prediction_rows, seed_all, smoke_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=EXTENSION_CONDITIONS)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--parent", choices=FUSION_CONDITIONS, default="fusion_gated")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous runs; choose a new output")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU is restricted to engineering smoke")
    config = extension_config(args.condition, args.seed, args.parent)
    config.update(batch_size=4 if args.smoke else 16, max_epochs=1 if args.smoke else 20)
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", {
        "config": config, "training_only_smoke": args.smoke, "validation_only": True,
        "cache": str(args.cache.resolve()),
        "cache_manifest_sha256": file_hash(args.cache / "manifest.json"),
    })
    sources = {}
    source_dir = args.output / "source"
    source_dir.mkdir()
    for name in (
        "train_fusion_extensions.py", "fusion_extensions.py", "edu_fusion.py",
        "same_window_controls.py", "train_same_window.py", "train_neural.py",
        "neural.py", "common.py", "paths.py",
    ):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        sources[name] = file_hash(source)
    write_json(args.output / "source_hashes.json", sources)
    backbone = resolve_recorded_path(config["architecture"]["backbone_model_path"])
    write_json(args.output / "provenance.json", {
        "initialization": "base-pretrained weights, not a task checkpoint",
        "files_sha256": {
            str(path): file_hash(path) for path in (
                backbone / "model.safetensors", backbone / "config.json",
                UPSTREAM / "models/flexible_model.py", UPSTREAM / "models/modules.py",
            )
        },
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "torch-geometric", "numpy")},
    })
    train = extension_dataset(args.cache, "train", args.condition)
    validation = extension_dataset(args.cache, "val", args.condition)
    selection_indices, calibration_indices = validation_partition(validation.records)
    write_json(args.output / "validation_partitions.json", {
        "rule": config["validation_partition"],
        "selection_ids": [validation.records[i]["item_id"] for i in selection_indices],
        "calibration_ids": [validation.records[i]["item_id"] for i in calibration_indices],
        "selection_group_ids": sorted({validation.records[i]["group_id"] for i in selection_indices}),
        "calibration_group_ids": sorted({validation.records[i]["group_id"] for i in calibration_indices}),
        "calibration_used_for_checkpoint_selection": False,
    })
    train_view = smoke_subset(train) if args.smoke else train
    if args.smoke:
        validation_view = Subset(validation, [
            next(i for i in selection_indices if int(validation.labels[i]) == label)
            for label in range(4)
        ])
    else:
        validation_view = Subset(validation, selection_indices)
    labels = [int(train_view[i]["label"]) for i in range(len(train_view))] if args.smoke else train.labels
    sampler = BalancedBatchSampler(labels, config["batch_size"], args.seed)
    loader = DataLoader(train_view, batch_sampler=sampler, collate_fn=collate_batch, num_workers=0)
    seed_all(args.seed)
    model = FusionExtensionModel(config).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in parameters),
    }
    write_json(args.output / "parameters.json", counts)
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps = config["max_epochs"] * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config["warmup_ratio"]), steps)
    before_classifier = model.classifier.out_proj.weight.detach().clone()
    before_encoder = model.bert_model.encoder.layer[-1].attention.self.query.weight.detach().clone()
    before_embedding = model.bert_model.embeddings.word_embeddings.weight.detach().clone()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best, best_epoch, stale = -1.0, -1, 0
    for epoch in range(config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        losses, ce_losses, divergences = [], [], []
        epoch_started = time.perf_counter()
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            first, second = model(batch)["logits"], model(batch)["logits"]
            loss, ce, kl = paired_dropout_loss(
                first, second, batch["labels"].to(args.device),
                config["extension"]["consistency_weight"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite paired-forward loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"],
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            ce_losses.append(float(ce.detach()))
            divergences.append(float(kl.detach()))
            if step % 100 == 0 or step == len(loader):
                print(json.dumps({
                    "event": "train_progress", "epoch": epoch + 1, "batch": step,
                    "batches": len(loader), "mean_loss": float(np.mean(losses)),
                }), flush=True)
        metrics, _, _ = evaluate(model, validation_view, config["batch_size"])
        improved = metrics["macro_f1"] > best
        if improved:
            best, best_epoch, stale = metrics["macro_f1"], epoch + 1, 0
            save_checkpoint(args.output / "best.pt", {
                "config": config, "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch, "validation": metrics,
            })
        else:
            stale += 1
        event = {
            "epoch": epoch + 1, "mean_loss": float(np.mean(losses)),
            "mean_ce": float(np.mean(ce_losses)), "mean_symmetric_kl": float(np.mean(divergences)),
            "validation_macro_f1": metrics["macro_f1"], "improved": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if stale >= config["early_stopping_patience"]:
            break
    if args.smoke:
        checks = {
            "classifier_updated": not torch.equal(before_classifier, model.classifier.out_proj.weight),
            "last_encoder_updated": not torch.equal(
                before_encoder, model.bert_model.encoder.layer[-1].attention.self.query.weight),
            "frozen_embeddings_unchanged": torch.equal(
                before_embedding, model.bert_model.embeddings.word_embeddings.weight),
        }
        if not all(checks.values()):
            raise ValueError("Smoke update/freeze checks failed")
        write_json(args.output / "smoke.json", {
            "status": "passed", "condition": args.condition, "training_only_smoke": True,
            "test_evaluated": False, "finite_gradients": True, **checks,
            "device": args.device, "parameters": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024 ** 2 if args.device == "cuda" else None),
        })
        return
    selected = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    metrics, probabilities, ids = evaluate(model, validation, config["batch_size"])
    write_jsonl(args.output / "validation_predictions.jsonl",
                prediction_rows(validation, probabilities, ids))
    write_json(args.output / "result.json", {
        "status": "validation_completed", "condition": args.condition, "seed": args.seed,
        "best_epoch": best_epoch, "validation": metrics, "test": None,
        "selection_validation": selected["validation"],
        "validation_partitions_sha256": file_hash(args.output / "validation_partitions.json"),
        "calibration_used_for_checkpoint_selection": False,
        "test_evaluated": False, "parameters": counts, "source_sha256": sources,
        "best_checkpoint_sha256": file_hash(args.output / "best.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "gpu": torch.cuda.get_device_name(0),
        "scope": "Matched segmentation x consistency; both use two training forwards, one at inference",
    })


if __name__ == "__main__":
    main()
