"""Train one fair-window CE condition from base-pretrained weights.

Formal runs require CUDA and a complete audited cache. CPU mode is restricted
to a four-class train/development smoke check and never opens the test split.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from experiments.common import LABELS, classification_metrics, file_hash, write_json, write_jsonl
from experiments.neural import BalancedBatchSampler, collate_batch
from experiments.same_window_controls import CONDITIONS, IndexedSameWindowDataset, config_for, model_for
from experiments.train_neural import save_checkpoint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_subset(dataset):
    return Subset(dataset, [int(np.flatnonzero(dataset.labels == label)[0]) for label in range(4)])


def evaluate(model, dataset, batch_size):
    model.eval()
    logits, labels, item_ids = [], [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=0):
            output = model(batch)["logits"].float().cpu()
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite evaluation logits")
            logits.append(output)
            labels.extend(batch["labels"].tolist())
            item_ids.extend(batch["id"])
    probabilities = torch.cat(logits).softmax(1).numpy()
    return classification_metrics(labels, probabilities), probabilities, item_ids


def prediction_rows(dataset, probabilities, item_ids):
    expected_ids = [row["item_id"] for row in dataset.records]
    if item_ids != expected_ids or len(probabilities) != len(expected_ids):
        raise ValueError("Prediction output order or size changed")
    return [{
        "item_id": row["item_id"],
        "group_id": row["group_id"],
        "domain": row["domain"],
        "true_label": row["label"],
        "prediction": LABELS[int(probability.argmax())],
        "probabilities": dict(zip(LABELS, probability.tolist())),
    } for row, probability in zip(dataset.records, probabilities)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="Defaults to CPU for smoke and CUDA for formal training")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve prior runs and choose a new directory")
    if args.batch_size <= 0 or args.batch_size % 4 or args.epochs <= 0:
        parser.error("Batch size must be positive and divisible by four; epochs must be positive")
    device_name = args.device or ("cpu" if args.smoke else "cuda")
    if not args.smoke and device_name != "cuda":
        parser.error("Formal same-window training requires --device cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    args.output.mkdir(parents=True)
    config = config_for(args.condition, args.seed)
    config.update(batch_size=4 if args.smoke else args.batch_size,
                  max_epochs=1 if args.smoke else args.epochs)
    evidence = args.cache / "manifest.json"
    write_json(args.output / "config.json", {
        "config": config,
        "cache": str(args.cache.resolve()),
        "cache_manifest_sha256": file_hash(evidence),
        "training_only_smoke": args.smoke,
    })

    train_data = IndexedSameWindowDataset(args.cache, "train", args.condition)
    val_data = IndexedSameWindowDataset(args.cache, "val", args.condition)
    if args.smoke:
        train_view, val_view = smoke_subset(train_data), smoke_subset(val_data)
        sampler_labels = [int(train_view[index]["label"]) for index in range(len(train_view))]
    else:
        train_view, val_view = train_data, val_data
        sampler_labels = train_data.labels
    sampler = BalancedBatchSampler(sampler_labels, config["batch_size"], args.seed)
    loader = DataLoader(train_view, batch_sampler=sampler, collate_fn=collate_batch, num_workers=0)

    seed_all(args.seed)
    device = torch.device(device_name)
    model = model_for(config).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(name.startswith("bert_model.embeddings") for name, _ in trainable):
        raise ValueError("Embedding layer must remain frozen")
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                  lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["max_epochs"] * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config["warmup_ratio"]), total_steps
    )
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for _, parameter in trainable),
        "trainable_names": [name for name, _ in trainable],
    }
    write_json(args.output / "parameters.json", parameters)
    initial_classifier = model.classifier.out_proj.weight.detach().clone()
    initial_last_encoder = model.bert_model.encoder.layer[-1].attention.self.query.weight.detach().clone()
    initial_embeddings = model.bert_model.embeddings.word_embeddings.weight[0].detach().clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_score = -1.0
    best_epoch = -1
    stale = 0
    started = time.perf_counter()
    for epoch in range(config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            loss = F.cross_entropy(logits.float(), batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable],
                                           config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        validation, _, _ = evaluate(model, val_view, config["batch_size"])
        improved = validation["macro_f1"] > best_score
        if improved:
            best_score, best_epoch, stale = validation["macro_f1"], epoch + 1, 0
            save_checkpoint(args.output / "best.pt", {
                "model_state_dict": model.state_dict(),
                "config": config,
                "validation": validation,
                "best_epoch": best_epoch,
            })
        else:
            stale += 1
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "epoch": epoch + 1,
                "mean_ce": float(np.mean(losses)),
                "validation_macro_f1": validation["macro_f1"],
                "improved": improved,
            }) + "\n")
        if stale >= config["early_stopping_patience"]:
            break

    if args.smoke:
        classifier_updated = not torch.equal(
            initial_classifier, model.classifier.out_proj.weight.detach()
        )
        last_encoder_updated = not torch.equal(
            initial_last_encoder,
            model.bert_model.encoder.layer[-1].attention.self.query.weight.detach(),
        )
        frozen_embeddings_unchanged = torch.equal(
            initial_embeddings,
            model.bert_model.embeddings.word_embeddings.weight[0].detach(),
        )
        if not classifier_updated or not last_encoder_updated or not frozen_embeddings_unchanged:
            raise ValueError("Smoke update/freeze invariant failed")
        write_json(args.output / "smoke.json", {
            "status": "passed",
            "condition": args.condition,
            "training_only_smoke": True,
            "test_evaluated": False,
            "finite_gradients": True,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None
            ),
            "classifier_updated": classifier_updated,
            "last_encoder_updated": last_encoder_updated,
            "frozen_embeddings_unchanged": frozen_embeddings_unchanged,
            "parameters": parameters,
            "elapsed_seconds": time.perf_counter() - started,
        })
        return

    selected = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    validation, val_probabilities, val_ids = evaluate(model, val_data, config["batch_size"])
    write_jsonl(args.output / "validation_predictions.jsonl",
                prediction_rows(val_data, val_probabilities, val_ids))
    test_data = IndexedSameWindowDataset(args.cache, "test", args.condition)
    test, test_probabilities, test_ids = evaluate(model, test_data, config["batch_size"])
    write_jsonl(args.output / "test_predictions.jsonl",
                prediction_rows(test_data, test_probabilities, test_ids))
    write_json(args.output / "result.json", {
        "status": "completed",
        "condition": args.condition,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation": validation,
        "test": test,
        "parameters": parameters,
        "best_checkpoint_sha256": file_hash(args.output / "best.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "scope": "Same visible 512-token prefix, CE objective, base-pretrained initialization.",
    })


if __name__ == "__main__":
    main()
