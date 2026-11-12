"""CPU smoke or CUDA CE training of a same-window condition, from base weights."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from window_controls import CONDITIONS, ROOT, WindowDataset, config_for, model_for
from experiments.common import LABELS, classification_metrics
from experiments.neural import BalancedBatchSampler, collate_batch
from experiments.train_neural import save_checkpoint
from same_window_cache import atomic_json
from screen_structure_cpu import digest


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def model_fingerprint(model):
    digest_state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest_state.update(name.encode())
        digest_state.update(bytes.fromhex(fingerprint(tensor)))
    return digest_state.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataset, batch_size):
    model.eval()
    logits, labels, ids = [], [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch):
            z = model(batch)["logits"].float().cpu()
            if not torch.isfinite(z).all():
                raise FloatingPointError("Non-finite evaluation logits")
            logits.append(z)
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["id"])
    p = torch.cat(logits).softmax(1).numpy()
    return classification_metrics(labels, p), p, ids


def smoke_subset(dataset):
    indices = [int(np.flatnonzero(dataset.labels == c)[0]) for c in range(4)]
    return Subset(dataset, indices)


def save_predictions(path, dataset, probabilities, ids):
    if ids != [r["item_id"] for r in dataset.records] or len(probabilities) != len(ids):
        raise ValueError("Prediction output order/size mismatch")
    with path.open("x") as stream:
        for row, p in zip(dataset.records, probabilities):
            stream.write(json.dumps({
                "item_id": row["item_id"], "group_id": row["group_id"], "domain": row["domain"],
                "true_label": row["label"], "prediction": LABELS[int(p.argmax())],
                "probabilities": dict(zip(LABELS, p.tolist())),
            }) + "\n")


def run(args, condition, output):
    device = torch.device(args.device)
    config = config_for(condition, args.seed)
    config.update(max_epochs=1 if args.mode == "smoke" else args.epochs,
                  batch_size=4 if args.mode == "smoke" else args.batch_size)
    cache = args.cache.resolve()
    evidence_file = cache / ("verification.json" if (cache / "verification.json").exists() else "manifest.json")
    signature = {
        "config": config, "cache": str(cache), "cache_evidence_sha256": digest(evidence_file),
        "mode": args.mode, "device": args.device,
        "source_sha256": {p.name: digest(p) for p in
                          (Path(__file__), ROOT / "window_controls.py")},
        "backbone_sha256": digest(ROOT / "resources/assets/roberta-base/model.safetensors"),
    }
    if args.resume:
        if not (output / "last.pt").exists() or json.loads((output / "config.json").read_text()) != signature:
            raise ValueError("Resume needs a matching completed-epoch checkpoint/config")
        if (output / "result.json").exists():
            raise ValueError("Completed runs cannot be resumed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", signature)
        for path in (Path(__file__), ROOT / "window_controls.py"):
            shutil.copy2(path, output / path.name)
    smoke = args.mode == "smoke"
    # Full training is rejected before constructing a model if the cache is partial.
    train = WindowDataset(cache, "train", condition, smoke=smoke)
    val = WindowDataset(cache, "val", condition, smoke=smoke)
    if smoke:
        train, val = smoke_subset(train), smoke_subset(val)
    labels = [int(train[i]["label"]) for i in range(len(train))] if smoke else train.labels
    sampler = BalancedBatchSampler(labels, config["batch_size"], args.seed)
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_batch)
    seed_all(args.seed)
    model = model_for(config).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for p in model.bert_model.embeddings.parameters()):
        raise ValueError("Embedding layer must be frozen")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["max_epochs"] * len(loader)
    warmup = int(total_steps * config["warmup_ratio"])
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    initial_head = fingerprint(model.classifier.out_proj.weight)
    initial_model = model_fingerprint(model) if smoke else None
    initial_frozen = fingerprint(model.bert_model.embeddings.word_embeddings.weight[0])
    initial_encoder = fingerprint(model.bert_model.encoder.layer[-1].attention.self.query.weight)
    parameter_counts = {"total": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in parameters)}
    start_epoch, best, stale = 0, -1., 0
    if args.resume:
        status = json.loads((output / "status.json").read_text())
        if digest(output / "last.pt") != status["last_checkpoint_sha256"]:
            raise ValueError("Resume checkpoint changed")
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, best, stale = saved["epoch"] + 1, saved["best"], saved["stale"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, config["max_epochs"]):
        if stale >= config["early_stopping_patience"]:
            break
        sampler.epoch = epoch
        model.train()
        losses, token_fingerprints = [], []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            z = model(batch)["logits"]
            loss = F.cross_entropy(z.float(), batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
            if smoke:
                if (model.classifier.out_proj.weight.grad is None
                        or model.bert_model.encoder.layer[-1].attention.self.query.weight.grad is None):
                    raise ValueError("Missing classifier/last-encoder gradient")
                if any(p.grad is not None for p in model.bert_model.embeddings.parameters()):
                    raise ValueError("Frozen embeddings received gradients")
                token_fingerprints.append(fingerprint(batch["full_text_input_ids"]))
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        validation, _, _ = evaluate(model, val, config["batch_size"])
        improved = validation["macro_f1"] > best
        if improved:
            best, stale = validation["macro_f1"], 0
        else:
            stale += 1
        event = {"epoch": epoch + 1, "mean_ce": float(np.mean(losses)),
                 "development_macro_f1": validation["macro_f1"], "best": best}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(f"{condition}: {event}", flush=True)
        if smoke:
            result = {
                "status": "passed", "condition": condition, "training_only_smoke": True,
                "test_evaluated": False, "finite_gradients": True,
                "classifier_updated": fingerprint(model.classifier.out_proj.weight) != initial_head,
                "last_encoder_updated": fingerprint(model.bert_model.encoder.layer[-1].attention.self.query.weight) != initial_encoder,
                "frozen_embeddings_unchanged": fingerprint(model.bert_model.embeddings.word_embeddings.weight[0]) == initial_frozen,
                "parameters": parameter_counts, "initial_classifier_sha256": initial_head,
                "initial_model_sha256": initial_model,
                "batch_token_sha256": token_fingerprints, "elapsed_seconds": time.perf_counter() - started,
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "torch-geometric")},
            }
            if not all(result[k] for k in ("classifier_updated", "last_encoder_updated", "frozen_embeddings_unchanged")):
                raise ValueError("Smoke update/freeze invariant failed")
            atomic_json(output / "smoke.json", result)
            return result
        saved = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best": best, "stale": stale, "signature": signature,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        if improved:
            save_checkpoint(output / "best.pt", {"model": model.state_dict(), "signature": signature,
                                                  "validation": validation, "epoch": epoch + 1})
        save_checkpoint(output / "last.pt", saved)
        atomic_json(output / "status.json", {
            "status": "running", "completed_epoch": epoch + 1,
            "last_checkpoint_sha256": digest(output / "last.pt"),
        })
        if stale >= config["early_stopping_patience"]:
            break
    # Best model is fixed before test is opened or evaluated.
    selected = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"])
    _, validation_probabilities, validation_ids = evaluate(model, val, config["batch_size"])
    save_predictions(output / "validation_predictions.jsonl", val, validation_probabilities, validation_ids)
    test = WindowDataset(cache, "test", condition)
    scores, probabilities, ids = evaluate(model, test, config["batch_size"])
    save_predictions(output / "test_predictions.jsonl", test, probabilities, ids)
    result = {
        "status": "completed", "test": scores, "validation": selected["validation"],
        "best_epoch": selected["epoch"], "parameters": parameter_counts,
        "best_checkpoint_sha256": digest(output / "best.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "scope": "New same-window CE condition; not equivalent to historical full-graph results.",
    }
    atomic_json(output / "result.json", result)
    status = json.loads((output / "status.json").read_text())
    atomic_json(output / "status.json", {**status, "status": "completed",
                                        "result_sha256": digest(output / "result.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "train"))
    parser.add_argument("--condition", required=True, choices=(*CONDITIONS, "all"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("Formal runs require a CUDA device; use --mode smoke for local CPU checks")
    if args.condition == "all" and (args.mode != "smoke" or args.resume):
        parser.error("'all' is only a fresh smoke run; formal runs have separate condition/seed outputs")
    if args.batch_size <= 0 or args.batch_size % 4 or args.epochs <= 0:
        parser.error("Batch size must be positive and divisible by 4; epochs must be positive")
    torch.set_num_threads(2)
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    results = {}
    for condition in conditions:
        results[condition] = run(args, condition, args.output / condition if len(conditions) > 1 else args.output)
        gc.collect()
    if len(conditions) > 1:
        if len({tuple(r["batch_token_sha256"]) for r in results.values()}) != 1:
            raise ValueError("Conditions used different lexical inputs")
        if results["graph"]["initial_model_sha256"] != results["shuffled"]["initial_model_sha256"]:
            raise ValueError("Graph control model initialization differs")
        atomic_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
