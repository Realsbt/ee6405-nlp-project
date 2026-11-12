"""Train the frozen-B0 CRE screen using train/validation only."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiments.common import DATA_ROOT, LABELS, classification_metrics, file_hash, write_json
from experiments.cre_data import CREDataset, collate_cre, SPLITTER_VERSION
from experiments.cre_losses import condition_probabilities, condition_scores_from_logits, pair_ranking_loss, residual_regularization
from experiments.cre_model import CREModel, MODES
from experiments.cre_sampling import build_sampling_plan
from experiments.neural import load_records, tokenizer_local


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = DATA_ROOT.parent / "cre_v1_runs"
BASELINE_CONFIG = PROJECT_ROOT / "experiments/glpc_runs/glpc_b0_s3407/config.json"


def atomic_torch_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_hashes():
    files = [
        Path(__file__), PROJECT_ROOT / "experiments/cre_model.py",
        PROJECT_ROOT / "experiments/cre_data.py", PROJECT_ROOT / "experiments/cre_losses.py",
        PROJECT_ROOT / "experiments/cre_sampling.py",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in files}


def make_effective_config(requested, mode, base_checkpoint):
    config = dict(requested)
    config["mode"] = mode
    config["lambda_pair"] = requested["lambda_pair"] if mode == "conditional_multi_pair" else 0.0
    config["base_checkpoint"] = str(base_checkpoint)
    config["labels"] = LABELS
    config["splitter_version"] = SPLITTER_VERSION
    return config


def batch_loader(plan, dataset, epoch, seed):
    order = np.random.default_rng(seed + epoch).permutation(len(plan["batches_plan"]))
    by_id = {item["id"]: item for item in dataset.items}
    for batch_index in order:
        specification = plan["batches_plan"][int(batch_index)]
        items = [by_id[item_id] for item_id in specification["item_ids"]]
        batch = collate_cre(items)
        pairs = [(pair["a"], pair["b"]) for pair in specification["pairs"]]
        pair_types = [pair["pair_type"] for pair in specification["pairs"]]
        yield batch, pairs, pair_types


def move_batch(batch):
    return {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model, dataset, batch_size):
    model.eval()
    logits, labels, ids, deltas, conditions = [], [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_cre)
        for batch in loader:
            output = model(move_batch(batch))
            if "logp" in output:
                probability = output["logp"].float().exp()
                score = output["condition_scores"].float()
            else:
                probability = output["logits"].float().softmax(dim=1)
                base = output["base_logits"].float()
                score = torch.stack(condition_scores_from_logits(base), dim=1)
            if not torch.isfinite(probability).all():
                raise FloatingPointError("Non-finite CRE validation probabilities")
            logits.append(probability.cpu())
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["id"])
            deltas.append(output["delta"].float().cpu())
            conditions.append(score.cpu())
    probabilities = torch.cat(logits).numpy()
    metrics = classification_metrics(labels, probabilities)
    delta_tensor = torch.cat(deltas)
    condition_tensor = torch.cat(conditions)
    diagnostics = {
        "delta_mean": delta_tensor.mean(0).tolist(),
        "delta_std": delta_tensor.std(0, unbiased=False).tolist(),
        "delta_abs_quantiles": torch.quantile(delta_tensor.abs(), torch.tensor([0.5, 0.9, 0.99])).tolist(),
        "condition_score_mean": condition_tensor.mean(0).tolist(),
        "seconds": time.perf_counter() - started,
    }
    return metrics, labels, ids, probabilities, diagnostics


def checkpoint_payload(model, config, metrics, epoch, global_step, optimizer=None, scheduler=None):
    return {
        "adapter_state_dict": model.adapter_state_dict(),
        "config": config,
        "validation": metrics,
        "epoch": epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "b0_checkpoint_sha256": file_hash(Path(config["base_checkpoint"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/cre_v1.json"))
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "experiments/glpc_runs/glpc_b0_s3407/best.pt")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CRE requires CUDA and BF16")
    requested = json.loads(args.config.read_text(encoding="utf-8"))
    config = make_effective_config(requested, args.mode, args.base_checkpoint.resolve())
    config["seed"] = args.seed
    if args.smoke:
        config.update(max_epochs=1, early_stopping_patience=1)
    output = RUN_ROOT / args.run_name
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if args.resume and config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume configuration differs")
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    base_architecture = dict(checkpoint["config"]["architecture"])
    base_architecture.update(config.get("architecture", {}))
    config["architecture"] = base_architecture
    config["base_checkpoint_sha256"] = file_hash(args.base_checkpoint)
    write_json(config_path, {"requested": requested, "effective": config})
    seed_everything(args.seed)
    tokenizer = tokenizer_local()
    train_records, val_records = load_records("train"), load_records("val")
    if args.smoke:
        train_records = [record for label in LABELS for record in train_records if record["label"] == label][:8]
    train_data = CREDataset(train_records, tokenizer, config["max_seq_length"])
    val_data = CREDataset(val_records, tokenizer, config["max_seq_length"])
    plan = build_sampling_plan(train_records, args.seed, config["batch_size"], config["samples_per_class"])
    write_json(output / "sampling_plan_manifest.json", plan)
    model = CREModel(config, checkpoint, args.mode).cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_info = {
        "total_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "new_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    write_json(output / "parameters.json", parameter_info)
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    updates_per_epoch = len(plan["batches_plan"])
    total_steps = config["max_epochs"] * max(1, updates_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    if args.resume:
        last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if last["config"] != config or last["b0_checkpoint_sha256"] != file_hash(args.base_checkpoint):
            raise ValueError("Resume checkpoint provenance differs")
        model.load_adapter_state_dict(last["adapter_state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        start_epoch = last["epoch"] + 1
        global_step = last["global_step"]
        best_overall = last["best_overall"]
        best_trained = last["best_trained"]
        stale = last["stale"]
    else:
        start_epoch, global_step, stale = 0, 0, 0
        best_overall, best_trained = None, None
    baseline_metrics, baseline_labels, baseline_ids, baseline_probs, baseline_diag = evaluate(model, val_data, config["batch_size"])
    if best_overall is None:
        best_overall = {"epoch": 0, "macro_f1": baseline_metrics["macro_f1"], "status": "ZERO_CORRECTION_SELECTED"}
        atomic_torch_save(checkpoint_payload(model, config, baseline_metrics, 0, 0), output / "best_overall.pt")
    write_json(output / "baseline_validation.json", {"metrics": baseline_metrics, "diagnostics": baseline_diag,
                                                     "test_partition_loaded": False, "samples": len(baseline_ids)})
    events = []
    started = time.perf_counter()
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        totals = {"four_class": 0.0, "pair": 0.0, "delta": 0.0, "total": 0.0}
        epoch_started = time.perf_counter()
        for batch, pair_indices, pair_types in batch_loader(plan, train_data, epoch, args.seed + 100000):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch = model(move_batch(batch))
                if "logp" in output_batch:
                    loss_four = F.nll_loss(output_batch["logp"].float(), batch["labels"].cuda())
                    logp = output_batch["logp"].float()
                else:
                    loss_four = F.cross_entropy(output_batch["logits"].float(), batch["labels"].cuda())
                    logp = output_batch["logits"].float().log_softmax(dim=1)
                loss_pair = pair_ranking_loss(logp, pair_indices, pair_types, config["pair_margin"], config["pair_temperature"]) if config["lambda_pair"] else logp.sum() * 0.0
                loss_delta = residual_regularization(output_batch["delta"], args.mode != "flat_cls")
                loss = loss_four + config["lambda_pair"] * loss_pair + config["lambda_delta"] * loss_delta
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CRE training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            global_step += 1
            for name, value in (("four_class", loss_four), ("pair", loss_pair), ("delta", loss_delta), ("total", loss)):
                totals[name] += float(value.detach())
        metrics, labels, ids, probabilities, diagnostics = evaluate(model, val_data, config["batch_size"])
        improved_overall = metrics["macro_f1"] > best_overall["macro_f1"]
        improved_trained = best_trained is None or metrics["macro_f1"] > best_trained["macro_f1"]
        if improved_overall:
            best_overall = {"epoch": epoch + 1, "macro_f1": metrics["macro_f1"], "status": "TRAINED_CORRECTION_SELECTED"}
            atomic_torch_save(checkpoint_payload(model, config, metrics, epoch + 1, global_step, optimizer, scheduler), output / "best_overall.pt")
        if improved_trained:
            best_trained = {"epoch": epoch + 1, "macro_f1": metrics["macro_f1"], "status": "TRAINED_CORRECTION_SELECTED"}
            atomic_torch_save(checkpoint_payload(model, config, metrics, epoch + 1, global_step, optimizer, scheduler), output / "best_trained.pt")
        stale = 0 if improved_trained else stale + 1
        event = {"epoch": epoch + 1, "global_step": global_step,
                 "mean_losses": {key: value / max(1, len(plan["batches_plan"])) for key, value in totals.items()},
                 "validation": metrics, "diagnostics": diagnostics, "improved_overall": improved_overall,
                 "improved_trained": improved_trained, "stale_epochs": stale,
                 "epoch_seconds": time.perf_counter() - epoch_started}
        events.append(event)
        atomic_torch_save(checkpoint_payload(model, config, metrics, epoch + 1, global_step, optimizer, scheduler), output / "last.pt")
        write_json(output / "events.json", events)
        if args.smoke or stale >= config["early_stopping_patience"]:
            break
    elapsed = time.perf_counter() - started
    if best_trained is None:
        best_trained = {"epoch": None, "macro_f1": None, "status": "NO_TRAINED_CHECKPOINT"}
    best_status = "ZERO_CORRECTION_SELECTED" if best_overall["epoch"] == 0 else "TRAINED_CORRECTION_SELECTED"
    write_json(output / "summary.json", {
        "status": "completed", "run_id": args.run_name, "mode": args.mode, "seed": args.seed,
        "epochs_completed": len(events), "best_overall": best_overall, "best_trained": best_trained,
        "selection_status": best_status, "test_partition_loaded": False,
        "training_seconds": elapsed, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "new_trainable_parameters": parameter_info["new_trainable_parameters"],
    })
    print(json.dumps({"run_id": args.run_name, "best_overall": best_overall, "best_trained": best_trained,
                      "epochs_completed": len(events), "selection_status": best_status}, indent=2))


if __name__ == "__main__":
    main()
