"""Train a neural comparator on frozen partitions, selecting on validation only.

Last-completed-epoch checkpoints support restart after interruption. No author
fine-tuned checkpoint is loaded. A smoke run uses train/validation subsets only.
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiments.common import DATA_ROOT, DOMAINS, LABELS, classification_metrics, file_hash, write_json, write_jsonl
from experiments.neural import (BACKBONE, PROJECT_ROOT, RUN_ROOT, BalancedBatchSampler, TextDataset,
                                balanced_smoke_records, build_model, collate_batch, import_upstream,
                                load_records, resolve_use_supcon, tokenizer_local)


STOP_REQUESTED = False
RUN_OUTPUT = None


def stop_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def save_checkpoint(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise argparse.ArgumentTypeError("Use 1–80 ASCII letters, digits, underscores or hyphens; start with a letter or digit")
    return value


def evaluate(model, dataset, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=0)
    logits, labels, ids = [], [], []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch)["logits"].float()
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite evaluation logits")
            logits.append(output.cpu())
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["id"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    probabilities = torch.cat(logits).softmax(dim=1).numpy()
    return classification_metrics(labels, probabilities), probabilities, ids, seconds


def main():
    global RUN_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/neural_v1.json"))
    parser.add_argument("--model-type", choices=["roberta", "race", "race_no_supcon"], default="roberta")
    supcon = parser.add_mutually_exclusive_group()
    supcon.add_argument("--use-supcon", dest="use_supcon", action="store_true",
                        help="Enable supervised contrastive loss for any model architecture")
    supcon.add_argument("--no-supcon", dest="use_supcon", action="store_false",
                        help="Disable supervised contrastive loss for any model architecture")
    parser.set_defaults(use_supcon=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name", type=run_name,
                        help="Alternative output name; existing runs are preserved; smoke runs add a smoke_ prefix")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["architecture"]["backbone_model_path"] = str(BACKBONE)
    config.update(model_type=args.model_type, seed=args.seed,
                  use_supcon=resolve_use_supcon(args.model_type, config.get("use_supcon", False), args.use_supcon))
    if args.smoke:
        config.update(max_epochs=1, early_stopping_patience=1)
    run_id = ("smoke_" if args.smoke else "") + (args.run_name or f"{args.model_type}_s{args.seed}")
    output = RUN_ROOT / run_id
    report_dir = PROJECT_ROOT / "experiments/reports" / run_id
    if not args.resume and (output.exists() or report_dir.exists()):
        raise FileExistsError(f"Preserving existing run {output} / {report_dir}; choose a new --run-name or use --resume for a stopped run")
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume configuration differs from the recorded run")
    if args.resume and (output / "result.json").exists():
        raise ValueError("This run is complete; verify it instead of resuming")
    if args.resume and (output / "status.json").exists():
        prior = json.loads((output / "status.json").read_text())
        try:
            os.kill(prior["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Recorded process {prior['pid']} is still alive; inspect it before resuming")
    RUN_OUTPUT = output
    source_files = [Path(__file__), PROJECT_ROOT / "experiments/neural.py", PROJECT_ROOT / "experiments/common.py"]
    if args.model_type != "roberta":
        source_files.append(PROJECT_ROOT / "experiments/graph_data.py")
    source_hashes = {}
    for path in source_files:
        relative = str(path.relative_to(PROJECT_ROOT))
        destination = output / "source" / relative
        source_hashes[relative] = file_hash(path)
        if destination.exists():
            if file_hash(destination) != source_hashes[relative]:
                raise ValueError("Training source differs from the archived run; preserve the previous experiment")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    status = {"status": "running", "run_id": run_id, "pid": os.getpid(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    write_json(output / "config.json", config)
    write_json(report_dir / "config.json", config)
    def event(kind, **values):
        record = {"event": kind, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    tokenizer = tokenizer_local()
    train_records, val_records = load_records("train"), load_records("val")
    if args.smoke:
        train_records = balanced_smoke_records(train_records, 8)
        val_records = balanced_smoke_records(val_records, 2)
    metadata = None
    if args.model_type == "roberta":
        train_data, val_data = TextDataset(train_records, tokenizer), TextDataset(val_records, tokenizer)
    else:
        from experiments.graph_data import GraphDataset
        train_data, val_data = GraphDataset("train", train_records), GraphDataset("val", val_records)
        metadata = train_data.metadata
    sampler = BalancedBatchSampler(train_data.labels, config["batch_size"], args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_batch, num_workers=0)
    model = build_model(config, metadata).cuda()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert all(not name.startswith("bert_model.") or name.startswith("bert_model.encoder.layer.11.") for name, _ in trainable)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["max_epochs"] * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    criterion_supcon = import_upstream()[2](temperature=config["supcon_temperature"])
    parameter_info = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for _, p in trainable),
                      "trainable_parameter_names": [name for name, _ in trainable]}
    write_json(output / "parameters.json", parameter_info)
    event("initialized", train_records=len(train_data), val_records=len(val_data), batches_per_epoch=len(loader),
          trainable_parameters=parameter_info["trainable"], gpu=torch.cuda.get_device_name(0))
    start_epoch, best_score, best_epoch, stale, global_step = 0, -1.0, -1, 0, 0
    frozen_before = model.bert_model.embeddings.word_embeddings.weight[0].detach().clone()
    trained_before = model.classifier.out_proj.weight.detach().clone()
    if args.resume and (output / "last.pt").exists():
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["completed_epoch"] + 1
        best_score, best_epoch, stale, global_step = (checkpoint[k] for k in ("best_score", "best_epoch", "stale", "global_step"))
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    elif args.resume:
        event("resume_from_base", reason="No completed epoch checkpoint; partial epoch is discarded")
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        epoch_started = time.perf_counter()
        running_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch)
                labels = batch["labels"].cuda()
                loss_ce = F.cross_entropy(prediction["logits"].float(), labels)
                loss = loss_ce
                if config["use_supcon"]:
                    loss = loss + config["supcon_weight"] * criterion_supcon(F.normalize(prediction["features"].float(), dim=1), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["gradient_clip"], error_if_nonfinite=True)
            if args.smoke:
                assert model.classifier.out_proj.weight.grad is not None
                assert model.bert_model.encoder.layer[-1].attention.self.query.weight.grad is not None
                assert all(p.grad is None for p in model.bert_model.encoder.layer[0].parameters())
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss.detach())
            if step % 50 == 0 or step == len(loader) - 1:
                event("train_step", epoch=epoch + 1, batch=step + 1, batches=len(loader), global_step=global_step,
                      loss=float(loss.detach()), gradient_norm=float(gradient_norm), learning_rate=scheduler.get_last_lr()[0])
            if STOP_REQUESTED:
                status.update(status="interrupted", last_completed_epoch=epoch, resume_checkpoint=str(output / "last.pt") if (output / "last.pt").exists() else None)
                write_json(output / "status.json", status)
                event("interrupted", partial_epoch=epoch + 1, completed_steps_in_partial_epoch=step + 1)
                return
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - epoch_started
        metrics, _, _, eval_seconds = evaluate(model, val_data, config["batch_size"])
        improved = metrics["macro_f1"] > best_score
        if improved:
            best_score, best_epoch, stale = metrics["macro_f1"], epoch + 1, 0
            save_checkpoint(output / "best.pt", {"model_state_dict": model.state_dict(), "config": config, "metadata": metadata,
                                                 "best_epoch": best_epoch, "validation": metrics})
        else:
            stale += 1
        save_checkpoint(output / "last.pt", {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "config": config, "completed_epoch": epoch,
                         "best_score": best_score, "best_epoch": best_epoch, "stale": stale, "global_step": global_step,
                         "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all()})
        event("epoch_complete", epoch=epoch + 1, mean_train_loss=running_loss / len(loader),
              fit_seconds=fit_seconds, validation_seconds=eval_seconds, validation=metrics,
              improved=improved, best_epoch=best_epoch, stale_epochs=stale)
        status.update(last_completed_epoch=epoch + 1, best_epoch=best_epoch, best_validation_macro_f1=best_score)
        write_json(output / "status.json", status)
        if STOP_REQUESTED or stale >= config["early_stopping_patience"]:
            break
    fit_total_seconds = time.perf_counter() - training_started
    assert torch.equal(frozen_before, model.bert_model.embeddings.word_embeddings.weight[0].detach())
    assert not torch.equal(trained_before, model.classifier.out_proj.weight.detach()), "Classifier was not updated"
    if args.smoke:
        smoke = {"status": "passed", "run_id": run_id, "training_only_smoke": True,
                 "test_partition_loaded": False, "finite_gradients": True, "frozen_layer_unchanged": True,
                 "classifier_updated": True, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                 "validation": metrics, "config": config}
        write_json(output / "smoke.json", smoke)
        write_json(report_dir / "smoke.json", smoke)
        status.update(status="smoke_passed")
        write_json(output / "status.json", status)
        event("smoke_passed", peak_cuda_allocated_mib=smoke["peak_cuda_allocated_mib"])
        return
    if STOP_REQUESTED:
        status.update(status="interrupted")
        write_json(output / "status.json", status)
        return
    # The selected model is frozen before opening the held-out test partition.
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_records = load_records("test")
    if args.model_type == "roberta":
        test_data = TextDataset(test_records, tokenizer)
    else:
        from experiments.graph_data import GraphDataset
        test_data = GraphDataset("test", test_records)
    metrics, probabilities, ids, test_seconds = evaluate(model, test_data, config["batch_size"])
    assert ids == [row["item_id"] for row in test_records]
    predictions = [{"item_id": row["item_id"], "group_id": row["group_id"], "domain": row["domain"],
                    "true_label": row["label"], "prediction": LABELS[int(probs.argmax())],
                    "probabilities": dict(zip(LABELS, probs.tolist()))} for row, probs in zip(test_records, probabilities)]
    write_jsonl(report_dir / "test_predictions.jsonl", predictions)
    by_domain = {}
    y_test = np.asarray([row["label_id"] for row in test_records])
    for domain in DOMAINS:
        mask = np.asarray([row["domain"] == domain for row in test_records])
        by_domain[domain] = classification_metrics(y_test[mask], probabilities[mask])
    result = {"status": "completed", "run_id": run_id, "config": config, "best_epoch": best_epoch,
              "validation": best["validation"], "test": metrics, "test_by_domain": by_domain,
              "initialization": {"kind": "base pretraining only", "path": str(BACKBONE),
                                 "model_safetensors_sha256": file_hash(BACKBONE / "model.safetensors")},
              "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"),
              "checkpoint_path": str(output / "best.pt"), "checkpoint_sha256": file_hash(output / "best.pt"),
              "parameters": parameter_info, "efficiency": {"training_validation_checkpoint_seconds": fit_total_seconds,
                  "test_forward_seconds_excluding_tokenization": test_seconds,
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2},
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "torch-geometric", "numpy")},
              "code_sha256": source_hashes, "training_source_snapshot": str(output / "source"),
              "upstream_commit": "61ce528430a8bb84fd1bfb3285ee34827182ef59",
              "evidence_boundary": "One seed of one HART grouped in-domain split; not independent-dataset or unseen-generator generalization. Test did not select checkpoints."}
    write_json(output / "result.json", result)
    write_json(report_dir / "result.json", result)
    status.update(status="completed")
    write_json(output / "status.json", status)
    event("completed", test_macro_f1=metrics["macro_f1"], test_auroc=metrics["macro_auroc_ovr"], best_epoch=best_epoch)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if RUN_OUTPUT is not None and (RUN_OUTPUT / "status.json").exists():
            status = json.loads((RUN_OUTPUT / "status.json").read_text())
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
            write_json(RUN_OUTPUT / "status.json", status)
        raise
