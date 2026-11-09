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
from experiments.research import validate, optimizer_groups, auxiliary_loss
from experiments.ablation import condition_name, resolve_config
from experiments.neural import (BACKBONE, PROJECT_ROOT, RUN_ROOT, BalancedBatchSampler, TextDataset,
                                balanced_smoke_records, build_model, collate_batch, import_upstream,
                                load_records, tokenizer_local)


STOP_REQUESTED = False
RUN_OUTPUT = None
TRACKER = None


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


def evaluate(model, dataset, batch_size, progress_interval=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=0)
    logits, labels, ids = [], [], []
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            output = model(batch)["logits"].float()
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite evaluation logits")
            logits.append(output.cpu())
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["id"])
            if progress_interval and (step % progress_interval == 0 or step == len(loader)):
                print(f"Evaluation {len(ids)}/{len(dataset)} samples ({time.perf_counter() - started:.1f}s)", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    probabilities = torch.cat(logits).softmax(dim=1).numpy()
    return classification_metrics(labels, probabilities), probabilities, ids, seconds


def main():
    global RUN_OUTPUT, TRACKER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/neural_v1.json"))
    parser.add_argument("--model-type", choices=["roberta", "race", "race_no_supcon"],
                        help="Architecture override; loss is independently read from JSON or --[no-]supcon")
    parser.add_argument("--supcon", action=argparse.BooleanOptionalAction, default=None,
                        help="Override use_supcon independently of the architecture")
    parser.add_argument("--seed", type=int, help="Override the JSON training seed")
    parser.add_argument("--run-name", type=run_name,
                        help="Alternative output name; existing runs are preserved; smoke runs add a smoke_ prefix")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Explicit held-out evaluation; never for screening")
    args = parser.parse_args()
    try:
        config = resolve_config(json.loads(args.config.read_text()), args.model_type, args.seed, args.supcon)
    except ValueError as error:
        parser.error(str(error))
    validate(config)
    seed = config["seed"]
    if args.smoke:
        config.update(max_epochs=1, early_stopping_patience=1)
    run_id = ("smoke_" if args.smoke else "") + (args.run_name or f"{condition_name(config)}_s{seed}")
    output = RUN_ROOT / run_id
    report_dir = RUN_ROOT / "reports" / run_id
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
    source_files = [Path(__file__), PROJECT_ROOT / "experiments/neural.py", PROJECT_ROOT / "experiments/common.py",
                    PROJECT_ROOT / "experiments/ablation.py", PROJECT_ROOT / "experiments/paths.py", PROJECT_ROOT / "experiments/research.py", PROJECT_ROOT / "experiments/tracking.py"]
    if config["model_type"] != "roberta":
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    status = {"status": "running", "run_id": run_id, "pid": os.getpid(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    write_json(output / "config.json", config)
    write_json(report_dir / "config.json", config)
    if config.get("wandb_enabled", False):
        from experiments.tracking import Tracker
        TRACKER = Tracker(output, config, resume=args.resume, job_type="smoke" if args.smoke else "train")
    def event(kind, **values):
        record = {"event": kind, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if TRACKER is not None:
            TRACKER.event(kind, record)
    tokenizer = tokenizer_local()
    train_records, val_records = load_records("train"), load_records("val")
    if args.smoke:
        train_records = balanced_smoke_records(train_records, 8)
        val_records = balanced_smoke_records(val_records, 2)
    metadata = None
    if config["model_type"] == "roberta":
        train_data, val_data = TextDataset(train_records, tokenizer), TextDataset(val_records, tokenizer)
    else:
        from experiments.graph_data import GraphDataset
        train_data, val_data = GraphDataset("train", train_records), GraphDataset("val", val_records)
        metadata = train_data.metadata
    sampler = BalancedBatchSampler(train_data.labels, config["batch_size"], seed)
    workers = config.get("dataloader_num_workers", 0)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_batch, num_workers=workers,
                        pin_memory=config.get("pin_memory", False),
                        persistent_workers=bool(workers and config.get("persistent_workers", False)))
    model = build_model(config, metadata).cuda()
    if TRACKER is not None:
        TRACKER.watch(model, config)
    forward_model = torch.compile(model) if config.get("torch_compile", False) else model
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, fused=config.get("fused_adamw", False))
    accumulation = config.get("gradient_accumulation_steps", 1)
    total_steps = config["max_epochs"] * math.ceil(len(loader) / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    criterion_supcon = import_upstream()[2](temperature=config["supcon_temperature"])
    parameter_info = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for _, p in trainable),
                      "trainable_parameter_names": [name for name, _ in trainable]}
    parameter_info["trainable_layers"] = list(range(12 - config["architecture"]["unfreeze_bert_layers"], 12))
    parameter_info["optimizer_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in groups]
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
    fit_epochs = []
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        epoch_started = time.perf_counter()
        running_loss = running_ce = running_supcon = 0.0
        last_step_done = time.perf_counter()
        for step, batch in enumerate(loader):
            step_started = time.perf_counter()
            data_wait = step_started - last_step_done
            if step % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = forward_model(batch)
                labels = batch["labels"].cuda()
                loss_ce = F.cross_entropy(prediction["logits"].float(), labels)
                loss = loss_ce
                loss_supcon = loss_ce.new_zeros(())
                if config["use_supcon"]:
                    features = prediction["features"]
                    if config.get("supcon_projection", False):
                        features = model.supcon_projection(features)
                    loss_supcon = criterion_supcon(F.normalize(features.float(), dim=1), labels)
                    loss = loss + config["supcon_weight"] * loss_supcon
                loss_aux = auxiliary_loss(model, prediction["features"], labels, config)
                loss = loss + loss_aux
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            window = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / window).backward()
            update = (step + 1) % accumulation == 0 or step == len(loader) - 1
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["gradient_clip"] if update else float("inf"), error_if_nonfinite=True)
            if args.smoke:
                assert model.classifier.out_proj.weight.grad is not None
                assert model.bert_model.encoder.layer[-1].attention.self.query.weight.grad is not None
                first = 12 - config["architecture"]["unfreeze_bert_layers"]
                for layer in model.bert_model.encoder.layer[:first]:
                    assert all(p.grad is None for p in layer.parameters())
                for layer in model.bert_model.encoder.layer[first:]:
                    assert layer.attention.self.query.weight.grad is not None
            if update:
                optimizer.step()
                scheduler.step()
                global_step += 1
            running_loss += float(loss.detach())
            running_ce += float(loss_ce.detach())
            running_supcon += float(loss_supcon.detach())
            if step % 50 == 0 or step == len(loader) - 1:
                event("train_step", epoch=epoch + 1, batch=step + 1, batches=len(loader), global_step=global_step,
                      loss=float(loss.detach()), loss_ce=float(loss_ce.detach()), loss_supcon=float(loss_supcon.detach()),
                      gradient_norm=float(gradient_norm), learning_rate=scheduler.get_last_lr()[0])
            if TRACKER is not None and (step % TRACKER.log_interval == 0 or step == len(loader)-1):
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - step_started
                telemetry = {"batch_step":epoch*len(loader)+step+1, "train/epoch":epoch+1,
                    "train/global_optimizer_step":global_step, "train/loss":float(loss.detach()),
                    "train/ce_loss":float(loss_ce.detach()),"train/supcon_loss":float(loss_supcon.detach()),
                    "train/auxiliary_loss":float(loss_aux.detach()),"train/gradient_norm_before_clip":float(gradient_norm),
                    "train/step_seconds":elapsed,"train/data_wait_seconds":data_wait,
                    "train/samples_per_second":len(batch["labels"])/elapsed,
                    "memory/allocated_mib":torch.cuda.memory_allocated()/1024**2,
                    "memory/reserved_mib":torch.cuda.memory_reserved()/1024**2,
                    "memory/peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
                    "memory/peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
                telemetry.update({f"lr/{g['name']}_{i}":g["lr"] for i,g in enumerate(optimizer.param_groups)})
                TRACKER.log(telemetry)
            last_step_done = time.perf_counter()
            if STOP_REQUESTED:
                status.update(status="interrupted", last_completed_epoch=epoch, resume_checkpoint=str(output / "last.pt") if (output / "last.pt").exists() else None)
                write_json(output / "status.json", status)
                event("interrupted", partial_epoch=epoch + 1, completed_steps_in_partial_epoch=step + 1)
                return
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - epoch_started
        fit_epochs.append(fit_seconds)
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
              mean_train_ce=running_ce / len(loader), mean_train_supcon=running_supcon / len(loader),
              fit_seconds=fit_seconds, validation_seconds=eval_seconds, validation=metrics,
              improved=improved, best_epoch=best_epoch, stale_epochs=stale)
        status.update(last_completed_epoch=epoch + 1, best_epoch=best_epoch, best_validation_macro_f1=best_score)
        write_json(output / "status.json", status)
        if STOP_REQUESTED or stale >= config["early_stopping_patience"]:
            break
    fit_total_seconds = time.perf_counter() - training_started
    write_json(output / "training_finished.json", {"pid":os.getpid(), "time":time.time(), "fit_seconds":fit_total_seconds})
    if not config.get("unfreeze_embeddings", False):
        assert torch.equal(frozen_before, model.bert_model.embeddings.word_embeddings.weight[0].detach())
    assert not torch.equal(trained_before, model.classifier.out_proj.weight.detach()), "Classifier was not updated"
    if args.smoke:
        smoke = {"status": "passed", "run_id": run_id, "training_only_smoke": True,
                 "test_partition_loaded": False, "finite_gradients": True, "frozen_layer_unchanged": not config.get("unfreeze_embeddings", False),
                 "classifier_updated": True, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                 "validation": metrics, "config": config,
                 "mean_train_ce": running_ce / len(loader), "mean_train_supcon": running_supcon / len(loader)}
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
    val_metrics, val_probs, val_ids, val_seconds = evaluate(model, val_data, config["batch_size"])
    assert val_ids == [r["item_id"] for r in val_records]
    write_jsonl(output / "validation_predictions.jsonl", [
        {"item_id": r["item_id"], "true_label": r["label"],
         "probabilities": dict(zip(LABELS, p.tolist()))} for r, p in zip(val_records, val_probs)])
    if TRACKER is not None:
        TRACKER.evaluation(val_metrics, val_probs, val_records)
    validation_result = {
        "status": "completed", "run_id": run_id, "config": config, "best_epoch": best_epoch,
        "validation": val_metrics, "test_partition_loaded": False,
        "checkpoint_path": str(output / "best.pt"), "checkpoint_sha256": file_hash(output / "best.pt"),
        "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"), "code_sha256": source_hashes,
        "parameters": parameter_info,
        "efficiency": {"training_validation_checkpoint_seconds": fit_total_seconds,
                       "fit_seconds_per_epoch": fit_epochs,
                       "validation_forward_seconds": val_seconds,
                       "validation_samples_per_second": len(val_records) / val_seconds,
                       "samples_per_second": len(loader) * config["batch_size"] * len(fit_epochs) / sum(fit_epochs) if fit_epochs else None,
                       "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                       "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}}
    if not args.evaluate_test:
        write_json(output / "result.json", validation_result)
        write_json(report_dir / "result.json", validation_result)
        status.update(status="completed", test_partition_loaded=False)
        write_json(output / "status.json", status)
        event("completed", validation_macro_f1=val_metrics["macro_f1"], test_partition_loaded=False)
        return
    test_records = load_records("test")
    if config["model_type"] == "roberta":
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
              "validation": best["validation"], "test_partition_loaded": True, "test": metrics, "test_by_domain": by_domain,
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
        if TRACKER is not None:
            TRACKER.log({"error/type":type(error).__name__, "error/message":str(error)})
        if RUN_OUTPUT is not None and (RUN_OUTPUT / "status.json").exists():
            status = json.loads((RUN_OUTPUT / "status.json").read_text())
            status.update(status="failed", error_type=type(error).__name__, error=str(error),
                          oom=isinstance(error, torch.OutOfMemoryError),
                          peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_initialized() else None,
                          peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_initialized() else None)
            write_json(RUN_OUTPUT / "status.json", status)
        raise
    finally:
        if TRACKER is not None:
            summary = None
            status = "failed"
            if RUN_OUTPUT is not None:
                if (RUN_OUTPUT / "status.json").exists():
                    status = json.loads((RUN_OUTPUT / "status.json").read_text())["status"]
                if (RUN_OUTPUT / "result.json").exists():
                    summary = json.loads((RUN_OUTPUT / "result.json").read_text())
            TRACKER.finish(status, summary)
