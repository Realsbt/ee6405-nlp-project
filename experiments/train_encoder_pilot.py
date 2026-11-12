"""Bounded all-layer encoder pilot; train/validation only, no test interface."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from experiments.common import LABELS, classification_metrics, file_hash, read_jsonl, write_json, write_jsonl
from experiments.neural import BalancedBatchSampler

DEBERTA_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def load_partitions(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    partitions = {}
    for split in ("train", "val"):
        path = root / f"{split}.jsonl"
        if file_hash(path) != protocol["partitions"][split]["sha256"]:
            raise ValueError(f"Pilot {split} partition changed")
        rows = read_jsonl(path)
        if len(rows) != protocol["partitions"][split]["count"]:
            raise ValueError(f"Pilot {split} count differs")
        if any(row["source_split"] != split or row["label"] != LABELS[row["label_id"]]
               for row in rows):
            raise ValueError("Pilot partition/label mismatch")
        ids = [row["item_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate pilot IDs")
        partitions[split] = rows
    for field in ("item_id", "group_id", "split_block_id"):
        if ({row[field] for row in partitions["train"]}
                & {row[field] for row in partitions["val"]}):
            raise ValueError(f"Training/validation overlap: {field}")
    return partitions, protocol


class EncodedDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.labels = np.array([row["label_id"] for row in records])
        # Each encoder retokenizes the same frozen visible character prefix.
        raw = tokenizer([row["article"] for row in records], truncation=False,
                        padding=False, return_attention_mask=False, verbose=False)
        self.truncated = sum(len(ids) > max_length for ids in raw["input_ids"])
        self.encoded = tokenizer([row["article"] for row in records], truncation=True,
                                 max_length=max_length, padding=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return ({key: value[index] for key, value in self.encoded.items()},
                int(self.labels[index]), self.records[index]["item_id"])


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        features, labels, ids = zip(*items)
        return self.tokenizer.pad(list(features), return_tensors="pt"), torch.tensor(labels), list(ids)


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, seed):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(.1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.classifier = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, features):
        hidden = self.encoder(**features).last_hidden_state[:, 0]
        return self.classifier(self.dropout(hidden))


def evaluate(model, data, collator, device, micro_batch, amp):
    model.eval()
    probabilities, labels, ids = [], [], []
    loader = DataLoader(data, batch_size=micro_batch, shuffle=False, collate_fn=collator)
    with torch.no_grad():
        for features, target, identifiers in loader:
            features = {key: value.to(device) for key, value in features.items()}
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                logits = model(features)
            probabilities.extend(logits.float().softmax(-1).cpu().tolist())
            labels.extend(target.tolist())
            ids.extend(identifiers)
    return classification_metrics(labels, probabilities), probabilities, ids


def train(args):
    if args.output.exists():
        raise FileExistsError("Preserve previous pilot outputs")
    if (args.epochs < 1 or args.micro_batch < 1 or args.batch_size < 4 or args.batch_size % 4
            or args.batch_size % args.micro_batch or args.max_seconds <= 0):
        raise ValueError("Invalid training budget")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    partitions, protocol = load_partitions(args.data_root)
    if args.smoke:
        partitions = {
            split: [next(row for row in rows if row["label_id"] == label) for label in range(4)]
            for split, rows in partitions.items()
        }
    source = (str(args.roberta) if args.condition == "roberta_full"
              else "microsoft/deberta-v3-base")
    revision = None if args.condition == "roberta_full" else DEBERTA_REVISION
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, use_fast=args.condition == "roberta_full",
        trust_remote_code=False,
        local_files_only=args.condition == "roberta_full",
    )
    datasets = {split: EncodedDataset(rows, tokenizer) for split, rows in partitions.items()}
    encoder = AutoModel.from_pretrained(
        source, revision=revision, trust_remote_code=False,
        local_files_only=args.condition == "roberta_full",
    )
    encoder.gradient_checkpointing_enable()
    model = EncoderClassifier(encoder, args.seed).to(device)
    initial_head = model.classifier.weight.detach().cpu().clone()
    collator = PadCollator(tokenizer)
    batch_size = 4 if args.smoke else args.batch_size
    micro_batch = min(args.micro_batch, batch_size)
    accumulation = batch_size // micro_batch
    sampler = BalancedBatchSampler(datasets["train"].labels, batch_size, args.seed)
    epochs = 1 if args.smoke else args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(.1 * epochs * len(sampler)), epochs * len(sampler),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.output.mkdir(parents=True)
    config = {
        "condition": args.condition, "seed": args.seed, "epochs": epochs,
        "model_source": source, "revision": revision, "recipe": "all_layers_cls_ce_v1",
        "effective_batch": batch_size, "micro_batch": micro_batch,
        "learning_rate": 2e-5, "weight_decay": .01, "warmup_fraction": .1,
        "max_length": 512, "all_encoder_layers_trainable": True,
        "max_training_seconds": args.max_seconds,
        "precision": "cuda_fp16" if amp else "cpu_fp32",
        "test_evaluated": False, "smoke": args.smoke,
        "data_protocol_sha256": file_hash(args.data_root / "protocol.json"),
        "source_sha256": file_hash(__file__),
        "source_cache_sha256": protocol["source_inputs_sha256"],
        "retokenization_truncated_records": {k: v.truncated for k, v in datasets.items()},
        "tokenizer_class": type(tokenizer).__name__,
        "source_weights_sha256": {
            p.name: file_hash(p) for p in Path(source).glob("*")
            if p.name in ("model.safetensors", "pytorch_model.bin", "config.json")
        } if args.condition == "roberta_full" else {"hub_commit": revision},
        "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
        "gpu": torch.cuda.get_device_name(0) if amp else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output / "config.json", config)
    tokenizer.save_pretrained(args.output / "tokenizer")
    encoder.config.save_pretrained(args.output / "encoder_config")
    started = time.perf_counter()
    best_score, best_epoch, stale, completed_epochs = -1.0, 0, 0, 0
    budget_stopped = False
    updates = 0
    if amp:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        sampler.epoch = epoch
        epoch_started = time.perf_counter()
        losses = []
        for step, indices in enumerate(sampler, 1):
            if time.perf_counter() - started > args.max_seconds:
                budget_stopped = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for offset in range(0, batch_size, micro_batch):
                features, target, _ = collator([datasets["train"][i]
                                               for i in indices[offset:offset + micro_batch]])
                features = {key: value.to(device) for key, value in features.items()}
                with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                    loss = nn.functional.cross_entropy(model(features).float(), target.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite CE loss")
                scaler.scale(loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                optimizer.zero_grad(set_to_none=True)
                if not amp:
                    raise ValueError("Non-finite gradients")
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            updates += 1
            losses.append(loss_value)
            if step == 1 or step % 50 == 0 or step == len(sampler):
                print(json.dumps({
                    "event": "train_progress", "condition": args.condition,
                    "epoch": epoch + 1, "batch": step, "batches": len(sampler),
                    "mean_ce": float(np.mean(losses)),
                    "epoch_elapsed_seconds": time.perf_counter() - epoch_started,
                }), flush=True)
        if budget_stopped:
            break
        metrics, _, _ = evaluate(model, datasets["val"], collator, device, micro_batch, amp)
        completed_epochs += 1
        improved = metrics["macro_f1"] > best_score
        if improved:
            best_score, best_epoch, stale = metrics["macro_f1"], epoch + 1, 0
            # CPU copies avoid duplicating the checkpoint in device memory.
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "condition": args.condition, "seed": args.seed, "epoch": best_epoch},
                       args.output / "best.pt")
        else:
            stale += 1
        event = {
            "event": "epoch_complete", "condition": args.condition, "epoch": epoch + 1,
            "validation_macro_f1": metrics["macro_f1"],
            "mean_ce": float(np.mean(losses)) if losses else None,
            "epoch_seconds": time.perf_counter() - epoch_started, "improved": improved,
        }
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if stale >= 2:
            break
    if not (args.output / "best.pt").exists():
        write_json(args.output / "result.json", {
            "status": "incomplete_no_epoch", "completed_epochs": 0,
            "condition": args.condition, "test_evaluated": False,
        })
        raise RuntimeError("Budget ended before a complete epoch; pilot is incomplete")
    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if updates == 0 or torch.equal(initial_head, model.classifier.weight.detach().cpu()):
        raise RuntimeError("No verified classifier update")
    metrics, p, ids = evaluate(model, datasets["val"], collator, device, micro_batch, amp)
    if ids != [row["item_id"] for row in datasets["val"].records]:
        raise ValueError("Validation order changed")
    if abs(metrics["macro_f1"] - best_score) > 1e-4:
        raise ValueError("Checkpoint reload changed validation score")
    write_jsonl(args.output / "validation_predictions.jsonl", [{
        "item_id": row["item_id"], "group_id": row["group_id"], "domain": row["domain"],
        "true_label": row["label"], "probabilities": dict(zip(LABELS, probability)),
    } for row, probability in zip(datasets["val"].records, p)])
    result = {
        "status": "smoke_completed" if args.smoke else (
            "partial_budget" if budget_stopped else "validation_completed"),
        "condition": args.condition, "seed": args.seed, "validation": metrics,
        "test": None, "test_evaluated": False, "best_epoch": best_epoch,
        "completed_epochs": completed_epochs, "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": updates, "classifier_update_verified": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if amp else None,
        "best_checkpoint_sha256": file_hash(args.output / "best.pt"),
        "config_sha256": file_hash(args.output / "config.json"),
    }
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("roberta_full", "deberta_full"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--roberta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
