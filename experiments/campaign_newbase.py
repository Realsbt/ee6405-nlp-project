"""Budgeted cold-start screen for E14/E15/E16 using train and S only."""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from experiments.common import classification_metrics, file_hash, write_json
from experiments.neural import load_records, tokenizer_local


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"
BACKBONE = ROOT / "assets/roberta-base"


class Records(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.encoded = tokenizer([row["article"] for row in rows], max_length=max_length,
                                 truncation=True, padding=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.encoded["input_ids"][index]),
                "label": torch.tensor(self.rows[index]["label_id"])}


def collate(items, pad_id=1):
    length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.stack([F.pad(item["input_ids"], (0, length-len(item["input_ids"])), value=pad_id)
                                  for item in items]),
        "attention_mask": torch.stack([F.pad(torch.ones_like(item["input_ids"]),
                                             (0, length-len(item["input_ids"])), value=0)
                                       for item in items]),
        "labels": torch.stack([item["label"] for item in items]),
    }


class ColdModel(nn.Module):
    def __init__(self, model_name, local_path=None):
        super().__init__()
        source = str(local_path) if local_path else model_name
        self.encoder = AutoModel.from_pretrained(source, local_files_only=local_path is not None)
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(hidden, 4)

    def forward(self, batch):
        output = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(self.dropout(self.norm(pooled)))


def evaluate(model, dataset, pad_id=1):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16,
                                collate_fn=partial(collate, pad_id=pad_id)):
            batch = {key: value.cuda() for key, value in batch.items()}
            logits.append(model(batch).float().cpu())
            labels.extend(batch["labels"].cpu().tolist())
    return classification_metrics(labels, torch.cat(logits).softmax(1).numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("E14", "E15", "E16"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--max-updates", type=int, default=320)
    args = parser.parse_args()
    if args.formal and args.task != "E16":
        raise ValueError("--formal is only supported for E16")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cold-start screen requires CUDA and BF16")
    started = time.time()
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    rows_train = load_records("train")
    rows_val = load_records("val")
    manifest = json.loads((CAMPAIGN / "protocol/split_manifest.json").read_text(encoding="utf-8"))
    s_ids = {row["item_id"] for row in manifest["records"] if row["partition"] == "S"}
    rows_s = [row for row in rows_val if row["item_id"] in s_ids]
    source = {"E14": "roberta-base", "E15": "microsoft/deberta-v3-small",
              "E16": "microsoft/deberta-v3-base"}[args.task]
    local_path = BACKBONE if args.task == "E14" else None
    try:
        if args.task == "E14":
            tokenizer = tokenizer_local()
        elif args.task == "E15":
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = ColdModel(source, local_path).cuda()
        if args.formal:
            model.encoder.gradient_checkpointing_enable()
    except (OSError, EnvironmentError) as error:
        run = CAMPAIGN / "tasks" / args.task
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "selection.json", {"task_id": args.task, "status": "RESOURCE_BLOCKED",
                                            "metrics": None, "optimizer_updates": 0,
                                            "stopped_reason": "model_not_available_locally",
                                            "error": str(error), "test_partition_loaded": False,
                                            "hc3_loaded": False})
        print(json.dumps({"task_id": args.task, "status": "RESOURCE_BLOCKED"}, indent=2))
        return
    train_data, s_data = Records(rows_train, tokenizer), Records(rows_s, tokenizer)
    pad_id = tokenizer.pad_token_id
    initial = evaluate(model, s_data, pad_id)
    if args.formal:
        run = CAMPAIGN / "top5_revision_01" / "E16_final_deberta_v3_base_s3407"
        run.mkdir(parents=True, exist_ok=True)
        microbatch, accumulation, max_epochs = 2, (2 if args.smoke else 8), (1 if args.smoke else 4)
        smoke_updates = 4 if args.smoke else None
        loader = DataLoader(
            train_data, batch_size=microbatch, shuffle=True,
            collate_fn=partial(collate, pad_id=pad_id),
        )
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if name.startswith("encoder."):
                    (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "lr": 1e-5, "weight_decay": .01},
            {"params": no_decay, "lr": 1e-5, "weight_decay": 0.0},
            {"params": list(model.norm.parameters()) + list(model.classifier.parameters()),
             "lr": 5e-5, "weight_decay": 0.0},
        ])
        total_updates = max_epochs * ((len(loader) + accumulation - 1) // accumulation)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=.1, end_factor=0.0, total_iters=max(1, total_updates)
        )
        best = initial
        best_epoch = 0
        events = [{"epoch": 0, "validation": initial}]
        updates = 0
        model.train()
        for epoch in range(1, max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader, 1):
                batch = {key: value.cuda() for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(batch).float(), batch["labels"]) / accumulation
                loss.backward()
                if step % accumulation == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if smoke_updates is not None and updates >= smoke_updates:
                        break
            if smoke_updates is not None and updates >= smoke_updates:
                current = evaluate(model, s_data, pad_id)
                events.append({"epoch": epoch, "updates": updates, "validation": current})
                best, best_epoch = current, epoch
                torch.save({"model_state_dict": model.state_dict(), "validation": best,
                            "epoch": epoch, "source": source, "seed": 3407},
                           run / "smoke.pt")
                break
            current = evaluate(model, s_data, pad_id)
            events.append({"epoch": epoch, "updates": updates, "validation": current})
            if (current["accuracy"] + current["macro_f1"]) > (best["accuracy"] + best["macro_f1"]):
                best, best_epoch = current, epoch
                torch.save({"model_state_dict": model.state_dict(), "validation": best,
                            "epoch": epoch, "source": source, "seed": 3407},
                           run / "best.pt")
            model.train()
        payload = {
            "task_id": args.task, "status": "COMPLETE", "formal_training": True,
            "metrics": best, "epoch0_metrics": initial, "best_epoch": best_epoch,
            "actual_epochs": max_epochs, "optimizer_updates": updates,
            "stopped_reason": "smoke_limit" if args.smoke else "max_epochs", "events": events,
            "seed": 3407, "microbatch": microbatch, "gradient_accumulation": accumulation,
            "effective_batch": microbatch * accumulation, "max_length": 512,
            "encoder_lr": 1e-5, "head_lr": 5e-5, "weight_decay": .01,
            "gradient_checkpointing": True, "model_source": source,
            "test_partition_loaded": False, "hc3_loaded": False,
            "wall_seconds": time.time() - started,
        }
        write_json(run / "selection.json", payload)
        print(json.dumps(payload, indent=2))
        return
    loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": list(model.norm.parameters()) + list(model.classifier.parameters()), "lr": 5e-5},
    ], weight_decay=.01)
    best = initial
    events = [{"updates": 0, "validation": initial}]
    updates = 0
    model.train()
    while updates < (8 if args.smoke else args.max_updates):
        for batch in loader:
            batch = {key: value.cuda() for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(batch).float(), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if args.smoke or updates % 40 == 0 or updates == args.max_updates:
                current = evaluate(model, s_data)
                events.append({"updates": updates, "validation": current, "loss": float(loss.detach())})
                if current["macro_f1"] > best["macro_f1"]:
                    best = current
            if updates >= (8 if args.smoke else args.max_updates):
                break
    run = CAMPAIGN / "tasks" / args.task
    run.mkdir(parents=True, exist_ok=True)
    payload = {"task_id": args.task, "status": "UNDERTRAINED" if updates < args.max_updates else "COMPLETE",
               "metrics": best, "epoch0_metrics": initial, "optimizer_updates": updates,
               "delta_accuracy_pp": 100*(best["accuracy"]-initial["accuracy"]),
               "delta_macro_f1_pp": 100*(best["macro_f1"]-initial["macro_f1"]),
               "stopped_reason": "smoke_limit" if args.smoke else "screen_update_budget",
               "wall_seconds": time.time()-started, "events": events,
               "model_source": source, "test_partition_loaded": False, "hc3_loaded": False}
    write_json(run / "selection.json", payload)
    torch.save({"model_state_dict": model.state_dict(), "validation": best, "source": source}, run / "screen_best.pt")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
