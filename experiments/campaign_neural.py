"""Budgeted E08/E09 B0 adaptation screen for the 36-task campaign."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.common import DATA_ROOT, classification_metrics, file_hash, write_json
from experiments.glpc_data import GLPCDataset, collate_glpc
from experiments.glpc_model import GLPCModel
from experiments.neural import BalancedBatchSampler, load_records, tokenizer_local


ROOT = Path(__file__).resolve().parents[1]
B0 = ROOT / "experiments/glpc_runs/glpc_b0_s3407/best.pt"
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"


def evaluate(model, dataset, indices, batch_size=16):
    subset = torch.utils.data.Subset(dataset, indices)
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=batch_size, collate_fn=collate_glpc):
            output = model({key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()})
            logits.append(output["logits"].float().cpu())
            labels.extend(batch["labels"].tolist())
    probabilities = torch.cat(logits).softmax(1).numpy()
    return classification_metrics(labels, probabilities), probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("E08", "E09", "E11", "E12", "E13"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-updates", type=int, default=160)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Campaign neural screen requires CUDA and BF16")
    started = time.time()
    train_records, val_records = load_records("train"), load_records("val")
    manifest = json.loads((CAMPAIGN / "protocol/split_manifest.json").read_text(encoding="utf-8"))
    s_ids = {row["item_id"] for row in manifest["records"] if row["partition"] == "S"}
    s_records = [record for record in val_records if record["item_id"] in s_ids]
    tokenizer = tokenizer_local()
    train_data, s_data = GLPCDataset(train_records, tokenizer), GLPCDataset(s_records, tokenizer)
    checkpoint = torch.load(B0, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["model_type"] = "roberta"
    config["classifier_type"] = "mlp"
    config["architecture"] = dict(config["architecture"])
    config["architecture"]["unfreeze_bert_layers"] = 4 if args.task in ("E08", "E11", "E12", "E13") else 12
    model = GLPCModel(config, mode="baseline").cuda()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sampler = BalancedBatchSampler(train_data.labels, 16, 3407)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_glpc)
    optimizer = torch.optim.AdamW(trainable, lr=5e-6 if args.task in ("E08", "E11", "E12", "E13") else 3e-6, weight_decay=.01)
    initial_metrics, _ = evaluate(model, s_data, list(range(len(s_data))))
    updates = 0
    best = initial_metrics
    events = [{"updates": 0, "validation": initial_metrics}]
    sampler.epoch = 0
    model.train()
    for batch in loader:
        if updates >= (8 if args.smoke else args.max_updates):
            break
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            if args.task == "E12":
                embedding = model.bert_model.embeddings(input_ids=batch["input_ids"])
                embedding.requires_grad_(True)
                clean = model(batch, inputs_embeds=embedding)["logits"]
                clean_loss = F.cross_entropy(clean.float(), batch["labels"])
                gradient = torch.autograd.grad(clean_loss, embedding, retain_graph=True)[0]
                delta = 0.01 * gradient / gradient.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
                delta = delta.detach()
                adversarial = model(batch, inputs_embeds=embedding + delta)["logits"]
                loss = 0.5 * (clean_loss + F.cross_entropy(adversarial.float(), batch["labels"]))
            elif args.task == "E11":
                second = model(batch)
                loss = 0.5 * (F.cross_entropy(output["logits"].float(), batch["labels"])
                              + F.cross_entropy(second["logits"].float(), batch["labels"]))
                loss = loss + 0.5 * F.mse_loss(
                    output["logits"].float().log_softmax(-1),
                    second["logits"].float().log_softmax(-1),
                )
            elif args.task == "E13":
                probabilities = output["logits"].float().softmax(-1)
                target_probability = probabilities.gather(1, batch["labels"].unsqueeze(1)).squeeze(1)
                loss = (-((1 - target_probability) ** 1.0)
                        * output["logits"].float().log_softmax(-1)
                        .gather(1, batch["labels"].unsqueeze(1)).squeeze(1)).mean()
            else:
                loss = F.cross_entropy(output["logits"].float(), batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        if updates % 40 == 0 or updates == (8 if args.smoke else args.max_updates):
            current, probabilities = evaluate(model, s_data, list(range(len(s_data))))
            events.append({"updates": updates, "validation": current, "loss": float(loss.detach())})
            if current["macro_f1"] > best["macro_f1"]:
                best = current
    run = CAMPAIGN / "tasks" / args.task
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": args.task, "status": "UNDERTRAINED" if updates < args.max_updates else "COMPLETE",
        "metrics": best, "base_metrics": initial_metrics,
        "delta_accuracy_pp": 100 * (best["accuracy"] - initial_metrics["accuracy"]),
        "delta_macro_f1_pp": 100 * (best["macro_f1"] - initial_metrics["macro_f1"]),
        "optimizer_updates": updates, "max_updates": args.max_updates,
        "stopped_reason": "smoke_limit" if args.smoke else "screen_update_budget",
        "wall_seconds": time.time() - started, "test_partition_loaded": False, "hc3_loaded": False,
        "b0_sha256": file_hash(B0), "events": events,
    }
    write_json(run / "selection.json", payload)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "validation": best}, run / "screen_best.pt")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
