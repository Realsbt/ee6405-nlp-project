"""Train GLPC-v0 on train/validation only; test evaluation is intentionally separate."""

import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiments.common import DATA_ROOT, file_hash, write_json
from experiments.glpc_data import GLPCDataset, collate_glpc
from experiments.glpc_model import GLPCModel, consistency_js
from experiments.neural import BalancedBatchSampler, balanced_smoke_records, import_upstream, load_records, tokenizer_local


MODES = ("baseline", "creator_aux", "creator_local", "glpc")


def evaluate(model, dataset, batch_size):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_glpc):
            model_output = model({key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()})
            logits.append(model_output["logits"].float().cpu())
            labels.extend(batch["labels"].tolist())
    probabilities = torch.cat(logits).softmax(dim=1).numpy()
    from experiments.common import classification_metrics
    return classification_metrics(labels, probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/glpc_v1.json"))
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GLPC-v0 requires CUDA with BF16 support")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from experiments.neural import BACKBONE, RUN_ROOT
    config["architecture"]["backbone_model_path"] = str(BACKBONE)
    config.update(mode=args.mode, seed=args.seed)
    if args.smoke:
        config.update(max_epochs=1, early_stopping_patience=1, batch_size=16)
    run_id = "glpc_" + (args.run_name or f"{args.mode}_s{args.seed}")
    output = RUN_ROOT.parent / "glpc_runs" / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    tokenizer = tokenizer_local()
    train_records, val_records = load_records("train"), load_records("val")
    if args.smoke:
        train_records, val_records = balanced_smoke_records(train_records, 8), balanced_smoke_records(val_records, 2)
    train_data, val_data = GLPCDataset(train_records, tokenizer), GLPCDataset(val_records, tokenizer)
    sampler = BalancedBatchSampler(train_data.labels, config["batch_size"], args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_glpc)
    model = GLPCModel(config, args.mode).cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps = max(1, config["max_epochs"] * len(loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config["warmup_ratio"]), steps)
    supcon = import_upstream()[2](temperature=config["supcon_temperature"])
    creator_targets = torch.tensor([0, 0, 1, 1], device="cuda")
    best = -1.0
    events = []
    for epoch in range(config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        totals = {"loss": 0.0, "ce": 0.0, "supcon": 0.0, "creator": 0.0, "consistency": 0.0}
        for batch in loader:
            batch_cuda = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = model(batch_cuda)
                labels = batch_cuda["labels"]
                loss_ce = F.cross_entropy(model_output["logits"].float(), labels)
                loss_supcon = supcon(F.normalize(model_output["features"].float(), dim=1), labels)
                loss_creator = torch.zeros((), device="cuda")
                if model_output["creator_logits"] is not None:
                    loss_creator = F.cross_entropy(model_output["creator_logits"].float(), creator_targets[labels])
                loss_consistency = torch.zeros((), device="cuda")
                if args.mode == "glpc":
                    loss_consistency = consistency_js(model_output["logits"], model_output["creator_logits"])
                loss = loss_ce + config["supcon_weight"] * loss_supcon
                loss = loss + config["creator_weight"] * loss_creator + config["consistency_weight"] * loss_consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for name, value in (("loss", loss), ("ce", loss_ce), ("supcon", loss_supcon),
                                ("creator", loss_creator), ("consistency", loss_consistency)):
                totals[name] += float(value.detach())
        metrics = evaluate(model, val_data, config["batch_size"])
        event = {"epoch": epoch + 1, "mean_losses": {key: value / len(loader) for key, value in totals.items()},
                 "validation": metrics}
        events.append(event)
        write_json(output / "validation.json", {"best_macro_f1": max(best, metrics["macro_f1"]), "events": events})
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"model_state_dict": model.state_dict(), "config": config, "validation": metrics}, output / "best.pt")
        if args.smoke:
            break
    write_json(output / "summary.json", {"status": "completed", "run_id": run_id, "mode": args.mode,
                                         "seed": args.seed, "test_partition_loaded": False,
                                         "best_validation_macro_f1": best})
    print(json.dumps({"run_id": run_id, "best_validation_macro_f1": best}, indent=2))


if __name__ == "__main__":
    main()
