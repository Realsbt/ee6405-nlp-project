"""Train frozen-B0 residual ablations with optional same-source ranking."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiments.common import classification_metrics, write_json
from experiments.glpc_data import GLPCDataset, PairDataset, collate_glpc, collate_pairs, hard_source_pairs
from experiments.glpc_model import GLPCV1ResidualModel, pair_ranking_loss
from experiments.neural import BalancedBatchSampler, balanced_smoke_records, load_records, tokenizer_local


MODES = ("r1_mlp", "r2_mlp_pair", "r3_local", "r4_local_pair")


def evaluate(model, dataset, batch_size):
    model.eval()
    probabilities, labels, predictions = [], [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_glpc):
            batch_cuda = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            model_output = model(batch_cuda)
            probabilities.append(model_output["logits"].float().softmax(dim=1).cpu())
            labels.extend(batch["labels"].tolist())
            predictions.extend(model_output["logits"].float().argmax(dim=1).cpu().tolist())
    probs = torch.cat(probabilities).numpy()
    return classification_metrics(labels, probs), labels, probs, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--b0-checkpoint", type=Path, default=Path("experiments/glpc_runs/glpc_b0_s3407/best.pt"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GLPC-v1 requires CUDA with BF16 support")
    config = json.loads(Path("experiments/configs/glpc_v1.json").read_text(encoding="utf-8"))
    config["architecture"]["backbone_model_path"] = str(__import__("experiments.neural", fromlist=["BACKBONE"]).BACKBONE)
    config.update(v1_mode=args.mode, seed=args.seed, pair_weight=0.2, pair_margin=0.1, residual_weight=1e-4)
    if args.smoke:
        config.update(max_epochs=1, batch_size=16)
    output = Path("experiments/glpc_runs") / f"glpc_v1_{args.run_name}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    tokenizer = tokenizer_local()
    train_records, val_records = load_records("train"), load_records("val")
    if args.smoke:
        train_records = balanced_smoke_records(train_records, 8)
    train_data, val_data = GLPCDataset(train_records, tokenizer), GLPCDataset(val_records, tokenizer)
    pair_records = hard_source_pairs(train_records) if "pair" in args.mode else []
    pair_data = PairDataset(pair_records, tokenizer) if pair_records else None
    sampler = BalancedBatchSampler(train_data.labels, config["batch_size"], args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_glpc)
    pair_loader = DataLoader(pair_data, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_pairs) if pair_data else None
    model = GLPCV1ResidualModel(config, "local" if "local" in args.mode else "mlp").cuda()
    model.load_b0(torch.load(args.b0_checkpoint, map_location="cpu", weights_only=False))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["max_epochs"] * max(1, len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config["warmup_ratio"]), total_steps
    )
    best = -1.0
    best_epoch = 0
    stale_epochs = 0
    events = []
    pair_iterator = iter(pair_loader) if pair_loader else None
    for epoch in range(config["max_epochs"]):
        model.train()
        sampler.epoch = epoch
        totals = {"ce": 0.0, "pair": 0.0, "residual_reg": 0.0}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_cuda = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = model(batch_cuda)
                loss_ce = F.cross_entropy(model_output["logits"].float(), batch_cuda["labels"])
                loss_reg = model_output["residual_logits"].float().square().mean()
                loss_pair = torch.zeros((), device="cuda")
                if pair_iterator is not None:
                    try:
                        pair_batch = next(pair_iterator)
                    except StopIteration:
                        pair_iterator = iter(pair_loader)
                        pair_batch = next(pair_iterator)
                    negative = {key: value.cuda() if torch.is_tensor(value) else value for key, value in pair_batch["negative"].items()}
                    positive = {key: value.cuda() if torch.is_tensor(value) else value for key, value in pair_batch["positive"].items()}
                    negative_output = model(negative)
                    positive_output = model(positive)
                    loss_pair = pair_ranking_loss(negative_output["logits"], positive_output["logits"],
                                                  pair_batch["pair_type"], config["pair_margin"])
                loss = loss_ce + config["pair_weight"] * loss_pair + config["residual_weight"] * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            totals["ce"] += float(loss_ce.detach())
            totals["pair"] += float(loss_pair.detach())
            totals["residual_reg"] += float(loss_reg.detach())
        metrics, labels, probs, predictions = evaluate(model, val_data, config["batch_size"])
        mean_losses = {key: value / len(loader) for key, value in totals.items()}
        improved = metrics["macro_f1"] > best
        if improved:
            best = metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "validation": metrics,
                "epoch": best_epoch,
            }, output / "best.pt")
        else:
            stale_epochs += 1
        events.append({
            "epoch": epoch + 1,
            "mean_losses": mean_losses,
            "validation": metrics,
            "improved": improved,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "learning_rate": scheduler.get_last_lr()[0],
        })
        write_json(output / "validation.json", {
            "best_macro_f1": best,
            "best_epoch": best_epoch,
            "events": events,
            "pair_groups": len(pair_records),
        })
        if args.smoke or stale_epochs >= config["early_stopping_patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics, labels, probs, predictions = evaluate(model, val_data, config["batch_size"])
    write_json(output / "summary.json", {
        "status": "completed",
        "run_id": output.name,
        "mode": args.mode,
        "seed": args.seed,
        "epochs_completed": len(events),
        "best_epoch": best_epoch,
        "pair_groups": len(pair_records),
        "test_partition_loaded": False,
        "best_validation_macro_f1": metrics["macro_f1"],
    })
    write_json(output / "predictions.json", {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probs.tolist(),
    })


if __name__ == "__main__":
    main()
