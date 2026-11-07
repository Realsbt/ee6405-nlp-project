"""Paired single-window and head/middle/tail inference for a main checkpoint.

This is an inference-only comparison. It never trains, selects a checkpoint,
or changes the evaluation records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from experiments.common import LABELS, classification_metrics, file_hash
from experiments.neural import build_model


WINDOW_LENGTH = 512


def windows_for_ids(input_ids: list[int], length: int = WINDOW_LENGTH) -> list[list[int]]:
    """Match the existing main-side head/middle/tail implementation."""
    if len(input_ids) <= length:
        return [input_ids]
    middle = max(0, (len(input_ids) - length) // 2)
    return [input_ids[:length], input_ids[middle:middle + length], input_ids[-length:]]


def probability_mean(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("At least one valid window probability is required")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Expected one probability vector per window")
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Window probabilities must be finite and normalized")
    result = values.mean(axis=0)
    return result / result.sum()


class WindowDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, mode: str):
        self.records = records
        self.mode = mode
        self.windows = []
        for record in records:
            encoded = tokenizer(
                record["article"], add_special_tokens=True, truncation=False,
                padding=False, return_attention_mask=False,
            )
            views = windows_for_ids(encoded["input_ids"])
            self.windows.append([views[0]] if mode == "single" else views)

    def __len__(self):
        return sum(len(views) for views in self.windows)

    def __getitem__(self, index):
        for record_index, views in enumerate(self.windows):
            if index < len(views):
                return {
                    "record_index": record_index,
                    "input_ids": torch.tensor(views[index], dtype=torch.long),
                }
            index -= len(views)
        raise IndexError(index)


def collate_windows(items, pad_id: int):
    max_length = max(len(item["input_ids"]) for item in items)
    return {
        "record_index": [item["record_index"] for item in items],
        "input_ids": torch.stack([
            torch.nn.functional.pad(item["input_ids"], (0, max_length - len(item["input_ids"])), value=pad_id)
            for item in items
        ]),
        "attention_mask": torch.stack([
            torch.nn.functional.pad(torch.ones_like(item["input_ids"]),
                                    (0, max_length - len(item["input_ids"])), value=0)
            for item in items
        ]),
    }


def predict(model, dataset: WindowDataset, tokenizer, batch_size: int, device: torch.device):
    probabilities = [[] for _ in dataset.windows]
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        loader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=lambda items: collate_windows(items, tokenizer.pad_token_id))
        for batch in loader:
            model_batch = {
                "full_text_input_ids": batch["input_ids"].to(device),
                "full_text_attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(model_batch)["logits"].float()
            values = logits.softmax(dim=1).cpu().numpy()
            for record_index, value in zip(batch["record_index"], values):
                probabilities[record_index].append(value)
    return [probability_mean(values) for values in probabilities], time.perf_counter() - started


def load_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8") if line.strip()]


def build_report(records, single, multi, multi_dataset, single_seconds, multi_seconds,
                 checkpoint, checkpoint_path):
    labels = np.asarray([record["label_id"] for record in records])
    single_probabilities = np.asarray(single)
    multi_probabilities = np.asarray(multi)
    single_metrics = classification_metrics(labels, single_probabilities)
    multi_metrics = classification_metrics(labels, multi_probabilities)
    single_prediction = single_probabilities.argmax(axis=1)
    multi_prediction = multi_probabilities.argmax(axis=1)
    return {
        "label_order": LABELS,
        "sample_count": len(records),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_hash(checkpoint_path),
            "config": checkpoint.get("config"),
            "seed": checkpoint.get("seed", "unknown"),
            "epoch": checkpoint.get("completed_epoch", checkpoint.get("epoch", "unknown")),
        },
        "input": {
            "window_length": WINDOW_LENGTH,
            "single_window": "first 512-token window",
            "multiview_windows": "first, centered, last 512-token windows; short documents use one window",
            "average": "arithmetic mean of per-window softmax probabilities",
            "single_window_count": len(records),
            "multiview_total_window_count": sum(len(views) for views in multi_dataset.windows),
            "multiview_window_count_distribution": {
                str(count): sum(len(views) == count for views in multi_dataset.windows)
                for count in sorted({len(views) for views in multi_dataset.windows})
            },
        },
        "single_window": {
            "metrics": single_metrics,
            "seconds": single_seconds,
            "predictions": [
                {"item_id": r["item_id"], "label_id": int(r["label_id"]),
                 "probabilities": [float(x) for x in p]}
                for r, p in zip(records, single_probabilities)
            ],
        },
        "multiview": {
            "metrics": multi_metrics,
            "seconds": multi_seconds,
            "predictions": [
                {"item_id": r["item_id"], "label_id": int(r["label_id"]),
                 "probabilities": [float(x) for x in p]}
                for r, p in zip(records, multi_probabilities)
            ],
        },
        "comparison": {
            "accuracy_delta_pp": 100 * (multi_metrics["accuracy"] - single_metrics["accuracy"]),
            "macro_f1_delta_pp": 100 * (multi_metrics["macro_f1"] - single_metrics["macro_f1"]),
            "wrong_to_right": int(((multi_prediction == labels) & (single_prediction != labels)).sum()),
            "right_to_wrong": int(((multi_prediction != labels) & (single_prediction == labels)).sum()),
            "documents_counted_once": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="Optional main config; must equal checkpoint config when provided.")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Tokenizer directory used by the checkpoint.")
    parser.add_argument("--data", type=Path, required=True,
                        help="JSONL records to evaluate; each record needs item_id, article, label_id.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and main config")
    config = checkpoint["config"]
    if args.config is not None:
        supplied = json.loads(args.config.read_text(encoding="utf-8"))
        if supplied != config:
            raise ValueError("Supplied config differs from the checkpoint config")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    device = torch.device(args.device)
    model = build_model(config, metadata=checkpoint.get("metadata")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    records = load_records(args.data)
    if not records or any(set(("item_id", "article", "label_id")) - set(record) for record in records):
        raise ValueError("Every record needs item_id, article, and label_id")
    single_dataset = WindowDataset(records, tokenizer, "single")
    multi_dataset = WindowDataset(records, tokenizer, "multi")
    single, single_seconds = predict(model, single_dataset, tokenizer, args.batch_size, device)
    multi, multi_seconds = predict(model, multi_dataset, tokenizer, args.batch_size, device)
    report = build_report(records, single, multi, multi_dataset, single_seconds,
                          multi_seconds, checkpoint, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
