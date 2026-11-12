"""Export validation predictions and confusion matrices for GLPC runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common import LABELS, classification_metrics, write_json
from experiments.glpc_data import GLPCDataset, collate_glpc
from experiments.glpc_model import GLPCModel
from experiments.neural import load_records, tokenizer_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    model = GLPCModel(config, config["mode"]).cuda()
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = GLPCDataset(load_records(args.split), tokenizer_local())
    logits, labels, ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_glpc):
            output = model({key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()})
            logits.append(output["logits"].float().cpu())
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["id"])
    probabilities = torch.cat(logits).softmax(dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((4, 4), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    write_json(args.run_dir / f"{args.split}_predictions.json", {
        "split": args.split, "metrics": classification_metrics(labels, probabilities),
        "labels": LABELS, "confusion_matrix": matrix.tolist(),
        "predictions": [{"item_id": item_id, "label_id": int(label), "prediction": int(prediction),
                         "probabilities": [float(x) for x in row]}
                        for item_id, label, prediction, row in zip(ids, labels, predictions, probabilities)]
    })
    print(json.dumps({"run_dir": str(args.run_dir), "split": args.split,
                      "macro_f1": classification_metrics(labels, probabilities)["macro_f1"],
                      "confusion_matrix": matrix.tolist()}, indent=2))


if __name__ == "__main__":
    main()
