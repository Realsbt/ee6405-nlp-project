"""Recompute CRE validation metrics from an adapter checkpoint in a fresh process."""

import argparse
import json
from pathlib import Path

import torch

from experiments.common import DATA_ROOT, LABELS, classification_metrics, write_json
from experiments.cre_data import CREDataset, collate_cre
from experiments.cre_model import CREModel
from experiments.neural import load_records, tokenizer_local
from experiments.train_cre import evaluate


def load_run(run_dir):
    effective = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["effective"]
    b0 = torch.load(effective["base_checkpoint"], map_location="cpu", weights_only=False)
    model = CREModel(effective, b0, effective["mode"]).cuda()
    checkpoint = torch.load(run_dir / "best_overall.pt", map_location="cpu", weights_only=False)
    model.load_adapter_state_dict(checkpoint["adapter_state_dict"])
    dataset = CREDataset(load_records("val"), tokenizer_local(), effective["max_seq_length"])
    return model, dataset, effective, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CRE evaluation requires CUDA")
    model, dataset, config, checkpoint = load_run(args.run_dir)
    metrics, labels, ids, probabilities, diagnostics = evaluate(model, dataset, config["batch_size"])
    predictions = probabilities.argmax(axis=1)
    result = {
        "run_id": args.run_dir.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "metrics": metrics,
        "diagnostics": diagnostics,
        "sample_ids_sha256": __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest(),
        "prediction_counts": {LABELS[i]: int((predictions == i).sum()) for i in range(4)},
        "test_partition_loaded": False,
    }
    output = args.output or args.run_dir / "independent_validation.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
