"""Replay the frozen HC3 set with published TF-IDF/RoBERTa checkpoints on CPU."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ["EE6405_STORAGE"] = str(ROOT / "resources")
os.environ["EE6405_UPSTREAM"] = str(ROOT / "resources/RACE")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(ROOT / "repository"))

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.common import LABELS, file_hash, read_jsonl
from experiments.neural import TextDataset, build_model, collate_batch, tokenizer_local
from scripts.restore_data import hc3
from analyze_saved_probabilities import binary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tfidf", "roberta"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions_output = args.output.with_suffix(".predictions.jsonl")
    if args.output.exists() or predictions_output.exists():
        raise FileExistsError("Preserving existing outputs; choose a new output path")
    manifest = json.loads((ROOT / "repository/resources/manifest.json").read_text())
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "resources/RACE"), "rev-parse", "HEAD"],
        text=True).strip()
    if upstream_commit != manifest["upstream"]["commit"]:
        raise ValueError("Upstream version differs from the frozen protocol")
    if subprocess.check_output(
        ["git", "-C", str(ROOT / "resources/RACE"), "status", "--porcelain"],
        text=True).strip():
        raise ValueError("Upstream code has unreviewed changes")
    torch.set_num_threads(2)
    hc3()
    records = read_jsonl(ROOT / "resources/experiments/hc3_aux_v1_1/test.jsonl")
    run = "tfidf_lr_v1" if args.model == "tfidf" else "roberta_s3407"
    report = ROOT / "repository/experiments/reports" / run
    original = read_jsonl(report / "hc3_aux_v1_1_predictions.jsonl")
    if [r["item_id"] for r in records] != [r["item_id"] for r in original]:
        raise ValueError("Frozen sample ordering differs")
    asset = next(r for r in manifest["files"] if r.get("run") == run)
    checkpoint = ROOT / "resources/checkpoints" / asset["asset"]
    if file_hash(checkpoint) != asset["sha256"]:
        raise ValueError("Checkpoint hash differs from the manifest")
    loading_started = time.perf_counter()
    if args.model == "tfidf":
        model = joblib.load(checkpoint)
        if list(model.classes_) != [0, 1, 2, 3]:
            raise ValueError("Unexpected classifier label order")
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(saved["config"], saved.get("metadata")).cpu().eval()
        model.load_state_dict(saved["model_state_dict"], strict=True)
    loading_seconds = time.perf_counter() - loading_started
    started = time.perf_counter()
    if args.model == "tfidf":
        probabilities = model.predict_proba([row["model_text"] for row in records])
    else:
        # TextDataset expects a training-label field; it is not used in forward.
        inference_rows = [{**r, "label_id": -1} for r in records]
        dataset = TextDataset(inference_rows, tokenizer_local())
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch)
        batches = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                batches.append(model(batch)["logits"].float().softmax(-1).numpy())
                if (index + 1) % 10 == 0:
                    print(f"CPU replay: {(index + 1) * 8}/{len(records)}", flush=True)
        probabilities = np.concatenate(batches)
    seconds = time.perf_counter() - started
    if probabilities.shape != (len(records), 4) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model outputs")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability sums")
    expected = np.array([[r["four_class_scores"][label] for label in LABELS]
                         for r in original])
    max_difference = float(np.abs(probabilities - expected).max())
    changed = int((probabilities.argmax(axis=1) != expected.argmax(axis=1)).sum())
    scores = probabilities[:, 2:].sum(axis=1).clip(0, 1)
    y = [r["true_binary_label"] for r in original]
    report_data = {
        "status": "passed" if max_difference <= 1e-4 and changed == 0 else "review_required",
        "scope": "Full frozen HC3 checkpoint replay on macOS CPU, not training, "
                 "new external validation or four-class accuracy on HC3.",
        "model": run, "n": len(records), "device": "cpu",
        "python": sys.version, "platform": platform.platform(),
        "upstream_commit": upstream_commit,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "repository"), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": file_hash(__file__),
        "checkpoint_sha256": file_hash(checkpoint),
        "frozen_data_sha256": file_hash(ROOT / "resources/experiments/hc3_aux_v1_1/test.jsonl"),
        "reference_predictions_sha256": file_hash(report / "hc3_aux_v1_1_predictions.jsonl"),
        "max_absolute_probability_difference": max_difference,
        "changed_argmax_predictions": changed,
        "load_seconds": loading_seconds, "inference_and_preprocessing_seconds": seconds,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "numpy", "scikit-learn", "joblib")},
        "binary_metrics": binary_metrics(list(zip(y, scores.tolist()))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("x") as stream:
        for row, target, p in zip(records, y, probabilities):
            stream.write(json.dumps({"item_id": row["item_id"], "true_binary_label": target,
                                     "four_class_scores": dict(zip(LABELS, p.tolist()))}) + "\n")
    with args.output.open("x") as stream:
        json.dump(report_data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report_data[k] for k in ("status", "model", "n",
                     "max_absolute_probability_difference", "changed_argmax_predictions",
                     "inference_and_preprocessing_seconds")}), flush=True)
    if report_data["status"] != "passed":
        raise SystemExit("Replay differs: inspect the preserved report")


if __name__ == "__main__":
    main()
