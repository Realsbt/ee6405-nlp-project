"""Package one private, validation-only paired encoder experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "experiments/common.py", "experiments/paths.py", "experiments/neural.py",
    "experiments/train_encoder_pilot.py", "cloud/kaggle/run_encoder_pilot.py",
    "cloud/kaggle/run_same_window_smoke.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, cache):
    if output.exists():
        raise FileExistsError("Preserve existing frozen pilot package")
    protocol = json.loads((cache / "protocol.json").read_text())
    if digest(cache / "inputs.jsonl") != protocol["inputs_sha256"]:
        raise ValueError("Original visible-input inventory changed")
    dataset, kernel = output / "dataset", output / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir()
    partition_hashes = {}
    groups = {"train": set(), "val": set()}
    handles = {split: (dataset / f"pilot_{split}.jsonl").open("w")
               for split in ("train", "val")}
    counts = {"train": 0, "val": 0}
    try:
        with (cache / "inputs.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                split = row["source_split"]
                if split not in handles:
                    continue
                handles[split].write(json.dumps(row) + "\n")
                counts[split] += 1
                groups[split].add(row["split_block_id"])
    finally:
        for handle in handles.values():
            handle.close()
    if groups["train"] & groups["val"]:
        raise ValueError("Train/validation source overlap")
    if counts != {"train": 11190, "val": 1600}:
        raise ValueError(f"Incomplete training/validation inputs: {counts}")
    for split in counts:
        partition_hashes[split] = {
            "sha256": digest(dataset / f"pilot_{split}.jsonl"), "count": counts[split],
        }
    plan = {
        "recipe": "all_layers_cls_ce_v1", "seed": 3407, "epochs": 4,
        "conditions": ["deberta_full", "roberta_full"],
        "effective_batch": 16, "micro_batch": 4, "max_seconds_per_arm": 14400,
        "learning_rate": 2e-5, "test_evaluated": False,
        "amp_initial_scale": 128.0, "smoke_max_overflow_attempts": 12,
        "selection": "Validation Macro-F1 only; retain all pilot results; no automatic final test",
        "expansion": "No extra seeds, models, learning-rate sweep or paid compute automatically",
        "input_note": "Same audited visible character prefix; each tokenizer capped at 512 tokens",
        "historical_comparison_note": "All-layer training differs from earlier one-layer fusion",
        "deberta_revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "source_inputs_sha256": protocol["inputs_sha256"],
        "partitions": partition_hashes,
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dataset / "pilot_protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    source_hashes = {name: digest(ROOT / name) for name in SOURCES}
    with zipfile.ZipFile(dataset / "encoder_pilot_code_v1.zip", "w",
                         zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    (dataset / "encoder_pilot_manifest.json").write_text(json.dumps({
        "code_files": source_hashes, "protocol_sha256": digest(dataset / "pilot_protocol.json"),
        "private_data": True, "contains_test_partition": False,
    }, indent=2) + "\n")
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "EE6405 Paired Encoder Pilot v1",
        "id": "zhengyingqi/ee6405-encoder-pilot-v1",
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    shutil.copy2(ROOT / "cloud/kaggle/run_encoder_pilot.py", kernel / "run_encoder_pilot.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": "zhengyingqi/ee6405-paired-encoder-pilot-v1",
        "title": "EE6405 Paired Encoder Pilot v1",
        "code_file": "run_encoder_pilot.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "zhengyingqi/ee6405-same-window-gpu-v1",
            "zhengyingqi/ee6405-encoder-pilot-v1",
        ],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.cache), indent=2))
