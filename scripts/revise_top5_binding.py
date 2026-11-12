import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.campaign12h import load_b0
from experiments.common import classification_metrics
from experiments.neural import load_records, tokenizer_local


ROOT = Path(r"D:\6405_prj\ee6405-nlp-project")
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"
OUT = CAMPAIGN / "top5_revision_01"
LABELS = ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    return sha256_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def predict(model, records, tokenizer):
    rows = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            part = records[start:start + 16]
            encoded = tokenizer(
                [record["article"] for record in part],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            output = model({
                "input_ids": encoded["input_ids"].cuda(),
                "attention_mask": encoded["attention_mask"].cuda(),
            })
            rows.extend(output["logits"].float().cpu().numpy())
    return torch.softmax(torch.tensor(np.asarray(rows)), dim=1).numpy()


def mc_dropout(model, records, tokenizer):
    model.train()
    torch.manual_seed(20261107)
    values = [predict(model, records, tokenizer) for _ in range(5)]
    model.eval()
    return np.mean(values, axis=0)


def multi_window(model, records, tokenizer):
    values = []
    model.eval()
    with torch.no_grad():
        for record in records:
            ids = tokenizer(
                record["article"], max_length=512, truncation=False,
                padding=False, return_tensors="pt"
            )["input_ids"][0]
            if len(ids) <= 512:
                windows = [ids]
            else:
                middle = max(0, (len(ids) - 512) // 2)
                windows = [ids[:512], ids[middle:middle + 512], ids[-512:]]
            probabilities = []
            for window in windows:
                logits = model({
                    "input_ids": window.unsqueeze(0).cuda(),
                    "attention_mask": torch.ones((1, len(window)), device="cuda"),
                })["logits"]
                probabilities.append(logits.float().softmax(1).cpu().numpy()[0])
            values.append(np.mean(probabilities, axis=0))
    return np.asarray(values)


def metrics(labels, probabilities):
    return classification_metrics(labels, probabilities)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CAMPAIGN / "protocol/split_manifest.json").read_text(encoding="utf-8-sig"))
    val = load_records("val")
    val_by_id = {record["item_id"]: record for record in val}
    ordered_ids = [row["item_id"] for row in manifest["records"] if row["partition"] == "S"]
    records = [val_by_id[item_id] for item_id in ordered_ids]
    assert len(records) == 960 and [record["item_id"] for record in records] == ordered_ids
    labels = np.asarray([record["label_id"] for record in records])
    model, _, _ = load_b0()
    tokenizer = tokenizer_local()
    b0_sha = sha256_bytes((ROOT / "experiments/glpc_runs/glpc_b0_s3407/best.pt").read_bytes())
    source_lock = json.loads((CAMPAIGN / "sources_lock.json").read_text(encoding="utf-8-sig"))
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    probabilities = {
        "E01": predict(model, records, tokenizer),
    }
    probabilities["E25"] = mc_dropout(model, records, tokenizer)
    probabilities["E26"] = multi_window(model, records, tokenizer)
    train = load_records("train")
    prior = np.bincount([record["label_id"] for record in train], minlength=4).astype(float)
    prior /= prior.sum()
    probabilities["E02"] = torch.softmax(
        torch.log(torch.tensor(probabilities["E01"]).clamp_min(1e-12))
        + 0.5 * torch.log(torch.tensor(prior)),
        dim=1,
    ).numpy()
    e03 = json.loads((CAMPAIGN / "tasks/E03/selection.json").read_text(encoding="utf-8-sig"))
    params = np.asarray(e03["config"]["parameters"])
    scale = np.exp(np.clip(params[:4], -0.7, 0.7))
    bias = params[4:] - params[4:].mean()
    probabilities["E03"] = torch.softmax(
        torch.tensor(np.log(np.clip(probabilities["E01"], 1e-12, 1)) * scale + bias),
        dim=1,
    ).numpy()
    probabilities["E04"] = probabilities["E01"].copy()

    candidates = []
    for task_id, value in probabilities.items():
        prediction_rows = [
            {"item_id": record["item_id"], "label_id": int(record["label_id"]),
             "probabilities": [float(x) for x in row]}
            for record, row in zip(records, value)
        ]
        binding = {
            "task_id": task_id,
            "ordered_item_ids": ordered_ids,
            "item_id_sha256": sha256_json(ordered_ids),
            "label_order": LABELS,
            "data_version": {
                "train_hash": source_lock["train_hash"],
                "val_hash": source_lock["val_hash"],
                "split_manifest_sha256": sha256_bytes(
                    (CAMPAIGN / "protocol/split_manifest.json").read_bytes()
                ),
                "partition": "S",
            },
            "model_version": {
                "base_checkpoint_sha256": b0_sha,
                "git_head": git_head,
                "tokenizer": str(tokenizer.name_or_path),
            },
            "prediction_rows": prediction_rows,
        }
        (OUT / f"{task_id}_prediction_with_ids.json").write_text(
            json.dumps(binding, ensure_ascii=False), encoding="utf-8"
        )
        current = metrics(labels, value)
        base_prediction = probabilities["E01"].argmax(1)
        prediction = value.argmax(1)
        candidates.append({
            "task_id": task_id,
            "status": "COMPLETE",
            "score_kind": "S_dev_selected",
            "samples": len(records),
            "accuracy": current["accuracy"],
            "macro_f1": current["macro_f1"],
            "J": (current["accuracy"] + current["macro_f1"]) / 2,
            "per_class_f1": {
                name: current["per_class"][name]["f1"] for name in LABELS
            },
            "wrong_to_right_vs_E01": int(
                ((prediction == labels) & (base_prediction != labels)).sum()
            ),
            "right_to_wrong_vs_E01": int(
                ((prediction != labels) & (base_prediction == labels)).sum()
            ),
            "binding_file": f"{task_id}_prediction_with_ids.json",
            "binding_file_sha256": sha256_bytes(
                (OUT / f"{task_id}_prediction_with_ids.json").read_bytes()
            ),
            "base_checkpoint_sha256": b0_sha,
        })

    report = {
        "revision": "top5_revision_01",
        "scope": "S_dev_selected",
        "sample_count": len(records),
        "ordered_item_id_sha256": sha256_json(ordered_ids),
        "label_order": LABELS,
        "data_version": {
            "train_hash": source_lock["train_hash"],
            "val_hash": source_lock["val_hash"],
            "split_manifest_sha256": sha256_bytes(
                (CAMPAIGN / "protocol/split_manifest.json").read_bytes()
            ),
        },
        "model_version": {"base_checkpoint_sha256": b0_sha, "git_head": git_head},
        "id_binding_status": "checkpoint_reinference_verified",
        "probability_recalculation_status": "recomputed",
        "s_oof": "not_available",
        "s_refit": "not_available",
        "s_dev_selected": candidates,
        "excluded": {
            "E05": "probability file has no generation-time ID binding and requires original sparse fit state",
            "E06": "probability file has no generation-time ID binding and requires original NB/SVM fit state",
            "E07": "probability file has no generation-time ID binding and requires original style fit state",
            "E08": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E09": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E10": "implementation/resource blocked; no candidate checkpoint",
            "E11": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E12": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E13": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E14": "checkpoint exists but no prior full S prediction file; snapshot listed separately",
            "E15": "resource blocked",
            "E16": "resource blocked in this round; no formal run",
            "E20": "no checkpoint or generation-time ID binding",
            "E21": "no checkpoint or generation-time ID binding",
            "E30": "depends on E05/E25 and original fusion output lacks generation-time ID binding",
            "E31": "depends on E05/E25 and original fusion output lacks generation-time ID binding",
        },
        "test_partition_loaded": False,
        "hc3_loaded": False,
    }
    (OUT / "top5_recomputed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verified_candidates": len(candidates), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
