"""Finalize the ID-bound S ranking without training or test access."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.common import classification_metrics
from experiments.neural import load_records


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"
OUT = CAMPAIGN / "top5_revision_01"
LABELS = ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"]


def load_binding(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if "prediction_rows" in value:
        ids = [row["item_id"] for row in value["prediction_rows"]]
        probs = np.asarray([row["probabilities"] for row in value["prediction_rows"]])
        labels = [row["label_id"] for row in value["prediction_rows"]]
    else:
        ids = value["ordered_item_ids"]
        probs = np.asarray(value["probabilities"], dtype=float)
        labels = value["label_ids"]
    return value, ids, probs, np.asarray(labels)


def main():
    manifest = json.loads(
        (CAMPAIGN / "protocol/split_manifest.json").read_text(encoding="utf-8-sig")
    )
    expected_ids = [row["item_id"] for row in manifest["records"] if row["partition"] == "S"]
    rows = {row["item_id"]: row for row in load_records("val")}
    expected_labels = np.asarray([rows[item_id]["label_id"] for item_id in expected_ids])
    candidates = []
    excluded = {}
    seen = {}
    for path in sorted(OUT.rglob("*_prediction_with_ids.json")):
        task_id = path.name.split("_", 1)[0]
        value, ids, probs, labels = load_binding(path)
        if ids != expected_ids or not np.array_equal(labels, expected_labels):
            excluded[task_id] = "prediction binding does not match ordered S manifest"
            continue
        if probs.shape != (len(expected_ids), 4) or not np.isfinite(probs).all():
            excluded[task_id] = "invalid probability shape or finite values"
            continue
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            excluded[task_id] = "probabilities are not normalized"
            continue
        fingerprint = hashlib.sha256(probs.tobytes()).hexdigest()
        if fingerprint in seen:
            excluded[task_id] = f"duplicate probability alias of {seen[fingerprint]}"
            continue
        seen[fingerprint] = task_id
        metrics = classification_metrics(expected_labels, probs)
        prediction = probs.argmax(axis=1)
        candidates.append({
            "task_id": task_id,
            "status": "COMPLETE",
            "score_kind": "S_dev_selected",
            "samples": len(expected_ids),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "J": (metrics["accuracy"] + metrics["macro_f1"]) / 2,
            "per_class_f1": {name: metrics["per_class"][name]["f1"] for name in LABELS},
            "binding_file": path.name,
            "binding_path": str(path),
            "binding_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "wrong_to_right_vs_E01": None,
            "right_to_wrong_vs_E01": None,
        })
    base = next((item for item in candidates if item["task_id"] == "E01"), None)
    if base:
        _, _, base_probs, _ = load_binding(Path(base["binding_path"]))
        base_prediction = base_probs.argmax(axis=1)
        for item in candidates:
            _, _, probs, _ = load_binding(Path(item["binding_path"]))
            prediction = probs.argmax(axis=1)
            item["wrong_to_right_vs_E01"] = int(
                ((prediction == expected_labels) & (base_prediction != expected_labels)).sum()
            )
            item["right_to_wrong_vs_E01"] = int(
                ((prediction != expected_labels) & (base_prediction == expected_labels)).sum()
            )
    source_lock = json.loads(
        (CAMPAIGN / "sources_lock.json").read_text(encoding="utf-8-sig")
    )
    report = {
        "revision": "top5_revision_01",
        "scope": "S_dev_selected",
        "sample_count": len(expected_ids),
        "ordered_item_id_sha256": hashlib.sha256(
            "\n".join(expected_ids).encode("utf-8")
        ).hexdigest(),
        "label_order": LABELS,
        "data_version": {
            "train_hash": source_lock["train_hash"],
            "val_hash": source_lock["val_hash"],
            "split_manifest_sha256": hashlib.sha256(
                (CAMPAIGN / "protocol/split_manifest.json").read_bytes()
            ).hexdigest(),
        },
        "id_binding_status": "checkpoint_reinference_verified",
        "probability_recalculation_status": "recomputed",
        "s_oof": "not_available",
        "s_refit": "not_available",
        "s_dev_selected": sorted(candidates, key=lambda x: (-x["J"], -x["macro_f1"], x["task_id"])),
        "excluded": excluded,
        "test_partition_loaded": False,
        "hc3_loaded": False,
    }
    for item in report["s_dev_selected"]:
        item.pop("binding_path", None)
    (OUT / "top5_recomputed.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "verified_candidates": [item["task_id"] for item in report["s_dev_selected"]],
        "excluded": report["excluded"],
    }, indent=2))


if __name__ == "__main__":
    main()
