"""Recheck split isolation and persisted baseline predictions without refitting."""

import argparse
from itertools import combinations
import json
from pathlib import Path

from experiments.paths import resolve_recorded_path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.common import DATA_ROOT, classification_metrics, file_hash, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/tfidf_lr_v1"))
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--report", type=Path, help="Write the new verification separately from archived evidence")
    args = parser.parse_args()
    result = json.loads((args.output / "result.json").read_text())
    assert file_hash(args.data / "protocol.json") == result["data_protocol_sha256"]
    partitions = {name: read_jsonl(args.data / f"{name}.jsonl") for name in ("train", "val", "test")}
    for name in partitions:
        assert file_hash(args.data / f"{name}.jsonl") == result["data_file_sha256"][name]
    for a, b in combinations(partitions, 2):
        for key in ("item_id", "group_id", "normalized_text_sha256", "split_block_id"):
            assert not ({r[key] for r in partitions[a]} & {r[key] for r in partitions[b]})
    assert file_hash(result["model_path"]) == result["model_sha256"]
    model = joblib.load(resolve_recorded_path(result["model_path"]))
    rows = partitions["test"]
    with threadpool_limits(limits=4):
        probs = model.predict_proba([r["model_text"] for r in rows])
    stored = read_jsonl(args.output / "test_predictions.jsonl")
    assert [r["item_id"] for r in rows] == [r["item_id"] for r in stored]
    expected = np.asarray([[r["probabilities"][name] for name in result["test"]["label_order"]] for r in stored])
    assert np.allclose(probs, expected, atol=1e-12, rtol=1e-12)
    metrics = classification_metrics([r["label_id"] for r in rows], probs)
    for key in ("accuracy", "macro_f1", "macro_auroc_ovr", "macro_tpr_at_1pct_fpr_roc"):
        assert abs(metrics[key] - result["test"][key]) < 1e-12
    checks = {
        "status": "passed", "retained_records": sum(map(len, partitions.values())),
        "cross_split_id_group_normalized_text_and_block_overlap": 0,
        "reloaded_model_prediction_count": len(rows), "max_probability_difference": float(abs(probs - expected).max()),
        "metrics_match_saved_results": True, "feature_count": len(model.named_steps["tfidf"].vocabulary_),
        "input_field": "model_text only; IDs, labels, domain and model_name excluded from features",
    }
    write_json(args.report or args.output / "verification.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
