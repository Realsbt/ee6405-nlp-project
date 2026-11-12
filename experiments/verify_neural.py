"""Reload a saved neural checkpoint and verify its complete test predictions."""

import argparse
import json
from pathlib import Path

from experiments.paths import resolve_recorded_path

import numpy as np
import torch

from experiments.common import DATA_ROOT, file_hash, read_jsonl, write_json
from experiments.neural import PROJECT_ROOT, RUN_ROOT, TextDataset, build_model, load_records, tokenizer_local
from experiments.train_neural import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", type=Path, help="Write the new verification separately from archived evidence")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = RUN_ROOT / args.run
    report_dir = PROJECT_ROOT / "experiments/reports" / args.run
    result = json.loads((report_dir / "result.json").read_text())
    assert result["status"] == "completed"
    assert file_hash(result["checkpoint_path"]) == result["checkpoint_sha256"]
    assert file_hash(DATA_ROOT / "protocol.json") == result["data_protocol_sha256"]
    records = load_records("test")
    if result["config"]["model_type"] == "roberta":
        data = TextDataset(records, tokenizer_local())
        metadata = None
    else:
        from experiments.graph_data import GraphDataset
        data = GraphDataset("test", records)
        metadata = data.metadata
    model = build_model(result["config"], metadata).cuda().eval()
    checkpoint = torch.load(resolve_recorded_path(result["checkpoint_path"]), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics, probabilities, ids, seconds = evaluate(model, data, result["config"]["batch_size"])
    stored = read_jsonl(report_dir / "test_predictions.jsonl")
    assert ids == [row["item_id"] for row in stored]
    expected = np.asarray([[row["probabilities"][name] for name in metrics["label_order"]] for row in stored])
    max_difference = float(np.abs(probabilities - expected).max())
    assert np.allclose(probabilities, expected, atol=1e-5, rtol=1e-5)
    assert np.array_equal(probabilities.argmax(1), expected.argmax(1))
    for key in ("accuracy", "macro_f1", "macro_auroc_ovr"):
        assert abs(metrics[key] - result["test"][key]) < 1e-8
    report = {"status": "passed", "strict_checkpoint_reload": True, "samples": len(ids),
              "same_ids_and_order": True, "predictions_identical": True,
              "max_probability_difference": max_difference, "metrics_match_saved_result": True,
              "evaluation_seconds": seconds, "checkpoint_sha256": result["checkpoint_sha256"]}
    write_json(args.report or report_dir / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
