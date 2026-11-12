"""Evaluate a completed detector on the frozen HC3 binary auxiliary protocol."""

import argparse
import json
from pathlib import Path

from experiments.paths import resolve_recorded_path
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from experiments.common import DATA_ROOT, LABELS, file_hash, read_jsonl, write_json, write_jsonl
from experiments.neural import PROJECT_ROOT


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= .5
    fpr, tpr, _ = roc_curve(labels, scores)
    return {"samples": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "ai_precision": float(precision_score(labels, predictions, zero_division=0)),
            "ai_recall": float(recall_score(labels, predictions, zero_division=0)),
            "human_false_positive_rate_at_0_5": float(predictions[labels == 0].mean()),
            "auroc": float(roc_auc_score(labels, scores)),
            "tpr_at_1pct_fpr_roc": float(tpr[fpr <= .01].max()),
            "confusion_matrix_human_ai": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tfidf_lr_v1")
    args = parser.parse_args()
    source = DATA_ROOT.parent / "hc3_aux_v1_1"
    protocol = json.loads((source / "protocol.json").read_text())
    assert file_hash(source / "test.jsonl") == protocol["answer_sha256"]
    records = read_jsonl(source / "test.jsonl")
    report_dir = PROJECT_ROOT / "experiments/reports" / args.run
    trained = json.loads((report_dir / "result.json").read_text())
    assert trained["status"] == "completed"
    timings = {}
    started = time.perf_counter()
    if args.run.startswith("tfidf"):
        assert file_hash(trained["model_path"]) == trained["model_sha256"]
        model = joblib.load(resolve_recorded_path(trained["model_path"]))
        with threadpool_limits(limits=4):
            probabilities = model.predict_proba([r["model_text"] for r in records])
        detector_hash = trained["model_sha256"]
    else:
        import torch
        from experiments.neural import TextDataset, build_model, tokenizer_local
        from experiments.train_neural import evaluate
        assert (report_dir / "verification.json").exists(), "Reload verification must pass before auxiliary evaluation"
        torch.set_num_threads(4)
        assert file_hash(trained["checkpoint_path"]) == trained["checkpoint_sha256"]
        checkpoint = torch.load(resolve_recorded_path(trained["checkpoint_path"]), map_location="cpu", weights_only=True)
        data_started = time.perf_counter()
        if trained["config"]["model_type"] == "roberta":
            dataset = TextDataset(records, tokenizer_local())
        else:
            from experiments.raw_graph import RawGraphDataset
            parsed_path = source / "test_parsed_aligned.jsonl"
            audit = json.loads(parsed_path.with_suffix(".audit.json").read_text())
            assert audit["status"] == "passed" and audit["records"] == len(records)
            assert "character_offset_adapter" in audit
            assert file_hash(parsed_path) == audit["output_sha256"]
            parsed = read_jsonl(parsed_path)
            assert len(parsed) == len(records)
            for row, graph in zip(records, parsed):
                assert all(graph[key] == value for key, value in row.items()), "Parsed data changed text, labels, IDs or order"
            dataset = RawGraphDataset(parsed, tokenizer_local())
            timings["rst_parse_model_load_seconds"] = audit["model_load_seconds"]
            timings["rst_parse_total_seconds"] = sum(audit["parse_seconds"])
            timings["rst_parse_device"] = "cpu" if audit["cuda_device"] < 0 else "cuda"
            timings["parsed_input_sha256"] = audit["output_sha256"]
        timings["tokenization_graph_preprocessing_seconds"] = time.perf_counter() - data_started
        model = build_model(trained["config"], checkpoint.get("metadata")).cuda().eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        _, probabilities, ids, forward_seconds = evaluate(model, dataset, trained["config"]["batch_size"])
        timings["detector_forward_seconds"] = forward_seconds
        assert ids == [row["item_id"] for row in records]
        detector_hash = trained["checkpoint_sha256"]
    assert probabilities.shape == (len(records), 4) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1)
    # This score and threshold were fixed in the protocol before evaluation.
    scores = probabilities[:, 2] + probabilities[:, 3]
    y = np.asarray([row["binary_label"] for row in records])
    metrics = binary_metrics(y, scores)
    per_source = {}
    for name in sorted({row["domain"] for row in records}):
        mask = np.asarray([row["domain"] == name for row in records])
        per_source[name] = binary_metrics(y[mask], scores[mask])
    result = {"status": "completed", "detector_run": args.run, "detector_sha256": detector_hash,
              "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(source / "protocol.json"),
              "score": "P(ai_generated)+P(ai_humanized)", "threshold": .5,
              "fit_or_tuning_on_hc3": False, "metrics": metrics, "per_source": per_source,
              "evaluation_seconds_excluding_separate_parser_worker": time.perf_counter() - started,
              "timing_details": timings,
              "evidence_boundary": protocol["evidence_boundary"], "script_sha256": file_hash(__file__)}
    write_json(report_dir / "hc3_aux_v1_1_result.json", result)
    write_jsonl(report_dir / "hc3_aux_v1_1_predictions.jsonl", [
        {"item_id": row["item_id"], "question_group": row["group_id"], "source": row["domain"],
         "true_binary_label": row["binary_label"], "ai_creator_score": float(score),
         "binary_prediction": int(score >= .5), "four_class_scores": dict(zip(LABELS, probs.tolist()))}
        for row, score, probs in zip(records, scores, probabilities)])
    print(json.dumps({"detector": args.run, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
