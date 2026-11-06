"""Final project evidence gate. Missing work cannot be marked completed."""

import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.common import DATA_ROOT, LABELS, classification_metrics, file_hash, read_jsonl, write_json
from experiments.neural import BACKBONE, PROJECT_ROOT, UPSTREAM


def verify_saved_predictions(folder, result, records):
    saved = read_jsonl(folder / "test_predictions.jsonl")
    assert [row["item_id"] for row in saved] == [row["item_id"] for row in records], "Test ID/order mismatch"
    assert [row["true_label"] for row in saved] == [row["label"] for row in records], "Test label mismatch"
    probabilities = np.asarray([[row["probabilities"][label] for label in LABELS] for row in saved])
    assert [row["prediction"] for row in saved] == [LABELS[index] for index in probabilities.argmax(1)]
    metrics = classification_metrics([row["label_id"] for row in records], probabilities)
    for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_auroc_ovr"):
        assert abs(metrics[key] - result["test"][key]) < 1e-9, f"Saved {key} mismatch"
    assert metrics["confusion_matrix"] == result["test"]["confusion_matrix"]
    for label in LABELS:
        assert metrics["per_class"][label]["support"] == result["test"]["per_class"][label]["support"]
        assert abs(metrics["per_class"][label]["f1"] - result["test"]["per_class"][label]["f1"]) < 1e-9


def verify_neural_recipe(run, result, verification, base_sha256):
    kind, seed = run.rsplit("_s", 1)
    expected = json.loads((PROJECT_ROOT / "experiments/configs/neural_v1.json").read_text())
    expected.update(model_type=kind, seed=int(seed), use_supcon=kind == "race")
    assert result["config"] == expected, "Run differs from the frozen comparison recipe"
    assert result["initialization"]["kind"] == "base pretraining only"
    assert result["initialization"]["path"] == str(BACKBONE)
    assert result["initialization"]["model_safetensors_sha256"] == base_sha256
    assert verification["samples"] == 3194 and verification["strict_checkpoint_reload"]
    assert verification["same_ids_and_order"] and verification["predictions_identical"]
    assert verification["metrics_match_saved_result"]
    snapshot = Path(result["training_source_snapshot"])
    for relative, expected_hash in result["code_sha256"].items():
        assert file_hash(snapshot / relative) == expected_hash, f"Training source snapshot changed: {relative}"
    events = read_jsonl(Path(result["checkpoint_path"]).parent / "events.jsonl")
    initialized = [event for event in events if event["event"] == "initialized"]
    assert initialized and all(event["train_records"] == 11190 and event["val_records"] == 1600 for event in initialized)
    epochs = [event for event in events if event["event"] == "epoch_complete"]
    assert epochs and len({event["epoch"] for event in epochs}) == len(epochs)
    selected = max(epochs, key=lambda event: event["validation"]["macro_f1"])
    assert selected["epoch"] == result["best_epoch"], "Selected checkpoint is not validation best"
    assert selected["validation"] == result["validation"]
    assert epochs[-1]["epoch"] <= expected["max_epochs"]
    assert epochs[-1]["epoch"] == expected["max_epochs"] or epochs[-1]["stale_epochs"] >= expected["early_stopping_patience"]
    assert len([event for event in events if event["event"] == "completed"]) == 1
    assert result["upstream_commit"] == "61ce528430a8bb84fd1bfb3285ee34827182ef59"
    return {"initialization": "same base pretrained weights", "recipe": "exact fixed comparison configuration",
            "selection": "maximum recorded validation Macro-F1; held-out test not a selection input",
            "completed_epochs": epochs[-1]["epoch"], "selected_epoch": selected["epoch"],
            "source_snapshot_hashes_match": True, "full_test_reload_samples": verification["samples"]}


def main():
    reports = PROJECT_ROOT / "experiments/reports"
    missing, checks = [], {}
    trained_models = {}
    protocol = json.loads((DATA_ROOT / "protocol.json").read_text())
    for split in ("train", "val", "test"):
        if file_hash(DATA_ROOT / f"{split}.jsonl") != protocol["partitions"][split]["sha256"]:
            missing.append(f"Frozen {split} data changed")
    auxiliary_root = DATA_ROOT.parent / "hc3_aux_v1_1"
    auxiliary_protocol = json.loads((auxiliary_root / "protocol.json").read_text())
    if file_hash(auxiliary_root / "test.jsonl") != auxiliary_protocol["answer_sha256"]:
        missing.append("Frozen HC3 auxiliary data changed")
    test_records = read_jsonl(DATA_ROOT / "test.jsonl")
    auxiliary_records = read_jsonl(auxiliary_root / "test.jsonl")
    base_sha256 = file_hash(BACKBONE / "model.safetensors")
    runs = ["tfidf_lr_v1", "race_no_supcon_s3407"] + [f"{kind}_s{seed}" for kind in ("roberta", "race") for seed in (3407, 42, 2026)]
    for run in runs:
        folder = reports / run
        try:
            result = json.loads((folder / "result.json").read_text())
            verification = json.loads((folder / "verification.json").read_text())
            assert result["status"] == "completed" and verification["status"] == "passed"
            assert result["data_protocol_sha256"] == file_hash(DATA_ROOT / "protocol.json")
            if run.startswith("tfidf"):
                assert file_hash(result["model_path"]) == result["model_sha256"]
                assert verification["reloaded_model_prediction_count"] == 3194
                assert verification["cross_split_id_group_normalized_text_and_block_overlap"] == 0
                assert verification["metrics_match_saved_results"]
            else:
                assert file_hash(result["checkpoint_path"]) == result["checkpoint_sha256"] == verification["checkpoint_sha256"]
                checks[f"{run}_recipe_and_selection"] = verify_neural_recipe(run, result, verification, base_sha256)
            verify_saved_predictions(folder, result, test_records)
            trained_models[run] = result
            checks[run] = "completed and reload-verified; frozen protocol and current checkpoint hash match"
        except (FileNotFoundError, KeyError, AssertionError) as error:
            missing.append(f"{run}: {type(error).__name__}: {error}")
    for run in ("tfidf_lr_v1", "roberta_s3407", "race_s3407", "race_no_supcon_s3407"):
        try:
            auxiliary = json.loads((reports / run / "hc3_aux_v1_1_result.json").read_text())
            assert auxiliary["status"] == "completed" and auxiliary["protocol_id"] == "hc3_aux_v1_1"
            assert auxiliary["metrics"]["samples"] == 1000 and not auxiliary["fit_or_tuning_on_hc3"]
            assert auxiliary["protocol_sha256"] == file_hash(auxiliary_root / "protocol.json")
            trained = trained_models[run]
            assert auxiliary["detector_sha256"] == trained.get("checkpoint_sha256", trained.get("model_sha256"))
            saved = read_jsonl(reports / run / "hc3_aux_v1_1_predictions.jsonl")
            assert [row["item_id"] for row in saved] == [row["item_id"] for row in auxiliary_records]
            assert [row["true_binary_label"] for row in saved] == [row["binary_label"] for row in auxiliary_records]
            scores = np.asarray([[row["four_class_scores"][label] for label in LABELS] for row in saved])
            assert scores.shape == (1000, 4) and np.isfinite(scores).all() and np.allclose(scores.sum(1), 1)
            binary_scores = scores[:, 2] + scores[:, 3]
            assert np.allclose(binary_scores, [row["ai_creator_score"] for row in saved])
            assert np.array_equal(binary_scores >= .5, [row["binary_prediction"] for row in saved])
            from experiments.evaluate_external import binary_metrics
            recomputed = binary_metrics([row["binary_label"] for row in auxiliary_records], binary_scores)
            assert all(abs(recomputed[key] - auxiliary["metrics"][key]) < 1e-9 for key in ("accuracy", "macro_f1", "auroc"))
            if run.startswith("race"):
                assert auxiliary["timing_details"]["parsed_input_sha256"] == file_hash(auxiliary_root / "test_parsed_aligned.jsonl")
            checks[f"{run}_hc3"] = "frozen binary auxiliary evaluation completed"
        except (FileNotFoundError, KeyError, AssertionError) as error:
            missing.append(f"{run} HC3: {type(error).__name__}")
    for name, status in (("comparison/summary.json", "completed"), ("demo_verification_all.json", "passed"),
                         ("demo_ui_verification.json", "passed"), ("graph_cache_verification.json", "passed"),
                         ("released_rst_span_audit.json", "audited_without_data_changes"),
                         ("parser_assets.json", "passed"),
                         ("retraining_entrypoint.json", "passed"),
                         ("auxiliary/summary.json", "completed"), ("comparison/case_studies.json", "completed"),
                         ("hc3_rst_aligned_span_audit.json", "passed")):
        try:
            record = json.loads((reports / name).read_text())
            assert record["status"] == status
            if name == "demo_ui_verification.json":
                assert Path(record["screenshot_path"]).is_file()
                assert file_hash(record["screenshot_path"]) == record["screenshot_sha256"]
                assert set(record["models_selected_and_run"]) == {"tfidf", "roberta", "race", "race_no_supcon"}
                assert record["all_four_choices_available"]
            if name == "parser_assets.json":
                assert all(file_hash(item["path"]) == item["sha256"] for item in record["files"])
            if name == "retraining_entrypoint.json":
                trainer = PROJECT_ROOT / "experiments/train_neural.py"
                assert record["training_entrypoint_sha256"] == file_hash(trainer)
                assert record["invocation_cwd"] == "/tmp" and record["existing_smoke_files_preserved"]
                assert set(record["smokes"]) == {"roberta", "race"}
                for kind, item in record["smokes"].items():
                    smoke_path = Path(item["smoke_path"])
                    assert file_hash(smoke_path) == item["smoke_sha256"]
                    smoke = json.loads(smoke_path.read_text())
                    assert smoke["status"] == "passed" and smoke["training_only_smoke"] and not smoke["test_partition_loaded"]
                    assert smoke["finite_gradients"] and smoke["frozen_layer_unchanged"] and smoke["classifier_updated"]
                    assert smoke["config"]["model_type"] == kind and smoke["config"]["seed"] == 3407
                    assert item["train_records"] == 32 and item["val_records"] == 8
                    assert file_hash(smoke_path.parent / "source/experiments/train_neural.py") == file_hash(trainer)
                assert record["guards"]["invalid_name"]["exit_code"] == 2
                assert record["guards"]["existing_name"]["exit_code"] == 1
                assert all(file_hash(item["log_path"]) == item["log_sha256"] for item in record["guards"].values())
                assert all(trained_models[run]["checkpoint_sha256"] == digest for run, digest in record["canonical_checkpoint_hashes_unchanged"].items())
            if name == "auxiliary/summary.json":
                assert record["hart_reference"]["samples"] == 3194
                assert record["hart_reference"]["test_file_sha256"] == file_hash(DATA_ROOT / "test.jsonl")
                assert all(item["hart_aligned_binary_reference"]["samples"] == 3194 for item in record["analyses"].values())
            if name == "demo_verification_all.json":
                assert set(record["models"]) == {"tfidf", "roberta", "race", "race_no_supcon"}
                assert record["empty_request_rejected"]
                assert all(record["checks"][model]["raw_text_parser_to_graph_to_checkpoint_verified"] for model in ("race", "race_no_supcon"))
            checks[name] = status
        except (FileNotFoundError, KeyError, AssertionError) as error:
            missing.append(f"{name}: {type(error).__name__}")
    for name in ("README.md", "SOURCES.md", "VIDEO_OUTLINE.md", "METHOD_NOTES.md", "requirements-core.txt", "requirements-rst.txt", "run_review.sh",
                 "reports/runtime_training.json", "reports/runtime_rst.json", "reports/comparison/REPORT.md",
                 "reports/comparison/main_metrics.png", "reports/comparison/confusion_matrices.png", "reports/comparison/domain_metrics.png",
                 "reports/comparison/learning_curves.png", "reports/auxiliary/aligned_binary_metrics.png"):
        path = PROJECT_ROOT / "experiments" / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    clean = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip() == ""
    upstream_commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if upstream_commit != "61ce528430a8bb84fd1bfb3285ee34827182ef59":
        missing.append("Upstream source revision changed")
    if not clean:
        missing.append("Tracked or untracked upstream snapshot changes need review")
    checks["upstream_tree_clean"] = clean
    result = {"status": "passed" if not missing else "incomplete", "checks": checks, "missing_or_invalid": missing,
              "scope": "Local implementation/evaluation/Demo/documentation gate. Human narration, signatures and course submission remain user tasks.",
              "verification_script_sha256": file_hash(__file__)}
    write_json(reports / "project_verification.json", result)
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
