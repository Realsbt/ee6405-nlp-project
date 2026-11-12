"""Replay RACE on released HART graphs or restored HC3 parses; calibrate on val only."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from replay_hc3_cpu import (
    ROOT, LABELS, build_model, collate_batch, file_hash, np, read_jsonl,
    subprocess, tokenizer_local, torch, DataLoader,
)
from experiments.common import classification_metrics
from experiments.raw_graph import RawGraphDataset
from evaluation_metrics import evaluate_creator, negative_only_threshold


def temperature_probabilities(logits, temperature):
    logits = np.asarray(logits, dtype=float)
    if (logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Need finite four-class logits and positive temperature")
    return softmax(logits / temperature, axis=1)


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    temperature_probabilities(logits, 1)
    if labels.shape != (len(logits),) or not len(labels) or not np.isin(labels, range(4)).all():
        raise ValueError("Need aligned four-class validation labels")
    def loss(log_t):
        values = logits / np.exp(log_t)
        return float(np.mean(logsumexp(values, axis=1) - values[np.arange(len(labels)), labels]))
    fit = minimize_scalar(loss, bounds=(np.log(.05), np.log(20)), method="bounded",
                          options={"xatol": 1e-8})
    if not fit.success:
        raise RuntimeError("Temperature optimization failed")
    log_t = fit.x if fit.fun <= loss(0) else 0.
    return {"temperature": float(np.exp(log_t)), "validation_nll_before": loss(0),
            "validation_nll_after": loss(log_t), "bounds": [.05, 20],
            "near_bound": bool(abs(log_t - np.log(.05)) < 1e-4 or abs(log_t - np.log(20)) < 1e-4)}


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def infer(model, records, output):
    started = time.perf_counter()
    dataset = RawGraphDataset(records, tokenizer_local())
    batches = []
    with torch.inference_mode(), output.open("x") as stream:
        for index, batch in enumerate(DataLoader(dataset, batch_size=8, collate_fn=collate_batch)):
            logits = model(batch)["logits"].float().numpy()
            batches.append(logits)
            for item_id, z in zip(batch["id"], logits):
                stream.write(json.dumps({"item_id": item_id, "logits": z.tolist(),
                                         "probabilities": dict(zip(LABELS, softmax(z.astype(float)).tolist()))}) + "\n")
            stream.flush()
            if (index + 1) % 25 == 0:
                print(f"{output.stem}: {min(8*(index+1),len(records))}/{len(records)}", flush=True)
    return np.concatenate(batches), time.perf_counter() - started


def evaluate(logits, labels, calibration, four_class=True):
    y = np.asarray(labels)
    creator = y >= 2 if four_class else y
    result = {}
    for name, t in (("uncalibrated", 1.), ("temperature_scaled", calibration["temperature"])):
        p = temperature_probabilities(logits, t)
        entry = {"creator": evaluate_creator(creator, p[:, 2:].sum(1),
                                             calibration[name + "_threshold"]["threshold"])}
        if four_class:
            entry["four_class"] = classification_metrics(y.tolist(), p)
            entry["nll"] = float(np.mean(logsumexp(logits / t, axis=1) - (logits / t)[np.arange(len(y)), y]))
        result[name] = entry
    return result


def compare(logits, reference, key):
    expected = np.array([[r[key][label] for label in LABELS] for r in reference])
    p = temperature_probabilities(logits, 1)
    difference = float(np.abs(p - expected).max())
    changed = int((p.argmax(1) != expected.argmax(1)).sum())
    return {"status": "passed" if difference <= 1e-4 and changed == 0 else "review_required",
            "max_absolute_probability_difference": difference,
            "changed_argmax": changed, "tolerance": 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["hart", "hc3"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = "race_s3407"
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "repository/resources/manifest.json").read_text())
    upstream = ROOT / "resources/RACE"
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() == manifest["upstream"]["commit"]
    assert not subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    asset = next(r for r in manifest["files"] if r.get("run") == run)
    checkpoint_path = ROOT / "resources/checkpoints" / asset["asset"]
    assert file_hash(checkpoint_path) == asset["sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"], checkpoint.get("metadata")).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    args.output.mkdir(exist_ok=True)
    evidence = {"model": run, "device": "cpu", "platform": platform.platform(),
                "checkpoint_sha256": asset["sha256"], "script_sha256": file_hash(__file__),
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "torch-geometric", "numpy", "scipy")},
                "scope": "Post-hoc CPU replay; fixed checkpoint. Calibration val was also used for training checkpoint selection."}
    archive = ROOT / "repository/experiments/reports" / run
    if args.phase == "hart":
        hart = ROOT / "resources/experiments/hart_grouped_v1"
        protocol = json.loads((hart / "protocol.json").read_text())
        # Freeze on val before opening test labels, probabilities or logits.
        assert file_hash(hart / "val.jsonl") == protocol["partitions"]["val"]["sha256"]
        validation = read_jsonl(hart / "val.jsonl")
        def graphs_for(records):
            selected = {r["item_id"]: r for r in records}
            found = {}
            for spec in manifest["files"]:
                if not spec["path"].startswith("assets/hart_official/"):
                    continue
                path = ROOT / "resources" / spec["path"]
                assert file_hash(path) == spec["sha256"]
                for line in path.open():
                    row = json.loads(line)
                    if row["item_id"] in selected:
                        original = selected[row["item_id"]]
                        assert row["article"] == original["article"] and row["label"] == original["label"]
                        assert row["item_id"] not in found
                        found[row["item_id"]] = row
            assert len(found) == len(records)
            return [found[r["item_id"]] for r in records]
        z_val, val_seconds = infer(model, graphs_for(validation), args.output / "validation_predictions.jsonl")
        y_val = np.array([r["label_id"] for r in validation])
        calibration = fit_temperature(z_val, y_val)
        for name, t in (("uncalibrated", 1), ("temperature_scaled", calibration["temperature"])):
            p = temperature_probabilities(z_val, t)
            calibration[name + "_threshold"] = negative_only_threshold(p[y_val < 2, 2:].sum(1))
        calibration.update({"validation_sha256": file_hash(hart / "val.jsonl"),
                            "validation_predictions_sha256": file_hash(args.output / "validation_predictions.jsonl"),
                            "checkpoint_sha256": asset["sha256"], "fit_on": "HART validation only"})
        write(args.output / "calibration.json", calibration)
        assert file_hash(hart / "test.jsonl") == protocol["partitions"]["test"]["sha256"]
        test = read_jsonl(hart / "test.jsonl")
        z_test, test_seconds = infer(model, graphs_for(test), args.output / "hart_test_predictions.jsonl")
        ref_path = archive / "test_predictions.jsonl"
        reference = read_jsonl(ref_path)
        assert [r["item_id"] for r in test] == [r["item_id"] for r in reference]
        evidence.update({"n_validation": len(validation), "n_test": len(test),
                         "replay": compare(z_test, reference, "probabilities"),
                         "metrics": evaluate(z_test, [r["label_id"] for r in test], calibration),
                         "seconds": {"validation": val_seconds, "test": test_seconds},
                         "reference_sha256": file_hash(ref_path),
                         "test_sha256": file_hash(hart / "test.jsonl")})
    else:
        calibration_path = args.output / "calibration.json"
        calibration = json.loads(calibration_path.read_text())
        assert calibration["checkpoint_sha256"] == asset["sha256"]
        source = ROOT / "resources/experiments/hc3_aux_v1_1"
        path = source / "test_parsed_aligned_cpu.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        assert audit["status"] == "passed" and audit["records"] == 1000
        assert file_hash(path) == audit["output_sha256"]
        records, parsed = read_jsonl(source / "test.jsonl"), read_jsonl(path)
        frozen = json.loads((source / "protocol.json").read_text())
        assert file_hash(source / "test.jsonl") == frozen["answer_sha256"]
        assert len(parsed) == len(records)
        for row, tree in zip(records, parsed):
            assert all(tree[key] == value for key, value in row.items())
        z_test, seconds = infer(model, parsed, args.output / "hc3_predictions.jsonl")
        ref_path = archive / "hc3_aux_v1_1_predictions.jsonl"
        reference = read_jsonl(ref_path)
        assert [r["item_id"] for r in parsed] == [r["item_id"] for r in reference]
        evidence.update({"n_test": len(parsed), "replay": compare(z_test, reference, "four_class_scores"),
                         "metrics": evaluate(z_test, [r["binary_label"] for r in records], calibration, False),
                         "seconds": seconds, "reference_sha256": file_hash(ref_path),
                         "parsed_sha256": file_hash(path), "calibration_sha256": file_hash(calibration_path),
                         "scope_parser": "New CPU parses; parse timing makes raw file hashes non-comparable. "
                                         "Numerical tolerance does not prove identical trees."})
    write(args.output / (args.phase + "_report.json"), evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
