"""CPU checkpoint validation inference and frozen MixSet transfer evaluation."""

import argparse
import json
from pathlib import Path
import time

from replay_hc3_cpu import (ROOT, LABELS, TextDataset, build_model, collate_batch,
                           file_hash, joblib, np, read_jsonl, subprocess,
                           tokenizer_local, torch, DataLoader)
from experiments.common import classification_metrics
from evaluation_metrics import evaluate_creator, negative_only_threshold


def predict(model, kind, rows, tokenizer):
    if kind == "tfidf":
        texts = []
        for row in rows:
            if "model_text" in row:
                texts.append(row["model_text"])
            else:
                tokens = tokenizer(row["article"], max_length=512, truncation=True,
                                   return_offsets_mapping=True)
                end = max(end for start, end in tokens["offset_mapping"])
                texts.append(row["article"][:end])
        return model.predict_proba(texts)
    dataset = TextDataset([{**r, "label_id": -1} for r in rows], tokenizer)
    results = []
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=8, collate_fn=collate_batch)):
            results.append(model(batch)["logits"].float().softmax(-1).numpy())
            if (index + 1) % 25 == 0:
                print(f"{kind}: {min((index+1)*8, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(results)


def save_predictions(path, rows, probabilities):
    with path.open("x") as stream:
        for row, p in zip(rows, probabilities):
            out = {k: row[k] for k in ("item_id", "group_id", "domain", "operation") if k in row}
            out["true_label"] = row["label"]
            out["probabilities"] = dict(zip(LABELS, map(float, p)))
            stream.write(json.dumps(out) + "\n")


def summarize(rows, probabilities, threshold, binary_labels=None):
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (len(rows), 4) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid probability matrix")
    p = p / p.sum(axis=1, keepdims=True)
    scores = p[:, 2:].sum(axis=1).clip(0, 1)
    y = binary_labels if binary_labels is not None else [
        int(row["label"] in LABELS[2:]) for row in rows]
    result = {"creator": evaluate_creator(y, scores, threshold)}
    if binary_labels is None:
        result["four_class"] = classification_metrics([LABELS.index(r["label"]) for r in rows], p)
        pure_human = np.array([r["label"] == "human_written" for r in rows])
        result["pure_human_fpr_at_locked_threshold"] = (
            float((scores[pure_human] >= threshold).mean()) if pure_human.any() else None)
        result["group_count"] = len({r["group_id"] for r in rows})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tfidf", "roberta"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving previous results")
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "repository/resources/manifest.json").read_text())
    upstream = ROOT / "resources/RACE"
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                               text=True).strip() != manifest["upstream"]["commit"]:
        raise ValueError("Wrong upstream revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream source changed")
    run = "tfidf_lr_v1" if args.model == "tfidf" else "roberta_s3407"
    asset = next(row for row in manifest["files"] if row.get("run") == run)
    path = ROOT / "resources/checkpoints" / asset["asset"]
    if file_hash(path) != asset["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if args.model == "tfidf":
        model = joblib.load(path)
        if list(model.classes_) != list(range(4)):
            raise ValueError("Wrong class order")
    else:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(saved["config"], saved.get("metadata")).cpu().eval()
        model.load_state_dict(saved["model_state_dict"], strict=True)
    tokenizer = tokenizer_local()
    hart = ROOT / "resources/experiments/hart_grouped_v1"
    protocol = json.loads((hart / "protocol.json").read_text())
    if file_hash(hart / "val.jsonl") != protocol["partitions"]["val"]["sha256"]:
        raise ValueError("Validation partition changed")
    validation = read_jsonl(hart / "val.jsonl")
    started = time.perf_counter()
    p_val = predict(model, args.model, validation, tokenizer)
    p_val = p_val / p_val.sum(axis=1, keepdims=True)
    negative_scores = [float(p[2:].sum().clip(0, 1)) for row, p in zip(validation, p_val)
                       if row["label"] in LABELS[:2]]
    threshold = negative_only_threshold(negative_scores)
    threshold.update({"run": run, "checkpoint_sha256": asset["sha256"],
                      "validation_sha256": file_hash(hart / "val.jsonl"),
                      "script_sha256": file_hash(__file__)})
    args.output.mkdir()
    (args.output / "threshold.json").write_text(json.dumps(threshold, indent=2) + "\n")
    save_predictions(args.output / "validation_predictions.jsonl", validation, p_val)
    # Test/external rows are opened only after the threshold is saved.
    cutoff = threshold["threshold"]
    test = read_jsonl(hart / "test.jsonl")
    if file_hash(hart / "test.jsonl") != protocol["partitions"]["test"]["sha256"]:
        raise ValueError("Test partition changed")
    archived_path = ROOT / "repository/experiments/reports" / run / "test_predictions.jsonl"
    archived = read_jsonl(archived_path)
    if [r["item_id"] for r in test] != [r["item_id"] for r in archived]:
        raise ValueError("Archived test IDs mismatch")
    p_test = [[r["probabilities"][label] for label in LABELS] for r in archived]
    hc3 = read_jsonl(ROOT / "repository/experiments/reports" / run / "hc3_aux_v1_1_predictions.jsonl")
    p_hc3 = [[r["four_class_scores"][label] for label in LABELS] for r in hc3]
    mix_root = ROOT / "resources/mixset_conservative_transfer_v1"
    mix_protocol = json.loads((mix_root / "protocol.json").read_text())
    if file_hash(mix_root / "test.jsonl") != mix_protocol["data_sha256"]:
        raise ValueError("Frozen MixSet changed")
    mix = read_jsonl(mix_root / "test.jsonl")
    p_mix = predict(model, args.model, mix, tokenizer)
    save_predictions(args.output / "mixset_predictions.jsonl", mix, p_mix)
    result = {
        "status": "completed", "model": run, "device": "cpu",
        "threshold": threshold, "elapsed_seconds": time.perf_counter() - started,
        "inference_scope": "Validation and MixSet: new checkpoint inference. HART test and HC3: "
                           "archived predictions; HC3 checkpoint replay separately verified.",
        "hart_test": summarize(test, p_test, cutoff),
        "hc3": summarize(hc3, p_hc3, cutoff, [r["true_binary_label"] for r in hc3]),
        "mixset": summarize(mix, p_mix, cutoff),
        "mixset_protocol_sha256": file_hash(mix_root / "protocol.json"),
        "hart_archived_predictions_sha256": file_hash(archived_path),
        "limitations": "Post-hoc extension after seeing old HART/HC3 results. Validation also used "
                       "for checkpoint selection. MixSet mapped subset is not the original binary task. "
                       "Empirical validation FPR is not guaranteed on target populations.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({dataset: {
        "f1_4class": result[dataset].get("four_class", {}).get("macro_f1"),
        "auroc_creator": result[dataset]["creator"]["auroc"],
        "locked": result[dataset]["creator"]["at_threshold"]["hart_validation_locked"],
    } for dataset in ("hart_test", "hc3", "mixset")}, indent=2))


if __name__ == "__main__":
    main()
