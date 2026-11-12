"""Summarize retained provenance and paired, source-cluster MixSet uncertainty."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from prepare_research_data import digest, write_new

ROOT = Path(__file__).resolve().parent
LABELS = ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"]


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def cluster_draw(groups, rng):
    names = sorted(set(groups))
    counts = rng.multinomial(len(names), np.full(len(names), 1 / len(names)))
    count_map = dict(zip(names, counts))
    return np.array([count_map[group] for group in groups])


def main():
    output = ROOT / "transfer_uncertainty_v1.json"
    if output.exists():
        raise FileExistsError(output)
    provenance = read_rows(ROOT / "hart_provenance_v1/provenance.jsonl")
    by_id = {r["item_id"]: r for r in provenance}
    if len(by_id) != len(provenance):
        raise ValueError("Nonunique provenance IDs")
    retained = {}
    for split in ("train", "val", "test"):
        rows = read_rows(ROOT / f"resources/experiments/hart_grouped_v1/{split}.jsonl")
        statuses = Counter(by_id[r["item_id"]]["status"] for r in rows)
        assert statuses == {"matched_label": len(rows)}
        retained[split] = dict(statuses)
    predictions = {}
    thresholds = {}
    inputs = {}
    for model in ("tfidf", "roberta"):
        path = ROOT / f"transfer_{model}_v1/mixset_predictions.jsonl"
        predictions[model] = read_rows(path)
        inputs[str(path.relative_to(ROOT))] = digest(path)
        thresholds[model] = json.loads(
            (path.parent / "threshold.json").read_text())["threshold"]
    rows = predictions["tfidf"]
    assert [(r["item_id"], r["group_id"], r["true_label"]) for r in rows] == [
        (r["item_id"], r["group_id"], r["true_label"]) for r in predictions["roberta"]]
    y = np.array([LABELS.index(r["true_label"]) for r in rows])
    groups = [r["group_id"] for r in rows]
    arrays = {}
    for model, records in predictions.items():
        p = np.array([[r["probabilities"][label] for label in LABELS] for r in records])
        p /= p.sum(axis=1, keepdims=True)
        arrays[model] = (p.argmax(axis=1), p[:, 2:].sum(axis=1) >= thresholds[model])
    rng = np.random.default_rng(20261106)
    samples = {model: {"macro_f1": [], "fpr": [], "tpr": []} for model in arrays}
    for _ in range(2000):
        weights = cluster_draw(groups, rng)
        for model, (pred, binary) in arrays.items():
            samples[model]["macro_f1"].append(float(f1_score(
                y, pred, labels=range(4), average="macro", sample_weight=weights,
                zero_division=0)))
            for metric, mask in (("fpr", y < 2), ("tpr", y >= 2)):
                denominator = weights[mask].sum()
                samples[model][metric].append(
                    float(np.dot(weights[mask], binary[mask]) / denominator)
                    if denominator else np.nan)
    intervals = {}
    for model, metrics in samples.items():
        intervals[model] = {}
        for metric, values in metrics.items():
            a = np.array(values)
            intervals[model][metric] = {
                "percentile_95": np.nanquantile(a, [0.025, 0.975]).tolist(),
                "valid_replicates": int(np.isfinite(a).sum())}
    delta = np.array(samples["roberta"]["macro_f1"]) - samples["tfidf"]["macro_f1"]
    report = {
        "retained_hart_provenance": retained,
        "mixset_groups": len(set(groups)), "replicates": 2000, "seed": 20261106,
        "intervals": intervals,
        "paired_roberta_minus_tfidf_macro_f1_95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "scope": "Post-hoc paired source-group percentile bootstrap, fixed checkpoints and "
                 "thresholds. Not seed, calibration-selection or population uncertainty; "
                 "not corrected for multiple comparisons. Labels fixed to four classes.",
        "input_sha256": inputs, "script_sha256": digest(Path(__file__)),
    }
    write_new(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
