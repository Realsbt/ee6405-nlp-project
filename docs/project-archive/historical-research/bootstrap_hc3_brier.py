"""Post-hoc paired Brier differences, stratified by source and clustered by question."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_saved_probabilities import probability_vector


def paired_interval(delta, sources, repetitions=5000, seed=20261106):
    delta = np.asarray(delta, dtype=float)
    sources = np.asarray(sources)
    if len(delta) == 0 or len(delta) != len(sources) or not np.isfinite(delta).all():
        raise ValueError("Invalid paired question-level differences")
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    for source in sorted(set(sources)):
        values = delta[sources == source]
        draws = rng.integers(0, len(values), size=(repetitions, len(values)))
        samples += values[draws].sum(axis=1) / len(delta)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean_brier_difference": float(delta.mean()),
            "percentile_95_interval": [float(low), float(high)]}


def load(path):
    groups, ids = {}, set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["item_id"] in ids:
            raise ValueError("Duplicate answer ID")
        ids.add(row["item_id"])
        p = probability_vector(row["four_class_scores"])
        score = p[2] + p[3]
        label = row["true_binary_label"]
        if label not in (0, 1):
            raise ValueError("Expected human/AI ground truth")
        group = groups.setdefault(row["question_group"], [])
        group.append((row["item_id"], label, row["source"], (score - label) ** 2))
    for group in groups.values():
        if len(group) != 2 or {r[1] for r in group} != {0, 1} or len({r[2] for r in group}) != 1:
            raise ValueError("Expected one human/AI pair per source-specific question")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path(__file__).parent / "repository")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        for values, expected in (([0, 0, 0, 0], 0), ([0.2] * 4, 0.2)):
            result = paired_interval(values, ["a", "a", "b", "b"], repetitions=100)
            assert np.allclose(result["percentile_95_interval"], [expected, expected])
            assert np.isclose(result["mean_brier_difference"], expected)
        assert paired_interval([0, 1], ["a", "a"]) == paired_interval([0, 1], ["a", "a"])
        print("PASS: zero effect, constant effect, and deterministic resampling")
        return
    if args.output is None:
        parser.error("--output is required unless --self-test")
    paths = sorted((args.repository / "experiments/reports").glob("*/hc3_aux_v1_1_predictions.jsonl"))
    if not paths:
        raise FileNotFoundError("No HC3 saved predictions")
    data = {path.parent.name: load(path) for path in paths}
    reference = data["roberta_s3407"]
    keys = sorted(reference)
    sources = [reference[key][0][2] for key in keys]
    losses = {}
    for name, groups in data.items():
        if set(groups) != set(reference):
            raise ValueError("Question group mismatch")
        for key in keys:
            if sorted(r[:3] for r in groups[key]) != sorted(r[:3] for r in reference[key]):
                raise ValueError("Mismatched paired samples")
        losses[name] = np.array([sum(r[3] for r in groups[key]) / 2 for key in keys])
    result = {
        "scope": "Post-hoc descriptive HC3 Brier comparisons. Frozen single-seed models, "
                 "question-pair bootstrap within each source, 5000 draws, seed 20261106. "
                 "Intervals omit retraining and unseen-source uncertainty; no multiplicity correction.",
        "n_questions": len(keys), "n_answers": len(keys) * 2,
        "source_question_counts": {s: sources.count(s) for s in sorted(set(sources))},
        "input_sha256": {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vs_roberta_ce": {name: paired_interval(loss - losses["roberta_s3407"], sources)
                          for name, loss in losses.items() if name != "roberta_s3407"},
        "vs_constant_half": {name: paired_interval(loss - 0.25, sources)
                             for name, loss in losses.items()},
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("n_questions", "vs_roberta_ce", "vs_constant_half")}, indent=2))


if __name__ == "__main__":
    main()
