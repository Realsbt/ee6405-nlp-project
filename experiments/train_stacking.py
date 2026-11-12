"""Portable CPU stacking of frozen predictions; no encoder training.

Adapted from Yingqi's 2026-11-09 local stack_cpu.py. The saved historical
results retain their original provenance; this entry point makes the core
four-fold fitting procedure reproducible without that workspace's helpers.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.common import LABELS, classification_metrics, file_hash, write_json, write_jsonl

SPECS = [{"kind": "logistic", "C": c} for c in (.01, .1, 1, 10)] + [
    {"kind": "histogram", "max_leaf_nodes": leaves, "l2_regularization": 10.0}
    for leaves in (3, 7)
]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def aligned_probabilities(records, path):
    predictions = rows(path)
    by_id = {row["item_id"]: row for row in predictions}
    ids = {row["item_id"] for row in records}
    if len(ids) != len(records) or len(by_id) != len(predictions) or set(by_id) != ids:
        raise ValueError(f"Prediction IDs differ: {path}")
    values = []
    for record in records:
        row = by_id[record["item_id"]]
        if ("true_label" in row
                and row["true_label"] != record.get("label", record.get("true_label"))):
            raise ValueError("Prediction labels differ")
        if "group_id" in row and row["group_id"] != record["group_id"]:
            raise ValueError("Prediction groups differ")
        values.append([row["probabilities"][label] for label in LABELS])
    p = np.asarray(values, dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid probabilities")
    return p / p.sum(1, keepdims=True)


def stack_features(probabilities):
    blocks = []
    for p in probabilities:
        z = np.log(np.clip(p, 1e-12, 1))
        blocks.extend([z - z.mean(1, keepdims=True), p])
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Invalid stacking features")
    return features


def model_for(spec):
    if spec["kind"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=spec["C"], max_iter=2000, random_state=3407,
        ))
    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=.05, max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=30, l2_regularization=spec["l2_regularization"],
        early_stopping=False, random_state=3407,
    )


def labels(records):
    return np.array([LABELS.index(row.get("label", row.get("true_label"))) for row in records])


def prediction_rows(records, probabilities):
    return [{
        "item_id": row["item_id"], "group_id": row["group_id"],
        "true_label": row.get("label", row.get("true_label")),
        "probabilities": dict(zip(LABELS, p.tolist())),
    } for row, p in zip(records, probabilities)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True,
                        help="Frozen dev metadata with label, group_id and split_block_id")
    parser.add_argument("--dev-predictions", type=Path, nargs="+", required=True,
                        help="CE, RACE SupCon, TF-IDF predictions in that fixed order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--test-predictions", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing outputs")
    if bool(args.test) != bool(args.test_predictions):
        parser.error("--test and --test-predictions must be provided together")
    if args.test and len(args.dev_predictions) != len(args.test_predictions):
        parser.error("Development/test component counts differ")
    dev = rows(args.development)
    y = labels(dev)
    groups = np.array([row["split_block_id"] for row in dev])
    x = stack_features([aligned_probabilities(dev, path) for path in args.dev_predictions])
    folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261109)
                 .split(x, y, groups))
    for fit, held in folds:
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Source overlap across meta-folds")
    args.output.mkdir(parents=True)
    candidates, best = [], None
    with threadpool_limits(limits=2):
        for spec in SPECS:
            oof = np.zeros((len(dev), len(LABELS)))
            for fit, held in folds:
                model = model_for(spec)
                model.fit(x[fit], y[fit])
                if model.classes_.tolist() != list(range(len(LABELS))):
                    raise ValueError("Missing class in fit")
                oof[held] = model.predict_proba(x[held])
            metrics = classification_metrics(y.tolist(), oof.tolist())
            candidates.append({"spec": spec, "oof_development": metrics})
            if best is None or metrics["macro_f1"] > best[0]:
                best = (metrics["macro_f1"], spec, oof.copy())
        model = model_for(best[1])
        model.fit(x, y)
    joblib.dump(model, args.output / "meta_model.joblib")
    write_json(args.output / "selection.json", {
        "selected": best[1], "candidates": candidates, "test_used_for_selection": False,
        "input_sha256": {str(p): file_hash(p) for p in [args.development, *args.dev_predictions]},
        "limitations": [
            "Base checkpoint selection already used the development labels.",
            "Historical test inspection makes this exploratory, not independent confirmation.",
            "CPU meta-model fitting reuses previously GPU-trained encoders.",
        ],
    })
    write_jsonl(args.output / "development_oof_predictions.jsonl",
                prediction_rows(dev, best[2]))
    # Test files are not opened until the selection has been saved.
    if args.test:
        test = rows(args.test)
        if set(groups) & {row["split_block_id"] for row in test}:
            raise ValueError("Development/test source overlap")
        xt = stack_features([aligned_probabilities(test, path) for path in args.test_predictions])
        p = model.predict_proba(xt)
        write_jsonl(args.output / "test_predictions.jsonl", prediction_rows(test, p))
        write_json(args.output / "result.json", classification_metrics(labels(test).tolist(), p.tolist()))
    print(f"Selected {best[1]}, development OOF Macro-F1={best[0]:.6f}")


if __name__ == "__main__":
    main()
