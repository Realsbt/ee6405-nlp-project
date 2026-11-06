"""Shared labels, provenance helpers, and explicit four-class metrics."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve


LABELS = ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"]
DOMAINS = ["arxiv", "essay", "news", "writing"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path(os.environ.get("EE6405_STORAGE", PROJECT_ROOT))
ASSETS_ROOT = Path(os.environ.get("EE6405_ASSETS", STORAGE_ROOT / "assets"))
DATA_ROOT = Path(os.environ.get("EE6405_DATA_ROOT", STORAGE_ROOT / "experiments/hart_grouped_v1"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(labels, probabilities):
    """Use all four fixed labels. Missing-class AUROC/TPR is null, never zero.

    TPR@1%FPR is descriptive ROC analysis of this evaluation partition, not a
    deployment threshold calibrated on an independent population.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 4):
        raise ValueError("Expected N x 4 probabilities")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid class probabilities")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(4), zero_division=0
    )
    per_class = {}
    tprs, aucs = [], []
    for class_id, name in enumerate(LABELS):
        target = labels == class_id
        auc = tpr = None
        if target.any() and (~target).any():
            auc = float(roc_auc_score(target, probabilities[:, class_id]))
            fpr, curve_tpr, _ = roc_curve(target, probabilities[:, class_id])
            tpr = float(curve_tpr[fpr <= 0.01].max())
            aucs.append(auc)
            tprs.append(tpr)
        per_class[name] = {"support": int(support[class_id]), "precision": float(precision[class_id]),
                           "recall": float(recall[class_id]), "f1": float(f1[class_id]),
                           "auroc_ovr": auc, "tpr_at_1pct_fpr_roc": tpr}
    return {
        "samples": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()), "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_auroc_ovr": float(np.mean(aucs)) if len(aucs) == 4 else None,
        "macro_tpr_at_1pct_fpr_roc": float(np.mean(tprs)) if len(tprs) == 4 else None,
        "per_class": per_class, "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(4)).tolist(),
        "label_order": LABELS,
    }
