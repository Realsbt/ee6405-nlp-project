"""Locked-threshold creator evaluation, separate from four-class provenance metrics."""

import math
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from analyze_saved_probabilities import binary_metrics


def negative_only_threshold(negative_scores, target_fpr=0.01):
    values = np.asarray(negative_scores, dtype=float)
    if (not 0 <= target_fpr < 1 or len(values) == 0
            or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
        raise ValueError("Need valid validation negatives and 0 <= target_fpr < 1")
    allowed = math.floor(target_fpr * len(values))
    threshold = float(np.nextafter(np.sort(values)[::-1][allowed], np.inf))
    return {"threshold": threshold, "n_validation_negatives": len(values),
            "allowed_false_positives": allowed,
            "actual_false_positives": int((values >= threshold).sum()),
            "target_empirical_validation_fpr": target_fpr,
            "scope": "Negative-only validation order statistic, ties excluded. "
                     "Not a population FPR guarantee or a test-set ROC threshold."}


def evaluate_creator(labels, scores, locked_threshold):
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    result = binary_metrics(list(zip(y.tolist(), p.tolist())))
    both_classes = len(set(y)) == 2
    result["auroc"] = float(roc_auc_score(y, p)) if both_classes else None
    result["auprc_ai_positive"] = float(average_precision_score(y, p)) if both_classes else None
    result["positive_prevalence"] = float(y.mean())
    result["at_threshold"] = {}
    for name, threshold in (("fixed_0_5", 0.5), ("hart_validation_locked", locked_threshold)):
        predicted = p >= threshold
        tn = int(((y == 0) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        tp = int(((y == 1) & predicted).sum())
        result["at_threshold"][name] = {
            "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fp / (tn + fp) if tn + fp else None,
            "tpr": tp / (fn + tp) if fn + tp else None,
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        }
    return result
