"""Train a first TF-IDF baseline; choose C on validation, then evaluate test once."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import warnings

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from experiments.common import DATA_ROOT, DOMAINS, LABELS, classification_metrics, file_hash, read_jsonl, write_json, write_jsonl


def save_confusion_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = np.asarray(metrics["confusion_matrix"])
    proportions = counts / counts.sum(axis=1, keepdims=True)
    names = ["Human", "Human + AI polish", "AI", "AI + human/tool edit"]
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    plot = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{counts[i, j]}\n{proportions[i, j]:.1%}", ha="center", va="center",
                    color="white" if proportions[i, j] > .55 else "black", fontsize=10)
    ax.set_xticks(range(4), names, rotation=25, ha="right")
    ax.set_yticks(range(4), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("TF-IDF + logistic regression\nHART grouped holdout; row percentages")
    fig.colorbar(plot, ax=ax, label="Fraction of true class", shrink=.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/tfidf_lr_v1"))
    parser.add_argument("--model", type=Path, default=DATA_ROOT.parent / "models/tfidf_lr_v1.joblib")
    args = parser.parse_args()
    if args.output.exists() or args.model.exists():
        raise FileExistsError("Use a fresh output/model path; existing experimental evidence is preserved")
    protocol = json.loads((args.data / "protocol.json").read_text())
    for split in ("train", "val", "test"):
        assert file_hash(args.data / f"{split}.jsonl") == protocol["partitions"][split]["sha256"]
    train, val = read_jsonl(args.data / "train.jsonl"), read_jsonl(args.data / "val.jsonl")
    y_train, y_val = np.asarray([r["label_id"] for r in train]), np.asarray([r["label_id"] for r in val])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                 sublinear_tf=True, lowercase=True, dtype=np.float64)
    vector_started = time.perf_counter()
    x_train = vectorizer.fit_transform([r["model_text"] for r in train])
    x_val = vectorizer.transform([r["model_text"] for r in val])
    vector_seconds = time.perf_counter() - vector_started
    print(f"Train matrix {x_train.shape}; {x_train.nnz} nonzeros; vectorization {vector_seconds:.2f}s", flush=True)
    candidates, models = [], {}
    with threadpool_limits(limits=4):
        for c in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=3407)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, y_train)
            convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise RuntimeError(f"C={c} did not converge: {convergence}")
            assert model.classes_.tolist() == list(range(4))
            metrics = classification_metrics(y_val, model.predict_proba(x_val))
            result = {"C": c, "fit_seconds": time.perf_counter() - started,
                      "iterations": model.n_iter_.tolist(), "validation": metrics, "convergence_warnings": convergence}
            candidates.append(result)
            models[c] = model
            print(f"C={c:g}: validation macro-F1={metrics['macro_f1']:.4f}, AUROC={metrics['macro_auroc_ovr']:.4f}", flush=True)
    selected = max(candidates, key=lambda row: (row["validation"]["macro_f1"], -row["C"]))
    classifier = models[selected["C"]]
    args.output.mkdir(parents=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    joblib.dump(pipeline, args.model)
    selection = {
        "selection_metric": "validation macro-F1; smaller C breaks exact ties", "candidate_C": [0.1, 1.0, 10.0],
        "selected_C": selected["C"], "candidates": candidates,
        "vocabulary_and_idf_fit_on": "train only", "classifier_fit_on": "train only",
        "test_used_for_selection": False, "refit_on_train_plus_val": False,
    }
    # Freeze selection and the model before loading test records.
    write_json(args.output / "selection.json", selection)
    test = read_jsonl(args.data / "test.jsonl")
    y_test = np.asarray([r["label_id"] for r in test])
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        probabilities = pipeline.predict_proba([r["model_text"] for r in test])
    predict_seconds = time.perf_counter() - started
    metrics = classification_metrics(y_test, probabilities)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y_train)
    dummy_probs = dummy.predict_proba(np.zeros((len(test), 1)))
    dummy_metrics = classification_metrics(y_test, dummy_probs)
    by_domain = {}
    for domain in DOMAINS:
        mask = np.asarray([r["domain"] == domain for r in test])
        by_domain[domain] = classification_metrics(y_test[mask], probabilities[mask])
    predictions = []
    for row, probs in zip(test, probabilities):
        predictions.append({"item_id": row["item_id"], "group_id": row["group_id"], "domain": row["domain"],
                            "true_label": row["label"], "prediction": LABELS[int(probs.argmax())],
                            "probabilities": dict(zip(LABELS, probs.tolist())), "input_truncated": row["input_truncated"]})
    write_jsonl(args.output / "test_predictions.jsonl", predictions)
    errors = [{**pred, "text_prefix": row["model_text"][:300]} for row, pred in zip(test, predictions)
              if pred["true_label"] != pred["prediction"]]
    write_jsonl(args.output / "errors.jsonl", errors)
    features = vectorizer.get_feature_names_out()
    top_features = {name: [{"feature": str(features[i]), "coefficient": float(classifier.coef_[class_id, i])}
                           for i in np.argsort(classifier.coef_[class_id])[-15:][::-1]]
                    for class_id, name in enumerate(LABELS)}
    write_json(args.output / "top_training_features.json", top_features)
    result = {
        "status": "completed", "model": "word TF-IDF (1,2) + balanced multinomial logistic regression",
        "protocol_id": protocol["protocol_id"], "data_protocol_sha256": file_hash(args.data / "protocol.json"),
        "data_file_sha256": {split: protocol["partitions"][split]["sha256"] for split in ("train", "val", "test")},
        "selected_C": selected["C"], "validation": selected["validation"], "test": metrics,
        "majority_class_reference": dummy_metrics, "test_by_domain": by_domain,
        "settings": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000, "sublinear_tf": True,
                     "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000,
                     "feature_text": "512-token-aligned model_text only", "cpu_thread_limit": 4},
        "efficiency": {"vectorizer_train_fit_plus_val_transform_seconds": vector_seconds,
                       "all_candidate_fit_and_validation_seconds": sum(c["fit_seconds"] for c in candidates),
                       "test_vectorize_and_predict_seconds": predict_seconds,
                       "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       "model_file_bytes": args.model.stat().st_size},
        "model_path": str(args.model), "model_sha256": file_hash(args.model), "train_script_sha256": file_hash(__file__),
        "packages": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")},
        "evidence_boundary": "One in-domain HART group-held-out test after validation-only tuning; not cross-benchmark, unseen-generator or unseen-domain validation. ROC TPR@1%FPR is descriptive, not a deployment threshold guarantee.",
    }
    write_json(args.output / "result.json", result)
    save_confusion_plot(metrics, args.output / "confusion_matrix.png")
    print(json.dumps({"selected_C": selected["C"], "test_accuracy": metrics["accuracy"],
                      "test_macro_f1": metrics["macro_f1"], "test_macro_auroc": metrics["macro_auroc_ovr"],
                      "test_per_class_f1": {name: values["f1"] for name, values in metrics["per_class"].items()},
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
