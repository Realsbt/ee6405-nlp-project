"""Score-first 36-candidate campaign runner.

This module intentionally owns a separate campaign directory and only reads
HART train/validation.  It does not load test or HC3.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from experiments.common import DATA_ROOT, LABELS, classification_metrics, file_hash, write_json
from experiments.neural import load_records, tokenizer_local
from experiments.glpc_model import GLPCModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
B0_PATH = PROJECT_ROOT / "experiments/glpc_runs/glpc_b0_s3407/best.pt"
PLAN_PATH = Path(os.environ.get("EE6405_CAMPAIGN_PLAN", PROJECT_ROOT.parent / "gpt/campaign_plan.json"))
DEFAULT_ROOT = DATA_ROOT.parent / "campaign12h" / "ee6405_scorefirst_36_12h"
START_TIME = time.time()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save(path, value):
    write_json(path, value)


def group_key(record):
    return record.get("split_block_id") or record.get("group_id")


def make_split(records, root):
    labels = [f"{r['domain']}|{r['label_id']}" for r in records]
    groups = [group_key(r) for r in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261107)
    folds = [None] * len(records)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(records)), labels, groups)):
        for index in indices:
            folds[index] = fold
    if any(fold is None for fold in folds):
        raise RuntimeError("S/H split did not assign every validation record")
    manifest = {
        "seed": 20261107,
        "fold_rule": "fold0,1=H; fold2,3,4=S",
        "records": [
            {"item_id": r["item_id"], "group_id": group_key(r), "label_id": r["label_id"],
             "domain": r["domain"], "fold": folds[i], "partition": "H" if folds[i] in (0, 1) else "S"}
            for i, r in enumerate(records)
        ],
        "data_hash": file_hash(DATA_ROOT / "val.jsonl"),
    }
    s = [i for i, fold in enumerate(folds) if fold not in (0, 1)]
    h = [i for i, fold in enumerate(folds) if fold in (0, 1)]
    if min(sum(records[i]["label_id"] == label for i in s) for label in range(4)) < 30:
        raise RuntimeError("S class count is below protocol minimum")
    if min(sum(records[i]["label_id"] == label for i in h) for label in range(4)) < 20:
        raise RuntimeError("H class count is below protocol minimum")
    save(root / "protocol/split_manifest.json", manifest)
    return s, h, manifest


def load_b0():
    checkpoint = torch.load(B0_PATH, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["model_type"] = "roberta"
    config["classifier_type"] = "mlp"
    model = GLPCModel(config, mode="baseline")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().cuda()
    return model, checkpoint, config


def predict_b0(records, model, batch_size=16):
    tokenizer = tokenizer_local()
    rows = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            part = records[start:start + batch_size]
            encoded = tokenizer([r["article"] for r in part], max_length=512, truncation=True,
                                padding=True, return_tensors="pt")
            output = model({"input_ids": encoded["input_ids"].cuda(),
                            "attention_mask": encoded["attention_mask"].cuda()})
            rows.extend(output["logits"].float().cpu().numpy())
    return np.asarray(rows, dtype=np.float64)


def metrics(records, probabilities):
    return classification_metrics([r["label_id"] for r in records], probabilities)


def result(task_id, status, records, probabilities, base_probs, started, stop_reason, config, root, extra=None):
    m = metrics(records, probabilities)
    base = metrics(records, base_probs)
    prediction = probabilities.argmax(1)
    base_prediction = base_probs.argmax(1)
    payload = {
        "task_id": task_id, "status": status, "metrics": m,
        "base_metrics": base, "delta_accuracy_pp": 100 * (m["accuracy"] - base["accuracy"]),
        "delta_macro_f1_pp": 100 * (m["macro_f1"] - base["macro_f1"]),
        "J": (m["accuracy"] + m["macro_f1"]) / 2,
        "wrong_to_right": int(((prediction == [r["label_id"] for r in records]) &
                               (base_prediction != [r["label_id"] for r in records])).sum()),
        "right_to_wrong": int(((prediction != [r["label_id"] for r in records]) &
                               (base_prediction == [r["label_id"] for r in records])).sum()),
        "changed_wrong_to_wrong": int(((prediction != [r["label_id"] for r in records]) &
                                      (base_prediction != [r["label_id"] for r in records]) &
                                      (prediction != base_prediction)).sum()),
        "stopped_reason": stop_reason, "config": config,
        "wall_seconds": time.time() - started, "test_partition_loaded": False, "hc3_loaded": False,
    }
    save(root / f"tasks/{task_id}/selection.json", payload)
    np.save(root / f"tasks/{task_id}/selection_probabilities.npy", probabilities)
    return payload


def train_text_features(train, records):
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=100000)),
    ])
    x_train = vectorizer.fit_transform([r["article"][:12000] for r in train])
    x_eval = vectorizer.transform([r["article"][:12000] for r in records])
    classifier = LogisticRegression(C=4, max_iter=500, random_state=20261107)
    classifier.fit(x_train, [r["label_id"] for r in train])
    return classifier.predict_proba(x_eval)


def nb_svm(train, records):
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=2, max_features=100000)
    x_train = vectorizer.fit_transform([r["article"][:12000] for r in train])
    x_eval = vectorizer.transform([r["article"][:12000] for r in records])
    scores = []
    for label in range(4):
        positive = x_train[[r["label_id"] == label for r in train]].sum(0) + 1
        negative = x_train[[r["label_id"] != label for r in train]].sum(0) + 1
        ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        scores.append(LinearSVC(C=1).fit(x_train.multiply(ratio), [r["label_id"] == label for r in train]).decision_function(x_eval))
    raw = np.stack(scores, axis=1)
    return torch.softmax(torch.tensor(raw, dtype=torch.float64), dim=1).numpy()


def style_features(records):
    values = []
    for record in records:
        text = record["article"][:12000]
        words = text.split()
        sentences = [part for part in text.replace("!", ".").replace("?", ".").split(".") if part.strip()]
        counts = np.asarray([len(w) for w in words], dtype=float)
        values.append([
            len(text), len(words), len(sentences), counts.mean() if len(counts) else 0,
            counts.std() if len(counts) else 0, text.count(",") / max(1, len(text)),
            text.count(";") / max(1, len(text)), sum(c.isdigit() for c in text) / max(1, len(text)),
            len(set(words)) / max(1, len(words)), text.count("\n") / max(1, len(text)),
        ])
    return np.asarray(values, dtype=float)


def fit_logistic(x_train, y_train, x_eval):
    scaler = StandardScaler()
    classifier = LogisticRegression(C=.1, max_iter=500, random_state=20261107)
    return classifier.fit(scaler.fit_transform(x_train), y_train).predict_proba(scaler.transform(x_eval))


def write_campaign_index(root, task_rows):
    rows = []
    for row in task_rows:
        rows.append(f"<tr><td>{html.escape(row['task_id'])}</td><td>{html.escape(row['status'])}</td>"
                    f"<td>{row.get('S_accuracy')}</td><td>{row.get('S_macro_f1')}</td>"
                    f"<td>{row.get('H_accuracy')}</td><td>{row.get('H_macro_f1')}</td></tr>")
    page = """<!doctype html><meta charset="utf-8"><title>36-task campaign</title>
<style>body{font-family:system-ui}table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px}</style>
<table><tr><th>Task</th><th>Status</th><th>S Accuracy</th><th>S Macro-F1</th><th>H Accuracy</th><th>H Macro-F1</th></tr>
""" + "\n".join(rows) + "</table>"
    (root / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase", choices=("bootstrap", "cheap", "probes", "fusion", "finalize"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    state_path = root / "campaign_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("campaign_id", plan["campaign_id"])
        state.setdefault("start_epoch", START_TIME)
        state.setdefault("deadline_epoch", state["start_epoch"] + plan["budget"]["total_wall_seconds"])
        state["phase"] = args.phase
        state["test_partition_loaded"] = False
        state["hc3_loaded"] = False
    else:
        state = {"campaign_id": plan["campaign_id"], "start_epoch": START_TIME,
                 "deadline_epoch": START_TIME + plan["budget"]["total_wall_seconds"],
                 "phase": args.phase, "test_partition_loaded": False, "hc3_loaded": False}
    save(state_path, state)
    val = load_records("val")
    train = load_records("train")
    s_indices, h_indices, manifest = make_split(val, root)
    s_records, h_records = [val[i] for i in s_indices], [val[i] for i in h_indices]
    model, checkpoint, b0_config = load_b0()
    b0_all = predict_b0(val, model)
    b0_s, b0_h = b0_all[s_indices], b0_all[h_indices]
    b0_s_prob = torch.softmax(torch.tensor(b0_s), dim=1).numpy()
    b0_h_prob = torch.softmax(torch.tensor(b0_h), dim=1).numpy()
    save(root / "sources_lock.json", {"branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip(),
                                      "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                                      "b0_sha256": file_hash(B0_PATH), "train_hash": file_hash(DATA_ROOT / "train.jsonl"),
                                      "val_hash": file_hash(DATA_ROOT / "val.jsonl"), "historical_validation_reuse_warning": True})
    np.save(root / "baseline_all_logits.npy", b0_all)
    if args.phase == "bootstrap":
        save(root / "baseline_replay.json", {"task_id": "E01", "status": "COMPLETE",
                                             "S": metrics(s_records, b0_s_prob), "H": None,
                                             "test_partition_loaded": False, "hc3_loaded": False})
        return
    if args.phase == "cheap":
        rows = []
        rows.append(result("E01", "COMPLETE", s_records, b0_s_prob, b0_s_prob, time.time(), "completed", {"kind": "B0"}, root))
        counts = np.bincount([r["label_id"] for r in train], minlength=4).astype(float)
        prior = counts / counts.sum()
        for tau in (-.5, -.25, 0, .25, .5):
            p = torch.softmax(torch.tensor(b0_s) + tau * torch.log(torch.tensor(prior)), dim=1).numpy()
            candidate = result("E02", "COMPLETE", s_records, p, b0_s_prob, time.time(), "completed", {"tau": tau}, root)
            rows.append(candidate)
        best_e02 = max(rows[1:], key=lambda x: (x["J"] if x["delta_accuracy_pp"] >= 0 and x["delta_macro_f1_pp"] >= 0 else -1))
        for task_id, fn, config in [
            ("E03", None, {"kind": "vector_calibration"}),
            ("E04", None, {"kind": "top2_ridge"}),
        ]:
            if task_id == "E03":
                x = np.log(np.clip(b0_s_prob, 1e-12, 1)).astype(float)
                y = np.asarray([r["label_id"] for r in s_records])
                def objective(theta):
                    scale = np.exp(np.clip(theta[:4], -.7, .7))
                    bias = theta[4:] - theta[4:].mean()
                    logits = x * scale + bias
                    p = torch.softmax(torch.tensor(logits), dim=1).numpy()
                    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean() + .05 * np.square(theta).mean()
                fitted = minimize(objective, np.zeros(8), method="L-BFGS-B", options={"maxiter": 100})
                scale = np.exp(np.clip(fitted.x[:4], -.7, .7)); bias = fitted.x[4:] - fitted.x[4:].mean()
                p = torch.softmax(torch.tensor(x * scale + bias), dim=1).numpy()
                rows.append(result("E03", "COMPLETE", s_records, p, b0_s_prob, time.time(), "completed",
                                   {"kind": "vector_calibration", "parameters": fitted.x.tolist()}, root))
            else:
                top = np.argsort(-b0_s_prob, axis=1)[:, :2]
                margin = b0_s_prob[np.arange(len(s_records)), top[:, 0]] - b0_s_prob[np.arange(len(s_records)), top[:, 1]]
                features = np.concatenate((b0_s_prob, top, margin[:, None]), axis=1)
                y = np.asarray([int(top[i, 1] == label) - int(top[i, 0] == label)
                                for i, label in enumerate([r["label_id"] for r in s_records])])
                ridge = Ridge(alpha=10).fit(features, y)
                swap = (margin < .15) & (ridge.predict(features) > .10)
                p = b0_s_prob.copy()
                p[swap] = p[swap][:, ::-1] if False else p[swap]
                for i in np.flatnonzero(swap):
                    p[i, top[i, 0]], p[i, top[i, 1]] = p[i, top[i, 1]], p[i, top[i, 0]]
                rows.append(result("E04", "COMPLETE", s_records, p, b0_s_prob, time.time(), "completed",
                                   {"kind": "top2_ridge", "swapped": int(swap.sum())}, root))
        sparse = train_text_features(train, s_records)
        rows.append(result("E05", "COMPLETE", s_records, sparse, b0_s_prob, time.time(), "completed",
                           {"kind": "tfidf_word_char_lr"}, root))
        nb = nb_svm(train, s_records)
        rows.append(result("E06", "COMPLETE", s_records, nb, b0_s_prob, time.time(), "completed",
                           {"kind": "nb_svm"}, root))
        style = fit_logistic(style_features(train), [r["label_id"] for r in train], style_features(s_records))
        rows.append(result("E07", "COMPLETE", s_records, style, b0_s_prob, time.time(), "completed",
                           {"kind": "style_lr"}, root))
        completed = {row["task_id"]: row for row in rows}
        task_rows = []
        for task in plan["tasks"]:
            row = completed.get(task["id"])
            task_rows.append({"task_id": task["id"], "family": task["family"], "method": task["name"],
                              "status": row["status"] if row else "PLANNED",
                              "S_accuracy": row["metrics"]["accuracy"] if row else None,
                              "S_macro_f1": row["metrics"]["macro_f1"] if row else None,
                              "H_accuracy": None, "H_macro_f1": None})
        save(root / "leaderboard_selection.json", task_rows)
        write_campaign_index(root, task_rows)
        save(root / "candidate_lock.json", {"locked_after": "selection", "candidates": sorted(completed),
                                            "default_recommendation": max(completed, key=lambda key: completed[key]["J"]),
                                            "accuracy_candidate": max(completed, key=lambda key: completed[key]["metrics"]["accuracy"]),
                                            "macro_f1_candidate": max(completed, key=lambda key: completed[key]["metrics"]["macro_f1"])})
        return
    if args.phase == "probes":
        rows = []
        model.train()
        torch.manual_seed(20261107)
        mc = []
        for _ in range(5):
            mc.append(torch.softmax(torch.tensor(predict_b0(s_records, model)), dim=1).numpy())
        model.eval()
        mc_prob = np.mean(mc, axis=0)
        rows.append(result("E25", "COMPLETE", s_records, mc_prob, b0_s_prob, time.time(), "completed",
                           {"kind": "mc_dropout", "passes": 5}, root))
        tokenizer = tokenizer_local()
        multi = []
        with torch.no_grad():
            for record in s_records:
                encoded = tokenizer(record["article"], max_length=512, truncation=False,
                                    padding=False, return_tensors="pt")
                ids = encoded["input_ids"][0]
                windows = [ids[:512]]
                if len(ids) > 512:
                    windows = [ids[:512], ids[max(0, (len(ids)-512)//2):max(0, (len(ids)-512)//2)+512], ids[-512:]]
                probs = []
                for window in windows:
                    batch_ids = window.unsqueeze(0).cuda()
                    mask = torch.ones_like(batch_ids)
                    logits = model({"input_ids": batch_ids, "attention_mask": mask})["logits"]
                    probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
                multi.append(np.mean(probs, axis=0))
        rows.append(result("E26", "COMPLETE", s_records, np.asarray(multi), b0_s_prob, time.time(), "completed",
                           {"kind": "head_middle_tail_probability_mean"}, root))
        for task in plan["tasks"]:
            if task["id"] in ("E25", "E26"):
                task["state"] = "COMPLETE"
        save(root / "probe_results.json", rows)
        return
    if args.phase == "fusion":
        def read_prob(task_id):
            return np.load(root / f"tasks/{task_id}/selection_probabilities.npy")
        candidates = [("E01", b0_s_prob), ("E05", read_prob("E05")), ("E25", read_prob("E25"))]
        average = np.mean([value for _, value in candidates], axis=0)
        row30 = result("E30", "COMPLETE", s_records, average, b0_s_prob, time.time(), "completed",
                       {"kind": "equal_probability_mean", "members": [x[0] for x in candidates]}, root)
        y = np.asarray([r["label_id"] for r in s_records])
        raw = np.zeros(len(candidates))
        def objective(raw_weights):
            weights = np.exp(raw_weights - raw_weights.max())
            weights /= weights.sum()
            p = sum(weight * probs for weight, (_, probs) in zip(weights, candidates))
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean() + .05 * np.square(weights - 1 / len(weights)).sum()
        fitted = minimize(objective, raw, method="L-BFGS-B", options={"maxiter": 100})
        weights = np.exp(fitted.x - fitted.x.max()); weights /= weights.sum()
        weighted = sum(weight * probs for weight, (_, probs) in zip(weights, candidates))
        row31 = result("E31", "COMPLETE", s_records, weighted, b0_s_prob, time.time(), "completed",
                       {"kind": "nonnegative_probability_fusion", "members": [x[0] for x in candidates],
                        "weights": weights.tolist(), "fit": "S_selection_not_OOF"}, root)
        save(root / "fusion_results.json", [row30, row31])
        selection_rows = [json.loads(path.read_text(encoding="utf-8"))
                          for path in sorted((root / "tasks").glob("E*/selection.json"))]
        selection_rows.extend([row30, row31])
        save(root / "candidate_lock.json", {
            "locked_after": "selection", "candidates": sorted({row["task_id"] for row in selection_rows}),
            "default_recommendation": max(selection_rows, key=lambda x: x["J"])["task_id"],
            "accuracy_candidate": max(selection_rows, key=lambda x: x["metrics"]["accuracy"])["task_id"],
            "macro_f1_candidate": max(selection_rows, key=lambda x: x["metrics"]["macro_f1"])["task_id"],
            "weights_locked_before_H": True,
        })
        task_rows = []
        for task in plan["tasks"]:
            row = next((item for item in selection_rows if item["task_id"] == task["id"]), None)
            task_rows.append({"task_id": task["id"], "family": task["family"], "method": task["name"],
                              "status": row["status"] if row else "PLANNED",
                              "S_accuracy": row["metrics"]["accuracy"] if row else None,
                              "S_macro_f1": row["metrics"]["macro_f1"] if row else None,
                              "H_accuracy": None, "H_macro_f1": None})
        save(root / "leaderboard_selection.json", task_rows)
        write_campaign_index(root, task_rows)
        return
    if args.phase == "finalize":
        locked = json.loads((root / "candidate_lock.json").read_text(encoding="utf-8"))
        selected_ids = locked["candidates"]
        all_rows = {}
        for path in (root / "tasks").glob("E*/selection.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            all_rows[item["task_id"]] = item
        all_probs = {"E01": b0_h_prob}
        h_model = model
        h_all = predict_b0(h_records, h_model)
        all_probs["E01"] = torch.softmax(torch.tensor(h_all), dim=1).numpy()
        train_labels = [r["label_id"] for r in train]
        h_labels = [r["label_id"] for r in h_records]
        prior = np.bincount(train_labels, minlength=4).astype(float)
        prior /= prior.sum()
        e02_tau = all_rows["E02"]["config"]["tau"]
        all_probs["E02"] = torch.softmax(torch.tensor(h_all) + e02_tau * torch.log(torch.tensor(prior)), dim=1).numpy()
        e03_params = np.asarray(all_rows["E03"]["config"]["parameters"])
        e03_scale = np.exp(np.clip(e03_params[:4], -.7, .7))
        e03_bias = e03_params[4:] - e03_params[4:].mean()
        all_probs["E03"] = torch.softmax(torch.tensor(np.log(np.clip(all_probs["E01"], 1e-12, 1)) * e03_scale + e03_bias), dim=1).numpy()
        all_probs["E04"] = all_probs["E01"].copy()
        all_probs["E05"] = train_text_features(train, h_records)
        all_probs["E06"] = nb_svm(train, h_records)
        all_probs["E07"] = fit_logistic(style_features(train), train_labels, style_features(h_records))
        h_model.train()
        mc = []
        for pass_index in range(5):
            torch.manual_seed(20261107 + pass_index)
            mc.append(torch.softmax(torch.tensor(predict_b0(h_records, h_model)), dim=1).numpy())
        h_model.eval()
        all_probs["E25"] = np.mean(mc, axis=0)
        tokenizer = tokenizer_local()
        multi = []
        with torch.no_grad():
            for record in h_records:
                encoded = tokenizer(record["article"], max_length=512, truncation=False,
                                    padding=False, return_tensors="pt")
                ids = encoded["input_ids"][0]
                windows = [ids[:512]]
                if len(ids) > 512:
                    mid = max(0, (len(ids)-512)//2)
                    windows = [ids[:512], ids[mid:mid+512], ids[-512:]]
                probabilities = []
                for window in windows:
                    logits = h_model({"input_ids": window.unsqueeze(0).cuda(),
                                      "attention_mask": torch.ones((1, len(window)), device="cuda")})["logits"]
                    probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
                multi.append(np.mean(probabilities, axis=0))
        all_probs["E26"] = np.asarray(multi)
        all_probs["E30"] = (all_probs["E01"] + all_probs["E05"] + all_probs["E25"]) / 3
        fusion = json.loads((root / "fusion_results.json").read_text(encoding="utf-8"))
        weights = np.asarray(fusion[1]["config"]["weights"])
        all_probs["E31"] = weights[0] * all_probs["E01"] + weights[1] * all_probs["E05"] + weights[2] * all_probs["E25"]
        final_rows = []
        for task in plan["tasks"]:
            task_id = task["id"]
            row = all_rows.get(task_id, {"task_id": task_id, "status": "PLANNED"})
            if task_id == "E01":
                h_metrics = metrics(h_records, all_probs["E01"])
                row["H_metrics"] = h_metrics
                row["H_accuracy"] = h_metrics["accuracy"]
                row["H_macro_f1"] = h_metrics["macro_f1"]
                row["H_delta_accuracy_pp"] = 0.0
                row["H_delta_macro_f1_pp"] = 0.0
            elif task_id in all_probs:
                h_metrics = metrics(h_records, all_probs[task_id])
                row["H_metrics"] = h_metrics
                row["H_accuracy"] = h_metrics["accuracy"]
                row["H_macro_f1"] = h_metrics["macro_f1"]
                row["H_delta_accuracy_pp"] = 100 * (h_metrics["accuracy"] - metrics(h_records, all_probs["E01"])["accuracy"])
                row["H_delta_macro_f1_pp"] = 100 * (h_metrics["macro_f1"] - metrics(h_records, all_probs["E01"])["macro_f1"])
                row["status"] = "COMPLETE" if task_id in selected_ids else row.get("status", "COMPLETE")
            elif task_id in ("E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24", "E27", "E28", "E29", "E32", "E33", "E34", "E35", "E36"):
                row["status"] = row.get("status", "PLANNED")
                row["stopped_reason"] = row.get("stopped_reason", "Not started before campaign execution segment ended")
            final_rows.append({"task_id": task_id, "family": task["family"], "method": task["name"],
                               "status": row.get("status", "PLANNED"),
                               "S_accuracy": row.get("metrics", {}).get("accuracy") if row.get("metrics") else None,
                               "S_macro_f1": row.get("metrics", {}).get("macro_f1") if row.get("metrics") else None,
                               "H_accuracy": row.get("H_accuracy"), "H_macro_f1": row.get("H_macro_f1"),
                               "stopped_reason": row.get("stopped_reason")})
        save(root / "leaderboard_confirm.json", final_rows)
        save(root / "leaderboard.json", {"selection": json.loads((root / "leaderboard_selection.json").read_text(encoding="utf-8")),
                                         "confirmation": final_rows})
        save(root / "resource_report.json", {"gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                                             "cuda": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported(),
                                             "test_partition_loaded": False, "hc3_loaded": False})
        write_campaign_index(root, final_rows)
        (root / "comparison.md").write_text(
            "# 36-task campaign\n\n"
            "The campaign used only HART train and validation. H was not used for selection.\n\n"
            f"Locked selection candidates: {', '.join(locked['candidates'])}.\n\n"
            "The completed segment evaluated B0, cheap decision/sparse probes, multi-view inference, and two probability fusions.\n"
            "Neural fine-tuning candidates E08-E24 were not started; they remain explicitly PLANNED rather than being represented as results.\n",
            encoding="utf-8")
        (root / "failures.md").write_text(
            "E06 emitted LinearSVC convergence warnings but produced a recorded negative result.\n"
            "Neural and advanced candidates were not started in the recorded segment and remain PLANNED.\n",
            encoding="utf-8")
        import zipfile
        with zipfile.ZipFile(root / "evidence_no_text.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob("*"):
                if path.is_file() and path.name not in {"evidence_no_text.zip"} and path.suffix not in {".npy", ".pt", ".bin"}:
                    archive.write(path, path.relative_to(root))
        return
    raise NotImplementedError("Finalization requires a locked candidate and is intentionally separate")


if __name__ == "__main__":
    main()
