"""Post-hoc paired diagnostics, not causal ablations or model selection."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score

from analyze_saved_probabilities import LABELS

ROOT = Path(__file__).resolve().parent


def read_index(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {row["item_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate item_id in {path}")
    return indexed


def compare(rows, text, graph):
    ids = [row["item_id"] for row in rows]
    if not ids:
        raise ValueError("Empty slice")
    for row in rows:
        for predictions in (text, graph):
            p = predictions[row["item_id"]]
            if p["true_label"] != row["label"] or p["domain"] != row["domain"]:
                raise ValueError("Prediction metadata mismatch")
    y = [row["label"] for row in rows]
    a = [text[i]["prediction"] for i in ids]
    b = [graph[i]["prediction"] for i in ids]
    if not set(y + a + b).issubset(LABELS):
        raise ValueError("Unknown class")
    wins = sum(v == t and u != t for t, u, v in zip(y, a, b))
    losses = sum(u == t and v != t for t, u, v in zip(y, a, b))
    result = {
        "n": len(ids),
        "source_blocks": len({r["split_block_id"] for r in rows}),
        "class_counts": dict(Counter(y)),
        "race_only_correct": wins, "roberta_only_correct": losses,
        "both_wrong": sum(u != t and v != t for t, u, v in zip(y, a, b)),
        "roberta_accuracy": sum(u == t for t, u in zip(y, a)) / len(y),
        "race_accuracy": sum(v == t for t, v in zip(y, b)) / len(y),
        "accuracy_delta_pp": 100 * (wins - losses) / len(y),
        "confusion_roberta": dict(Counter(f"{t}->{u}" for t, u in zip(y, a))),
        "confusion_race": dict(Counter(f"{t}->{v}" for t, v in zip(y, b))),
    }
    # Single-class slices use recall (= accuracy here), not misleading macro-F1.
    if set(y) == set(LABELS):
        result["macro_f1_delta_pp"] = 100 * (
            f1_score(y, b, labels=LABELS, average="macro")
            - f1_score(y, a, labels=LABELS, average="macro"))
    return result


def main():
    test_path = ROOT / "resources/experiments/hart_grouped_v1/test.jsonl"
    provenance_path = ROOT / "hart_provenance_v1/provenance.jsonl"
    quality_path = ROOT / "repository/experiments/reports/released_rst_item_quality.jsonl"
    data = read_index(test_path)
    provenance = read_index(provenance_path)
    quality = read_index(quality_path)
    slices = {"all": list(data.values())}
    for row in data.values():
        item = row["item_id"]
        gap = quality[item]["uncovered_nonspace_in_lexical_window"]
        categories = [
            f"domain:{row['domain']}", f"class:{row['label']}",
            f"truncated:{row['input_truncated']}",
            f"domain_class:{row['domain']}:{row['label']}",
            f"window_span_gap:{'unknown' if gap is None else gap > 0}",
        ]
        if row["label"] == "ai_humanized":
            actors = {c["language_source"] for c in provenance[item]["candidates"]}
            actor = next(iter(actors)) if len(actors) == 1 else "ambiguous"
            categories.append(f"humanized_actor:{actor}")
        for category in categories:
            slices.setdefault(category, []).append(row)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (test_path, provenance_path, quality_path)}
    comparisons = {}
    for loss, a, b in (("CE", "roberta", "race_no_supcon"),
                       ("SupCon", "roberta_supcon", "race")):
        for seed in (3407, 42, 2026):
            predictions = []
            for prefix in (a, b):
                path = ROOT / f"repository/experiments/reports/{prefix}_s{seed}/test_predictions.jsonl"
                indexed = read_index(path)
                if set(indexed) != set(data):
                    raise ValueError(f"Unmatched prediction IDs: {path}")
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                predictions.append(indexed)
            comparisons[f"{loss}_s{seed}"] = {
                name: compare(rows, *predictions) for name, rows in slices.items()}
    result = {
        "scope": "Post-hoc descriptive analysis; no causal claim, no new training. "
                 "Slices overlap; seeds share test items. No multiple-testing p-values.",
        "hashes": hashes, "comparisons": comparisons,
    }
    (ROOT / "paired_gap_diagnostics_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 逐样本差距诊断：实际运行结果", "",
        "2026-11-06；RACE减RoBERTa。按同损失、同种子配对。",
        "下表均为准确率差的百分点；单类别行等于召回率差，不是Macro-F1。",
        "仅描述已见测试集，不用于调参或因果归因；切片相关且存在领域/长度混杂。",
        "", "| 切片 | 样本/来源块 | CE: 3407 / 42 / 2026 | SupCon: 3407 / 42 / 2026 |",
        "|---|---:|---:|---:|",
    ]
    for name, rows in slices.items():
        values = []
        for loss in ("CE", "SupCon"):
            values.append(" / ".join(
                f"{comparisons[f'{loss}_s{seed}'][name]['accuracy_delta_pp']:+.2f}"
                for seed in (3407, 42, 2026)))
        lines.append(f"| {name} | {len(rows)}/{len({r['split_block_id'] for r in rows})} | "
                     + " | ".join(values) + " |")
    lines += ["", "## 总体配对计数", "",
              "| 配置 | RACE独对 | RoBERTa独对 | 两者都错 | Macro-F1差，百分点 |",
              "|---|---:|---:|---:|---:|"]
    for name, slices_result in comparisons.items():
        s = slices_result["all"]
        lines.append(f"| {name} | {s['race_only_correct']} | {s['roberta_only_correct']} | "
                     f"{s['both_wrong']} | {s['macro_f1_delta_pp']:+.3f} |")
    lines += ["", "完整混淆计数、类别支持数与输入SHA256见 `paired_gap_diagnostics_v1.json`。",
              "重跑：`.venv-cpu/bin/python diagnose_model_gap.py`。"]
    (ROOT / "paired_gap_diagnostics_v1.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
