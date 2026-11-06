"""Audit the author's HART release and freeze a group-disjoint local protocol.

No model selection uses this script. Cleaning and split choices are fixed before
the first baseline: normalized-text conflicts are excluded, same-label duplicates
are collapsed, and both source groups and exact normalized-text links are blocked.
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata
import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from experiments.common import ASSETS_ROOT, DATA_ROOT, DOMAINS, LABELS, file_hash, write_json, write_jsonl


def text_hash(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


class UnionFind:
    def __init__(self, values):
        self.parents = {value: value for value in values}

    def find(self, value):
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def load_release(source):
    records = []
    for split in ("train", "val", "test"):
        path = source / f"{split}_graph.jsonl"
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                original = json.loads(line)
                article = original["article"]
                label, group = original["label"], original["group_id"]
                domain = group.split("-", 1)[0]
                if label not in LABELS or domain not in DOMAINS:
                    raise ValueError((original["item_id"], label, domain))
                if not isinstance(article, str) or not article.strip():
                    raise ValueError("Empty article")
                records.append({
                    "item_id": original["item_id"], "group_id": group, "domain": domain,
                    "label": label, "label_id": LABELS.index(label), "article": article,
                    "model_name": original.get("model_name"), "source_dataset": original.get("source_dataset"),
                    "original_split": split, "source_file": str(path), "source_line": line_number,
                    "normalized_text_sha256": text_hash(article), "has_rst": bool(original.get("rst_structure")),
                })
    if len({row["item_id"] for row in records}) != len(records):
        raise ValueError("Duplicate item IDs")
    return records


def clean_and_group(records):
    by_text = defaultdict(list)
    for row in records:
        by_text[row["normalized_text_sha256"]].append(row)
    groups = UnionFind(row["group_id"] for row in records)
    for matches in by_text.values():
        for row in matches[1:]:
            groups.union(matches[0]["group_id"], row["group_id"])
    retained, exclusions = [], []
    for matches in by_text.values():
        conflicting = len({row["label"] for row in matches}) > 1
        eligible = sorted((row for row in matches if row["has_rst"]), key=lambda row: row["item_id"])
        keep_id = eligible[0]["item_id"] if eligible and not conflicting else None
        for row in matches:
            if row["item_id"] == keep_id:
                retained.append({**row, "split_block_id": groups.find(row["group_id"])})
            else:
                reason = "conflicting_normalized_text_labels" if conflicting else (
                    "missing_rst" if not row["has_rst"] else "same_label_normalized_duplicate")
                exclusions.append({"item_id": row["item_id"], "group_id": row["group_id"],
                                   "label": row["label"], "reason": reason,
                                   "normalized_text_sha256": row["normalized_text_sha256"],
                                   "retained_item_id": keep_id})
    return sorted(retained, key=lambda row: row["item_id"]), exclusions


def overlap_counts(partitions):
    result = {}
    for a, b in combinations(partitions, 2):
        result[f"{a}__{b}"] = {
            key: len({row[key] for row in partitions[a]} & {row[key] for row in partitions[b]})
            for key in ("group_id", "normalized_text_sha256")
        }
    return result


def make_partitions(records):
    strata = np.asarray([row["domain"] + "|" + row["label"] for row in records])
    groups = np.asarray([row["split_block_id"] for row in records])
    # Select fold 0 in advance. Five folds reserve ~20%; 1/8 of the remainder
    # reserves ~10% of the full corpus for validation. No fold is chosen by score.
    remaining, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(len(records)), strata, groups))
    train_in_remaining, val_in_remaining = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=43).split(
        np.zeros(len(remaining)), strata[remaining], groups[remaining]))
    indices = {"train": remaining[train_in_remaining], "val": remaining[val_in_remaining], "test": test}
    partitions = {name: [records[i] for i in rows] for name, rows in indices.items()}
    for values in overlap_counts(partitions).values():
        assert not any(values.values()), "Cross-partition source/text leakage"
    for a, b in combinations(partitions, 2):
        assert not ({r["split_block_id"] for r in partitions[a]} & {r["split_block_id"] for r in partitions[b]})
    assert sum(map(len, partitions.values())) == len(records)
    for rows in partitions.values():
        assert {row["label"] for row in rows} == set(LABELS)
        assert {row["domain"] for row in rows} == set(DOMAINS)
    return partitions


def describe(rows):
    return {"records": len(rows), "source_groups": len({r["group_id"] for r in rows}),
            "labels": dict(Counter(r["label"] for r in rows)), "domains": dict(Counter(r["domain"] for r in rows)),
            "domain_label_counts": {domain: dict(Counter(r["label"] for r in rows if r["domain"] == domain)) for domain in DOMAINS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSETS_ROOT / "hart_official")
    parser.add_argument("--output", type=Path, default=DATA_ROOT)
    parser.add_argument("--report", type=Path, default=Path("experiments/reports/data_v1.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Frozen dataset already exists: {args.output}")
    manifest = json.loads((args.source / "download_manifest.json").read_text())
    for file in manifest:
        assert file_hash(file["path"]) == file["sha256"], "Source file changed"
    raw = load_release(args.source)
    original = {split: [r for r in raw if r["original_split"] == split] for split in ("train", "val", "test")}
    retained, exclusions = clean_and_group(raw)
    partitions = make_partitions(retained)
    tokenizer_path = ASSETS_ROOT / "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    for rows in partitions.values():
        for start in range(0, len(rows), 128):
            chunk = rows[start:start + 128]
            tokens = tokenizer([row["article"] for row in chunk], truncation=True, max_length=512,
                               return_offsets_mapping=True, return_attention_mask=False)
            for row, offsets, input_ids in zip(chunk, tokens["offset_mapping"], tokens["input_ids"]):
                text_end = max(end for begin, end in offsets if end > begin)
                row["model_text"] = row["article"][:text_end]
                row["input_token_count_with_specials"] = len(input_ids)
                row["input_truncated"] = text_end < len(row["article"].rstrip())
                assert row["model_text"].strip()
    args.output.mkdir(parents=True)
    for name, rows in partitions.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    write_jsonl(args.output / "exclusions.jsonl", exclusions)
    report = {
        "protocol_id": "hart_grouped_v1", "source": "RACE author's processed English HART four-class release",
        "raw": describe(raw), "original_partition_overlap": overlap_counts(original),
        "retained": describe(retained), "exclusions": dict(Counter(row["reason"] for row in exclusions)),
        "cleaning": {"normalization": "NFKC + collapse whitespace + casefold",
                     "conflicts": "Exclude all records sharing normalized text with different labels",
                     "duplicates": "Keep lexicographically first eligible item ID per same-label normalized text",
                     "comparison_pool": "Require nonempty RST for every model's common sample pool"},
        "split": {"algorithm": "StratifiedGroupKFold; first fold; strata domain|label; group source/duplicate connected component",
                  "test_n_splits": 5, "test_seed": 42, "validation_n_splits": 8, "validation_seed": 43,
                  "target_train_val_test": [0.7, 0.1, 0.2], "selection_used_model_scores": False},
        "partitions": {name: {**describe(rows), "split_blocks": len({r["split_block_id"] for r in rows}),
                              "truncated_at_512_tokens": sum(r["input_truncated"] for r in rows),
                              "fraction": len(rows) / len(retained), "sha256": file_hash(args.output / f"{name}.jsonl")}
                       for name, rows in partitions.items()},
        "new_partition_overlap": overlap_counts(partitions),
        "lexical_input": "TF-IDF uses original character prefix covered by the first 512 RoBERTa tokens including specials",
        "tokenizer_revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "source_files": manifest, "prepare_script_sha256": file_hash(__file__),
        "data_root": str(args.output),
        "checkpoint_boundary": "Author fine-tuned checkpoint is for smoke checks only; it may have seen the new test groups and must not be used for an independent evaluation on this split.",
        "remaining_limitations": ["Only one benchmark with four domains", "No semantic near-duplicate detection beyond source IDs and normalized text", "In-domain grouped holdout; not unseen-domain or unseen-generator validation"],
    }
    write_json(args.output / "protocol.json", report)
    write_json(args.report, report)
    print(json.dumps({"raw": len(raw), "retained": len(retained), "exclusions": report["exclusions"],
                      "partitions": {name: len(rows) for name, rows in partitions.items()},
                      "overlap": report["new_partition_overlap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
