"""Build a train/dev-only same-window parser smoke cache, not full training data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_hc3_cpu import ROOT, tokenizer_local
from screen_structure_cpu import digest, rows, write
from complete_external_cpu import check_parsed
from experiments.raw_graph import RawGraphDataset
from experiments.rst_worker import leaves


def visible_prefix(text, tokenizer, limit=512):
    encoded = tokenizer(text, max_length=limit, truncation=True, return_offsets_mapping=True)
    ends = sorted({end for start, end in encoded["offset_mapping"] if end > start})
    if not ends:
        raise ValueError("No visible non-special tokens")
    # Retokenization at a prefix boundary can change the last subword.
    for end in reversed(ends):
        visible = text[:end]
        ids = tokenizer(visible, truncation=False)["input_ids"]
        if len(ids) <= limit:
            return visible, ids
    raise ValueError("Could not form a bounded visible prefix")


def select_cells(records, count):
    if count < 1:
        raise ValueError("Need positive samples per domain/class cell")
    cells = {}
    for row in sorted(records, key=lambda r: r["item_id"]):
        cells.setdefault((row["domain"], row["label_id"]), []).append(row)
    if len(cells) != 16 or any(len(cell) < count for cell in cells.values()):
        raise ValueError("Need all four domains and four classes with sufficient support")
    return [row for key in sorted(cells) for row in cells[key][:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--nfkc-alignment", action="store_true",
                        help="Opt in to verified Unicode-equivalent offsets; original text is not changed")
    parser.add_argument("--output", type=Path, default=ROOT / "resources/experiments/same_window_smoke_v1")
    args = parser.parse_args()
    args.output.mkdir()
    data = ROOT / "resources/experiments/hart_grouped_v1"
    original_protocol = json.loads((data / "protocol.json").read_text())
    tokenizer = tokenizer_local()
    selected, hashes, groups = [], {}, {}
    for split in ("train", "val"):
        path = data / f"{split}.jsonl"
        hashes[split] = digest(path)
        if hashes[split] != original_protocol["partitions"][split]["sha256"]:
            raise ValueError("Source partition changed")
        current = select_cells(rows(path), args.per_cell)
        groups[split] = {r["split_block_id"] for r in current}
        for row in current:
            visible, ids = visible_prefix(row["article"], tokenizer)
            selected.append({
                "item_id": row["item_id"], "group_id": row["group_id"],
                "split_block_id": row["split_block_id"], "source_split": split,
                "domain": row["domain"], "label": row["label"], "label_id": row["label_id"],
                "article": visible, "original_article_sha256": hashlib.sha256(row["article"].encode()).hexdigest(),
                "visible_article_sha256": hashlib.sha256(visible.encode()).hexdigest(),
                "visible_token_ids": ids, "token_count_with_specials": len(ids),
            })
    if groups["train"] & groups["val"]:
        raise ValueError("Train/development source overlap")
    input_path = args.output / "inputs.jsonl"
    with input_path.open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    write(args.output / "protocol.json", {
        "date": "2026-11-09", "scope": "Train/dev-only parser and token alignment smoke; no training or test evaluation.",
        "max_tokens_with_specials": 512, "samples_per_domain_class_per_split": args.per_cell,
        "selection": "Lexicographic item_id within each domain/class; no model predictions used.",
        "graph": "Fresh parse of EXACT visible article, not pruning a full-text tree.",
        "input_sha256": digest(input_path), "source_partition_sha256": hashes,
        "script_sha256": digest(__file__), "full_cache_ready": False,
        "offset_alignment": "strict_then_verified_nfkc" if args.nfkc_alignment else "strict_exact",
    })
    env = dict(os.environ, PYTHONPATH=str(ROOT / "repository"), PYTHONDONTWRITEBYTECODE="1")
    parsed_path = args.output / "parsed.jsonl"
    entry = [str(ROOT / "parse_nfkc_cpu.py")] if args.nfkc_alignment else ["-m", "experiments.rst_worker"]
    with (args.output / "parser.log").open("x") as log:
        subprocess.run([
            sys.executable, "-B", *entry, "--cuda-device", "-1",
            "--input", str(input_path), "--output", str(parsed_path),
        ], env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    parsed = rows(parsed_path)
    coverage = check_parsed(selected, parsed)
    dataset = RawGraphDataset(parsed, tokenizer)
    invisible = 0
    for row, item in zip(parsed, dataset):
        ids = item["full_text_input_ids"].tolist()
        if ids != row["visible_token_ids"] or len(ids) > 512:
            raise ValueError("Model and parser visible text/token mismatch")
        spans = item["edu_token_spans"]
        if not spans or any(not 0 <= start < end <= len(ids) for start, end in spans):
            raise ValueError("Invalid visible EDU token span")
        invisible += len(leaves(row["rst_structure"])) - len(spans)
    if invisible:
        raise ValueError("Some EDU nodes are not visible in the lexical window")
    write(args.output / "verification.json", {
        **coverage, "all_model_tokens_equal_visible_prefix": True, "invisible_edu_nodes": invisible,
        "source_blocks_disjoint": True, "parsed_sha256": digest(parsed_path),
        "protocol_sha256": digest(args.output / "protocol.json"),
        "n_train": sum(r["source_split"] == "train" for r in parsed),
        "n_development": sum(r["source_split"] == "val" for r in parsed),
        "full_cache_ready": False, "training_control_implementations_ready": False,
    })
    print(json.dumps(coverage), flush=True)


if __name__ == "__main__":
    main()
