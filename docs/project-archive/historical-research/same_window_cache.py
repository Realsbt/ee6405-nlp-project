"""Resumable, bounded CPU construction of the full same-window input/graph cache."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from replay_hc3_cpu import ROOT, torch, tokenizer_local
from prepare_same_window_cpu import visible_prefix
from complete_external_cpu import check_parsed
from parse_nfkc_cpu import FALLBACKS, remap_verified
from screen_structure_cpu import digest, rows, write
from experiments.raw_graph import RawGraphDataset
from experiments.rst_worker import tree_dict

DEFAULT = ROOT / "resources/experiments/same_window_full_v2"
TOKEN_OFFSET_FALLBACKS = []


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_json(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def parser_on_cpu(unknown_offsets=False):
    from transformers import PreTrainedModel
    from isanlp_rst.parser import Parser
    from isanlp_rst.base_predictor import BasePredictor
    load, resize, align = torch.load, PreTrainedModel.resize_token_embeddings, BasePredictor.remap_tree_offsets

    def mapped(path, *args, **kwargs):
        if str(path).endswith("best_weights.pt"):
            kwargs["mmap"] = True
        return load(path, *args, **kwargs)

    def resized(model, *args, **kwargs):
        return resize(model, *args, **{**kwargs, "mean_resizing": False})

    try:
        torch.load = mapped
        PreTrainedModel.resize_token_embeddings = resized
        parser = Parser(model_dir=str(ROOT / "resources/assets/isanlp_rst_local"),
                        hf_model_version="rstdt", cuda_device=-1)
        def remap(self, unit, positions, originals, original_text):
            try:
                return remap_verified(self, unit, positions, originals, original_text)
            except ValueError:
                if not unknown_offsets:
                    raise
                from repair_window_unknowns import remap_from_tokens
                count = remap_from_tokens(self, unit, original_text)
                TOKEN_OFFSET_FALLBACKS.append(count)

        BasePredictor.remap_tree_offsets = remap
        yield parser
    finally:
        torch.load, PreTrainedModel.resize_token_embeddings = load, resize
        BasePredictor.remap_tree_offsets = align


def initialize(output, tokenizer):
    protocol_path = output / "protocol.json"
    scripts = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / "parse_nfkc_cpu.py", ROOT / "prepare_same_window_cpu.py",
        ROOT / "repair_window_unknowns.py",
        ROOT / "repository/experiments/rst_worker.py", ROOT / "repository/experiments/rst_alignment.py",
        ROOT / "resources/RACE/utils/flexible_dataset.py",
        ROOT / "resources/assets/roberta-base/tokenizer.json",
        ROOT / "resources/assets/isanlp_rst_local/best_weights.pt",
        ROOT / "resources/assets/isanlp_rst_local/config.json",
    )}
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["code_sha256"] != scripts or digest(output / "inputs.jsonl") != protocol["inputs_sha256"]:
            raise ValueError("Cache code/input changed; preserve this version and choose a new output")
        return rows(output / "inputs.jsonl"), protocol
    output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "resources/experiments/hart_grouped_v1"
    source_protocol = json.loads((data / "protocol.json").read_text())
    all_rows, blocks, hashes = [], {}, {}
    for split in ("train", "val", "test"):
        path = data / f"{split}.jsonl"
        hashes[split] = digest(path)
        if hashes[split] != source_protocol["partitions"][split]["sha256"]:
            raise ValueError("Frozen source partition changed")
        records = rows(path)
        blocks[split] = {r["split_block_id"] for r in records}
        for r in records:
            text, ids = visible_prefix(r["article"], tokenizer)
            all_rows.append({
                "item_id": r["item_id"], "group_id": r["group_id"], "split_block_id": r["split_block_id"],
                "domain": r["domain"], "label": r["label"], "label_id": r["label_id"],
                "source_split": split, "article": text, "visible_token_ids": ids,
                "original_article_sha256": hashlib.sha256(r["article"].encode()).hexdigest(),
                "visible_article_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_count_with_specials": len(ids),
            })
        print(f"Prepared {split}: {len(records)} visible inputs", flush=True)
    if (blocks["train"] & blocks["val"] or blocks["train"] & blocks["test"]
            or blocks["val"] & blocks["test"]):
        raise ValueError("Source-block overlap")
    if len({r["item_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate IDs")
    with (output / "inputs.jsonl").open("x") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    prefix_splits = {}
    for r in all_rows:
        prefix_splits.setdefault(r["visible_article_sha256"], set()).add(r["source_split"])
    protocol = {
        "date": "2026-11-09", "inputs_sha256": digest(output / "inputs.jsonl"),
        "source_partition_sha256": hashes, "code_sha256": scripts,
        "offset_alignment": "strict, verified NFKC, then replayed actual tokenizer offsets for unknowns; unchanged original text",
        "text_budget": "512 tokens including specials; reparse the exact visible prefix",
        "counts": dict(Counter(r["source_split"] for r in all_rows)),
        "failure_policy": "Record failures; no silent exclusion; full training blocked until all records verified.",
        "test_scope": "Test input preprocessing only, no model fitting, evaluation or test-based selection.",
        "cross_split_exact_visible_duplicates": sum(len(splits) > 1 for splits in prefix_splits.values()),
    }
    write(protocol_path, protocol)
    return all_rows, protocol


def checked_graph(row, tokenizer):
    check_parsed([row], [row])
    item = RawGraphDataset([row], tokenizer)[0]
    if item["full_text_input_ids"].tolist() != row["visible_token_ids"]:
        raise ValueError("Lexical tokens differ from visible input")
    n_edu = sum(n["type"] == "edu_node" for n in item["nodes"])
    if n_edu != len(item["edu_token_spans"]):
        raise ValueError("Invisible EDU nodes")
    if any(not 0 <= a < b <= len(row["visible_token_ids"]) for a, b in item["edu_token_spans"]):
        raise ValueError("Invalid EDU token spans")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            item[key] = value.tolist()
    return item


def promote(output, source, repairs, records, tokenizer):
    old = json.loads((source / "protocol.json").read_text())
    if digest(source / "inputs.jsonl") != old["inputs_sha256"] or digest(output / "inputs.jsonl") != old["inputs_sha256"]:
        raise ValueError("Promotion requires identical verified visible inputs")
    if digest(source / "same_window_cache.source.py") != old["code_sha256"]["same_window_cache.py"]:
        raise ValueError("Missing or changed frozen source snapshot")
    for name, sha in old["code_sha256"].items():
        if name == "same_window_cache.py":
            continue
        if digest(ROOT / name) != sha:
            snapshot = source / "dependency_snapshots" / sha
            if not snapshot.exists() or digest(snapshot) != sha:
                raise ValueError("Original cache dependency changed without a frozen snapshot")
    expected, done = {r["item_id"]: r for r in records}, set()
    for row in rows(source / "parsed.jsonl"):
        if row["item_id"] in done or any(row.get(k) != v for k, v in expected[row["item_id"]].items()):
            raise ValueError("Promoted parse differs from expected input or is duplicated")
        checked_graph(row, tokenizer)
        append_json(output / "parsed.jsonl", row)
        done.add(row["item_id"])
    original_successes = len(done)
    history = rows(source / "failures.jsonl") if (source / "failures.jsonl").exists() else []
    for failure in history:
        append_json(output / "failures.jsonl", failure)
    repair_count = 0
    if repairs:
        audit = json.loads((repairs / "verification.json").read_text())
        if (audit["cache_protocol_sha256"] != digest(source / "protocol.json")
                or audit["adapter_sha256"] != digest(ROOT / "repair_window_unknowns.py")
                or audit["parsed_sha256"] != digest(repairs / "parsed.jsonl")):
            raise ValueError("Repair provenance changed")
        allowed = {r["item_id"] for r in audit["records"] if r["status"] == "passed"}
        failed_ids = {r["item_id"] for r in history}
        for row in rows(repairs / "parsed.jsonl"):
            key = row["item_id"]
            if key in done or key not in allowed or key not in failed_ids:
                raise ValueError("Unexpected or duplicate repaired record")
            if any(row.get(k) != v for k, v in expected[key].items()):
                raise ValueError("Repaired source metadata changed")
            checked_graph(row, tokenizer)
            append_json(output / "parsed.jsonl", row)
            done.add(key)
            repair_count += 1
        if repair_count != len(allowed):
            raise ValueError("Missing repaired record")
    atomic_json(output / "promotion.json", {
        "source": str(source.resolve()), "source_protocol_sha256": digest(source / "protocol.json"),
        "source_parsed_sha256": digest(source / "parsed.jsonl"), "source_records": original_successes,
        "repairs": str(repairs.resolve()) if repairs else None, "repaired_records": repair_count,
        "repair_verification_sha256": digest(repairs / "verification.json") if repairs else None,
        "source_preserved": True, "failures_history_preserved": len(history),
    })


def compile_graphs(output, expected, tokenizer):
    """Rebuild a bounded-memory indexed graph cache from verified durable parses."""
    index, counts, unchanged = {}, Counter(), Counter()
    temporary = output / "graphs.building"
    with temporary.open("wb") as stream, (output / "parsed.jsonl").open() as raw:
        for line in raw:
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id in index or any(row.get(k) != v for k, v in expected[item_id].items()):
                raise ValueError("Duplicate parse or changed source metadata")
            item = checked_graph(row, tokenizer)
            payload = (json.dumps(item) + "\n").encode()
            index[item_id] = [stream.tell(), len(payload)]
            stream.write(payload)
            counts[row["source_split"]] += 1
            relations = {n["relation"] for n in item["nodes"] if n["type"] == "relation_node"}
            unchanged["single_or_zero_relation_graphs"] += len(relations) <= 1
    os.replace(temporary, output / "graphs.jsonl")
    atomic_json(output / "index.json", index)
    return counts, unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--max-records", type=int, default=256,
                        help="Bound this invocation; 0 explicitly processes all remaining records")
    parser.add_argument("--retry-failures", action="store_true", help="Retry recorded unresolved failures; retain the history")
    parser.add_argument("--promote-from", type=Path, help="Explicitly migrate a frozen previous version into a new output")
    parser.add_argument("--repairs", type=Path, help="Verified separate repair output; requires --promote-from")
    parser.add_argument("--compile-only", action="store_true", help="Compile existing/promoted parses without parsing remaining records")
    args = parser.parse_args()
    if args.max_records < 0:
        parser.error("--max-records must be >= 0")
    if args.repairs and not args.promote_from:
        parser.error("--repairs requires --promote-from")
    if args.promote_from and args.output.exists():
        parser.error("Promotion requires a new output; preserve the old version")
    torch.set_num_threads(2)
    tokenizer = tokenizer_local()
    records, protocol = initialize(args.output, tokenizer)
    if args.promote_from:
        promote(args.output, args.promote_from, args.repairs, records, tokenizer)
    expected = {r["item_id"]: r for r in records}
    parsed_path, failures_path = args.output / "parsed.jsonl", args.output / "failures.jsonl"
    def ids(path):
        if not path.exists():
            return set()
        with path.open() as stream:
            return {json.loads(line)["item_id"] for line in stream}
    done = ids(parsed_path)
    failed = ids(failures_path) - done
    if not done and not failed:
        seed = ROOT / "resources/experiments/same_window_smoke_nfkc_v2"
        audit = json.loads((seed / "parsed.audit.json").read_text())
        if digest(seed / "parsed.jsonl") != audit["output_sha256"]:
            raise ValueError("Seed smoke parse changed")
        for row in rows(seed / "parsed.jsonl"):
            item_id = row["item_id"]
            if any(row.get(k) != v for k, v in expected[item_id].items()):
                raise ValueError("Seed/input mismatch")
            checked_graph(row, tokenizer)
            append_json(parsed_path, row)
            done.add(item_id)
    pending = [r for r in records if r["item_id"] not in done
               and (args.retry_failures or r["item_id"] not in failed)]
    if args.max_records:
        pending = pending[:args.max_records]
    if args.compile_only:
        pending = []
    relation_names = {line.split("_")[0] for line in
                      (ROOT / "resources/RACE/utils/relation_table.txt").read_text().splitlines() if line.strip()}
    canonical = {name.casefold(): name for name in relation_names}
    started = time.perf_counter()
    if pending:
        with parser_on_cpu(unknown_offsets=True) as rst:
            for i, row in enumerate(pending):
                try:
                    n_fallbacks = len(FALLBACKS)
                    n_token_fallbacks = len(TOKEN_OFFSET_FALLBACKS)
                    result = rst(row["article"])
                    if len(result.get("rst", [])) != 1:
                        raise ValueError("Expected exactly one RST tree")
                    parsed = {**row, "rst_structure": tree_dict(result["rst"][0], row["article"], canonical),
                              "alignment_mode": ("verified_parser_token_offsets" if len(TOKEN_OFFSET_FALLBACKS) > n_token_fallbacks
                                                 else "verified_nfkc" if len(FALLBACKS) > n_fallbacks else "strict_exact")}
                    checked_graph(parsed, tokenizer)
                except Exception as error:
                    append_json(failures_path, {
                        "item_id": row["item_id"], "source_split": row["source_split"],
                        "error_type": type(error).__name__, "error": str(error),
                        "visible_article_sha256": row["visible_article_sha256"],
                    })
                    failed.add(row["item_id"])
                    print(f"Recorded failure: {row['item_id']}: {error}", flush=True)
                else:
                    append_json(parsed_path, parsed)
                    done.add(row["item_id"])
                    failed.discard(row["item_id"])
                if (i + 1) % 25 == 0 or i + 1 == len(pending):
                    print(f"Attempted {i+1}/{len(pending)} this run; total verified={len(done)} failed={len(failed)}", flush=True)
    counts, graph_stats = compile_graphs(args.output, expected, tokenizer)
    remaining = len(records) - len(done) - len(failed)
    status = ("ready" if not remaining and not failed
              and not protocol["cross_split_exact_visible_duplicates"] else "partial")
    atomic_json(args.output / "manifest.json", {
        "status": status, "verified": len(done), "failed": len(failed), "pending": remaining,
        "counts": dict(counts), "expected_counts": protocol["counts"], "graph_stats": dict(graph_stats),
        "source_blocks_disjoint": True, "invisible_edu_nodes": 0,
        "protocol_sha256": digest(args.output / "protocol.json"),
        "parsed_sha256": digest(parsed_path), "graphs_sha256": digest(args.output / "graphs.jsonl"),
        "index_sha256": digest(args.output / "index.json"),
        "invocation_seconds": time.perf_counter() - started,
        "full_cache_ready": status == "ready",
    })
    print(json.dumps(json.loads((args.output / "manifest.json").read_text())), flush=True)


if __name__ == "__main__":
    main()
