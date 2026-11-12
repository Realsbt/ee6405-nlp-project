"""Verify unknown-token offsets in separate artifacts; never alter frozen caches."""

import argparse
import json
from pathlib import Path

import razdel

from parse_nfkc_cpu import compact
from replay_hc3_cpu import ROOT, tokenizer_local, torch
from same_window_cache import atomic_json, checked_graph, parser_on_cpu
from screen_structure_cpu import digest, rows
from experiments.rst_alignment import tree_leaves
from experiments.rst_worker import tree_dict


def token_offset_spans(contents, pieces, offsets, original_indices, raw):
    if len(pieces) != len(offsets) or "<unk>" not in pieces:
        raise ValueError("Expected actual unknown token and matching offset inventory")
    tokens, cursor = [], 0
    for piece, (a, b) in zip(pieces, offsets):
        text = compact(piece.replace("\u2581", " "))
        if not text:
            continue
        if not 0 <= a < b <= len(original_indices):
            raise ValueError("Invalid tokenizer offset")
        indices = [i for i in original_indices[a:b] if i is not None]
        if not indices:
            raise ValueError("Nonempty piece has no original characters")
        tokens.append((cursor, cursor + len(text), indices[0], indices[-1] + 1, text))
        cursor += len(text)
    if "".join(compact(s) for s in contents) != "".join(t[4] for t in tokens):
        raise ValueError("Leaf strings differ from replayed actual tokenizer pieces")
    boundaries = {t[1]: i + 1 for i, t in enumerate(tokens)}
    cursor, start_token, spans = 0, 0, []
    for content in contents:
        size = len(compact(content))
        if not size:
            raise ValueError("Empty EDU")
        cursor += size
        if cursor not in boundaries:
            raise ValueError("EDU boundary splits a tokenizer piece")
        stop_token = boundaries[cursor]
        selected = tokens[start_token:stop_token]
        if not selected:
            raise ValueError("Empty token interval")
        start, end = min(t[2] for t in selected), max(t[3] for t in selected)
        if spans and spans[-1][1] > start:
            raise ValueError("EDU boundary splits overlapping tokenizer offsets")
        spans.append((start, end))
        start_token = stop_token
    covered = {i for start, end in spans for i in range(start, end)}
    if any(not c.isspace() and i not in covered for i, c in enumerate(raw)):
        raise ValueError("Original nonspace characters omitted")
    return spans


def remap_from_tokens(predictor, unit, raw):
    words = list(razdel.tokenize(raw))
    joined, original_indices = [], []
    for word in words:
        if joined:
            joined.append(" ")
            original_indices.append(None)
        joined.append(word.text)
        original_indices.extend(range(word.start, word.stop))
    joined = "".join(joined)
    encoded = predictor.tokenizer(joined, add_special_tokens=False, return_offsets_mapping=True)
    pieces = predictor.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    leaves = tree_leaves(unit)
    spans = token_offset_spans([leaf.text for leaf in leaves], pieces, encoded["offset_mapping"],
                              original_indices, raw)
    for leaf, (start, end) in zip(leaves, spans):
        leaf.start, leaf.end, leaf.text = start, end, raw[start:end]

    def update(node):
        children = [c for c in (getattr(node, "left", None), getattr(node, "right", None)) if c is not None]
        for child in children:
            update(child)
        if children:
            node.start, node.end = children[0].start, children[-1].end
            node.text = raw[node.start:node.end]

    update(unit)
    return pieces.count("<unk>")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    failed = {r["item_id"] for r in rows(args.cache / "failures.jsonl")}
    selected = [r for r in rows(args.cache / "inputs.jsonl")
                if r["item_id"] in failed and r["source_split"] in ("train", "val")]
    canonical = {s.split("_")[0].casefold(): s.split("_")[0] for s in
                 (ROOT / "resources/RACE/utils/relation_table.txt").read_text().splitlines() if s}
    reports, tokenizer = [], tokenizer_local()
    torch.set_num_threads(2)
    with parser_on_cpu() as parser:
        from isanlp_rst.base_predictor import BasePredictor
        verified = BasePredictor.remap_tree_offsets
        for row in selected:
            used = []

            def remap(self, unit, positions, originals, original_text):
                try:
                    return verified(self, unit, positions, originals, original_text)
                except ValueError:
                    used.append(remap_from_tokens(self, unit, original_text))

            BasePredictor.remap_tree_offsets = remap
            try:
                result = parser(row["article"])
                if len(result.get("rst", [])) != 1:
                    raise ValueError("Expected one tree")
                parsed = {**row, "rst_structure": tree_dict(result["rst"][0], row["article"], canonical),
                          "alignment_mode": "verified_parser_token_offsets"}
                checked_graph(parsed, tokenizer)
                with (args.output / "parsed.jsonl").open("a") as stream:
                    stream.write(json.dumps(parsed) + "\n")
                reports.append({"item_id": row["item_id"], "status": "passed",
                                "unknown_tokens": sum(used)})
            except Exception as error:
                reports.append({"item_id": row["item_id"], "status": "failed", "error": str(error)})
            finally:
                BasePredictor.remap_tree_offsets = verified
    report = {"records": reports, "original_cache_unchanged": True,
              "scope": "Separate train/dev repair verification, not integrated or complete full cache",
              "cache_protocol_sha256": digest(args.cache / "protocol.json"),
              "adapter_sha256": digest(__file__)}
    if (args.output / "parsed.jsonl").exists():
        report["parsed_sha256"] = digest(args.output / "parsed.jsonl")
    atomic_json(args.output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
