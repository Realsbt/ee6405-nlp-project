"""Inspect failed train/dev parses without accepting or rewriting any tree."""

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

from parse_nfkc_cpu import compact
from replay_hc3_cpu import torch
from same_window_cache import atomic_json, parser_on_cpu
from screen_structure_cpu import digest, rows
from experiments.rst_alignment import tree_leaves


def differences(original, parsed):
    a, b = compact(original), compact(parsed)
    changes = []
    for kind, i, j, k, end in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if kind != "equal":
            changes.append({"operation": kind, "original_span": [i, j], "parsed_span": [k, end],
                            "original": a[i:j], "parsed": b[k:end],
                            "original_codepoints": [f"U+{ord(c):04X}" for c in a[i:j]],
                            "parsed_codepoints": [f"U+{ord(c):04X}" for c in b[k:end]]})
    return {"nfkc_nonspace_equal": a == b, "original_characters": len(a),
            "parsed_characters": len(b), "differences": changes}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving previous diagnostics")
    completed = {r["item_id"] for r in rows(args.cache / "parsed.jsonl")}
    failures = {r["item_id"] for r in rows(args.cache / "failures.jsonl")} - completed
    selected = [r for r in rows(args.cache / "inputs.jsonl") if r["item_id"] in failures]
    reports = []
    torch.set_num_threads(2)
    with parser_on_cpu() as parser:
        from isanlp_rst.base_predictor import BasePredictor
        verified = BasePredictor.remap_tree_offsets
        for row in selected:
            captured = {}

            def inspect(self, unit, positions, originals, original_text):
                text = "".join(leaf.text for leaf in tree_leaves(unit))
                captured.update(differences(original_text, text))
                captured["raw_leaf_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                return verified(self, unit, positions, originals, original_text)

            BasePredictor.remap_tree_offsets = inspect
            try:
                parser(row["article"])
            except ValueError as error:
                captured["error"] = str(error)
            finally:
                BasePredictor.remap_tree_offsets = verified
            reports.append({"item_id": row["item_id"], "source_split": row["source_split"], **captured})
    report = {"cache_protocol_sha256": digest(args.cache / "protocol.json"),
              "diagnostics_only": True, "trees_accepted": 0, "records": reports}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
