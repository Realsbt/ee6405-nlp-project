"""Raw English text -> JSON RST trees, executed in the separate RST environment.

Use CPU while neural training occupies the GPU. The worker preserves original
text and returns character spans suitable for the official RACE preprocessing.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from experiments.rst_alignment import exact_tree_alignment


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path("/data/botao/ee6405-race-validation/assets")


def tree_dict(node, text, relation_names):
    left, right = getattr(node, "left", None), getattr(node, "right", None)
    relation = getattr(node, "relation", None)
    if left is None and right is None:
        relation = "elementary"
    else:
        relation = relation_names.get(str(relation).casefold(), relation)
        if relation not in relation_names.values():
            raise ValueError(f"Unsupported relation: {relation}")
    start, end = int(node.start), int(node.end)
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Invalid parser character span {start}:{end} for {len(text)} chars")
    return {"id": getattr(node, "id", None), "relation": relation,
            "nuclearity": getattr(node, "nuclearity", None), "start": start, "end": end,
            "text": text[start:end],
            "left": tree_dict(left, text, relation_names) if left is not None else None,
            "right": tree_dict(right, text, relation_names) if right is not None else None}


def leaves(tree):
    if tree["left"] is None and tree["right"] is None:
        return [tree]
    return (leaves(tree["left"]) if tree["left"] else []) + (leaves(tree["right"]) if tree["right"] else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing parsed output: {args.output}")
    if args.smoke:
        records = [
            {"item_id": "raw_smoke_single", "article": "Clear explanations help readers understand scientific results."},
            {"item_id": "raw_smoke_multiple", "article": "An author first develops an argument. An editor may then improve its wording, while the main logical structure remains. However, extensive rewriting can also change the argument itself."},
        ]
    elif args.input:
        records = [json.loads(line) for line in args.input.open() if line.strip()]
    else:
        raise ValueError("Provide --input or --smoke")
    assert len({r["item_id"] for r in records}) == len(records)
    assert all(isinstance(r.get("article"), str) and r["article"].strip() for r in records)
    import torch
    from transformers import PreTrainedModel
    from isanlp_rst.parser import Parser
    torch.set_num_threads(2)
    original_load = torch.load
    original_resize = PreTrainedModel.resize_token_embeddings
    def mapped_load(path, *values, **kwargs):
        if args.cuda_device < 0 and str(path).endswith("best_weights.pt"):
            kwargs["mmap"] = True
        return original_load(path, *values, **kwargs)
    def resize_without_unused_mean(self, *values, **kwargs):
        # The parser immediately strictly loads every trained parameter. New-row
        # initialization is discarded; avoid a large temporary covariance matrix.
        kwargs["mean_resizing"] = False
        return original_resize(self, *values, **kwargs)
    load_started = time.perf_counter()
    try:
        torch.load = mapped_load
        PreTrainedModel.resize_token_embeddings = resize_without_unused_mean
        rst = Parser(model_dir=str(ASSETS / "isanlp_rst_local"), hf_model_version="rstdt", cuda_device=args.cuda_device)
    finally:
        torch.load = original_load
        PreTrainedModel.resize_token_embeddings = original_resize
    load_seconds = time.perf_counter() - load_started
    names = {line.split("_")[0] for line in (ROOT / "local_validation/upstream/RACE/utils/relation_table.txt").read_text().splitlines() if line.strip()}
    relation_names = {name.casefold(): name for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timings, edu_counts = [], []
    with args.output.open("w") as stream:
        for index, record in enumerate(records):
            text = record["article"]
            started = time.perf_counter()
            with exact_tree_alignment():
                output = rst(text)
            if not output.get("rst") or len(output["rst"]) != 1:
                raise ValueError("Expected one parsed discourse tree")
            tree = tree_dict(output["rst"][0], text, relation_names)
            units = leaves(tree)
            assert units and all(a["end"] <= b["start"] for a, b in zip(units, units[1:]))
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            edu_counts.append(len(units))
            stream.write(json.dumps({**record, "rst_structure": tree, "rst_parse_seconds": elapsed,
                                     "article_sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(records) - 1:
                print(json.dumps({"parsed": index + 1, "total": len(records), "seconds": elapsed, "edus": len(units)}), flush=True)
    report = {"status": "passed", "records": len(records), "cuda_device": args.cuda_device,
              "model_load_seconds": load_seconds, "parse_seconds": timings, "edu_counts": edu_counts,
              "character_spans_valid_and_ordered": True, "canonical_relation_inventory": sorted(names),
              "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "parser_revision": "5141450ac3eb7decad0b8735fdbf5ec6928ec97a",
              "isanlp_git_commit": "2a102e59f9718acc7fe259dd8d83c66d5da39794",
              "packages": {name: importlib.metadata.version(name) for name in ["isanlp-rst", "isanlp", "torch", "transformers", "tokenizers"]},
              "cpu_loading_adaptations": "mmap checkpoint; skip unused mean initialization before complete strict state load",
              "character_offset_adapter": "Exact decoded-EDU to original non-whitespace alignment; mismatches fail; predicted segmentation/relations unchanged",
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), ROOT / "experiments/rst_alignment.py"]},
              "scope": "Raw-text RST parsing only; not yet a final trained RACE end-to-end prediction"}
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
