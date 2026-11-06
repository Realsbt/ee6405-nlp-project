"""Cache the official RST-to-graph preprocessing for the frozen retained IDs.

The source RST trees cover the complete article, while lexical token features use
the first 512 tokens. That information difference is retained and reported.
"""

from collections import Counter
import json
from pathlib import Path

from experiments.common import DATA_ROOT, LABELS, file_hash, write_json
from experiments.neural import PROJECT_ROOT, UPSTREAM, import_upstream, load_records, tokenizer_local


CACHE_ROOT = DATA_ROOT / "graph_cache_v1"


def main():
    import_upstream()
    from utils.flexible_dataset import FlexibleGraphDataset
    if CACHE_ROOT.exists():
        raise FileExistsError(f"Preserving existing graph cache: {CACHE_ROOT}")
    temporary = CACHE_ROOT.with_suffix(".building")
    temporary.mkdir(parents=True)
    records = {split: load_records(split) for split in ("train", "val", "test")}
    expected = {row["item_id"]: (split, row) for split, rows in records.items() for row in rows}
    assert len(expected) == sum(map(len, records.values()))
    processor = object.__new__(FlexibleGraphDataset)
    processor.config = {"backbone_model_path": "/data/botao/ee6405-race-validation/assets/roberta-base", "max_seq_length": 512}
    processor.tokenizer = tokenizer_local()
    handles = {split: (temporary / f"{split}.jsonl").open("wb") for split in records}
    indices = {split: {} for split in records}
    counts = {split: Counter() for split in records}
    relation_counts = Counter()
    source_files = json.loads((DATA_ROOT / "protocol.json").read_text())["source_files"]
    try:
        processed = 0
        for source in source_files:
            assert file_hash(source["path"]) == source["sha256"]
            with Path(source["path"]).open() as stream:
                for line in stream:
                    raw = json.loads(line)
                    match = expected.get(raw["item_id"])
                    if match is None:
                        continue
                    split, record = match
                    assert raw["article"] == record["article"] and raw["label"] == record["label"]
                    item = processor._process_contextual_rst_item(raw)
                    assert item is not None and int(item["label"]) == record["label_id"]
                    assert len(item["root_node_ids"]) == 1 and item["edu_token_spans"]
                    for start, end in item["edu_token_spans"]:
                        assert 0 <= start < end <= len(item["full_text_input_ids"])
                    # Text fields are not read by the official model after spans
                    # are formed. Drop repeated span strings to reduce RAM/I/O.
                    for node in item["nodes"]:
                        node.pop("text", None)
                        if node["type"] == "relation_node":
                            relation_counts[node["relation"]] += 1
                    for key in ("full_text_input_ids", "full_text_attention_mask"):
                        item[key] = item[key].tolist()
                    item["label"] = int(item["label"])
                    item["domain_label"] = int(item["domain_label"])
                    n_edu = sum(node["type"] == "edu_node" for node in item["nodes"])
                    counts[split]["articles"] += 1
                    counts[split]["total_edu_nodes"] += n_edu
                    counts[split]["visible_edu_nodes"] += len(item["edu_token_spans"])
                    counts[split]["articles_with_unseen_edu_nodes"] += n_edu > len(item["edu_token_spans"])
                    payload = (json.dumps(item, ensure_ascii=False) + "\n").encode()
                    offset = handles[split].tell()
                    handles[split].write(payload)
                    indices[split][item["id"]] = [offset, len(payload)]
                    processed += 1
                    if processed % 2000 == 0:
                        print(f"Cached {processed}/{len(expected)} graphs", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    for split, rows in records.items():
        assert set(indices[split]) == {row["item_id"] for row in rows}
        write_json(temporary / f"{split}_index.json", indices[split])
    names = {line.strip().split("_")[0] for line in (UPSTREAM / "utils/relation_table.txt").read_text().splitlines() if line.strip()}
    assert set(relation_counts) <= names, set(relation_counts) - names
    edge_types = set()
    for name in names:
        for target in ("edu_node", "relation_node"):
            edge_types.add(("relation_node", name, target))
            edge_types.add((target, name, "relation_node"))
    metadata = {"edge_types": sorted(edge_types)}
    write_json(temporary / "metadata.json", metadata)
    report = {
        "status": "completed", "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"),
        "counts": {split: dict(values) for split, values in counts.items()},
        "relation_counts": dict(relation_counts), "relation_types": len(names), "edge_types": len(edge_types),
        "source_code_sha256": {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in
                               (Path(__file__), UPSTREAM / "utils/flexible_dataset.py")},
        "cache_file_sha256": {split: file_hash(temporary / f"{split}.jsonl") for split in records},
        "information_boundary": "Original full-article RST and first-512-token lexical features. Hidden EDUs remain as structure; compare untruncated subsets before attributing gains purely to graph modeling.",
    }
    write_json(temporary / "manifest.json", report)
    temporary.rename(CACHE_ROOT)
    write_json(PROJECT_ROOT / "experiments/reports/graph_cache_v1.json", report)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
