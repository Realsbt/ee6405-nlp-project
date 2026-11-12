"""Audit actual relation reassignment and reject formal use of a partial cache."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from same_window_cache import atomic_json
from window_controls import WindowDataset, config_for, import_upstream, metadata, shuffled_relations


def audit(cache, smoke=False):
    totals, splits = Counter(), {}
    manifest = None if smoke else json.loads((cache / "manifest.json").read_text())
    import_upstream()
    from utils.flexible_graph_builder import FlexibleGraphBuilder
    inventory = metadata()
    inventory["edge_type_to_idx"] = {k: i for i, k in enumerate(inventory["edge_types"])}
    builder = FlexibleGraphBuilder(config_for("graph")["architecture"], inventory)
    for split in ("train", "val") if smoke else ("train", "val", "test"):
        if manifest is not None and not manifest["counts"].get(split, 0):
            splits[split] = {"graphs": 0}
            continue
        dataset = WindowDataset(cache, split, "graph", smoke=True)
        counts = Counter()
        for item in dataset:
            changed = shuffled_relations(item)
            before = [n["relation"] for n in item["nodes"] if n["type"] == "relation_node"]
            after = [n["relation"] for n in changed["nodes"] if n["type"] == "relation_node"]
            if Counter(before) != Counter(after):
                raise ValueError("Relation histogram changed")
            if any(not torch.equal(item[k], changed[k]) if isinstance(item[k], torch.Tensor)
                   else item[k] != changed[k] for k in item if k != "nodes"):
                raise ValueError("Non-relation graph content changed")
            delta = sum(a != b for a, b in zip(before, after))
            graph, control = builder.build(item), builder.build(changed)
            if graph.num_nodes != control.num_nodes or not torch.equal(graph.edge_index, control.edge_index):
                raise ValueError("Rebuilt graph topology changed")
            counts.update(graphs=1, relation_nodes=len(before), changed_relation_nodes=delta,
                          unchanged_graphs=int(not delta),
                          single_or_zero_relation_graphs=int(len(set(before)) <= 1),
                          directed_edges=graph.edge_type.numel(),
                          changed_edge_types=int((graph.edge_type != control.edge_type).sum()))
        splits[split] = dict(counts)
        totals.update(counts)
    report = {"cache": str(cache.resolve()), "scope": "smoke" if smoke else "currently compiled records",
              "splits": splits, "totals": dict(totals),
              "changed_relation_node_fraction": totals["changed_relation_nodes"] / max(1, totals["relation_nodes"]),
              "changed_edge_type_fraction": totals["changed_edge_types"] / max(1, totals["directed_edges"]),
              "unchanged_graph_fraction": totals["unchanged_graphs"] / max(1, totals["graphs"])}
    if not smoke:
        report["full_cache_ready"] = manifest["full_cache_ready"]
        if not manifest["full_cache_ready"]:
            try:
                WindowDataset(cache, "train", "graph")
            except ValueError as error:
                if "complete, failure-free cache" not in str(error):
                    raise
                report["partial_cache_training_rejected"] = True
            else:
                raise AssertionError("Partial cache incorrectly accepted")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving previous audit; choose a new output")
    report = audit(args.cache, args.smoke)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
