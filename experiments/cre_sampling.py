"""Deterministic train-only group-pair-friendly batch plan."""

import hashlib
import json
from collections import defaultdict

import numpy as np

from experiments.cre_data import pair_inventory


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_sampling_plan(records, seed=3407, batch_size=16, samples_per_class=4):
    if batch_size != 16 or samples_per_class != 4:
        raise ValueError("CRE v1 fixes batch_size=16 and samples_per_class=4")
    by_id = {record["item_id"]: record for record in records}
    pairs = pair_inventory(records)
    pair_by_type = defaultdict(list)
    for pair in pairs:
        pair_by_type[pair["pair_type"]].append(pair)
    rng = np.random.default_rng(seed)
    for values in pair_by_type.values():
        rng.shuffle(values)
    pair_pointers = {pair_type: 0 for pair_type in pair_by_type}
    queues = {label: list(rng.permutation([r["item_id"] for r in records if r["label_id"] == label])) for label in range(4)}
    pointers = {label: 0 for label in range(4)}
    num_batches = len(records) // batch_size
    batches = []
    for batch_index in range(num_batches):
        selected = []
        selected_set = set()
        selected_pairs = []
        for pair_type in ("H-P", "P-G", "G-U"):
            candidates = pair_by_type[pair_type]
            chosen = None
            for offset in range(len(candidates)):
                candidate = candidates[(pair_pointers[pair_type] + offset) % len(candidates)]
                if candidate["a"] not in selected_set and candidate["b"] not in selected_set:
                    chosen = candidate
                    pair_pointers[pair_type] = (pair_pointers[pair_type] + offset + 1) % len(candidates)
                    break
            if chosen:
                selected_pairs.append(chosen)
                selected.extend((chosen["a"], chosen["b"]))
                selected_set.update((chosen["a"], chosen["b"]))
        by_class = defaultdict(list)
        for item_id in selected:
            by_class[by_id[item_id]["label_id"]].append(item_id)
        for label in range(4):
            while len(by_class[label]) < samples_per_class:
                queue = queues[label]
                found = None
                for _ in range(len(queue)):
                    candidate = queue[pointers[label] % len(queue)]
                    pointers[label] += 1
                    if candidate not in selected_set:
                        found = candidate
                        break
                if found is None:
                    raise RuntimeError(f"Unable to fill class {label} in batch {batch_index}")
                by_class[label].append(found)
                selected_set.add(found)
        ordered = [item_id for label in range(4) for item_id in by_class[label]]
        rng.shuffle(ordered)
        index_by_id = {item_id: i for i, item_id in enumerate(ordered)}
        batches.append({
            "item_ids": ordered,
            "pairs": [
                {"pair_type": pair["pair_type"], "a": index_by_id[pair["a"]], "b": index_by_id[pair["b"]],
                 "group_id": pair["group_id"]}
                for pair in selected_pairs
            ],
        })
    manifest = {
        "version": "shared_group_pair_balanced_v1",
        "seed": seed,
        "batch_size": batch_size,
        "samples_per_class": samples_per_class,
        "batches": len(batches),
        "draws_per_epoch": len(batches) * batch_size,
        "batch_plan_sha256": _hash(batches),
        "batches_plan": batches,
    }
    return manifest
