"""Compare cached graph inputs and outputs with the official live preprocessing."""

import argparse
import json
from pathlib import Path

from experiments.paths import resolve_recorded_path

import torch

from experiments.common import write_json
from experiments.graph_data import GraphDataset
from experiments.neural import PROJECT_ROOT, balanced_smoke_records, build_model, collate_batch, import_upstream, load_records, tokenizer_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=PROJECT_ROOT / 'experiments/reports/graph_cache_verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    records = balanced_smoke_records(load_records("train"), 1)
    cached = GraphDataset("train", records)
    needed = {row["item_id"] for row in records}
    raw_by_id = {}
    for source in {row["source_file"] for row in records}:
        with resolve_recorded_path(source).open() as stream:
            for line in stream:
                raw = json.loads(line)
                if raw["item_id"] in needed:
                    raw_by_id[raw["item_id"]] = raw
    import_upstream()
    from utils.flexible_dataset import FlexibleGraphDataset
    processor = object.__new__(FlexibleGraphDataset)
    processor.config = {"max_seq_length": 512}
    processor.tokenizer = tokenizer_local()
    originals, loaded = [], []
    for index, row in enumerate(records):
        original = processor._process_contextual_rst_item(raw_by_id[row["item_id"]])
        item = cached[index]
        for node in original["nodes"]:
            node.pop("text", None)
        for key in ("nodes", "parent_child_edges", "root_node_ids", "descendant_map", "edu_token_spans", "id"):
            assert original[key] == item[key], (row["item_id"], key)
        for key in ("full_text_input_ids", "full_text_attention_mask", "label"):
            assert torch.equal(original[key], item[key])
        originals.append(original)
        loaded.append(item)
    config = json.loads((PROJECT_ROOT / "experiments/configs/neural_v1.json").read_text())
    config["architecture"]["backbone_model_path"] = str(PROJECT_ROOT / "assets/roberta-base")
    config.update(model_type="race")
    model = build_model(config, cached.metadata).eval()
    with torch.inference_mode():
        expected = model(collate_batch(originals))["logits"]
        actual = model(collate_batch(loaded))["logits"]
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    report = {"status": "passed", "samples": len(records), "split": "train only",
              "cached_fields_equal_official_processing": True, "four_class_output_shape": list(actual.shape),
              "max_abs_logit_difference": float((actual - expected).abs().max()),
              "weights": "base pretrained RoBERTa plus randomly initialized RACE layers; no author fine-tuned checkpoint",
              "device": "cpu", "training_performed": False}
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
