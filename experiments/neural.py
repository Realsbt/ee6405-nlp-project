"""Shared data/model interfaces for the RoBERTa and RACE comparisons."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

from experiments.common import ASSETS_ROOT, DATA_ROOT, LABELS, file_hash, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "local_validation/upstream/RACE"
BACKBONE = ASSETS_ROOT / "roberta-base"
RUN_ROOT = DATA_ROOT.parent / "neural_runs"


def resolve_use_supcon(model_type, configured, requested):
    """Resolve the loss independently from the model architecture."""
    if requested is not None:
        return requested
    if configured:
        return True
    return model_type == "race"


def import_upstream():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from models.flexible_model import FlexibleBaselineModel, RACEModel
    from models.supcon_loss import SupConLoss
    return FlexibleBaselineModel, RACEModel, SupConLoss


def build_model(config, metadata=None):
    baseline, race, _ = import_upstream()
    model_class = baseline if config["model_type"] == "roberta" else race
    return model_class(feature_dim=128, gnn_hidden_dim=512, num_heads=4, num_classes=4,
                       config=config["architecture"], metadata=metadata, output_features=True)


class TextDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.labels = np.asarray([r["label_id"] for r in records])
        self.encoded = tokenizer([r["article"] for r in records], max_length=512,
                                 truncation=True, padding=False, return_attention_mask=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        ids = self.encoded["input_ids"][index]
        return {"full_text_input_ids": torch.tensor(ids, dtype=torch.long),
                "full_text_attention_mask": torch.ones(len(ids), dtype=torch.long),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
                "id": self.records[index]["item_id"]}


def collate_batch(items):
    # RoBERTa uses pad ID 1. Real-token attention masks are preserved.
    max_length = max(len(item["full_text_input_ids"]) for item in items)
    batch = {"id": [item["id"] for item in items], "labels": torch.stack([item["label"] for item in items])}
    for key, pad in (("full_text_input_ids", 1), ("full_text_attention_mask", 0)):
        batch[key] = torch.stack([torch.nn.functional.pad(item[key], (0, max_length - len(item[key])), value=pad)
                                  for item in items])
    for key in ("nodes", "parent_child_edges", "root_node_ids", "descendant_map", "edu_token_spans"):
        if key in items[0]:
            batch[key] = [item[key] for item in items]
    return batch


class BalancedBatchSampler(Sampler):
    """Same class-balanced sample budget for every neural model and epoch.

    Minority classes cycle; majority classes use a random subset each epoch.
    At batch 16 each class supplies 4 examples; 699 batches = 11,184 draws.
    """
    def __init__(self, labels, batch_size, seed):
        if batch_size % 4:
            raise ValueError("Balanced four-class batches require batch_size divisible by 4")
        self.indices = [np.flatnonzero(np.asarray(labels) == i) for i in range(4)]
        if any(not len(indices) for indices in self.indices):
            raise ValueError("Every training class must be present")
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.num_batches = len(labels) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = [rng.permutation(x) for x in self.indices]
        pointers = [0] * 4
        for _ in range(self.num_batches):
            selected = []
            for label in range(4):
                for _ in range(self.batch_size // 4):
                    if pointers[label] == len(indices[label]):
                        indices[label] = rng.permutation(indices[label])
                        pointers[label] = 0
                    selected.append(int(indices[label][pointers[label]]))
                    pointers[label] += 1
            rng.shuffle(selected)
            yield selected


def load_records(split, data_root=DATA_ROOT):
    protocol = json.loads((data_root / "protocol.json").read_text())
    path = data_root / f"{split}.jsonl"
    if file_hash(path) != protocol["partitions"][split]["sha256"]:
        raise ValueError("Frozen data partition changed")
    return read_jsonl(path)


def balanced_smoke_records(records, per_class):
    return [row for label in LABELS for row in [r for r in records if r["label"] == label][:per_class]]


def tokenizer_local():
    return AutoTokenizer.from_pretrained(str(BACKBONE), local_files_only=True)
