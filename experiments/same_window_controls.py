"""Fair-window controls for isolating RST graph contributions.

All four conditions consume the same tokenized 512-token prefix. The graph
conditions differ only in the structural operation applied after parsing that
prefix. The cache itself is external because it contains source text.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.common import file_hash
from experiments.neural import build_model, import_upstream
from experiments.paths import PROJECT_ROOT, UPSTREAM, resolve_recorded_path


CONDITIONS = ("text", "graph", "shuffled", "edu")
CONTROL_SEED = 20261109


def relation_metadata():
    table = resolve_recorded_path(UPSTREAM / "utils/relation_table.txt")
    names = {line.split("_")[0] for line in table.read_text().splitlines() if line.strip()}
    edge_types = set()
    for name in names:
        for target in ("edu_node", "relation_node"):
            edge_types.add(("relation_node", name, target))
            edge_types.add((target, name, "relation_node"))
    return {"edge_types": sorted(edge_types)}


def shuffled_relations(item, control_seed=CONTROL_SEED):
    """Deterministically permute relation assignments within one graph.

    The graph topology, relation histogram, text, label and all other fields
    are unchanged. The seed uses only the fixed control seed and item ID.
    """
    result = {**item, "nodes": [dict(node) for node in item["nodes"]]}
    selected = [node for node in result["nodes"] if node["type"] == "relation_node"]
    labels = [node["relation"] for node in selected]
    payload = f"{control_seed}:{item['id']}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    for node, relation in zip(selected, np.random.default_rng(seed).permutation(labels)):
        node["relation"] = str(relation)
    return result


def mean_edu(hidden, spans):
    """Mean tokens within each EDU, then give every EDU equal weight."""
    if not spans or any(not 0 <= start < end <= len(hidden) for start, end in spans):
        raise ValueError("EDU spans must be nonempty and within visible hidden states")
    return torch.stack([hidden[start:end].mean(0) for start, end in spans]).mean(0)


class IndexedSameWindowDataset(Dataset):
    """Read an audited external same-window graph cache by byte offset."""

    def __init__(self, cache, split, condition, allow_partial=False):
        if split not in ("train", "val", "test") or condition not in CONDITIONS:
            raise ValueError("Unknown split or condition")
        self.cache = Path(cache)
        self.condition = condition
        self._handle = None
        manifest = json.loads((self.cache / "manifest.json").read_text())
        if not allow_partial and not manifest["full_cache_ready"]:
            raise ValueError("Formal training requires a complete, failure-free cache")
        for name, key in (("graphs.jsonl", "graphs_sha256"), ("index.json", "index_sha256")):
            if file_hash(self.cache / name) != manifest[key]:
                raise ValueError(f"Same-window cache changed: {name}")
        if file_hash(self.cache / "protocol.json") != manifest["protocol_sha256"]:
            raise ValueError("Same-window protocol changed")
        protocol = json.loads((self.cache / "protocol.json").read_text())
        if file_hash(self.cache / "inputs.jsonl") != protocol["inputs_sha256"]:
            raise ValueError("Visible-input inventory changed")
        self.index = json.loads((self.cache / "index.json").read_text())
        with (self.cache / "inputs.jsonl").open() as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.records = [row for row in self.records
                        if row["source_split"] == split and row["item_id"] in self.index]
        self.labels = np.asarray([row["label_id"] for row in self.records])
        if not len(self.records):
            raise ValueError(f"No cached records for split {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = (self.cache / "graphs.jsonl").open("rb")
        offset, length = self.index[self.records[index]["item_id"]]
        self._handle.seek(offset)
        item = json.loads(self._handle.read(length))
        if item["id"] != self.records[index]["item_id"]:
            raise ValueError("Indexed graph ID mismatch")
        item["descendant_map"] = {int(key): value for key, value in item["descendant_map"].items()}
        for key in ("full_text_input_ids", "full_text_attention_mask", "label", "domain_label"):
            item[key] = torch.tensor(item[key], dtype=torch.long)
        return shuffled_relations(item) if self.condition == "shuffled" else item

    def __getstate__(self):
        return {**self.__dict__, "_handle": None}


class EDUPoolingModel(nn.Module):
    """EDU pooling and a matched-width head, without graph message passing."""

    def __init__(self, config):
        super().__init__()
        baseline = build_model({**config, "model_type": "roberta"})
        self.bert_model = baseline.bert_model
        import_upstream()
        from models.modules import ClassificationHead
        architecture = config["architecture"]
        self.projection = nn.Sequential(
            nn.Linear(architecture["bert_dim"], architecture["gnn_hidden_dim"]),
            nn.LayerNorm(architecture["gnn_hidden_dim"]),
            nn.Dropout(architecture["proj_dropout"]),
        )
        self.classifier = ClassificationHead(
            input_dim=architecture["gnn_hidden_dim"],
            hidden_dim=architecture["gnn_hidden_dim"],
            num_classes=4,
            dropout_prob=architecture["classifier_dropout"],
        )
        baseline._init_classifier(self.classifier)

    def forward(self, batch):
        device = next(self.parameters()).device
        hidden = self.bert_model(
            input_ids=batch["full_text_input_ids"].to(device),
            attention_mask=batch["full_text_attention_mask"].to(device),
        ).last_hidden_state
        pooled = torch.stack([mean_edu(row, spans)
                              for row, spans in zip(hidden, batch["edu_token_spans"])])
        features = self.projection(pooled)
        return {"features": features, "logits": self.classifier(features)}


def config_for(condition, seed=3407):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    config = json.loads((PROJECT_ROOT / "experiments/configs/neural_v1.json").read_text())
    config.update(
        condition=condition,
        seed=seed,
        use_supcon=False,
        model_type="roberta" if condition in ("text", "edu") else "race",
        precision="float32",
        control_seed=CONTROL_SEED,
    )
    return config


def model_for(config):
    if config["condition"] == "edu":
        return EDUPoolingModel(config)
    metadata = None if config["condition"] == "text" else relation_metadata()
    return build_model(config, metadata)
