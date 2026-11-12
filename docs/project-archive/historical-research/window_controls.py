"""Same-window datasets and four independently trained mechanism conditions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from replay_hc3_cpu import ROOT, build_model, tokenizer_local
from experiments.neural import import_upstream
from experiments.raw_graph import RawGraphDataset
from screen_structure_cpu import digest, rows

CONDITIONS = ("text", "graph", "shuffled", "edu")
CONTROL_SEED = 20261109


def metadata():
    names = {s.split("_")[0] for s in (ROOT / "resources/RACE/utils/relation_table.txt").read_text().splitlines() if s}
    edge_types = set()
    for name in names:
        for target in ("edu_node", "relation_node"):
            edge_types.add(("relation_node", name, target))
            edge_types.add((target, name, "relation_node"))
    return {"edge_types": sorted(edge_types)}


def shuffled_relations(item, control_seed=CONTROL_SEED):
    """Shuffle assignments within each graph; preserve its relation histogram."""
    result = {**item, "nodes": [dict(n) for n in item["nodes"]]}
    selected = [n for n in result["nodes"] if n["type"] == "relation_node"]
    labels = [n["relation"] for n in selected]
    seed = int.from_bytes(hashlib.sha256(f"{control_seed}:{item['id']}".encode()).digest()[:8], "big")
    shuffled = np.random.default_rng(seed).permutation(labels)
    for node, relation in zip(selected, shuffled):
        node["relation"] = str(relation)
    return result


class WindowDataset(Dataset):
    def __init__(self, cache, split, condition, smoke=False):
        if split not in ("train", "val", "test") or condition not in CONDITIONS:
            raise ValueError("Unknown split or condition")
        self.condition, self._handle = condition, None
        self.path = Path(cache)
        if (self.path / "manifest.json").exists():
            manifest = json.loads((self.path / "manifest.json").read_text())
            if not smoke and not manifest["full_cache_ready"]:
                raise ValueError("Full training requires a complete, failure-free cache")
            for name, key in (("graphs.jsonl", "graphs_sha256"), ("index.json", "index_sha256")):
                if digest(self.path / name) != manifest[key]:
                    raise ValueError("Graph cache changed")
            protocol = json.loads((self.path / "protocol.json").read_text())
            if (digest(self.path / "protocol.json") != manifest["protocol_sha256"]
                    or digest(self.path / "inputs.jsonl") != protocol["inputs_sha256"]):
                raise ValueError("Visible input protocol changed")
            self.index = json.loads((self.path / "index.json").read_text())
            self.records = [r for r in rows(self.path / "inputs.jsonl")
                            if r["source_split"] == split and r["item_id"] in self.index]
            self.items = None
        else:
            if not smoke or split == "test":
                raise ValueError("A smoke cache cannot be used for full training/test evaluation")
            report = json.loads((self.path / "verification.json").read_text())
            if report["status"] != "passed" or digest(self.path / "parsed.jsonl") != report["parsed_sha256"]:
                raise ValueError("Unverified smoke cache")
            self.records = [r for r in rows(self.path / "parsed.jsonl") if r["source_split"] == split]
            self.items = RawGraphDataset(self.records, tokenizer_local())
        self.labels = np.array([r["label_id"] for r in self.records])
        if not len(self.labels):
            raise ValueError("No cached records for requested split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.items is not None:
            item = self.items[index]
        else:
            if self._handle is None:
                self._handle = (self.path / "graphs.jsonl").open("rb")
            offset, length = self.index[self.records[index]["item_id"]]
            self._handle.seek(offset)
            item = json.loads(self._handle.read(length))
            item["descendant_map"] = {int(k): v for k, v in item["descendant_map"].items()}
            for key in ("full_text_input_ids", "full_text_attention_mask", "label", "domain_label"):
                item[key] = torch.tensor(item[key], dtype=torch.long)
        if item["id"] != self.records[index]["item_id"]:
            raise ValueError("Indexed graph ID mismatch")
        return shuffled_relations(item) if self.condition == "shuffled" else item

    def __getstate__(self):
        return {**self.__dict__, "_handle": None}


def mean_edu(hidden, spans):
    if not spans or any(not 0 <= a < b <= len(hidden) for a, b in spans):
        raise ValueError("EDU spans must be nonempty and within visible hidden states")
    return torch.stack([hidden[a:b].mean(0) for a, b in spans]).mean(0)


class EDUPoolingModel(nn.Module):
    """Unweighted EDU mean, projection to RACE head width, no message passing."""
    def __init__(self, config):
        super().__init__()
        baseline, _, _ = import_upstream()
        base = baseline(feature_dim=128, gnn_hidden_dim=512, num_heads=4, num_classes=4,
                        config=config["architecture"], output_features=True)
        self.bert_model = base.bert_model
        from models.modules import ClassificationHead
        self.projection = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512),
                                        nn.Dropout(config["architecture"]["proj_dropout"]))
        self.classifier = ClassificationHead(input_dim=512, hidden_dim=512, num_classes=4,
                                             dropout_prob=config["architecture"]["classifier_dropout"])
        base._init_classifier(self.classifier)

    def forward(self, batch):
        device = next(self.parameters()).device
        hidden = self.bert_model(input_ids=batch["full_text_input_ids"].to(device),
                                 attention_mask=batch["full_text_attention_mask"].to(device)).last_hidden_state
        pooled = torch.stack([mean_edu(h, spans) for h, spans in zip(hidden, batch["edu_token_spans"])])
        features = self.projection(pooled)
        return {"features": features, "logits": self.classifier(features)}


def config_for(condition, seed=3407):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    config = json.loads((ROOT / "repository/experiments/configs/neural_v1.json").read_text())
    config.update(condition=condition, seed=seed, use_supcon=False,
                  model_type="roberta" if condition in ("text", "edu") else "race",
                  precision="float32", control_seed=CONTROL_SEED)
    config["architecture"]["backbone_model_path"] = str(ROOT / "resources/assets/roberta-base")
    return config


def model_for(config):
    return EDUPoolingModel(config) if config["condition"] == "edu" else build_model(config, metadata())
