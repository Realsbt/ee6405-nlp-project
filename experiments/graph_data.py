"""Memory-bounded access to cached official graph preprocessing."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.common import DATA_ROOT, file_hash
from experiments.prepare_graph_cache import CACHE_ROOT


class GraphDataset(Dataset):
    def __init__(self, split, records):
        self.records = records
        self.labels = np.asarray([row["label_id"] for row in records])
        self.path = CACHE_ROOT / f"{split}.jsonl"
        self.index = json.loads((CACHE_ROOT / f"{split}_index.json").read_text())
        manifest = json.loads((CACHE_ROOT / "manifest.json").read_text())
        assert manifest["data_protocol_sha256"] == file_hash(DATA_ROOT / "protocol.json")
        assert manifest["cache_file_sha256"][split] == file_hash(self.path)
        assert all(row["item_id"] in self.index for row in records)
        self.metadata = json.loads((CACHE_ROOT / "metadata.json").read_text())
        self.metadata["edge_types"] = [tuple(edge) for edge in self.metadata["edge_types"]]
        self._handle = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item_id = self.records[index]["item_id"]
        if self._handle is None:
            self._handle = self.path.open("rb")
        offset, size = self.index[item_id]
        self._handle.seek(offset)
        item = json.loads(self._handle.read(size))
        assert item["id"] == item_id
        item["parent_child_edges"] = [tuple(edge) for edge in item["parent_child_edges"]]
        item["descendant_map"] = {int(key): value for key, value in item["descendant_map"].items()}
        for key in ("full_text_input_ids", "full_text_attention_mask", "label"):
            item[key] = torch.tensor(item[key], dtype=torch.long)
        return item

    def __getstate__(self):
        return {**self.__dict__, "_handle": None}
