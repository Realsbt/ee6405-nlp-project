"""Apply official RACE preprocessing to newly parsed texts, without training."""

import numpy as np
from torch.utils.data import Dataset

from experiments.neural import BACKBONE, import_upstream


class RawGraphDataset(Dataset):
    def __init__(self, records, tokenizer):
        import_upstream()
        from utils.flexible_dataset import FlexibleGraphDataset
        self.records = records
        self.labels = np.asarray([row.get("label_id", -1) for row in records])
        processor = object.__new__(FlexibleGraphDataset)
        processor.config = {"backbone_model_path": str(BACKBONE), "max_seq_length": 512}
        processor.tokenizer = tokenizer
        self.items = []
        for record in records:
            item = processor._process_contextual_rst_item(record)
            if item is None or not item["edu_token_spans"] or not item["root_node_ids"]:
                raise ValueError(f"Text has no usable RST/token spans: {record.get('item_id')}")
            for node in item["nodes"]:
                node.pop("text", None)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
