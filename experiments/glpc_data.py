"""GLPC text-window dataset with explicit sentence-to-token alignment."""

import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset


_SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s|$)|\n+|$)", re.DOTALL)


def sentence_spans(text):
    if not text or not text.strip():
        raise ValueError("GLPC does not accept blank text")
    spans = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        if text[start:end].strip():
            spans.append((start, end))
    return spans or [(0, len(text))]


def visible_token_spans(text, tokenizer, max_length=512):
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    attention = encoded["attention_mask"]
    special = encoded.get("special_tokens_mask")
    if special is None:
        special = tokenizer.get_special_tokens_mask(encoded["input_ids"], already_has_special_tokens=True)
    sentence_tokens = []
    for sentence_id, (start, end) in enumerate(sentence_spans(text)):
        token_ids = [
            index for index, ((token_start, token_end), mask, is_special) in enumerate(
                zip(offsets, attention, special)
            )
            if mask and not is_special and token_end > token_start and token_start < end and token_end > start
        ]
        if token_ids:
            sentence_tokens.append({"sentence_id": sentence_id, "token_indices": token_ids})
    visible = [
        index for index, (offset, mask, is_special) in enumerate(zip(offsets, attention, special))
        if mask and not is_special and offset[1] > offset[0]
    ]
    if not visible:
        raise ValueError("GLPC input has no visible non-special tokens")
    if not sentence_tokens:
        sentence_tokens = [{"sentence_id": -1, "token_indices": visible}]
    return encoded, sentence_tokens


class GLPCDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.labels = [record["label_id"] for record in records]
        self.items = []
        for record in records:
            encoded, sentence_tokens = visible_token_spans(record["article"], tokenizer, max_length)
            self.items.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "sentence_tokens": sentence_tokens,
                    "id": record["item_id"],
                    "label": torch.tensor(record["label_id"], dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_glpc(items, pad_token_id=1):
    max_length = max(len(item["input_ids"]) for item in items)
    batch = {
        "input_ids": torch.stack(
            [torch.nn.functional.pad(item["input_ids"], (0, max_length - len(item["input_ids"])), value=pad_token_id)
             for item in items]
        ),
        "attention_mask": torch.stack(
            [torch.nn.functional.pad(item["attention_mask"], (0, max_length - len(item["attention_mask"])), value=0)
             for item in items]
        ),
        "labels": torch.stack([item["label"] for item in items]),
        "id": [item["id"] for item in items],
        "sentence_tokens": [item["sentence_tokens"] for item in items],
    }
    return batch


def hard_source_pairs(records):
    """Return confirmed same-group cross-label pairs for training supervision."""
    groups = defaultdict(dict)
    for record in records:
        groups[record["group_id"]][record["label_id"]] = record
    pairs = []
    for group_id, versions in groups.items():
        for negative, positive, pair_type in ((2, 1, "polished_vs_generated"),
                                              (2, 3, "humanized_vs_generated")):
            if negative in versions and positive in versions:
                pairs.append({
                    "group_id": group_id,
                    "pair_type": pair_type,
                    "negative": versions[negative],
                    "positive": versions[positive],
                })
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.items = []
        for pair in pairs:
            negative, positive = pair["negative"], pair["positive"]
            negative_encoded, negative_sentences = visible_token_spans(negative["article"], tokenizer, max_length)
            positive_encoded, positive_sentences = visible_token_spans(positive["article"], tokenizer, max_length)
            self.items.append({
                "pair_type": pair["pair_type"],
                "group_id": pair["group_id"],
                "negative": _encoded_item(negative, negative_encoded, negative_sentences),
                "positive": _encoded_item(positive, positive_encoded, positive_sentences),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _encoded_item(record, encoded, sentence_tokens):
    return {
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        "sentence_tokens": sentence_tokens,
        "id": record["item_id"],
        "label": torch.tensor(record["label_id"], dtype=torch.long),
    }


def collate_pairs(items, pad_token_id=1):
    negative = collate_glpc([item["negative"] for item in items], pad_token_id)
    positive = collate_glpc([item["positive"] for item in items], pad_token_id)
    return {"negative": negative, "positive": positive,
            "pair_type": [item["pair_type"] for item in items],
            "group_id": [item["group_id"] for item in items]}
