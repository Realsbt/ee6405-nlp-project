"""CRE input alignment and train-only record batching."""

import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset


SPLITTER_VERSION = "deterministic_terminal_punctuation_v2"
_SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s|$)|\n+|$)", re.DOTALL)


def sentence_spans(text):
    if not text or not text.strip():
        raise ValueError("CRE does not accept blank text")
    spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if text[m.start():m.end()].strip()]
    return spans or [(0, len(text))]


def encode_record(record, tokenizer, max_length=512):
    text = record["article"]
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = encoded["offset_mapping"]
    attention = encoded["attention_mask"]
    special = encoded["special_tokens_mask"]
    visible = [
        i for i, (offset, mask, is_special) in enumerate(zip(offsets, attention, special))
        if mask and not is_special and offset[1] > offset[0]
    ]
    if not visible:
        raise ValueError(f"No visible tokens for {record['item_id']}")
    visible_end = max(offsets[i][1] for i in visible)
    spans = sentence_spans(text[:visible_end])
    sentence_tokens = []
    assigned = set()
    for sentence_id, (start, end) in enumerate(spans):
        candidates = []
        for token_index in visible:
            token_start, token_end = offsets[token_index]
            overlap = max(0, min(token_end, end) - max(token_start, start))
            if overlap:
                candidates.append((overlap, token_index))
        chosen = []
        for _, token_index in sorted(candidates, key=lambda pair: (-pair[0], pair[1])):
            if token_index not in assigned:
                chosen.append(token_index)
                assigned.add(token_index)
        if chosen:
            sentence_tokens.append({"sentence_id": sentence_id, "token_indices": sorted(chosen)})
    if assigned != set(visible):
        by_sentence = {item["sentence_id"]: item["token_indices"] for item in sentence_tokens}
        for token_index in visible:
            if token_index in assigned:
                continue
            token_start, token_end = offsets[token_index]
            sentence_id = min(
                range(len(spans)),
                key=lambda index: (
                    0 if spans[index][0] <= token_start <= spans[index][1] else
                    min(abs(token_start - spans[index][0]), abs(token_start - spans[index][1])),
                    index,
                ),
            )
            by_sentence.setdefault(sentence_id, []).append(token_index)
            assigned.add(token_index)
        sentence_tokens = [
            {"sentence_id": sentence_id, "token_indices": sorted(token_indices)}
            for sentence_id, token_indices in sorted(by_sentence.items())
        ]
    fallback = not sentence_tokens
    if fallback:
        sentence_tokens = [{"sentence_id": -1, "token_indices": visible}]
        assigned = set(visible)
    if assigned != set(visible):
        raise ValueError(f"Sentence alignment omitted visible tokens for {record['item_id']}")
    return {
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        "sentence_tokens": sentence_tokens,
        "offset_mapping": offsets,
        "special_tokens_mask": special,
        "visible_token_indices": visible,
        "fallback": fallback,
        "truncated": visible_end < len(text.rstrip()),
        "id": record["item_id"],
        "group_id": record["group_id"],
        "domain": record["domain"],
        "label": torch.tensor(record["label_id"], dtype=torch.long),
    }


class CREDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.items = [encode_record(record, tokenizer, max_length) for record in records]
        self.labels = [int(item["label"]) for item in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_cre(items, pad_token_id=1):
    max_length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.stack([
            torch.nn.functional.pad(item["input_ids"], (0, max_length - len(item["input_ids"])), value=pad_token_id)
            for item in items
        ]),
        "attention_mask": torch.stack([
            torch.nn.functional.pad(item["attention_mask"], (0, max_length - len(item["attention_mask"])), value=0)
            for item in items
        ]),
        "labels": torch.stack([item["label"] for item in items]),
        "id": [item["id"] for item in items],
        "group_id": [item["group_id"] for item in items],
        "domain": [item["domain"] for item in items],
        "sentence_tokens": [item["sentence_tokens"] for item in items],
        "visible_token_indices": [item["visible_token_indices"] for item in items],
    }


def pair_inventory(records):
    groups = defaultdict(dict)
    for record in records:
        groups[record["group_id"]][record["label_id"]] = record
    pair_specs = (("H-P", 0, 1), ("P-G", 1, 2), ("G-U", 2, 3))
    pairs = []
    for group_id, versions in sorted(groups.items()):
        for pair_type, a, b in pair_specs:
            if a in versions and b in versions:
                pairs.append({
                    "pair_type": pair_type,
                    "group_id": group_id,
                    "a": versions[a]["item_id"],
                    "b": versions[b]["item_id"],
                })
    return pairs


def audit_pair_inventory(records):
    pairs = pair_inventory(records)
    by_type = defaultdict(set)
    groups = defaultdict(set)
    for pair in pairs:
        by_type[pair["pair_type"]].update((pair["a"], pair["b"]))
        groups[pair["pair_type"]].add(pair["group_id"])
    return {
        "pairs": len(pairs),
        "groups_by_type": {key: len(value) for key, value in sorted(groups.items())},
        "items_by_type": {key: len(value) for key, value in sorted(by_type.items())},
        "pair_types": sorted({pair["pair_type"] for pair in pairs}),
        "relation_limit": "Same group is treated as same-source cross-label evidence, not a verified edit lineage.",
    }
