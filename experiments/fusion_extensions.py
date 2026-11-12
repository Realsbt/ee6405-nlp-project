"""A matched 2x2 of segmentation and same-text dropout consistency."""

import hashlib

import torch
from torch.nn import functional as F

from experiments.edu_fusion import EDUFusionModel, fusion_config
from experiments.same_window_controls import IndexedSameWindowDataset


EXTENSION_CONDITIONS = (
    "ext_edu_ce2", "ext_chunks_ce2",
    "ext_edu_consistent", "ext_chunks_consistent",
)
CHUNK_SIZE = 32


def chunk_spans(input_ids, attention_mask, chunk_size=CHUNK_SIZE):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("Expected matching batch/token axes")
    result = []
    for ids, mask in zip(input_ids, attention_mask):
        count = int(mask.sum().item())
        if (count < 3 or not torch.all(mask[:count] == 1)
                or not torch.all(mask[count:] == 0)):
            raise ValueError("Expected nonempty body and contiguous right padding")
        if ids[0].item() != 0 or ids[count - 1].item() != 2:
            raise ValueError("Expected the audited RoBERTa BOS/body/EOS convention")
        result.append([(start, min(start + chunk_size, count - 1))
                       for start in range(1, count - 1, chunk_size)])
    return result


def validation_partition(records):
    """Freeze disjoint source-group selection/calibration halves before fitting."""
    groups = sorted(
        {row["group_id"] for row in records},
        key=lambda group: hashlib.sha256(f"fusion-calibration-v1:{group}".encode()).digest(),
    )
    if len(groups) < 2:
        raise ValueError("Need at least two validation source groups")
    selection_groups = set(groups[:len(groups) // 2])
    selection = [i for i, row in enumerate(records) if row["group_id"] in selection_groups]
    calibration = [i for i, row in enumerate(records) if row["group_id"] not in selection_groups]
    for indices in (selection, calibration):
        if {records[i]["label_id"] for i in indices} != {0, 1, 2, 3}:
            raise ValueError("Each fixed validation half must contain all four labels")
    return selection, calibration


class FusionExtensionModel(EDUFusionModel):
    def __init__(self, config):
        if config["condition"] not in EXTENSION_CONDITIONS:
            raise ValueError("Unknown fusion extension")
        super().__init__({**config, "condition": config["extension"].get("parent", "fusion_gated")})
        self.use_chunks = config["extension"]["segmentation"] == "chunks"
        self.chunk_size = config["extension"]["chunk_size"]

    def forward(self, batch):
        if self.use_chunks:
            batch = {**batch, "edu_token_spans": chunk_spans(
                batch["full_text_input_ids"], batch["full_text_attention_mask"],
                self.chunk_size,
            )}
        return super().forward(batch)


def paired_dropout_loss(first, second, labels, consistency_weight):
    """Both conditions use two CE forwards; only the symmetric KL differs."""
    if consistency_weight < 0:
        raise ValueError("Consistency weight must be nonnegative")
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Logit shapes must match")
    first, second = first.float(), second.float()
    ce = (F.cross_entropy(first, labels) + F.cross_entropy(second, labels)) / 2
    log_first, log_second = first.log_softmax(-1), second.log_softmax(-1)
    symmetric_kl = (
        F.kl_div(log_first, log_second.exp(), reduction="batchmean")
        + F.kl_div(log_second, log_first.exp(), reduction="batchmean")
    ) / 2
    return ce + consistency_weight * symmetric_kl, ce, symmetric_kl


def extension_config(condition, seed=3407, parent="fusion_gated"):
    if condition not in EXTENSION_CONDITIONS:
        raise ValueError("Unknown fusion extension")
    config = fusion_config(parent, seed)
    config.update(
        condition=condition, recipe_id="fusion_segmentation_consistency_v1",
        selection_metric="selection_half_validation_macro_f1",
        validation_partition="sha256 source-group halves, fusion-calibration-v1 salt",
    )
    config["extension"] = {
        "segmentation": "chunks" if "_chunks_" in condition else "edu",
        "chunk_size": CHUNK_SIZE, "training_forwards_per_batch": 2,
        "consistency_weight": 1.0 if condition.endswith("_consistent") else 0.0,
        "objective": "mean(two CE) + weight * symmetric_KL",
        "parent": parent,
    }
    return config


def extension_dataset(cache, split, condition):
    if condition not in EXTENSION_CONDITIONS:
        raise ValueError("Unknown fusion extension")
    return IndexedSameWindowDataset(cache, split, "text")
