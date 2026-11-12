"""Residual text/EDU fusion over the frozen same-window input."""

import torch
from torch import nn

from experiments.neural import build_model, import_upstream
from experiments.same_window_controls import IndexedSameWindowDataset, config_for


FUSION_CONDITIONS = ("fusion_cls", "fusion_mean", "fusion_attention", "fusion_gated")


class EDUFusion(nn.Module):
    """Aggregate valid EDUs and add a fixed or learned residual to CLS."""

    def __init__(self, dimension, condition, attention_dim=128):
        super().__init__()
        if condition not in FUSION_CONDITIONS:
            raise ValueError(f"Unknown fusion condition: {condition}")
        self.condition = condition
        if condition in ("fusion_attention", "fusion_gated"):
            self.attention = nn.Sequential(
                nn.Linear(dimension, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1, bias=False),
            )
        if condition == "fusion_gated":
            self.gate = nn.Linear(2 * dimension, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, hidden, attention_mask, spans_by_item):
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("Hidden states and attention mask must have matching batch/token axes")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("Attention mask must be binary")
        if not torch.all(attention_mask[:, 0] == 1):
            raise ValueError("CLS must be visible")
        if len(spans_by_item) != len(hidden):
            raise ValueError("One EDU span list is required per item")
        cls = hidden[:, 0]
        if self.condition == "fusion_cls":
            return {"pooled": cls, "edu_weights": [], "edu_gate": cls.new_zeros((len(cls), 1))}

        contexts, weights = [], []
        for row, mask, spans in zip(hidden, attention_mask, spans_by_item):
            if not spans:
                raise ValueError("EDU spans must not be empty")
            previous_start, previous_end = 0, 0
            for start, end in spans:
                if (not 0 <= start < end <= len(row)
                        or start < previous_start or end < previous_end):
                    raise ValueError("EDU spans must be ordered and in bounds")
                if not torch.all(mask[start:end] == 1):
                    raise ValueError("EDU spans must not include padding")
                # Adjacent character spans can share a boundary BPE token.
                previous_start, previous_end = start, end
            edus = torch.stack([row[start:end].mean(0) for start, end in spans])
            if self.condition == "fusion_mean":
                weight = edus.new_full((len(edus),), 1.0 / len(edus))
            else:
                weight = self.attention(edus).squeeze(-1).softmax(0)
            contexts.append((weight[:, None] * edus).sum(0))
            weights.append(weight)
        context = torch.stack(contexts)
        gate = (self.gate(torch.cat((cls, context), dim=-1)).sigmoid()
                if self.condition == "fusion_gated" else cls.new_ones((len(cls), 1)))
        return {"pooled": cls + gate * context, "edu_weights": weights, "edu_gate": gate}


class EDUFusionModel(nn.Module):
    """One encoder, a shared-width head, and an optional EDU residual."""

    def __init__(self, config):
        super().__init__()
        baseline = build_model({**config, "model_type": "roberta"})
        self.bert_model = baseline.bert_model
        import_upstream()
        from models.modules import ClassificationHead

        architecture = config["architecture"]
        dimension = architecture["bert_dim"]
        width = architecture["gnn_hidden_dim"]
        # Initialize the common head before variant-specific modules.
        self.projection = nn.Sequential(
            nn.Linear(dimension, width),
            nn.LayerNorm(width),
            nn.Dropout(architecture["proj_dropout"]),
        )
        self.classifier = ClassificationHead(
            input_dim=width, hidden_dim=width, num_classes=4,
            dropout_prob=architecture["classifier_dropout"],
        )
        baseline._init_classifier(self.classifier)
        self.fusion = EDUFusion(dimension, config["condition"], config["fusion"]["attention_dim"])

    def forward(self, batch):
        device = next(self.parameters()).device
        mask = batch["full_text_attention_mask"].to(device)
        hidden = self.bert_model(
            input_ids=batch["full_text_input_ids"].to(device), attention_mask=mask,
        ).last_hidden_state
        output = self.fusion(hidden, mask, batch["edu_token_spans"])
        features = self.projection(output.pop("pooled"))
        return {**output, "features": features, "logits": self.classifier(features)}


def fusion_config(condition, seed=3407):
    if condition not in FUSION_CONDITIONS:
        raise ValueError(f"Unknown fusion condition: {condition}")
    config = config_for("text", seed)
    config.update(
        condition=condition,
        recipe_id="edu_residual_fusion_v1",
        fusion={
            "attention_dim": 128,
            "combination": "cls + gate * edu_context",
            "gate_initial_bias": -2.0,
            "head_width": config["architecture"]["gnn_hidden_dim"],
        },
    )
    return config


def fusion_dataset(cache, split, condition):
    if condition not in FUSION_CONDITIONS:
        raise ValueError(f"Unknown fusion condition: {condition}")
    return IndexedSameWindowDataset(cache, split, "text")
