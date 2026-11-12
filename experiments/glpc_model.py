"""GLPC-v0: shared RoBERTa global/local branches and creator supervision."""

import torch
from torch import nn
from transformers import AutoModel

from experiments.neural import import_upstream


CREATOR_LABELS = torch.tensor([0, 0, 1, 1])


class GLPCModel(nn.Module):
    def __init__(self, config, mode="glpc"):
        super().__init__()
        self.config = config
        self.mode = mode
        architecture = config["architecture"]
        self.bert_model = AutoModel.from_pretrained(architecture["backbone_model_path"])
        self.bert_dim = architecture.get("bert_dim", 768)
        unfreeze_layers = architecture.get("unfreeze_bert_layers", 1)
        for parameter in self.bert_model.parameters():
            parameter.requires_grad = False
        if unfreeze_layers > 0:
            for layer in self.bert_model.encoder.layer[-unfreeze_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        _, _, supcon_class = import_upstream()
        self.supcon_class = supcon_class
        self.classifier = self._classification_head(self.bert_dim, 4, architecture)
        if self.mode in ("creator_aux", "creator_local", "glpc"):
            self.creator_head = self._classification_head(self.bert_dim, 2, architecture)
        else:
            self.creator_head = None
        if self.mode in ("creator_local", "glpc"):
            self.local_stats = nn.Sequential(
                nn.Linear(self.bert_dim * 3, self.bert_dim),
                nn.GELU(),
                nn.Dropout(architecture.get("classifier_dropout", 0.1)),
            )
            self.local_residual = nn.Linear(self.bert_dim, 4)
            nn.init.zeros_(self.local_residual.weight)
            nn.init.zeros_(self.local_residual.bias)
        else:
            self.local_stats = None
            self.local_residual = None

    @staticmethod
    def _classification_head(input_dim, num_classes, architecture):
        _, _, _ = import_upstream()
        from models.modules import ClassificationHead
        return ClassificationHead(
            input_dim=input_dim,
            hidden_dim=input_dim,
            num_classes=num_classes,
            dropout_prob=architecture.get("classifier_dropout", 0.1),
        )

    def forward(self, batch, inputs_embeds=None):
        encoder_inputs = {"attention_mask": batch["attention_mask"]}
        if inputs_embeds is None:
            encoder_inputs["input_ids"] = batch["input_ids"]
        else:
            encoder_inputs["inputs_embeds"] = inputs_embeds
        outputs = self.bert_model(**encoder_inputs)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]
        logits = self.classifier(cls)
        local_logits = torch.zeros_like(logits)
        if self.mode in ("creator_local", "glpc"):
            descriptors = []
            for row, sentence_list in enumerate(batch["sentence_tokens"]):
                vectors = []
                for sentence in sentence_list:
                    indices = torch.tensor(sentence["token_indices"], device=hidden.device)
                    vectors.append(hidden[row, indices].mean(dim=0))
                local = torch.stack(vectors)
                mean = local.mean(dim=0)
                std = local.std(dim=0, unbiased=False) if len(vectors) > 1 else torch.zeros_like(mean)
                adjacent = (local[1:] - local[:-1]).abs().mean(dim=0) if len(vectors) > 1 else torch.zeros_like(mean)
                descriptors.append(torch.cat((mean, std, adjacent)))
            local_logits = self.local_residual(self.local_stats(torch.stack(descriptors)))
            logits = logits + local_logits
        creator_logits = self.creator_head(cls) if self.creator_head is not None else None
        return {"features": cls, "logits": logits, "creator_logits": creator_logits, "local_logits": local_logits}


class GLPCV1ResidualModel(nn.Module):
    """Frozen B0 plus a trainable residual correction branch."""

    def __init__(self, config, mode="local"):
        super().__init__()
        if mode not in ("mlp", "local"):
            raise ValueError("GLPC-v1 mode must be mlp or local")
        self.mode = mode
        self.config = config
        self.base = GLPCModel(config, mode="baseline")
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()
        architecture = config["architecture"]
        self.bert_dim = architecture.get("bert_dim", 768)
        if mode == "mlp":
            self.residual_features = nn.Sequential(
                nn.Linear(self.bert_dim, self.bert_dim),
                nn.GELU(),
                nn.Dropout(architecture.get("classifier_dropout", 0.1)),
            )
        else:
            self.residual_features = nn.Sequential(
                nn.Linear(self.bert_dim * 3, self.bert_dim),
                nn.GELU(),
                nn.Dropout(architecture.get("classifier_dropout", 0.1)),
            )
        self.residual_head = nn.Linear(self.bert_dim, 4)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def load_b0(self, checkpoint):
        self.base.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def _local_descriptors(self, hidden, sentence_tokens):
        descriptors = []
        for row, sentence_list in enumerate(sentence_tokens):
            vectors = []
            for sentence in sentence_list:
                indices = torch.tensor(sentence["token_indices"], device=hidden.device)
                vectors.append(hidden[row, indices].mean(dim=0))
            local = torch.stack(vectors)
            mean = local.mean(dim=0)
            std = local.std(dim=0, unbiased=False) if len(vectors) > 1 else torch.zeros_like(mean)
            adjacent = (local[1:] - local[:-1]).abs().mean(dim=0) if len(vectors) > 1 else torch.zeros_like(mean)
            descriptors.append(torch.cat((mean, std, adjacent)))
        return torch.stack(descriptors)

    def forward(self, batch):
        with torch.no_grad():
            hidden = self.base.bert_model(input_ids=batch["input_ids"],
                                          attention_mask=batch["attention_mask"]).last_hidden_state
            cls = hidden[:, 0]
            base_logits = self.base.classifier(cls)
        features = cls if self.mode == "mlp" else self._local_descriptors(hidden, batch["sentence_tokens"])
        residual = self.residual_head(self.residual_features(features))
        return {"features": features, "logits": base_logits + residual,
                "base_logits": base_logits, "residual_logits": residual}


def pair_ranking_loss(negative_logits, positive_logits, pair_types, margin):
    scores = []
    for index, pair_type in enumerate(pair_types):
        target_class = 1 if pair_type == "polished_vs_generated" else 3
        scores.append(torch.relu(
            margin - ((positive_logits[index, target_class] - positive_logits[index, 2])
                      - (negative_logits[index, target_class] - negative_logits[index, 2]))
        ))
    return torch.stack(scores).mean() if scores else negative_logits.sum() * 0.0


def consistency_js(four_logits, creator_logits):
    q = torch.softmax(four_logits.float(), dim=1)
    creator = torch.softmax(creator_logits.float(), dim=1)
    q_creator = torch.stack((q[:, 0] + q[:, 1], q[:, 2] + q[:, 3]), dim=1)
    midpoint = (q_creator + creator) / 2
    return 0.5 * (
        torch.sum(q_creator * (q_creator.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log()), dim=1)
        + torch.sum(creator * (creator.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log()), dim=1)
    ).mean()
