"""CRE frozen-B0 adapter and conditional residual expert models."""

import torch
from torch import nn

from experiments.cre_losses import combine_condition_scores, condition_scores_from_logits
from experiments.glpc_model import GLPCModel


MODES = ("flat_cls", "conditional_cls", "conditional_multi", "conditional_multi_pair")


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value):
        return self.output(self.activation(self.input(value)))


class CREModel(nn.Module):
    def __init__(self, config, base_checkpoint, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unsupported CRE mode: {mode}")
        self.mode = mode
        architecture = dict(config.get("architecture", {}))
        architecture.setdefault("bert_dim", 768)
        architecture.setdefault("classifier_type", "mlp")
        architecture.setdefault("classifier_dropout", 0.1)
        base_config = {"architecture": architecture}
        self.base = GLPCModel(base_config, mode="baseline")
        self.base.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()
        self.hidden_dim = int(config["hidden_dim"])
        if self.base.bert_dim != 768:
            raise ValueError(f"CRE v1 requires B0 hidden size 768, got {self.base.bert_dim}")
        if mode == "flat_cls":
            self.flat = _ResidualMLP(768, 469, 4)
        else:
            self.shared_projection = nn.Sequential(
                nn.Linear(768, 128),
                nn.LayerNorm(128),
            )
            if mode == "conditional_cls":
                self.experts = nn.ModuleList([_ResidualMLP(128, 677, 1) for _ in range(3)])
            else:
                self.sentence_projection = nn.Linear(384, 128)
                self.sentence_norm = nn.LayerNorm(128)
                self.position = nn.Embedding(config["max_sentence_positions"], 128)
                self.token_queries = nn.Parameter(torch.empty(3, 128))
                self.sentence_queries = nn.Parameter(torch.empty(3, 128))
                nn.init.normal_(self.token_queries, std=0.02)
                nn.init.normal_(self.sentence_queries, std=0.02)
                self.expert_layers = nn.ModuleList([_ResidualMLP(384, 128, 1) for _ in range(3)])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_state_dict(self):
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if not key.startswith("base.")}

    def load_adapter_state_dict(self, state_dict):
        current = self.adapter_state_dict()
        if set(current) != set(state_dict):
            raise ValueError("CRE adapter checkpoint keys do not match the model")
        base_state = {f"base.{key}": value for key, value in self.base.state_dict().items()}
        self.load_state_dict({**base_state, **state_dict}, strict=True)

    def _masked_attention(self, values, query, mask):
        scores = torch.matmul(values.float(), query.float()) / (128 ** 0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * values.float(), dim=0)

    def _multi_evidence(self, hidden, batch):
        projected = self.shared_projection(hidden.float())
        global_value = projected[:, 0]
        deltas = []
        for row, sentence_list in enumerate(batch["sentence_tokens"]):
            visible = batch["visible_token_indices"][row]
            token_values = projected[row, visible]
            token_mask = torch.ones(len(visible), dtype=torch.bool, device=hidden.device)
            sentence_values = []
            previous = None
            for sentence in sentence_list:
                indices = torch.tensor(sentence["token_indices"], device=hidden.device)
                current = projected[row, indices].mean(dim=0)
                difference = torch.zeros_like(current) if previous is None else current - previous
                position_index = min(sentence["sentence_id"], self.position.num_embeddings - 1)
                sentence_values.append(self.sentence_norm(
                    self.sentence_projection(torch.cat((current, difference, difference.abs()))) +
                    self.position.weight[position_index]
                ))
                previous = current
            sentence_values = torch.stack(sentence_values)
            sentence_mask = torch.ones(len(sentence_values), dtype=torch.bool, device=hidden.device)
            expert_values = []
            for expert in range(3):
                token_value = self._masked_attention(token_values, self.token_queries[expert], token_mask)
                sentence_value = self._masked_attention(sentence_values, self.sentence_queries[expert], sentence_mask)
                expert_values.append(self.expert_layers[expert](torch.cat((global_value[row], token_value, sentence_value))))
            deltas.append(torch.cat(expert_values))
        return torch.stack(deltas)

    def forward(self, batch):
        with torch.no_grad():
            hidden = self.base.bert_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).last_hidden_state
            hidden = hidden.detach()
            base_logits = self.base.classifier(hidden[:, 0]).float()
        if self.mode == "flat_cls":
            delta = self.flat(hidden[:, 0].float())
            final_logits = base_logits + delta
            return {"base_logits": base_logits, "logits": final_logits, "delta": delta}
        if self.mode == "conditional_cls":
            cls_value = self.shared_projection(hidden[:, 0].float())
            delta = torch.cat([expert(cls_value) for expert in self.experts], dim=1)
        else:
            delta = self._multi_evidence(hidden, batch)
        s_c, s_h, s_a = condition_scores_from_logits(base_logits)
        adjusted = combine_condition_scores(s_c + delta[:, 0], s_h + delta[:, 1], s_a + delta[:, 2])
        return {
            "base_logits": base_logits,
            "logp": adjusted,
            "probabilities": adjusted.exp(),
            "logits": adjusted,
            "delta": delta,
            "condition_scores": torch.stack((s_c + delta[:, 0], s_h + delta[:, 1], s_a + delta[:, 2]), dim=1),
        }
