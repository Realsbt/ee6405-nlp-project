"""Controlled E10 Linear-LoRA compatibility smoke without changing dependencies."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.glpc_data import GLPCDataset, collate_glpc
from experiments.glpc_model import GLPCModel
from experiments.neural import load_records, tokenizer_local


ROOT = Path(__file__).resolve().parents[1]
B0 = ROOT / "experiments/glpc_runs/glpc_b0_s3407/best.pt"
OUT = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h/tasks/E10"


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + (inputs @ self.lora_a.t() @ self.lora_b.t()) * self.scale


def main():
    checkpoint = torch.load(B0, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["model_type"] = "roberta"
    config["classifier_type"] = "mlp"
    config["architecture"] = dict(config["architecture"])
    config["architecture"]["unfreeze_bert_layers"] = 0
    model = GLPCModel(config, mode="baseline").cuda()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    matched = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name in ("query", "value"):
            child = getattr(parent, child_name, None)
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child).cuda())
                matched += 1
    if not matched:
        raise RuntimeError("no query/value modules matched")
    tokenizer = tokenizer_local()
    dataset = GLPCDataset(load_records("train")[:16], tokenizer)
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_glpc)))
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    before = [p.detach().clone() for p in trainable]
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output["logits"].float(), batch["labels"])
    loss.backward()
    if not any(p.grad is not None and torch.any(p.grad != 0) for p in trainable):
        raise RuntimeError("LoRA gradients are zero")
    optimizer.step()
    changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
    if not changed:
        raise RuntimeError("LoRA parameters did not change")
    result = {"status": "SMOKE_PASS", "matched_query_value_modules": matched,
              "b0_sha256": __import__("hashlib").sha256(B0.read_bytes()).hexdigest(),
              "adapter_b_zero_initialized": True, "adapter_gradient_nonzero": True,
              "adapter_parameters_changed": changed, "test_partition_loaded": False,
              "hc3_loaded": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "linear_lora_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
