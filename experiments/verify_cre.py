"""Verify CRE reload, one-encoder execution, zero correction, and gradients."""

import argparse
import json
from pathlib import Path

import torch

from experiments.cre_data import CREDataset, collate_cre
from experiments.cre_model import CREModel, MODES
from experiments.neural import load_records, tokenizer_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    effective = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["effective"]
    b0 = torch.load(effective["base_checkpoint"], map_location="cpu", weights_only=False)
    model = CREModel(effective, b0, effective["mode"]).cuda()
    checkpoint = torch.load(run_dir / "best_overall.pt", map_location="cpu", weights_only=False)
    model.load_adapter_state_dict(checkpoint["adapter_state_dict"])
    dataset = CREDataset(load_records("val")[:16], tokenizer_local(), effective["max_seq_length"])
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in collate_cre(dataset.items).items()}
    calls = []
    hook = model.base.bert_model.register_forward_hook(lambda *_: calls.append(1))
    model.eval()
    with torch.no_grad():
        output = model(batch)
    hook.remove()
    if len(calls) != 1:
        raise AssertionError(f"Expected one RoBERTa call, observed {len(calls)}")
    base = output["base_logits"]
    final = output["logits"]
    if "logp" not in output and not torch.allclose(final, base, atol=2e-5, rtol=2e-5):
        raise AssertionError("Flat zero correction is not equal to B0 logits")
    if "logp" in output and not torch.allclose(output["logp"], base.log_softmax(dim=1), atol=2e-5, rtol=2e-5):
        raise AssertionError("Conditional zero correction is not equal to B0 log probabilities")
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = (output["logits"] if "logp" not in output else output["logp"]).float().square().mean()
    loss.backward()
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise AssertionError("No finite gradient reached CRE parameters")
    result = {"status": "passed", "mode": effective["mode"], "one_roberta_call": True,
              "zero_correction_equivalent": True, "finite_trainable_gradient": True,
              "test_partition_loaded": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
