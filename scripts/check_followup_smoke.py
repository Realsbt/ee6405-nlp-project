"""Strict engineering reload checks; optionally discard only CPU-smoke weights."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import file_hash, write_json
from experiments.fusion_extensions import EXTENSION_CONDITIONS, FusionExtensionModel, extension_dataset
from experiments.fusion_scale_controls import SCALE_CONDITIONS, ScaleControlModel, scale_dataset
from experiments.neural import collate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("scale", "extensions"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discard-weights", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous reload evidence")
    if args.discard_weights and "cpu" not in args.input.name:
        parser.error("Only explicitly named CPU-smoke directories may discard weights")
    conditions, factory, dataset_factory = (
        (SCALE_CONDITIONS, ScaleControlModel, scale_dataset) if args.kind == "scale"
        else (EXTENSION_CONDITIONS, FusionExtensionModel, extension_dataset)
    )
    results = {}
    for condition in conditions:
        run = args.input / condition
        recorded = json.loads((run / "config.json").read_text())
        smoke = json.loads((run / "smoke.json").read_text())
        if (not recorded["training_only_smoke"] or smoke["test_evaluated"]
                or smoke["status"] != "passed" or smoke["device"] != "cpu"):
            raise ValueError("Not a passed CPU engineering smoke")
        if file_hash(args.cache / "manifest.json") != recorded["cache_manifest_sha256"]:
            raise ValueError("Smoke cache manifest changed")
        for name, expected in json.loads((run / "source_hashes.json").read_text()).items():
            source = Path(__file__).resolve().parents[1] / "experiments" / name
            if file_hash(source) != expected:
                raise ValueError(f"Smoke source changed: {name}")
        checkpoint_path = run / "best.pt"
        checkpoint_hash = file_hash(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != recorded["config"]:
            raise ValueError("Smoke checkpoint configuration changed")
        model = factory(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        dataset = dataset_factory(args.cache, "val", condition)
        indices = [int(np.flatnonzero(dataset.labels == label)[0]) for label in range(4)]
        with torch.inference_mode():
            output = model(collate_batch([dataset[i] for i in indices]))
        if not torch.isfinite(output["logits"]).all():
            raise ValueError("Nonfinite reloaded smoke logits")
        for weights in output["edu_weights"]:
            if not torch.isfinite(weights).all() or not torch.allclose(weights.sum(), weights.new_tensor(1.)):
                raise ValueError("Invalid reloaded local attention")
        results[condition] = {
            "status": "passed", "strict_reload": True, "finite_forward": True,
            "checkpoint_sha256": checkpoint_hash,
            "attention_normalized": True, "weights_discarded": False,
            "training_only_smoke": True, "test_evaluated": False,
        }
        del checkpoint, model, output, dataset
    # Only discard after ALL checkpoints have been checked and their hashes saved.
    write_json(args.output, {"status": "passed", "conditions": results, "performance_claim": False})
    if args.discard_weights:
        for condition in conditions:
            (args.input / condition / "best.pt").unlink()
            results[condition]["weights_discarded"] = True
            write_json(args.output, {"status": "passed", "conditions": results, "performance_claim": False})


if __name__ == "__main__":
    main()
