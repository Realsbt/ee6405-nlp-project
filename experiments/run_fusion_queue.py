"""Run a validation-only fusion matrix on an already configured CUDA host."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.common import write_json
from experiments.edu_fusion import FUSION_CONDITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=FUSION_CONDITIONS,
                        default=list(FUSION_CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Conditions and seeds must be unique")
    if args.output.exists():
        parser.error("Queue output already exists; preserve it and choose a new directory")
    if args.epochs <= 0 or args.batch_size <= 0 or args.batch_size % 4:
        parser.error("Epochs must be positive; batch size must be positive and divisible by four")
    import torch
    if not torch.cuda.is_available():
        parser.error("This queue requires CUDA; use train_edu_fusion --smoke for CPU checks")
    if not (args.cache / "manifest.json").is_file():
        parser.error("Cache manifest is missing")
    args.output.mkdir(parents=True)
    results = []
    for seed in args.seeds:
        for condition in args.conditions:
            target = args.output / f"{condition}_s{seed}"
            command = [
                sys.executable, "-m", "experiments.train_edu_fusion",
                "--cache", str(args.cache.resolve()), "--output", str(target.resolve()),
                "--condition", condition, "--seed", str(seed), "--device", "cuda",
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--validation-only",
            ]
            if args.smoke:
                command.append("--smoke")
            write_json(args.output / "queue.json", {
                "status": "running", "current": target.name, "finished": results,
                "test_evaluated": False,
            })
            print("+", " ".join(command), flush=True)
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                write_json(args.output / "queue.json", {
                    "status": "failed", "failed": target.name, "returncode": error.returncode,
                    "finished": results, "test_evaluated": False,
                })
                raise
            result = json.loads((target / ("smoke.json" if args.smoke else "result.json")).read_text())
            results.append({
                "condition": condition, "seed": seed, "output": str(target.resolve()),
                "status": result["status"],
                "validation_macro_f1": result.get("validation", {}).get("macro_f1"),
            })
    write_json(args.output / "queue.json", {
        "status": "completed", "finished": results, "test_evaluated": False,
        "scope": "Engineering smoke only" if args.smoke else "Validation screening only",
    })


if __name__ == "__main__":
    main()
