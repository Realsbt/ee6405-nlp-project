"""Create a compact CRE evidence package containing no正文、权重 or tokens."""

import argparse
import json
import shutil
from pathlib import Path

from experiments.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/cre_v1_runs"))
    parser.add_argument("--output", type=Path, default=Path("experiments/cre_v1_evidence"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = {"summary.json", "baseline_validation.json", "independent_validation.json",
               "parameters.json", "events.json", "sampling_plan_manifest.json"}
    copied = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        destination = args.output / run_dir.name
        destination.mkdir(exist_ok=True)
        for name in allowed:
            source = run_dir / name
            if source.exists():
                shutil.copy2(source, destination / name)
                copied.append(str(destination / name))
    write_json(args.output / "manifest.json", {"files": copied, "contains_weights": False,
                                                "contains_raw_text": False, "contains_token_ids": False})
    print(json.dumps({"output": str(args.output), "files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
