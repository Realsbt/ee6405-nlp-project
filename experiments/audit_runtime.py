"""Record the exercised interpreter and package versions without mutating it."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("training", "rst"), required=True)
    args = parser.parse_args()
    packages = ["torch", "numpy", "transformers", "tokenizers", "huggingface-hub", "safetensors", "tqdm", "requests"]
    packages += (["torch-geometric", "scikit-learn", "scipy", "joblib", "threadpoolctl", "matplotlib"] if args.kind == "training" else
                 ["isanlp-rst", "isanlp", "razdel", "jsonnet", "nltk", "sentencepiece", "fire", "termcolor", "lxml", "networkx", "pillow"])
    versions = {name: importlib.metadata.version(name) for name in packages}
    report = {"status": "recorded", "kind": args.kind, "python": sys.version, "executable": sys.executable,
              "prefix": sys.prefix, "base_prefix": sys.base_prefix, "platform": platform.platform(),
              "packages": versions, "fresh_system_install_validated": False,
              "scope": "Existing local overlay runtime; inherited base packages were not installed or changed by this script"}
    if args.kind == "rst":
        report["isanlp_direct_url"] = json.loads(importlib.metadata.distribution("isanlp").read_text("direct_url.json"))
    path = Path(__file__).parent / f"reports/runtime_{args.kind}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"kind": args.kind, "packages": versions, "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
