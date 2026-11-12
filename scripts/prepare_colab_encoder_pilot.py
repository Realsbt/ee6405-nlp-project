"""Create an allowlisted Colab upload without test data or model weights."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def prepare(package, backbone, output):
    if output.exists():
        raise FileExistsError("Preserve immutable Colab upload")
    entries = {}
    frozen_names = (
        "encoder_pilot_code_v1.zip", "pilot_train.jsonl", "pilot_val.jsonl",
        "pilot_protocol.json", "encoder_pilot_manifest.json",
    )
    for name in frozen_names:
        entries[f"frozen/{name}"] = package / "dataset" / name
    source_manifest = json.loads((package / "dataset/encoder_pilot_manifest.json").read_text())
    for name, expected in source_manifest["code_files"].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
        entries[f"code/{name}"] = path
    for name in ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        entries[f"roberta_config/{name}"] = backbone / name
    entries["run_encoder_pilot.py"] = ROOT / "cloud/colab/run_encoder_pilot.py"
    manifest = {
        "private": True, "test_partition_included": False,
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in entries.items()},
        "roberta_weights": {
            "repo": "FacebookAI/roberta-base",
            "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            "sha256": "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("colab_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.package, args.backbone, args.output), indent=2))
