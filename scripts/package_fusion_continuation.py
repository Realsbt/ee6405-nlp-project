"""Package verified own-stage outputs without unused partial checkpoints."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.kaggle.resume_edu_fusion import digest, validate_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous archives")
    code = Path(__file__).resolve().parents[1]
    cache_hash = digest(args.cache / "manifest.json")
    complete = set()
    for path in args.input.rglob("result.json"):
        result = json.loads(path.read_text())
        if not validate_completed(path.parent, result["condition"], result["seed"], code, cache_hash):
            raise ValueError(path)
        complete.add(path.parent)
    if len(complete) != 5:
        raise ValueError("Expected the five verified complete first-stage runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(args.input.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pt":
                if path.parent not in complete:
                    continue
            elif path.suffix not in (".json", ".jsonl", ".py"):
                continue
            name = str(Path("edu_fusion_experiment_v1") / path.relative_to(args.input))
            archive.write(path, name)
            manifest[name] = digest(path)
        archive.writestr("stage1_files_sha256.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(args.output), "sha256": digest(args.output),
                      "verified_complete_runs": len(complete), "files": len(manifest)}))


if __name__ == "__main__":
    main()
