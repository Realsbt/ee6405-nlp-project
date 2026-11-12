"""Package code only for a private Kaggle follow-up, never data or weights."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
def prepare(destination, kind="scale"):
    if kind not in ("scale", "extensions", "dfs"):
        raise ValueError("Unknown follow-up kind")
    version = {"scale": 1, "extensions": 2, "dfs": 3}[kind]
    dataset_id = f"zhengyingqi/ee6405-fusion-followup-code-v{version}"
    kernel_id = f"zhengyingqi/ee6405-fusion-{kind}-experiment-v1"
    if destination.exists():
        raise FileExistsError("Preserve the existing frozen package")
    dataset = destination / "dataset"
    kernel = destination / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir()
    sources = sorted({
        *ROOT.glob("experiments/*.py"),
        *ROOT.glob("cloud/kaggle/*.py"),
        ROOT / "experiments/configs/neural_v1.json",
    })
    manifest = {}
    archive = dataset / f"fusion_followup_code_v{version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source in sources:
            name = str(source.relative_to(ROOT))
            manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            bundle.write(source, name)
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": f"EE6405 Fusion Followup Code v{version}", "id": dataset_id,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    bootstrap = f"run_fusion_{kind}.py"
    shutil.copy2(ROOT / "cloud/kaggle" / bootstrap, kernel / bootstrap)
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": f"EE6405 Fusion {kind.title()} Experiment v1",
        "code_file": bootstrap, "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": ["zhengyingqi/ee6405-same-window-gpu-v1", dataset_id],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
    content_manifest = json.dumps({
        "files": manifest, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "scope": "Python source and one base config only; no text, checkpoints or credentials",
    }, indent=2) + "\n"
    (destination / "code_manifest.json").write_text(content_manifest)
    if kind == "dfs":
        (dataset / "fusion_dfs_code_manifest.json").write_text(content_manifest)
    return {"dataset": str(dataset), "kernel": str(kernel), "source_files": len(sources)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("scale", "extensions", "dfs"), default="scale")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.kind), indent=2))
