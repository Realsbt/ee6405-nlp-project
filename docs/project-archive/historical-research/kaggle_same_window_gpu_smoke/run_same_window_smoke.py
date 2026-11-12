"""Kaggle GPU smoke for all four same-window conditions."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


DATASET_SLUG = "ee6405-same-window-gpu-v1"
INPUT_ROOT = Path("/kaggle/input")
EXTRACTED = Path("/kaggle/working/ee6405_inputs")
OUTPUT = Path("/kaggle/working/same_window_gpu_smoke")


def run(*command):
    print("+", *command, flush=True)
    subprocess.run(command, check=True)


def find_dataset():
    expected = INPUT_ROOT / DATASET_SLUG
    if expected.is_dir():
        return expected
    candidates = {path.parent for path in INPUT_ROOT.rglob("repository.zip")}
    candidates.update(path.parent for path in INPUT_ROOT.rglob("repository") if path.is_dir())
    if len(candidates) != 1:
        available = [str(path.relative_to(INPUT_ROOT)) for path in INPUT_ROOT.rglob("*")]
        raise FileNotFoundError(
            f"Could not uniquely locate the attached dataset; /kaggle/input contains {available[:30]}")
    return candidates.pop()


def materialize(dataset, name):
    directory = dataset / name
    if directory.is_dir():
        return directory
    archive = dataset / f"{name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {directory} or {archive}")
    target = EXTRACTED / name
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    return target


def main():
    dataset = find_dataset()
    print(f"Using dataset mount: {dataset}", flush=True)
    repository = materialize(dataset, "repository")
    resources = materialize(dataset, "resources")
    cache = resources / "experiments/same_window_full_v3"
    os.environ["EE6405_STORAGE"] = str(resources)
    os.environ["EE6405_UPSTREAM"] = str(resources / "RACE")
    sys.path.insert(0, str(repository))
    os.chdir(repository)
    run(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "torch-geometric==2.6.1")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
    results = {}
    for condition in ("text", "graph", "shuffled", "edu"):
        target = OUTPUT / condition
        run(sys.executable, "-m", "experiments.train_same_window",
            "--condition", condition,
            "--cache", str(cache),
            "--output", str(target),
            "--smoke",
            "--device", "cuda")
        results[condition] = json.loads((target / "smoke.json").read_text())
    summary = {
        "status": "passed",
        "cuda": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "conditions": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
