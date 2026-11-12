"""Private Kaggle execution of the frozen fusion screen/confirmation protocol."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import zipfile


ARCHIVE_NAME = "edu_fusion_code_v3.zip"
OUTPUT = Path("/kaggle/working/edu_fusion_experiment_v1")
EPOCHS = 20
BATCH_SIZE = 16
SEEDS = (3407, 42, 2026)
BASELINE = "fusion_cls"
CANDIDATES = ("fusion_mean", "fusion_attention", "fusion_gated")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_module(module, *arguments):
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def select_candidate(screen):
    if set(screen) != {BASELINE, *CANDIDATES}:
        raise ValueError("Screening requires the baseline and all three fusion candidates")
    for condition, result in screen.items():
        if (result["condition"] != condition or result["seed"] != 3407
                or result["status"] != "validation_completed"
                or result["test_evaluated"] or result["test"] is not None):
            raise ValueError("Screening must be complete, matched-seed and validation-only")
    selected = min(
        CANDIDATES,
        key=lambda name: (-screen[name]["validation"]["macro_f1"],
                          screen[name]["parameters"]["trainable"], name),
    )
    return {
        "selected": selected,
        "baseline": BASELINE,
        "rule": "Best validation Macro-F1 among fusion variants; ties: fewer trainable parameters, then name",
        "validation_gap": screen[selected]["validation"]["macro_f1"]
                          - screen[BASELINE]["validation"]["macro_f1"],
        "all_validation_macro_f1": {
            name: result["validation"]["macro_f1"] for name, result in screen.items()
        },
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_used_for_selection": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    archives = list(Path("/kaggle/input").rglob(ARCHIVE_NAME))
    code = Path("/kaggle/working/edu_fusion_code_v3")
    archive_sha256 = None
    if len(archives) == 1:
        with zipfile.ZipFile(archives[0]) as bundle:
            bundle.extractall(code)
        archive_sha256 = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    elif not archives:
        candidates = {path.parent.parent for path in Path("/kaggle/input").rglob("edu_fusion.py")
                      if path.parent.name == "experiments"}
        if len(candidates) != 1:
            raise ValueError(f"Expected one unpacked fusion code root, found {len(candidates)}")
        shutil.copytree(candidates.pop(), code)
    else:
        raise ValueError(f"Expected at most one {ARCHIVE_NAME}, found {len(archives)}")
    content_manifest = {
        str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(code.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".json", ".md")
    }
    code_content_sha256 = hashlib.sha256(
        json.dumps(content_manifest, sort_keys=True).encode()
    ).hexdigest()
    sys.path.insert(0, str(code))
    os.chdir(code)
    # Reuse the prior working resource-discovery and archive-extraction code.
    from cloud.kaggle.run_same_window_smoke import find_dataset, materialize

    resources = materialize(find_dataset(), "resources")
    os.environ.update(
        EE6405_STORAGE=str(resources),
        EE6405_UPSTREAM=str(resources / "RACE"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "torch-geometric==2.6.1"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; formal fusion training was not started")
    cache = resources / "experiments/same_window_full_v3"
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve existing pipeline output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "code_manifest.json", content_manifest)

    def stage(name, **extra):
        payload = {
            "stage": name, "time_utc": datetime.now(timezone.utc).isoformat(),
            "code_archive_sha256": archive_sha256,
            "code_content_sha256": code_content_sha256,
            "bootstrap_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resource_dataset": "zhengyingqi/ee6405-same-window-gpu-v1",
            "gpu": torch.cuda.get_device_name(0), **extra,
        }
        write_json(OUTPUT / "pipeline.json", payload)
        print(json.dumps(payload), flush=True)

    try:
        stage("cuda_smoke")
        run_module("experiments.run_fusion_queue", "--cache", cache,
                   "--output", OUTPUT / "smoke", "--smoke")
        smoke = json.loads((OUTPUT / "smoke/queue.json").read_text())
        if smoke["status"] != "completed" or smoke["test_evaluated"]:
            raise ValueError("All four CUDA smoke conditions must pass without opening test")
        stage("validation_screen")
        run_module("experiments.run_fusion_queue", "--cache", cache,
                   "--output", OUTPUT / "screen", "--seeds", 3407,
                   "--epochs", EPOCHS, "--batch-size", BATCH_SIZE)
        screen = {
            name: json.loads((OUTPUT / f"screen/{name}_s3407/result.json").read_text())
            for name in (BASELINE, *CANDIDATES)
        }
        selection = select_candidate(screen)
        selection["code_archive_sha256"] = archive_sha256
        selection["code_content_sha256"] = code_content_sha256
        write_json(OUTPUT / "selection.json", selection)
        selected = selection["selected"]
        stage("confirmation", selected=selected)
        run_module("experiments.run_fusion_queue", "--cache", cache,
                   "--output", OUTPUT / "confirmation",
                   "--conditions", BASELINE, selected, "--seeds", 42, 2026,
                   "--epochs", EPOCHS, "--batch-size", BATCH_SIZE)
        stage("frozen_checkpoint_evaluation", selected=selected)
        metrics = {}
        for condition in (BASELINE, selected):
            metrics[condition] = {}
            for seed in SEEDS:
                source = OUTPUT / ("screen" if seed == 3407 else "confirmation")
                run = source / f"{condition}_s{seed}"
                destination = OUTPUT / f"evaluation/{condition}_s{seed}"
                run_module("experiments.evaluate_edu_fusion", "--run", run,
                           "--cache", cache, "--output", destination,
                           "--device", "cuda", "--batch-size", BATCH_SIZE, "--evaluate-test")
                result = json.loads((destination / "result.json").read_text())
                metrics[condition][str(seed)] = result["splits"]["test"]["metrics"]
        differences = [
            metrics[selected][str(seed)]["macro_f1"] - metrics[BASELINE][str(seed)]["macro_f1"]
            for seed in SEEDS
        ]
        summary = {
            "status": "completed", "selection": selection, "test": metrics,
            "macro_f1": {
                condition: {
                    "mean": statistics.mean(row["macro_f1"] for row in values.values()),
                    "sample_sd": statistics.stdev(row["macro_f1"] for row in values.values()),
                } for condition, values in metrics.items()
            },
            "paired_candidate_minus_baseline": {
                "seeds": list(SEEDS), "differences": differences,
                "mean": statistics.mean(differences),
                "sample_sd": statistics.stdev(differences),
            },
            "formal_training_runs": 8,
            "test_scope": "Repeated existing HART holdout, not a new independent dataset",
            "performance_claim": "Report the measured difference, including zero or negative results",
        }
        write_json(OUTPUT / "summary.json", summary)
        stage("completed", selected=selected)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        stage("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
