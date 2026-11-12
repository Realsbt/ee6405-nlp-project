"""Run the predeclared three-scale, three-seed mechanism matrix privately."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


OUTPUT_NAME = "fusion_scale_experiment_v1"
ARCHIVE_NAME = "fusion_followup_code_v1.zip"


def main():
    archives = list(Path("/kaggle/input").rglob(ARCHIVE_NAME))
    if len(archives) != 1:
        raise ValueError(f"Expected one {ARCHIVE_NAME}, found {len(archives)}")
    code = Path("/kaggle/working/fusion_followup_code_v1")
    with zipfile.ZipFile(archives[0]) as bundle:
        bundle.extractall(code)
    os.chdir(code)
    sys.path.insert(0, str(code))
    from cloud.kaggle.run_same_window_smoke import find_dataset, materialize

    resources = materialize(find_dataset(), "resources")
    os.environ.update(
        EE6405_STORAGE=str(resources), EE6405_UPSTREAM=str(resources / "RACE"),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "torch-geometric==2.6.1"], check=True)
    import torch
    from experiments.common import file_hash, write_json
    from experiments.fusion_scale_controls import SCALE_CONDITIONS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no formal scale training started")
    cache = resources / "experiments/same_window_full_v3"
    output = Path("/kaggle/working") / OUTPUT_NAME
    output.mkdir()
    code_hash = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    write_json(output / "protocol.json", {
        "conditions": list(SCALE_CONDITIONS), "seeds": [3407, 42, 2026],
        "epochs": 20, "batch_size": 16, "selection": "All nine conditions predeclared",
        "code_archive_sha256": code_hash,
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "scope": "Post-hoc repeated HART holdout; no candidate selection using test",
    })
    finished = []

    def stage(name, **extra):
        write_json(output / "pipeline.json", {
            "stage": name, "finished": finished, "gpu": torch.cuda.get_device_name(0),
            "code_archive_sha256": code_hash, **extra,
        })
        print(json.dumps({"stage": name, **extra}), flush=True)

    def run(module, *arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    try:
        for condition in SCALE_CONDITIONS:
            stage("cuda_smoke", condition=condition)
            smoke = output / "smoke" / condition
            run("experiments.train_fusion_scale", "--condition", condition,
                "--cache", cache, "--output", smoke, "--smoke", "--device", "cuda")
            evidence = json.loads((smoke / "smoke.json").read_text())
            if evidence["status"] != "passed" or evidence["test_evaluated"]:
                raise ValueError("Invalid scale smoke evidence")
            # Smoke weights are disposable; retain config, hashes and checks.
            (smoke / "best.pt").unlink()
        for seed in (3407, 42, 2026):
            for condition in SCALE_CONDITIONS:
                name = f"{condition}_s{seed}"
                target = output / "runs" / name
                stage("validation_training", run=name)
                run("experiments.train_fusion_scale", "--condition", condition,
                    "--seed", seed, "--cache", cache, "--output", target,
                    "--device", "cuda", "--validation-only")
                stage("frozen_evaluation", run=name)
                run("experiments.evaluate_fusion_scale", "--run", target,
                    "--cache", cache, "--output", output / "evaluation" / name,
                    "--device", "cuda", "--evaluate-test")
                result = json.loads((output / "evaluation" / name / "result.json").read_text())
                finished.append({
                    "condition": condition, "seed": seed,
                    "test_macro_f1": result["splits"]["test"]["metrics"]["macro_f1"],
                    "best_checkpoint_sha256": result["checkpoint_sha256"],
                })
        write_json(output / "summary.json", {
            "status": "completed", "runs": finished, "formal_training_runs": 9,
            "test_used_for_selection": False,
            "claim": "Measure adaptive versus global and global versus fixed; no promised gain",
        })
        stage("completed")
    except Exception as error:
        stage("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
