"""Private full 2x2x3 segmentation/consistency matrix, with held-out calibration."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    archives = list(Path("/kaggle/input").rglob("fusion_followup_code_v2.zip"))
    if len(archives) != 1:
        raise ValueError("Expected one frozen extension code archive")
    code = Path("/kaggle/working/fusion_followup_code_v2")
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
    from experiments.fusion_extensions import EXTENSION_CONDITIONS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no formal extension training started")
    cache = resources / "experiments/same_window_full_v3"
    output = Path("/kaggle/working/fusion_extensions_experiment_v1")
    output.mkdir()
    code_hash = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    write_json(output / "protocol.json", {
        "conditions": list(EXTENSION_CONDITIONS), "seeds": [3407, 42, 2026],
        "batch_size": 16, "epochs": 20, "chunk_size": 32,
        "training_forwards_per_batch": 2, "consistency_weights": [0, 1],
        "code_archive_sha256": code_hash,
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "validation": "Source-group disjoint selection/calibration halves; calibration never selects checkpoint",
        "selection": "All 12 factor/seed runs predeclared; no test-based candidate selection",
        "scope": "Existing HART source groups; post-hoc study, not new independent data",
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
        for condition in EXTENSION_CONDITIONS:
            stage("cuda_smoke", condition=condition)
            target = output / "smoke" / condition
            run("experiments.train_fusion_extensions", "--condition", condition,
                "--cache", cache, "--output", target, "--device", "cuda", "--smoke")
            smoke = json.loads((target / "smoke.json").read_text())
            if smoke["status"] != "passed" or smoke["test_evaluated"]:
                raise ValueError("Invalid extension smoke evidence")
            (target / "best.pt").unlink()
        for seed in (3407, 42, 2026):
            for condition in EXTENSION_CONDITIONS:
                name = f"{condition}_s{seed}"
                target = output / "runs" / name
                stage("validation_training", run=name)
                run("experiments.train_fusion_extensions", "--condition", condition,
                    "--seed", seed, "--cache", cache, "--output", target, "--device", "cuda")
                stage("frozen_evaluation", run=name)
                run("experiments.evaluate_fusion_extensions", "--run", target,
                    "--cache", cache, "--output", output / "evaluation" / name,
                    "--device", "cuda", "--evaluate-test")
                result = json.loads((output / "evaluation" / name / "result.json").read_text())
                finished.append({
                    "condition": condition, "seed": seed,
                    "test_macro_f1": result["test"]["macro_f1"],
                    "best_checkpoint_sha256": result["checkpoint_sha256"],
                })
        write_json(output / "summary.json", {
            "status": "completed", "runs": finished, "formal_training_runs": 12,
            "test_used_for_selection": False,
            "claim": "Report both matched factors, interaction, every seed and every unsuccessful condition",
        })
        stage("completed")
    except Exception as error:
        stage("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
