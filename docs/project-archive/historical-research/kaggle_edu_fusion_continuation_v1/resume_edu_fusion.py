"""Continue the frozen protocol after a terminal first-stage job."""

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


OUTPUT = Path("/kaggle/working/edu_fusion_experiment_v1")
PREVIOUS_KERNEL = "zhengyingqi/ee6405-text-edu-fusion-experiment-v1"
CORE_SHA256 = {
    "edu_fusion.py": "a345c0247b8139fb86a2bd910d2b8964f27eaef9e9486aad5a0eff9c44d23a55",
    "train_same_window.py": "165979c9a49bc317a8195ef7e81d02f6031f131b0fbc05a707ef04968611060d",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_completed(run, condition, seed, code, cache_hash):
    """Absent results mean incomplete; malformed completed runs are errors."""
    if not (run / "result.json").is_file():
        return False
    result, recorded = load(run / "result.json"), load(run / "config.json")
    config = recorded["config"]
    if (result["status"] != "validation_completed"
            or result["condition"] != condition or result["seed"] != seed
            or result["test_evaluated"] or result["test"] is not None
            or recorded["training_only_smoke"] or not recorded["validation_only"]
            or config["condition"] != condition or config["seed"] != seed
            or config["max_epochs"] != 20 or config["batch_size"] != 16):
        raise ValueError(f"Invalid completed training evidence: {run}")
    if recorded["cache_manifest_sha256"] != cache_hash:
        raise ValueError(f"Cache hash mismatch: {run}")
    if digest(run / "best.pt") != result["best_checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {run}")
    sources = load(run / "source_hashes.json")
    if not CORE_SHA256.items() <= sources.items():
        raise ValueError(f"Frozen core source hashes missing or changed: {run}")
    for name, expected in sources.items():
        if (digest(code / "experiments" / name) != expected
                or digest(run / "source" / name) != expected):
            raise ValueError(f"Training source hash mismatch: {run / name}")
    return True


def main():
    previous = [path.parent for path in Path("/kaggle/input").rglob("selection.json")
                if path.parent.name == "edu_fusion_experiment_v1"]
    if not previous:
        archives = list(Path("/kaggle/input").rglob("stage1_verified.zip"))
        if len(archives) == 1:
            extracted = Path("/kaggle/working/first_stage_inputs")
            with zipfile.ZipFile(archives[0]) as archive:
                archive.extractall(extracted)
            previous = [extracted / "edu_fusion_experiment_v1"]
    if len(previous) != 1:
        raise ValueError(f"Expected one prior experiment output, found {len(previous)}")
    previous = previous[0]
    if (previous / "summary.json").exists() or (previous / "evaluation").exists():
        raise ValueError("Continuation expects unfinished training with no final test access")
    selection = load(previous / "selection.json")
    if (selection["baseline"] != "fusion_cls" or selection["selected"] != "fusion_gated"
            or selection["seeds"] != [3407, 42, 2026]
            or selection["test_used_for_selection"]):
        raise ValueError("The previously frozen candidate/protocol differs")
    roots = sorted({
        path.parent.parent for path in Path("/kaggle/input").rglob("edu_fusion.py")
        if path.parent.name == "experiments"
        and all((path.parent / name).is_file()
                and digest(path.parent / name) == expected
                for name, expected in CORE_SHA256.items())
    })
    if not roots:
        raise ValueError("The frozen v3 training code was not mounted")
    code = Path("/kaggle/working/edu_fusion_code_v3")
    shutil.copytree(roots[0], code)
    expected_manifest = load(previous / "code_manifest.json")
    for name, expected in expected_manifest.items():
        if digest(code / name) != expected:
            raise ValueError(f"Code differs from first-stage execution: {name}")
    if hashlib.sha256(json.dumps(expected_manifest, sort_keys=True).encode()).hexdigest() != selection["code_content_sha256"]:
        raise ValueError("Code manifest differs from the frozen selection")
    sys.path.insert(0, str(code))
    os.chdir(code)
    from cloud.kaggle.run_same_window_smoke import find_dataset, materialize
    from cloud.kaggle.run_edu_fusion import select_candidate

    resources = materialize(find_dataset(), "resources")
    cache = resources / "experiments/same_window_full_v3"
    os.environ.update(
        EE6405_STORAGE=str(resources), EE6405_UPSTREAM=str(resources / "RACE"),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "torch-geometric==2.6.1"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    prior_pipeline = load(previous / "pipeline.json")
    if torch.cuda.get_device_name(0) != prior_pipeline["gpu"]:
        raise ValueError("GPU type differs from the first stage")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    screen = {condition: load(previous / f"screen/{condition}_s3407/result.json")
              for condition in selection["all_validation_macro_f1"]}
    recalculated_selection = select_candidate(screen)
    for key in ("selected", "baseline", "all_validation_macro_f1", "validation_gap",
                "seeds", "epochs", "batch_size", "test_used_for_selection"):
        if recalculated_selection[key] != selection[key]:
            raise ValueError(f"Frozen selection mismatch: {key}")
    shutil.copy2(previous / "selection.json", OUTPUT / "selection.json")
    shutil.copy2(previous / "code_manifest.json", OUTPUT / "code_manifest.json")
    smoke = load(previous / "smoke/queue.json")
    if smoke["status"] != "completed" or smoke["test_evaluated"]:
        raise ValueError("Prior CUDA smoke was not completed without test access")
    shutil.copytree(previous / "smoke", OUTPUT / "smoke",
                    ignore=shutil.ignore_patterns("*.pt"))
    continuation = {
        "source_kernel": PREVIOUS_KERNEL,
        "prior_terminal_state": "cancel_acknowledged",
        "prior_pipeline": prior_pipeline,
        "selection_sha256": digest(previous / "selection.json"),
        "reused_complete_runs": [], "restarted_incomplete_runs": [],
        "new_runs": [], "test_accessed_before_continuation": False,
        "policy": "Preserve completed runs; restart incomplete runs from base pretraining",
    }

    def stage(name, **extra):
        write_json(OUTPUT / "continuation.json", continuation)
        payload = {
            "stage": name, "time_utc": datetime.now(timezone.utc).isoformat(),
            "source_kernel": PREVIOUS_KERNEL,
            "code_content_sha256": selection["code_content_sha256"],
            "bootstrap_sha256": digest(Path(__file__)),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(), **extra,
        }
        write_json(OUTPUT / "pipeline.json", payload)
        print(json.dumps(payload), flush=True)

    def run_module(module, *arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    try:
        stage("verify_and_reuse")
        cache_hash = digest(cache / "manifest.json")
        conditions = list(selection["all_validation_macro_f1"])
        expected_runs = [("screen", condition, 3407) for condition in conditions]
        expected_runs += [("confirmation", condition, seed)
                          for seed in (42, 2026)
                          for condition in (selection["baseline"], selection["selected"])]
        missing = []
        for phase, condition, seed in expected_runs:
            relative = Path(phase) / f"{condition}_s{seed}"
            source, target = previous / relative, OUTPUT / relative
            if validate_completed(source, condition, seed, code, cache_hash):
                shutil.copytree(source, target)
                continuation["reused_complete_runs"].append(str(relative))
            else:
                if phase == "screen":
                    raise ValueError("Every screening run must already be complete")
                missing.append((condition, seed, target))
                if source.exists():
                    continuation["restarted_incomplete_runs"].append(str(relative))
        stage("train_missing_runs", missing=len(missing))
        for condition, seed, target in missing:
            run_module("experiments.train_edu_fusion", "--cache", cache,
                       "--output", target, "--condition", condition, "--seed", seed,
                       "--device", "cuda", "--epochs", 20, "--batch-size", 16,
                       "--validation-only")
            if not validate_completed(target, condition, seed, code, cache_hash):
                raise ValueError(f"Continuation run did not finish: {target}")
            continuation["new_runs"].append(str(target.relative_to(OUTPUT)))
            stage("train_missing_runs", last_completed=target.name)
        stage("frozen_checkpoint_evaluation")
        metrics = {}
        for condition in (selection["baseline"], selection["selected"]):
            metrics[condition] = {}
            for seed in selection["seeds"]:
                phase = "screen" if seed == 3407 else "confirmation"
                run = OUTPUT / phase / f"{condition}_s{seed}"
                destination = OUTPUT / "evaluation" / f"{condition}_s{seed}"
                run_module("experiments.evaluate_edu_fusion", "--run", run,
                           "--cache", cache, "--output", destination,
                           "--device", "cuda", "--batch-size", 16, "--evaluate-test")
                metrics[condition][str(seed)] = load(destination / "result.json")["splits"]["test"]["metrics"]
        baseline, candidate = selection["baseline"], selection["selected"]
        differences = [metrics[candidate][str(seed)]["macro_f1"]
                       - metrics[baseline][str(seed)]["macro_f1"]
                       for seed in selection["seeds"]]
        write_json(OUTPUT / "summary.json", {
            "status": "completed", "selection": selection, "test": metrics,
            "macro_f1": {
                condition: {
                    "mean": statistics.mean(row["macro_f1"] for row in values.values()),
                    "sample_sd": statistics.stdev(row["macro_f1"] for row in values.values()),
                } for condition, values in metrics.items()
            },
            "paired_candidate_minus_baseline": {
                "seeds": selection["seeds"], "differences": differences,
                "mean": statistics.mean(differences),
                "sample_sd": statistics.stdev(differences),
            },
            "formal_training_runs": 8, "continuation": continuation,
            "interrupted_training_attempts": len(continuation["restarted_incomplete_runs"]),
            "test_scope": "Repeated existing HART holdout, not a new independent dataset",
            "performance_claim": "Report the measured difference, including zero or negative results",
        })
        stage("completed")
    except Exception as error:
        stage("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
