"""Audit the current independent loss controls and wrapper from outside the repo.

Run after run_ablation_queue. The two small wrapper smokes read train/val only.
"""

import json
import os
from pathlib import Path
import subprocess

from experiments.ablation import CONDITIONS, recipe_for_run
from experiments.common import file_hash, read_jsonl, write_json
from experiments.paths import PROJECT_ROOT, STORAGE


def read(path):
    return json.loads(path.read_text())


def main():
    reports = PROJECT_ROOT / "experiments/reports"
    runs = STORAGE / "experiments/neural_runs"
    trainer_hash = file_hash(PROJECT_ROOT / "experiments/train_neural.py")
    base = read(PROJECT_ROOT / "experiments/configs/neural_v1.json")
    smokes = {}
    for kind in CONDITIONS:
        path = runs / f"smoke_ablation_v1_{kind}_s3407/smoke.json"
        smoke = read(path)
        expected = recipe_for_run(base, f"{kind}_s3407")
        expected.update(max_epochs=1, early_stopping_patience=1)
        assert smoke["status"] == "passed" and smoke["config"] == expected
        assert smoke["training_only_smoke"] and not smoke["test_partition_loaded"]
        assert smoke["finite_gradients"] and smoke["frozen_layer_unchanged"] and smoke["classifier_updated"]
        assert (smoke["mean_train_supcon"] > 0) == expected["use_supcon"]
        assert file_hash(path.parent / "source/experiments/train_neural.py") == trainer_hash
        smokes[kind] = {"smoke_path": str(path), "smoke_sha256": file_hash(path),
                        "use_supcon": expected["use_supcon"], "mean_train_supcon": smoke["mean_train_supcon"]}
    write_json(reports / "ablation_entrypoint.json", {"status": "passed", "training_entrypoint_sha256": trainer_hash,
               "smokes": smokes, "test_partition_loaded": False,
               "scope": "All four architecture/loss combinations actually optimized on train/val smoke subsets"})
    wrapper = str(PROJECT_ROOT / "experiments/run_neural.sh")
    wrapper_smokes = {}
    environment = dict(os.environ, EE6405_STORAGE=str(STORAGE), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for kind in ("roberta", "race"):
        alias = f"io_ablation_v1_{kind}_s3407"
        path = runs / f"smoke_{alias}/smoke.json"
        if not path.exists():
            with (reports / f"smoke_{alias}.log").open("w") as log:
                subprocess.run(["bash", wrapper, "--model-type", kind, "--supcon" if kind == "race" else "--no-supcon",
                                "--seed", "3407", "--run-name", alias, "--smoke"],
                               cwd="/tmp", env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke = read(path)
        assert smoke["status"] == "passed" and not smoke["test_partition_loaded"]
        assert file_hash(path.parent / "source/experiments/train_neural.py") == trainer_hash
        initialized = [e for e in read_jsonl(path.parent / "events.jsonl") if e["event"] == "initialized"]
        assert len(initialized) == 1 and initialized[0]["train_records"] == 32 and initialized[0]["val_records"] == 8
        wrapper_smokes[kind] = {"run_id": f"smoke_{alias}", "smoke_path": str(path), "smoke_sha256": file_hash(path),
                                "train_records": 32, "val_records": 8, "test_partition_loaded": False}
    guards = {}
    for name, alias, expected_code in (("invalid_name", "../invalid", 2), ("existing_name", "io_ablation_v1_roberta_s3407", 1)):
        path = reports / f"ablation_retraining_{name}.log"
        with path.open("w") as log:
            process = subprocess.run(["bash", wrapper, "--model-type", "roberta", "--no-supcon", "--seed", "3407",
                                      "--run-name", alias, "--smoke"], cwd="/tmp", env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        assert process.returncode == expected_code
        guards[name] = {"exit_code": process.returncode, "log_path": str(path), "log_sha256": file_hash(path)}
    original = read(STORAGE / "experiments/ablation_2x2_v1/preserved_checkpoints.json")
    for item in original.values():
        assert file_hash(item["path"]) == item["sha256"]
    old = reports / "retraining_entrypoint.json"
    archive = reports / "retraining_entrypoint_before_2x2.json"
    if not archive.exists():
        archive.write_bytes(old.read_bytes())
    # Retain and recheck the old smokes as historical evidence.
    historical = read(archive)
    for item in historical["smokes"].values():
        assert file_hash(item["smoke_path"]) == item["smoke_sha256"]
    write_json(old, {"status": "passed", "training_entrypoint_sha256": trainer_hash, "invocation_cwd": "/tmp",
                    "smokes": wrapper_smokes, "guards": guards, "existing_smoke_files_preserved": True,
                    "canonical_checkpoint_hashes_unchanged": {run: item["sha256"] for run, item in original.items()},
                    "historical_evidence": str(archive),
                    "scope": "Current wrapper smokes from /tmp, invalid-name/overwrite rejection; original smokes and checkpoints preserved"})
    print(json.dumps({"status": "passed", "four_condition_smokes": list(smokes), "wrapper_smokes": list(wrapper_smokes),
                      "guards": {key: value["exit_code"] for key, value in guards.items()}}, indent=2))


if __name__ == "__main__":
    main()
