"""Complete the five missing runs of the fixed 2x2 ablation, sequentially."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ablation import CONDITIONS, MISSING_RUNS, NEURAL_RUNS, SEEDS, recipe_for_run, resolve_config, training_arguments
from experiments.paths import PROJECT_ROOT, STORAGE, resolve_recorded_path


REPORTS = PROJECT_ROOT / "experiments/reports"
RUNS = STORAGE / "experiments/neural_runs"
QUEUE = STORAGE / "experiments/ablation_2x2_v1"


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with resolve_recorded_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status(**values):
    document = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "queue_pid": os.getpid(), **values}
    temporary = QUEUE / "status.tmp"
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(QUEUE / "status.json")
    print(json.dumps(document), flush=True)


def execute(module, arguments, log_name):
    command = [sys.executable, "-B", "-u", "-m", module, *arguments]
    log_path = QUEUE / log_name
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        status(status="running", command=command, child_pid=child.pid, log=str(log_path))
        code = child.wait()
    if code:
        raise RuntimeError(f"Exit {code}: {log_path}")


def verify_complete(run, base):
    result = read(REPORTS / run / "result.json")
    assert result["status"] == "completed"
    assert resolve_config(result["config"]) == recipe_for_run(base, run), f"Recipe mismatch: {run}"
    assert sha256(result["checkpoint_path"]) == result["checkpoint_sha256"], f"Checkpoint changed: {run}"
    verification = REPORTS / run / "verification.json"
    if not verification.exists() or read(verification).get("checkpoint_sha256") != result["checkpoint_sha256"]:
        execute("experiments.verify_neural", ["--run", run], f"{run}_verify.log")
    check = read(verification)
    assert check["status"] == "passed" and check["samples"] == 3194
    assert check["strict_checkpoint_reload"] and check["predictions_identical"] and check["metrics_match_saved_result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Read-only plan; starts no process")
    parser.add_argument("--status", action="store_true", help="Read-only progress for the five added runs")
    parser.add_argument("--smoke-only", action="store_true", help="Check all four conditions without loading test data")
    args = parser.parse_args()
    if args.status:
        progress = []
        for run in MISSING_RUNS:
            path = RUNS / run / "status.json"
            item = read(path) if path.exists() else {"status": "not_started"}
            check = REPORTS / run / "verification.json"
            latest = {}
            events = RUNS / run / "events.jsonl"
            if item["status"] == "running" and events.exists():
                for line in reversed(events.read_text().splitlines()):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A concurrent append may leave the last line incomplete.
                    latest = {key: event[key] for key in ("event", "epoch", "batch", "batches") if key in event}
                    break
            progress.append({"run": run, "status": item["status"], "completed_epochs": item.get("last_completed_epoch", 0),
                             "best_epoch": item.get("best_epoch"),
                             "latest": latest,
                             "reload_verified": check.exists() and read(check).get("status") == "passed"})
        print(json.dumps({"verified": sum(item["reload_verified"] for item in progress), "total": len(MISSING_RUNS),
                          "runs": progress}, indent=2))
        return
    base = read(PROJECT_ROOT / "experiments/configs/neural_v1.json")
    plan = {"protocol_id": base["protocol_id"], "recipe_id": base["recipe_id"], "seeds": SEEDS,
            "conditions": CONDITIONS, "required_runs": NEURAL_RUNS, "additional_runs": MISSING_RUNS,
            "commands": {run: training_arguments(run) for run in MISSING_RUNS},
            "selection": "Validation Macro-F1 only; same fixed split, recipe, batches and base weights",
            "scope": "HART 2x2 comparison; existing HC3 evidence remains a separate seed-3407 analysis"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan_path = QUEUE / "plan.json"
            if plan_path.exists():
                assert read(plan_path) == json.loads(json.dumps(plan)), "Recorded plan differs"
            else:
                plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            # Hash the original seven completed checkpoints before training.
            preserved_path = QUEUE / "preserved_checkpoints.json"
            if preserved_path.exists():
                preserved = read(preserved_path)
            else:
                preserved = {run: {"path": read(REPORTS / run / "result.json")["checkpoint_path"],
                                   "sha256": read(REPORTS / run / "result.json")["checkpoint_sha256"]}
                             for run in NEURAL_RUNS if run not in MISSING_RUNS}
                preserved_path.write_text(json.dumps(preserved, indent=2) + "\n")
            for item in preserved.values():
                assert sha256(item["path"]) == item["sha256"], "Existing checkpoint changed"
            for kind in CONDITIONS:
                run = f"{kind}_s3407"
                smoke_id = f"smoke_ablation_v1_{run}"
                smoke_path = REPORTS / smoke_id / "smoke.json"
                if not smoke_path.exists():
                    execute("experiments.train_neural", [*training_arguments(run), "--smoke", "--run-name", f"ablation_v1_{run}"],
                            f"{smoke_id}.log")
                smoke = read(smoke_path)
                expected = recipe_for_run(base, run)
                expected.update(max_epochs=1, early_stopping_patience=1)
                assert smoke["status"] == "passed" and smoke["config"] == expected
                assert not smoke["test_partition_loaded"] and smoke["finite_gradients"] and smoke["classifier_updated"]
                assert (smoke["mean_train_supcon"] > 0) == expected["use_supcon"]
                assert sha256(RUNS / smoke_id / "source/experiments/train_neural.py") == sha256(PROJECT_ROOT / "experiments/train_neural.py")
            if args.smoke_only:
                status(status="smoke_passed", test_partition_loaded=False)
                return
            for run in MISSING_RUNS:
                if not (REPORTS / run / "result.json").exists():
                    arguments = training_arguments(run)
                    if (RUNS / run).exists():
                        previous = read(RUNS / run / "status.json")
                        cmdline = Path(f"/proc/{previous['pid']}/cmdline")
                        if cmdline.exists() and b"experiments.train_neural" in cmdline.read_bytes():
                            raise RuntimeError(f"Training process still running for {run}")
                        arguments.append("--resume")
                    execute("experiments.train_neural", arguments, f"{run}.log")
                verify_complete(run, base)
                status(status="run_verified", run=run)
            for item in preserved.values():
                assert sha256(item["path"]) == item["sha256"], "Existing checkpoint changed"
            status(status="completed", verified_additional_runs=MISSING_RUNS, original_checkpoints_unchanged=True,
                   scope="Five added HART training runs and full checkpoint reloads; analysis is a separate step")
        except Exception as error:
            status(status="failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
