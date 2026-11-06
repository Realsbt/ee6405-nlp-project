"""Run the predeclared GPU comparisons sequentially, preserving each run.

This local queue is narrower than the active Goal: a completed queue means the
listed training and reload checks finished, not that the whole project is done.
The 2x2 loss/architecture comparison is represented by explicit jobs so the
loss switch cannot be inferred accidentally from the model name.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:
    fcntl = None


ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("EE6405_STORAGE", ROOT))
RUNS = STORAGE / "experiments/neural_runs"
REPORTS = ROOT / "experiments/reports"
QUEUE = STORAGE / "experiments/comparison_queue_v1"
JOBS = [
    {"run": "roberta_supcon_s3407", "model_type": "roberta", "seed": 3407, "supcon": True},
    {"run": "roberta_supcon_s42", "model_type": "roberta", "seed": 42, "supcon": True},
    {"run": "roberta_supcon_s2026", "model_type": "roberta", "seed": 2026, "supcon": True},
    {"run": "race_no_supcon_s42", "model_type": "race", "seed": 42, "supcon": False},
    {"run": "race_no_supcon_s2026", "model_type": "race", "seed": 2026, "supcon": False},
]


def read(path):
    return json.loads(path.read_text())


def write_status(**values):
    document = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "queue_pid": os.getpid(), **values}
    temporary = QUEUE / "status.tmp"
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(QUEUE / "status.json")
    print(json.dumps(document), flush=True)


def process_is_training(pid, job):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    switch = "--use-supcon" if job["supcon"] else "--no-supcon"
    return ("experiments.train_neural" in command
            and f"--model-type {job['model_type']}" in command
            and f"--seed {job['seed']}" in command
            and switch in command)


def wait_for_existing(job):
    status_path = RUNS / job["run"] / "status.json"
    status = read(status_path)
    while process_is_training(status["pid"], job):
        write_status(status="waiting_for_existing_training", run=job["run"], training_pid=status["pid"])
        time.sleep(30)
        status = read(status_path)
    if status["status"] != "completed" or not (REPORTS / job["run"] / "result.json").exists():
        raise RuntimeError(f"Existing run did not complete: {job['run']}: {status}")


def execute(module, arguments, log_name):
    command = [sys.executable, "-u", "-m", module, *arguments]
    with (REPORTS / log_name).open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        write_status(status="running", command=command, child_pid=child.pid, log=str(REPORTS / log_name))
        code = child.wait()
    if code:
        raise RuntimeError(f"Command exited {code}; inspect {REPORTS / log_name}")


def verify(run):
    path = REPORTS / run / "verification.json"
    result = read(REPORTS / run / "result.json")
    if path.exists():
        check = read(path)
        if check["status"] == "passed" and check["checkpoint_sha256"] == result["checkpoint_sha256"]:
            return
    execute("experiments.verify_neural", ["--run", run], f"{run}_verify.log")


def auxiliary(run):
    path = REPORTS / run / "hc3_aux_v1_1_result.json"
    if path.exists() and read(path)["status"] == "completed":
        return
    execute("experiments.evaluate_external", ["--run", run], f"{run}_hc3_aux.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"wait_for": "race_s3407", "remaining_training": JOBS,
                          "selection": "validation Macro-F1 only; frozen recipe for every listed run",
                          "verification": "strict reload and all 3194 test predictions after each run",
                          "matrix": {
                              "roberta_no_supcon": ["roberta_s3407", "roberta_s42", "roberta_s2026"],
                              "roberta_supcon": ["roberta_supcon_s3407", "roberta_supcon_s42", "roberta_supcon_s2026"],
                              "race_no_supcon": ["race_no_supcon_s3407", "race_no_supcon_s42", "race_no_supcon_s2026"],
                              "race_supcon": ["race_s3407", "race_s42", "race_s2026"],
                          }}, indent=2))
        return
    if fcntl is None:
        raise RuntimeError("The training queue requires a POSIX host with fcntl; use --plan on Windows")
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (RUNS / "race_s3407/result.json").exists():
                assert read(RUNS / "race_s3407/status.json")["status"] == "completed"
            else:
                wait_for_existing({"run": "race_s3407", "model_type": "race", "seed": 3407, "supcon": True})
            verify("race_s3407")
            auxiliary("roberta_s3407")
            for job in JOBS:
                run = job["run"]
                if (RUNS / run).exists():
                    if (RUNS / run / "result.json").exists():
                        assert read(RUNS / run / "status.json")["status"] == "completed"
                    else:
                        wait_for_existing(job)
                else:
                    switch = "--use-supcon" if job["supcon"] else "--no-supcon"
                    execute("experiments.train_neural",
                            ["--model-type", job["model_type"], "--seed", str(job["seed"]),
                             "--run-name", run, switch], f"{run}.log")
                verify(run)
            write_status(status="completed", scope="Training/reload/auxiliary queue only; remaining Goal deliverables require review")
        except Exception as error:
            write_status(status="failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
