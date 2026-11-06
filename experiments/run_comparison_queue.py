"""Run the predeclared GPU comparisons sequentially, preserving each run.

This local queue is narrower than the active Goal: a completed queue means the
listed training and reload checks finished, not that the whole project is done.
"""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path

from experiments.paths import STORAGE
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]

RUNS = STORAGE / "experiments/neural_runs"
REPORTS = ROOT / "experiments/reports"
QUEUE = STORAGE / "experiments/comparison_queue_v1"
PLAN = [("race_no_supcon", 3407), ("roberta", 42), ("race", 42),
        ("roberta", 2026), ("race", 2026)]


def read(path):
    return json.loads(path.read_text())


def write_status(**values):
    document = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "queue_pid": os.getpid(), **values}
    temporary = QUEUE / "status.tmp"
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(QUEUE / "status.json")
    print(json.dumps(document), flush=True)


def process_is_training(pid, run):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    kind, seed = run.rsplit("_s", 1)
    return "experiments.train_neural" in command and f"--model-type {kind}" in command and f"--seed {seed}" in command


def wait_for_existing(run):
    status_path = RUNS / run / "status.json"
    status = read(status_path)
    while process_is_training(status["pid"], run):
        write_status(status="waiting_for_existing_training", run=run, training_pid=status["pid"])
        time.sleep(30)
        status = read(status_path)
    if status["status"] != "completed" or not (REPORTS / run / "result.json").exists():
        raise RuntimeError(f"Existing run did not complete: {run}: {status}")


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
        print(json.dumps({"wait_for": "race_s3407", "remaining_training": PLAN,
                          "selection": "validation Macro-F1 only; frozen recipe for every listed run",
                          "verification": "strict reload and all 3194 test predictions after each run",
                          "auxiliary": "HC3 binary evaluation for the three seed-3407 neural conditions"}, indent=2))
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (RUNS / "race_s3407/result.json").exists():
                assert read(RUNS / "race_s3407/status.json")["status"] == "completed"
            else:
                wait_for_existing("race_s3407")
            verify("race_s3407")
            auxiliary("roberta_s3407")
            for kind, seed in PLAN:
                run = f"{kind}_s{seed}"
                if (RUNS / run).exists():
                    if (RUNS / run / "result.json").exists():
                        assert read(RUNS / run / "status.json")["status"] == "completed"
                    else:
                        wait_for_existing(run)
                else:
                    execute("experiments.train_neural", ["--model-type", kind, "--seed", str(seed)], f"{run}.log")
                verify(run)
                if seed == 3407:
                    # CPU parsing proceeds separately while the ablation trains.
                    # A complete, aligned audit is required; a partial JSONL is
                    # never an evaluation-ready file.
                    audit = STORAGE / "experiments/hc3_aux_v1_1/test_parsed_aligned.audit.json"
                    deadline = time.monotonic() + 3600
                    while not audit.exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError("HC3 parser did not finish within one hour; inspect its log")
                        write_status(status="waiting_for_hc3_parser")
                        time.sleep(30)
                    assert read(audit)["status"] == "passed"
                    auxiliary("race_s3407")
                    auxiliary(run)
            write_status(status="completed", scope="Training/reload/auxiliary queue only; remaining Goal deliverables require review")
        except Exception as error:
            write_status(status="failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
