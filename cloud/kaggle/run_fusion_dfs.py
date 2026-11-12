"""Private validation-gated DFS; record failures, never search using test scores."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def materialize_code(inputs, destination):
    if destination.exists():
        raise FileExistsError("Preserve previous materialized code")
    manifests = list(inputs.rglob("fusion_dfs_code_manifest.json"))
    archives = list(inputs.rglob("fusion_followup_code_v3.zip"))
    if len(manifests) != 1 or len(archives) > 1:
        raise ValueError("Expected one frozen code manifest and at most one archive")
    manifest = json.loads(manifests[0].read_text())
    for name in manifest["files"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe code manifest path")
    archive_verified = False
    if archives:
        if hashlib.sha256(archives[0].read_bytes()).hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Frozen code archive hash differs")
        with zipfile.ZipFile(archives[0]) as bundle:
            if set(bundle.namelist()) != set(manifest["files"]):
                raise ValueError("Code archive members differ from manifest")
            bundle.extractall(destination)
        archive_verified = True
    else:
        roots = {path.parent.parent for path in inputs.rglob("fusion_dfs_policy.py")
                 if path.parent.name == "experiments"}
        if len(roots) != 1:
            raise ValueError("Expected one unpacked DFS source root")
        source = roots.pop()
        for name in manifest["files"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
    actual = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
              for name in manifest["files"]}
    if actual != manifest["files"]:
        raise ValueError("Frozen source content hash differs")
    return {
        "code_archive_sha256_expected": manifest["archive_sha256"],
        "code_archive_verified": archive_verified,
        "code_content_verified": True,
        "code_content_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
        "source_files": actual,
    }


def main():
    plans = list(Path("/kaggle/input").rglob("fusion_dfs_plan.json"))
    if len(plans) != 1:
        raise ValueError("Expected exactly one frozen DFS plan")
    plan = json.loads(plans[0].read_text())
    parent = plan["parent"]
    code = Path("/kaggle/working/fusion_followup_code_v3")
    bootstrap = materialize_code(Path("/kaggle/input"), code)
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
    from experiments.edu_fusion import FUSION_CONDITIONS
    from experiments.fusion_dfs_policy import SEEDS, continuation_gate, extension_tree

    if parent not in FUSION_CONDITIONS:
        raise ValueError("Invalid frozen parent")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no DFS training started")
    output = Path("/kaggle/working/fusion_dfs_experiment_v1")
    output.mkdir()
    cache = resources / "experiments/same_window_full_v3"
    ledger = {
        "status": "running", "parent": parent, "runs": [], "decisions": [], "failed_attempts": [],
        "test_used_for_decisions": False,
        **{key: value for key, value in bootstrap.items() if key != "source_files"},
        "plan_sha256": hashlib.sha256(plans[0].read_bytes()).hexdigest(),
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "priority": ["root_contribution", "gate_mechanism_if_supported",
                     "consistency", "segmentation_descendant", "segmentation_backtrack"],
        "scope": "Adaptive validation search on repeatedly inspected HART; no independent confirmation claim",
        "calibration_history": (
            "Parent architecture and historical checkpoints used full validation; "
            "only this run's checkpoint selection and DFS gates exclude calibration IDs"
        ),
    }
    write_json(output / "code_manifest.json", bootstrap["source_files"])
    write_json(output / "protocol.json", {**ledger, "plan": plan})
    finished = {}
    smokes = set()

    def persist(event, **extra):
        write_json(output / "ledger.json", ledger)
        write_json(output / "pipeline.json", {"stage": event, **extra})
        print(json.dumps({"event": event, **extra}), flush=True)

    def run(module, *arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    def train(kind, condition, seed):
        key = (kind, condition, seed)
        if key in finished:
            return finished[key]
        ledger["current_run"] = {
            "kind": kind, "condition": condition, "seed": seed, "phase": "smoke",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        module = "experiments.train_fusion_scale" if kind == "scale" else "experiments.train_fusion_extensions"
        extra = ["--validation-only"] if kind == "scale" else ["--parent", parent]
        if (kind, condition) not in smokes:
            smoke = output / "smoke" / condition
            persist("cuda_smoke", condition=condition)
            run(module, "--condition", condition, "--cache", cache, "--output", smoke,
                "--device", "cuda", "--smoke", *extra)
            checks = json.loads((smoke / "smoke.json").read_text())
            if checks["status"] != "passed" or checks["test_evaluated"]:
                raise ValueError("Invalid DFS CUDA smoke")
            (smoke / "best.pt").unlink()
            smokes.add((kind, condition))
        name = f"{condition}_s{seed}"
        target = output / "runs" / name
        ledger["current_run"].update(
            phase="formal_validation_training",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        persist("validation_training", run=name)
        run(module, "--condition", condition, "--seed", seed, "--cache", cache,
            "--output", target, "--device", "cuda", *extra)
        result = json.loads((target / "result.json").read_text())
        if result["test_evaluated"] or result["test"] is not None:
            raise ValueError("Search training must never inspect test")
        score = result["validation" if kind == "scale" else "selection_validation"]["macro_f1"]
        row = {
            "kind": kind, "condition": condition, "seed": seed, "parent": parent,
            "selection_validation_macro_f1": score,
            "selection_scope": "full_validation" if kind == "scale" else "frozen_selection_half",
            "best_epoch": result["best_epoch"],
            "checkpoint_sha256": result["best_checkpoint_sha256"],
            "elapsed_seconds": result["elapsed_seconds"],
            "trainable_parameters": result["parameters"]["trainable"],
            "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
            "gpu": result["gpu"],
            "test_evaluated_during_search": False,
        }
        ledger["runs"].append(row)
        ledger.pop("current_run", None)
        finished[key] = row
        persist("result_recorded", **row)
        return row

    def compare(kind, candidate, reference, seeds, pilot):
        scores = {}
        for condition in (reference, candidate):
            scores[condition] = {
                seed: train(kind, condition, seed)["selection_validation_macro_f1"]
                for seed in seeds
            }
        decision = continuation_gate(scores[candidate], scores[reference], pilot=pilot)
        decision.update(kind=kind, candidate=candidate, reference=reference)
        ledger["decisions"].append(decision)
        persist("branch_decision", **decision)
        return decision["decision"] == "continue"

    def explore(kind, candidate, reference):
        if not compare(kind, candidate, reference, SEEDS[:1], pilot=True):
            return False
        return compare(kind, candidate, reference, SEEDS, pilot=False)

    def prefer(candidate, incumbent):
        def ranking(condition):
            rows = [finished[("extensions", condition, seed)] for seed in SEEDS]
            return (-sum(r["selection_validation_macro_f1"] for r in rows) / len(rows),
                    rows[0]["trainable_parameters"], condition)
        winner = min((candidate, incumbent), key=ranking)
        ledger["decisions"].append({
            "branch": "backtrack_finalist_choice", "decision": winner,
            "reason": "Highest three-seed mean selection F1; ties: fewer parameters, then name",
            "compared_conditions": [candidate, incumbent],
            "mean_validation_macro_f1": {
                c: -ranking(c)[0] for c in (candidate, incumbent)},
            "test_used_for_decision": False,
        })
        persist("backtrack_finalist_choice", selected=winner)
        return winner == candidate

    try:
        scale_finalist = None
        if parent == "fusion_gated":
            for condition in ("scale_adaptive", "scale_global", "scale_fixed"):
                train("scale", condition, SEEDS[0])
            # All pilot outcomes are retained, including the unchosen alternative.
            eligible = []
            for condition in ("scale_global", "scale_fixed"):
                if compare("scale", condition, "scale_adaptive", SEEDS[:1], pilot=True):
                    eligible.append(condition)
            if eligible:
                best = min(eligible, key=lambda name: (
                    -finished[("scale", name, SEEDS[0])]["selection_validation_macro_f1"],
                    finished[("scale", name, SEEDS[0])]["trainable_parameters"], name))
                for other in eligible:
                    if other != best:
                        ledger["decisions"].append({
                            "branch": other, "decision": "deferred",
                            "reason": "Passed pilot, but lower-priority than the strongest pilot; not a negative result",
                            "test_used_for_decision": False,
                        })
                if compare("scale", best, "scale_adaptive", SEEDS, pilot=False):
                    scale_finalist = best
            ledger["decisions"].append({
                "branch": "scale", "decision": "close_mechanism_study",
                "reason": "Mechanism diagnostic stays separate; no untested transfer into consistency models",
                "test_used_for_decision": False,
            })
        else:
            ledger["decisions"].append({
                "branch": "scale", "decision": "prune",
                "reason": "Selected validated parent is not gated; do not expand a rejected architecture",
                "test_used_for_decision": False,
            })
        reference = "ext_edu_ce2"
        incumbent = extension_tree(parent, lambda candidate, baseline: explore(
            "extensions", candidate, baseline), prefer=prefer)
        if incumbent == "ext_chunks_consistent":
            if not compare("extensions", incumbent, reference, SEEDS, pilot=False):
                incumbent = "ext_edu_consistent"
                ledger["decisions"].append({
                    "branch": "terminal_root_check", "decision": incumbent,
                    "reason": "Combined terminal must also pass the three-seed gate against the root CE2 reference",
                    "test_used_for_decision": False,
                })
        if parent == "fusion_cls":
            ledger["decisions"].append({
                "branch": "segmentation", "decision": "prune",
                "reason": "CLS parent does not consume segment boundaries; this change has no effect",
                "test_used_for_decision": False,
            })
        ledger["selected_extension"] = incumbent
        ledger["selected_scale"] = scale_finalist
        # Freeze the search before any test predictions are requested.
        ledger["search_frozen"] = True
        persist("search_frozen", selected=incumbent)
        evaluations = []
        if scale_finalist:
            for condition in ("scale_adaptive", scale_finalist):
                for seed in SEEDS:
                    name = f"{condition}_s{seed}"
                    target = output / "evaluation" / name
                    run("experiments.evaluate_fusion_scale",
                        "--run", output / "runs" / name, "--cache", cache,
                        "--output", target, "--device", "cuda", "--evaluate-test")
                    result = json.loads((target / "result.json").read_text())
                    evaluations.append({
                        "condition": condition, "seed": seed,
                        "test_macro_f1": result["splits"]["test"]["metrics"]["macro_f1"],
                        "checkpoint_sha256": result["checkpoint_sha256"],
                    })
                    persist("frozen_evaluation_recorded", run=name)
        if incumbent != reference:
            for condition in (reference, incumbent):
                for seed in SEEDS:
                    name = f"{condition}_s{seed}"
                    target = output / "evaluation" / name
                    run("experiments.evaluate_fusion_extensions",
                        "--run", output / "runs" / name, "--cache", cache,
                        "--output", target, "--device", "cuda", "--evaluate-test")
                    result = json.loads((target / "result.json").read_text())
                    evaluations.append({
                        "condition": condition, "seed": seed,
                        "test_macro_f1": result["test"]["macro_f1"],
                        "checkpoint_sha256": result["checkpoint_sha256"],
                    })
                    persist("frozen_evaluation_recorded", run=name)
        ledger["status"] = "completed"
        ledger["evaluations"] = evaluations
        ledger["formal_training_runs"] = len(ledger["runs"])
        write_json(output / "summary.json", ledger)
        persist("completed", runs=len(ledger["runs"]), selected=incumbent)
    except Exception as error:
        ledger["status"] = "failed"
        ledger["error"] = {"type": type(error).__name__, "message": str(error)}
        if "current_run" in ledger:
            ledger["failed_attempts"].append({
                **ledger["current_run"], "status": "failed", **ledger["error"],
                "performance_metrics": None,
            })
        persist("failed", **ledger["error"])
        raise


if __name__ == "__main__":
    main()
