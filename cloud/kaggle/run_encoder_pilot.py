"""Run two bounded encoder pilots in a separate private Kaggle job."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe pilot source path")
    return path


def materialize_sources(mount, code, manifest):
    if code.exists():
        raise FileExistsError("Preserve previous pilot code")
    archive = mount / "encoder_pilot_code_v1.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                safe_relative(name)
            bundle.extractall(code)
    else:
        roots = {
            path.parent.parent for path in mount.rglob("train_encoder_pilot.py")
            if path.parent.name == "experiments"
        }
        if len(roots) != 1:
            raise ValueError("Expected exactly one unpacked encoder source root")
        source_root = roots.pop()
        for name in manifest["code_files"]:
            relative = safe_relative(name)
            target = code.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root.joinpath(*relative.parts), target)
    actual = {}
    for name, expected in manifest["code_files"].items():
        relative = safe_relative(name)
        actual[name] = digest(code.joinpath(*relative.parts))
        if actual[name] != expected:
            raise ValueError(f"Pilot source changed: {name}")
    return actual


def main():
    manifests = list(Path("/kaggle/input").rglob("encoder_pilot_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one private encoder pilot dataset")
    mount = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    code = Path("/kaggle/working/encoder_pilot_code")
    actual = materialize_sources(mount, code, manifest)
    if digest(mount / "pilot_protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Pilot protocol changed")
    protocol = json.loads((mount / "pilot_protocol.json").read_text())
    if (manifest["contains_test_partition"] or protocol["test_evaluated"]
            or set(protocol["partitions"]) != {"train", "val"}):
        raise ValueError("Pilot must not include test")
    data = Path("/kaggle/working/encoder_pilot_data")
    data.mkdir()
    shutil.copy2(mount / "pilot_protocol.json", data / "protocol.json")
    for split in ("train", "val"):
        source = mount / f"pilot_{split}.jsonl"
        if digest(source) != protocol["partitions"][split]["sha256"]:
            raise ValueError("Pilot data changed")
        shutil.copy2(source, data / f"{split}.jsonl")
    sys.path.insert(0, str(code))
    os.chdir(code)
    from cloud.kaggle.run_same_window_smoke import find_dataset, materialize
    resources = materialize(find_dataset(), "resources")
    os.environ.update(
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
        EE6405_STORAGE=str(resources), EE6405_UPSTREAM=str(resources / "RACE"),
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "sentencepiece==0.2.1", "protobuf==5.29.6"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no formal encoder training started")
    output = Path("/kaggle/working/encoder_pilot_v1")
    output.mkdir()
    (output / "code_manifest.json").write_text(json.dumps(actual, indent=2) + "\n")
    shutil.copy2(data / "protocol.json", output / "protocol.json")

    def record(stage, **extra):
        payload = {
            "stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0), "test_evaluated": False, **extra,
        }
        (output / "pipeline.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload), flush=True)

    def run(condition, phase):
        destination = output / phase / condition
        command = [
            sys.executable, "-B", "-m", "experiments.train_encoder_pilot",
            "--condition", condition, "--data-root", str(data),
            "--roberta", str(resources / "assets/roberta-base"),
            "--output", str(destination), "--epochs", str(protocol["epochs"]),
            "--batch-size", str(protocol["effective_batch"]),
            "--micro-batch", str(protocol["micro_batch"]),
            "--seed", str(protocol["seed"]),
            "--max-seconds", str(protocol["max_seconds_per_arm"]), "--device", "cuda",
        ]
        if phase == "smoke":
            command.append("--smoke")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((destination / "result.json").read_text())
        expected_status = "smoke_completed" if phase == "smoke" else "validation_completed"
        if result["test_evaluated"] or result["status"] != expected_status:
            raise RuntimeError(f"Incomplete {phase}: {condition}: {result['status']}")
        if phase == "smoke":
            (destination / "best.pt").unlink()
        return result

    results = {}
    try:
        for condition in protocol["conditions"]:
            record("cuda_smoke", condition=condition)
            run(condition, "smoke")
        for condition in protocol["conditions"]:
            record("validation_pilot", condition=condition)
            results[condition] = run(condition, "formal")
            (output / "partial_summary.json").write_text(json.dumps(results, indent=2) + "\n")
        summary = {
            "status": "completed", "test_evaluated": False,
            "validation_results": results,
            "deberta_minus_roberta_validation_macro_f1": (
                results["deberta_full"]["validation"]["macro_f1"]
                - results["roberta_full"]["validation"]["macro_f1"]),
            "scope": "One-seed, four-epoch validation pilot; not a test improvement or a 97% result",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        record("completed")
    except Exception as error:
        record("incomplete", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
