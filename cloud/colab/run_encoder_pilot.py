"""Run the frozen paired pilot on a user-approved free Colab runtime."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path("/content/ee6405_encoder_pilot_v3")
    if root.exists():
        raise FileExistsError("Preserve previous pilot; do not launch duplicates")
    root.mkdir()
    bundle = Path("/content/ee6405_colab_pilot_v3.zip")
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe package path")
        archive.extractall(root)
    manifest = json.loads((root / "colab_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Package changed: {name}")
    code = root / "code"
    sys.path.insert(0, str(code))
    os.chdir(code)
    from cloud.kaggle.run_encoder_pilot import materialize_sources
    frozen = root / "frozen"
    sources = json.loads((frozen / "encoder_pilot_manifest.json").read_text())
    materialize_sources(frozen, root / "verified_sources", sources)
    protocol = json.loads((frozen / "pilot_protocol.json").read_text())
    if digest(frozen / "pilot_protocol.json") != sources["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if sources["contains_test_partition"] or protocol["test_evaluated"]:
        raise ValueError("Test partition forbidden")
    data = root / "data"
    data.mkdir()
    shutil.copy2(frozen / "pilot_protocol.json", data / "protocol.json")
    for split in ("train", "val"):
        path = frozen / f"pilot_{split}.jsonl"
        if digest(path) != protocol["partitions"][split]["sha256"]:
            raise ValueError("Data changed")
        shutil.copy2(path, data / f"{split}.jsonl")
    os.environ.update(USE_TF="0", USE_TORCH="1", USE_FLAX="0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU; no formal training started")
    from huggingface_hub import hf_hub_download
    backbone = root / "roberta"
    backbone.mkdir()
    weights = manifest["roberta_weights"]
    downloaded = Path(hf_hub_download(
        repo_id=weights["repo"], filename="model.safetensors",
        revision=weights["revision"],
    ))
    if digest(downloaded) != weights["sha256"]:
        raise ValueError("RoBERTa differs from the existing local control")
    shutil.copy2(downloaded, backbone / "model.safetensors")
    for path in (root / "roberta_config").iterdir():
        shutil.copy2(path, backbone / path.name)
    os.environ.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                      MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    output = root / "results"
    output.mkdir()
    shutil.copy2(frozen / "pilot_protocol.json", output / "protocol.json")
    shutil.copy2(root / "colab_manifest.json", output / "colab_manifest.json")
    shutil.copy2(Path(__file__), output / "launcher.py")

    def record(stage, **extra):
        payload = dict(stage=stage, time_utc=datetime.now(timezone.utc).isoformat(),
                       gpu=torch.cuda.get_device_name(0), test_evaluated=False, **extra)
        (output / "pipeline.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload), flush=True)

    def run(condition, phase):
        destination = output / phase / condition
        command = [
            sys.executable, "-B", "-m", "experiments.train_encoder_pilot",
            "--condition", condition, "--data-root", str(data),
            "--roberta", str(backbone), "--output", str(destination),
            "--seed", str(protocol["seed"]), "--epochs", str(protocol["epochs"]),
            "--batch-size", str(protocol["effective_batch"]),
            "--micro-batch", str(protocol["micro_batch"]),
            "--max-seconds", str(protocol["max_seconds_per_arm"]), "--device", "cuda",
        ]
        if phase == "smoke":
            command.append("--smoke")
        with (output / f"{phase}_{condition}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"Failed {phase}: {condition}; inspect saved log")
        result = json.loads((destination / "result.json").read_text())
        expected = "smoke_completed" if phase == "smoke" else "validation_completed"
        if result["status"] != expected or result["test_evaluated"]:
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
            "scope": "Single-seed validation pilot; not a test result or 97% claim",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        record("completed")
    except Exception as error:
        record("incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # Small evidence only; exclude private text, tokenizer files and weights.
        with zipfile.ZipFile(root / "pilot_evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                    archive.write(path, path.relative_to(output))
        print(f"Evidence saved: {root / 'pilot_evidence.zip'}", flush=True)


if __name__ == "__main__":
    main()
