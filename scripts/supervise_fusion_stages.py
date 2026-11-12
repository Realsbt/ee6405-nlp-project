"""Supervise declared private Kaggle stages; stop dispatch only on real quota errors."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quota_error(message):
    return bool(re.search(
        r"(gpu|accelerator)[ \t]+(?:weekly[ \t]+)?(?:quota|usage[ \t]+limit)"
        r"|(?:quota|weekly[ \t]+limit)[^\n]{0,50}(gpu|accelerator)"
        r"|(?:gpu|accelerator)[ \t]+hours?[^\n]{0,50}(?:exhausted|exceeded|insufficient)"
        r"|(?:insufficient|not enough|exhausted)[^\n]{0,50}(?:gpu|accelerator)[ \t]+hours?",
        message, re.IGNORECASE,
    ))


def safe_target(name, output, prefix):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe remote output path")
    if not path.parts or path.parts[0] != prefix:
        return None
    if path.suffix not in (".pt", ".json", ".jsonl", ".py"):
        return None
    if path.suffix == ".pt" and "smoke" in path.parts:
        return None
    return output.joinpath(*path.parts)


def collect(api, kernel, output, prefix):
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "supervised_download_manifest.json"
    prior_path = manifest_path if manifest_path.exists() else output / "download_manifest.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"files": []}
    downloaded = {}
    for entry in prior["files"]:
        name = entry.get("remote_name")
        if name is None:
            name = str(Path(entry["path"]).resolve().relative_to(output.resolve()))
        downloaded[name] = {**entry, "remote_name": name}
    owner, slug = kernel.split("/", 1)
    page_token = ""
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.page_size, request.page_token = 100, page_token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        if not page_token and response.log:
            lines = json.loads(response.log)
            (output / "kernel.log").write_text("".join(entry.get("data", "") for entry in lines))
        for item in response.files or []:
            target = safe_target(item.file_name, output, prefix)
            if target is None:
                continue
            if item.file_name in downloaded and target.is_file():
                with target.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == downloaded[item.file_name]["sha256"]:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download")
            digest = hashlib.sha256()
            with requests.get(item.url, stream=True, timeout=(30, 120)) as result:
                result.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in result.iter_content(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
            temporary.replace(target)
            downloaded[item.file_name] = {
                "remote_name": item.file_name, "path": str(target),
                "sha256": digest.hexdigest(),
            }
            write_json(manifest_path, {"kernel": kernel, "files": list(downloaded.values())})
            print(json.dumps({"event": "downloaded", "file": item.file_name}), flush=True)
        page_token = response.next_page_token
        if not page_token:
            break
    summary = output / prefix / "summary.json"
    return summary.exists() and json.loads(summary.read_text()).get("status") == "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("Use at least 60 seconds between remote checks")
    from kaggle.api.kaggle_api_extended import KaggleApi

    plan = json.loads(args.plan.read_text())
    if plan.get("superseded_by"):
        parser.error(f"This exhaustive plan was retired; use {plan['superseded_by']}")
    api = KaggleApi()
    api.authenticate()
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / "journal.json"
    state = json.loads(journal.read_text()) if journal.exists() else {"completed": [], "submitted": []}

    def event(status, **extra):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(journal, state)
        print(json.dumps({"status": status, **extra}), flush=True)

    for stage in plan["stages"]:
        kernel = stage["kernel"]
        if kernel in state["completed"]:
            continue
        if stage.get("submit") and kernel not in state["submitted"]:
            event("submitting", kernel=kernel)
            metadata = json.loads((Path(stage["package"]) / "kernel-metadata.json").read_text())
            if metadata["id"] != kernel or str(metadata["is_private"]).lower() != "true":
                raise ValueError("Kernel identity/private scope differs from the frozen plan")
            try:
                response = api.kernels_push(stage["package"])
                if response.error:
                    raise RuntimeError(response.error)
            except Exception as error:
                if quota_error(str(error)):
                    event("gpu_quota_exhausted", kernel=kernel, reason=str(error))
                    return
                event("repair_required", kernel=kernel, reason=str(error))
                raise
            state["submitted"].append(kernel)
            event("submitted", kernel=kernel, version=response.version_number)
        previous = None
        while True:
            response = api.kernels_status(kernel)
            status = str(response.status).rsplit(".")[-1].lower()
            if status != previous:
                event("monitoring", kernel=kernel, remote_state=status)
                previous = status
            if status not in ("running", "queued"):
                break
            time.sleep(args.interval)
        failure = response.failure_message or ""
        if quota_error(failure):
            event("gpu_quota_exhausted", kernel=kernel, reason=failure)
            return
        target = Path(stage["output"])
        collector_pid = stage.get("collector_pid")
        if collector_pid:
            announced = False
            while True:
                if (target / "download_manifest.json").exists():
                    break
                try:
                    command = subprocess.check_output(
                        ["ps", "-p", str(collector_pid), "-o", "args="], text=True,
                    )
                except subprocess.CalledProcessError:
                    break
                if "collect_fusion_run.py" not in command:
                    break
                if not announced:
                    event("waiting_existing_collector", kernel=kernel)
                    announced = True
                time.sleep(args.interval)
        if not collect(api, kernel, target, stage["prefix"]):
            event("repair_required", kernel=kernel, reason=failure or "No completed experiment summary")
            raise RuntimeError("Preserved available outputs; inspect kernel.log and repair the stage")
        if status not in ("complete", "completed"):
            event("repair_required", kernel=kernel, reason=failure or status)
            raise RuntimeError("Remote task did not complete successfully")
        if stage.get("postprocess"):
            event("verifying_results", kernel=kernel)
            subprocess.run([sys.executable, *stage["postprocess"]], check=True)
        state["completed"].append(kernel)
        event("stage_completed", kernel=kernel, collected=str(target))
    event("all_gpu_stages_completed")


if __name__ == "__main__":
    main()
