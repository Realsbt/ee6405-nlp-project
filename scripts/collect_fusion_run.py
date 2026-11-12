"""Wait for the private fusion kernel and collect its own evidence only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import time


def output_path(name, root, checkpoints=False):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe output path")
    if not relative.parts or relative.parts[0] != "edu_fusion_experiment_v1":
        return None
    if relative.suffix == ".pt":
        if not checkpoints or "smoke" in relative.parts:
            return None
    elif relative.suffix not in (".json", ".jsonl", ".py"):
        return None
    return root.joinpath(*relative.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="zhengyingqi/ee6405-text-edu-fusion-experiment-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("Use at least 30 seconds between remote checks")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests

    api = KaggleApi()
    api.authenticate()
    owner, slug = args.kernel.split("/", 1)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = None
    while True:
        response = api.kernels_status(args.kernel)
        state = str(response.status).rsplit(".")[-1].lower()
        if state != previous:
            event = {
                "kernel": args.kernel, "state": state,
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(event), flush=True)
            with (args.output / "status_history.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            previous = state
        if state not in ("running", "queued") or not args.watch:
            break
        time.sleep(args.interval)

    files = []
    page_token = ""
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.page_size, request.page_token = 100, page_token
        with api.build_kaggle_client() as client:
            output = client.kernels.kernels_api_client.list_kernel_session_output(request)
        files.extend(output.files or [])
        if not page_token and output.log:
            entries = json.loads(output.log)
            (args.output / "kernel.log").write_text(
                "".join(entry.get("data", "") for entry in entries)
            )
        page_token = output.next_page_token
        if not page_token:
            break
    downloaded = []
    for item in files:
        target = output_path(item.file_name, args.output, args.checkpoints)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(item.url, stream=True, timeout=(30, 120)) as result:
            result.raise_for_status()
            temporary = target.with_suffix(target.suffix + ".download")
            digest = hashlib.sha256()
            with temporary.open("wb") as stream:
                for chunk in result.iter_content(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
            temporary.replace(target)
        downloaded.append({"path": str(target), "sha256": digest.hexdigest()})
        print(json.dumps({"downloaded": item.file_name}), flush=True)
    (args.output / "download_manifest.json").write_text(json.dumps({
        "kernel": args.kernel, "terminal_state": state, "files": downloaded,
        "scope": "Fusion outputs only; no other kernel modified",
    }, indent=2) + "\n")
    if state not in ("complete", "completed"):
        raise SystemExit(f"Kernel is {state}; inspect saved logs and partial evidence")
    summary = args.output / "edu_fusion_experiment_v1/summary.json"
    if not summary.exists() or json.loads(summary.read_text())["status"] != "completed":
        raise RuntimeError("Kernel ended without the completed experiment summary")


if __name__ == "__main__":
    main()
