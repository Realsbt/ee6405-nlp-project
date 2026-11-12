"""Collect small, text-free evidence from a completed Kaggle same-window run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="zhengyingqi/ee6405-same-window-seed-3407")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests

    api = KaggleApi()
    api.authenticate()
    state = str(api.kernels_status(args.kernel).status).rsplit(".")[-1].lower()
    if state not in ("complete", "completed"):
        raise RuntimeError(f"Refusing to publish incomplete run: {state}")
    owner, slug = args.kernel.split("/", 1)
    allowed = {
        "summary.json", "config.json", "parameters.json", "result.json",
        "test_predictions.jsonl", "validation_predictions.jsonl",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    token = ""
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.page_size, request.page_token = 100, token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            relative = PurePosixPath(item.file_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe remote file path")
            if (not relative.parts or relative.parts[0] != "same_window_seed_3407"
                    or relative.name not in allowed):
                continue
            target = args.output.joinpath(*relative.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            result = requests.get(item.url, timeout=(30, 120))
            result.raise_for_status()
            target.write_bytes(result.content)
            files.append({
                "path": target.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(result.content).hexdigest(),
            })
            print(f"Collected {relative}", flush=True)
        token = response.next_page_token
        if not token:
            break
    if not (args.output / "summary.json").exists():
        raise RuntimeError("Completed job has no saved summary")
    manifest = {
        "kernel": args.kernel, "status": state,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded": ["model weights", "raw source text", "credentials"],
        "files": files,
    }
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
