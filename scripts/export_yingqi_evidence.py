"""Export explicitly allowlisted local evidence, without weights or source text."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error("Use a new evidence directory")
    args.output.mkdir(parents=True)

    def copy(source, relative):
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == ".csv":
            target.write_text(source.read_text())

    fusion = repo / "outputs/kaggle_fusion_experiment_v1/edu_fusion_experiment_v1"
    for phase in ("screen", "confirmation"):
        for run in sorted((fusion / phase).glob("fusion_*_s*")):
            # Partial runs retain their recorded config, but no invented result.
            for name in ("result.json", "config.json", "parameters.json",
                         "provenance.json", "source_hashes.json", "events.jsonl",
                         "validation_predictions.jsonl"):
                if (run / name).exists():
                    copy(run / name, Path("fusion") / phase / run.name / name)
    copy(fusion / "selection.json", Path("fusion/selection.json"))
    for name in ("validation_screen.csv", "validation_screen.png", "verification.json"):
        copy(repo / "outputs/fusion_screen_report_v1" / name, Path("fusion") / name)
    for run in sorted((fusion / "smoke").glob("fusion_*_s*")):
        copy(run / "smoke.json", Path("fusion/smoke") / run.name / "smoke.json")
    for name in ("protocol.json", "selection.json", "results.json",
                 "test_predictions.jsonl", "development_oof_predictions.jsonl"):
        copy(args.research_root / "cpu_stacking_v1" / name, Path("stacking") / name)
    for name in ("results.json", "REPORT.md", "stack_all_original_predictions.jsonl"):
        copy(args.research_root / "external_completed_cpu_v1" / name,
             Path("external") / name)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    kernels = [
        "zhengyingqi/ee6405-same-window-seed-3407",
        "zhengyingqi/ee6405-text-edu-fusion-experiment-v1",
        "zhengyingqi/ee6405-text-edu-fusion-continuation-v1",
    ]
    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_note": "Singapore = UTC+08:00; this is a snapshot, not live monitoring",
        "kernels": {kernel: str(api.kernels_status(kernel).status).rsplit(".")[-1]
                    for kernel in kernels},
        "same_window": {"completed_seeds": [3407], "uncompleted_seeds": [42, 2026]},
        "fusion": {
            "completed_training": [
                "fusion_cls_s3407", "fusion_mean_s3407", "fusion_attention_s3407",
                "fusion_gated_s3407", "fusion_cls_s42",
            ],
            "uncompleted_training_in_first_job": [
                "fusion_gated_s42", "fusion_cls_s2026", "fusion_gated_s2026",
            ],
            "final_three_seed_test_results_available": False,
        },
    }
    (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[path.relative_to(args.output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps({
        "files_sha256": files,
        "scope": "Allowlisted metrics, ID/label/probability predictions and provenance only",
        "excluded": ["weights", "source articles", "credentials", "validation explanation excerpts"],
    }, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
