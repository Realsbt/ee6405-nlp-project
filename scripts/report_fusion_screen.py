"""Report completed validation screening without opening the test split."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.kaggle.run_edu_fusion import select_candidate
from experiments.common import LABELS, classification_metrics, file_hash, write_json
from scripts.report_fusion_results import metrics_close, render_validation_screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing reports")
    selection = json.loads((args.input / "selection.json").read_text())
    screen, table, hashes = {}, [], {}
    for condition in selection["all_validation_macro_f1"]:
        run = args.input / "screen" / f"{condition}_s3407"
        result = json.loads((run / "result.json").read_text())
        predictions = run / "validation_predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text().splitlines()]
        if len(rows) != 1600 or len({row["item_id"] for row in rows}) != 1600:
            raise ValueError("Expected all 1,600 unique validation samples")
        recomputed = classification_metrics(
            [LABELS.index(row["true_label"]) for row in rows],
            [[row["probabilities"][label] for label in LABELS] for row in rows],
        )
        if not metrics_close(recomputed, result["validation"]):
            raise ValueError("Validation predictions differ from saved metrics")
        screen[condition] = result
        hashes[condition] = file_hash(predictions)
        table.append({
            "condition": condition, "seed": 3407, "validation_macro_f1": recomputed["macro_f1"],
            "best_epoch": result["best_epoch"],
            "run_minutes": result["elapsed_seconds"] / 60,
            "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
            "trainable_parameters": result["parameters"]["trainable"],
        })
    checked = select_candidate(screen)
    for key in ("selected", "all_validation_macro_f1", "validation_gap"):
        if checked[key] != selection[key]:
            raise ValueError("Frozen selection does not match completed screening")
    args.output.mkdir(parents=True)
    with (args.output / "validation_screen.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(args.output / "verification.json", {
        "status": "passed", "scope": "Single-seed validation screening only",
        "test_opened": False, "prediction_sha256": hashes,
        "selection_sha256": file_hash(args.input / "selection.json"),
    })
    render_validation_screen(selection, args.output / "validation_screen.png")


if __name__ == "__main__":
    main()
