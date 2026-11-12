"""Screen B0 first/last and three-window inference candidates on S."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.campaign12h import load_b0, metrics, B0_PATH
from experiments.common import file_hash, write_json
from experiments.neural import load_records, tokenizer_local

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"


def window_logits(records, model, count):
    tokenizer = tokenizer_local()
    result = []
    with torch.no_grad():
        for record in records:
            encoded = tokenizer(record["article"], max_length=512, truncation=False,
                                padding=False, return_tensors="pt")
            ids = encoded["input_ids"][0]
            if len(ids) <= 512:
                windows = [ids]
            elif count == 2:
                windows = [ids[:512], ids[-512:]]
            else:
                middle = max(0, (len(ids) - 512) // 2)
                windows = [ids[:512], ids[middle:middle + 512], ids[-512:]]
            logits = []
            for window in windows:
                mask = torch.ones_like(window)
                output = model({"input_ids": window.unsqueeze(0).cuda(),
                                "attention_mask": mask.unsqueeze(0).cuda()})
                logits.append(output["logits"].float().cpu())
            result.append(torch.stack(logits).mean(0).squeeze(0).numpy())
    return torch.softmax(torch.tensor(np.asarray(result)), dim=1).numpy()


def main():
    started = time.time()
    model, _, _ = load_b0()
    val = load_records("val")
    manifest = json.loads((CAMPAIGN / "protocol/split_manifest.json").read_text(encoding="utf-8"))
    s_ids = {row["item_id"] for row in manifest["records"] if row["partition"] == "S"}
    records = [row for row in val if row["item_id"] in s_ids]
    tokenizer = tokenizer_local()
    base = []
    with torch.no_grad():
        for record in records:
            encoded = tokenizer(record["article"], max_length=512, truncation=True,
                                padding=False, return_tensors="pt")
            output = model({"input_ids": encoded["input_ids"].cuda(),
                            "attention_mask": encoded["attention_mask"].cuda()})
            base.append(output["logits"].float().cpu().squeeze(0))
    base_prob = torch.stack(base).softmax(1).numpy()
    runs = []
    for task_id, count in (("E20", 2), ("E21", 3)):
        probability = window_logits(records, model, count)
        current = metrics(records, probability)
        baseline = metrics(records, base_prob)
        payload = {
            "task_id": task_id, "status": "COMPLETE", "metrics": current,
            "base_metrics": baseline, "delta_accuracy_pp": 100 * (current["accuracy"] - baseline["accuracy"]),
            "delta_macro_f1_pp": 100 * (current["macro_f1"] - baseline["macro_f1"]),
            "windows": count, "wall_seconds": time.time() - started,
            "b0_sha256": file_hash(B0_PATH), "test_partition_loaded": False, "hc3_loaded": False,
        }
        run = CAMPAIGN / "tasks" / task_id
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "selection.json", payload)
        runs.append(payload)
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
