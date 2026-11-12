"""Re-infer the completed E16 checkpoint and bind probabilities to S item IDs."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.campaign_newbase import ColdModel, Records, collate
from experiments.common import file_hash, write_json
from experiments.neural import DATA_ROOT, load_records
from torch.utils.data import DataLoader
from functools import partial


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h"
RUN = CAMPAIGN / "top5_revision_01/E16_final_deberta_v3_base_s3407"
LABEL_ORDER = ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"]


def main():
    torch.manual_seed(3407)
    manifest_path = CAMPAIGN / "protocol/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    s_ids = [row["item_id"] for row in manifest["records"] if row["partition"] == "S"]
    s_id_set = set(s_ids)
    rows = [row for row in load_records("val") if row["item_id"] in s_id_set]
    rows.sort(key=lambda row: s_ids.index(row["item_id"]))
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    dataset = Records(rows, tokenizer)
    model = ColdModel("microsoft/deberta-v3-base").cuda()
    checkpoint = torch.load(RUN / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=16,
                            collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
        for batch in loader:
            batch = {key: value.cuda() for key, value in batch.items()}
            probabilities.append(model(batch).float().softmax(1).cpu())
            labels.extend(batch["labels"].cpu().tolist())
    probs = torch.cat(probabilities).numpy()
    prediction = {
        "task_id": "E16",
        "status": "checkpoint_reinference_verified",
        "scope": "S_dev_selected",
        "sample_count": len(rows),
        "ordered_item_ids": s_ids,
        "ordered_item_id_sha256": hashlib.sha256(
            "\n".join(s_ids).encode("utf-8")
        ).hexdigest(),
        "label_ids": labels,
        "label_order": LABEL_ORDER,
        "probabilities": probs.tolist(),
        "data_version": {
            "train_hash": file_hash(DATA_ROOT / "train.jsonl"),
            "val_hash": file_hash(DATA_ROOT / "val.jsonl"),
            "split_manifest_sha256": file_hash(manifest_path),
        },
        "model_version": {
            "checkpoint_sha256": file_hash(RUN / "best.pt"),
            "git_head": "5288c03bcbd0d6cfe835c352195f3f32e5553888",
            "model_source": "microsoft/deberta-v3-base",
        },
        "test_partition_loaded": False,
        "hc3_loaded": False,
    }
    write_json(RUN / "E16_prediction_with_ids.json", prediction)
    print(json.dumps({"path": str(RUN / "E16_prediction_with_ids.json"),
                      "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
