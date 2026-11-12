import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(r"D:\6405_prj\ee6405-nlp-project")
OUT = ROOT / "experiments/campaign12h/ee6405_scorefirst_36_12h/top5_revision_01"


def main():
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    before = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    result = {
        "task_id": "E16_final_deberta_v3_base_s3407",
        "attempt_kind": "official_resource_check",
        "model_id": "microsoft/deberta-v3-base",
        "revision": "main",
        "started_utc": started_utc,
        "environment_before": before,
        "environment_for_attempt": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "formal_training_started": False,
        "test_partition_loaded": False,
        "hc3_loaded": False,
    }
    try:
        config = AutoConfig.from_pretrained(
            "microsoft/deberta-v3-base", revision="main", local_files_only=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base", revision="main", local_files_only=False
        )
        result.update({
            "status": "RESOURCE_AVAILABLE",
            "config_model_type": config.model_type,
            "hidden_size": getattr(config, "hidden_size", None),
            "tokenizer_class": tokenizer.__class__.__name__,
        })
    except Exception as error:
        result.update({
            "status": "RESOURCE_BLOCKED",
            "stopped_reason": "official_resource_check_failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        })
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["wall_seconds"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e16_resource_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
