"""Audit CRE workspace provenance without loading HART test or HC3."""

import argparse
import json
import subprocess
from pathlib import Path

from experiments.common import DATA_ROOT, file_hash, write_json
from experiments.cre_data import audit_pair_inventory
from experiments.neural import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/cre_v1_audit.json"))
    args = parser.parse_args()
    train, val = load_records("train"), load_records("val")
    status = subprocess.run(["git", "status", "--short", "--branch"], capture_output=True, text=True, check=True)
    audit = {
        "branch_status": status.stdout,
        "train_samples": len(train), "validation_samples": len(val),
        "train_pair_inventory": audit_pair_inventory(train),
        "validation_pair_inventory": audit_pair_inventory(val),
        "test_partition_loaded": False, "hc3_loaded": False,
        "historical_equivalence": "unverified",
        "data_root": str(DATA_ROOT),
    }
    write_json(args.output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
