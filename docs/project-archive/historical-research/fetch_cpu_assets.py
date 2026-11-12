"""Fetch selected pinned CPU assets, reusing the repository downloader."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ["EE6405_STORAGE"] = str(ROOT / "resources")
os.environ["EE6405_UPSTREAM"] = str(ROOT / "resources/RACE")
sys.path.insert(0, str(ROOT / "repository"))
from scripts.fetch_resources import configure_parser, digest, fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hart", action="store_true",
                        help="Fetch only the three pinned HART graph releases")
    group.add_argument("--parser", action="store_true", help="Fetch and configure pinned RST parser")
    group.add_argument("--race", action="store_true", help="Fetch RACE SupCon seed 3407 checkpoint")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "repository/resources/manifest.json").read_text())
    for row in manifest["files"]:
        selected = (row["group"] == "parser" if args.parser else
                    row.get("run") == "race_s3407" if args.race else
                    row["path"].startswith("assets/hart_official/") if args.hart
                    else row["group"] == "backbone" or row["path"] == "assets/hc3/all.jsonl")
        if not selected:
            continue
        target = (ROOT / "resources/checkpoints" / row["asset"] if args.race
                  else ROOT / "resources" / row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fetch(row, target)
        if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
            raise ValueError(f"Asset failed size/hash check: {target}")
        print(f"Verified {row['path']}", flush=True)
    if args.parser:
        configure_parser()


if __name__ == "__main__":
    main()
