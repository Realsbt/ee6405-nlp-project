"""Read-only audit of author-supplied RST character spans; never changes data."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.common import DATA_ROOT, file_hash, write_json, write_jsonl
from experiments.neural import PROJECT_ROOT, load_records
from experiments.rst_worker import leaves


def main():
    records = {row["item_id"]: (split, row) for split in ("train", "val", "test") for row in load_records(split)}
    protocol = json.loads((DATA_ROOT / "protocol.json").read_text())
    totals = defaultdict(Counter)
    examples, quality = [], []
    for source in protocol["source_files"]:
        assert file_hash(source["path"]) == source["sha256"]
        for line in Path(source["path"]).open():
            raw = json.loads(line)
            if raw["item_id"] not in records:
                continue
            split, row = records[raw["item_id"]]
            assert row["article"] == raw["article"]
            text, units = raw["article"], leaves(raw["rst_structure"])
            counts = totals[split]
            counts["records"] += 1
            counts["edu_nodes"] += len(units)
            valid = all(isinstance(u.get("start"), int) and isinstance(u.get("end"), int) and 0 <= u["start"] < u["end"] <= len(text) for u in units)
            ordered = valid and all(a["end"] <= b["start"] for a, b in zip(units, units[1:]))
            item_quality = {"item_id": row["item_id"], "split": split, "valid_and_ordered_spans": bool(ordered),
                            "uncovered_nonspace_characters": None, "uncovered_nonspace_in_lexical_window": None,
                            "all_nonspace_covered": False}
            quality.append(item_quality)
            counts["records_with_invalid_spans"] += not valid
            counts["records_with_overlapping_or_unordered_spans"] += valid and not ordered
            if not ordered:
                if len(examples) < 20:
                    examples.append({"item_id": row["item_id"], "issue": "invalid or unordered spans"})
                continue
            counts["records_with_leaf_text_not_exact_original_slice"] += any(u.get("text") != text[u["start"]:u["end"]] for u in units)
            gaps = [(0, units[0]["start"]), (units[-1]["end"], len(text))]
            gaps += [(a["end"], b["start"]) for a, b in zip(units, units[1:])]
            missing = sum(sum(not char.isspace() for char in text[a:b]) for a, b in gaps)
            end = len(row["model_text"])
            missing_visible = sum(sum(not char.isspace() for char in text[a:min(b,end)]) for a, b in gaps if a < end)
            item_quality.update(uncovered_nonspace_characters=missing, uncovered_nonspace_in_lexical_window=missing_visible,
                                all_nonspace_covered=missing == 0)
            counts["records_with_uncovered_nonspace"] += missing > 0
            counts["records_with_uncovered_nonspace_in_lexical_window"] += missing_visible > 0
            counts["uncovered_nonspace_characters"] += missing
            counts["uncovered_nonspace_in_lexical_window"] += missing_visible
            counts["total_nonspace_characters"] += sum(not char.isspace() for char in text)
            if missing and len(examples) < 20:
                examples.append({"item_id": row["item_id"], "issue": "uncovered non-whitespace characters", "missing": missing, "missing_in_window": missing_visible})
    assert sum(values["records"] for values in totals.values()) == len(records)
    quality_path = PROJECT_ROOT / "experiments/reports/released_rst_item_quality.jsonl"
    write_jsonl(quality_path, sorted(quality, key=lambda row: row["item_id"]))
    report = {"status": "audited_without_data_changes", "counts": {split: dict(values) for split, values in totals.items()},
              "item_quality_sha256": file_hash(quality_path),
              "first_20_issue_examples": examples, "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"),
              "frozen_data_or_model_changed": False,
              "scope": "Audits released tree coordinates, not discourse annotation correctness. Findings are limitations, not a reason to retune on test or silently replace frozen trees.",
              "audit_script_sha256": file_hash(__file__)}
    write_json(PROJECT_ROOT / "experiments/reports/released_rst_span_audit.json", report)
    print(json.dumps({key:value for key,value in report.items() if key != "first_20_issue_examples"}, indent=2))


if __name__ == "__main__":
    main()
