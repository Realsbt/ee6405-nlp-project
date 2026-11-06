"""Check that raw-text parser leaf spans cover all non-whitespace characters."""

import json
from pathlib import Path

from experiments.common import DATA_ROOT, file_hash, read_jsonl, write_json
from experiments.neural import PROJECT_ROOT
from experiments.rst_worker import leaves


def main():
    path = DATA_ROOT.parent / "hc3_aux_v1_1/test_parsed_aligned.jsonl"
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    assert audit["status"] == "passed" and audit["output_sha256"] == file_hash(path)
    rows = read_jsonl(path)
    uncovered = []
    for row in rows:
        units = leaves(row["rst_structure"])
        text = row["article"]
        assert units and all(0 <= unit["start"] < unit["end"] <= len(text) for unit in units)
        assert all(unit["text"] == text[unit["start"]:unit["end"]] for unit in units)
        assert all(a["end"] <= b["start"] for a, b in zip(units, units[1:]))
        gaps = text[:units[0]["start"]] + text[units[-1]["end"]:]
        gaps += "".join(text[a["end"]:b["start"]] for a, b in zip(units, units[1:]))
        if any(not char.isspace() for char in gaps):
            uncovered.append({"item_id": row["item_id"], "uncovered_nonspace_characters": sum(not c.isspace() for c in gaps),
                              "gap_excerpt": gaps[:100]})
    report = {"status": "passed" if not uncovered else "needs_review", "records": len(rows),
              "parsed_sha256": audit["output_sha256"], "valid_ordered_exact_character_spans": True,
              "records_with_uncovered_nonwhitespace": len(uncovered), "cases": uncovered,
              "audit_script_sha256": file_hash(__file__)}
    write_json(PROJECT_ROOT / "experiments/reports/hc3_rst_aligned_span_audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
