"""Check full HC3 CPU parse coverage and correct Darwin RSS reporting separately."""

import json
import platform
from replay_hc3_cpu import ROOT, read_jsonl, file_hash
from experiments.rst_worker import leaves
from race_cpu_replay import write


def main():
    path = ROOT / "resources/experiments/hc3_aux_v1_1/test_parsed_aligned_cpu.jsonl"
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    assert audit["status"] == "passed" and audit["output_sha256"] == file_hash(path)
    rows = read_jsonl(path)
    bad = []
    for row in rows:
        units, text = leaves(row["rst_structure"]), row["article"]
        assert units and all(0 <= u["start"] < u["end"] <= len(text) for u in units)
        assert all(u["text"] == text[u["start"]:u["end"]] for u in units)
        assert all(a["end"] <= b["start"] for a, b in zip(units, units[1:]))
        gaps = text[:units[0]["start"]] + text[units[-1]["end"]:]
        gaps += "".join(text[a["end"]:b["start"]] for a, b in zip(units, units[1:]))
        if any(not c.isspace() for c in gaps):
            bad.append(row["item_id"])
    report = {"status": "passed" if not bad else "review_required", "records": len(rows),
              "uncovered_nonwhitespace_ids": bad, "parsed_sha256": file_hash(path),
              "script_sha256": file_hash(__file__), "platform": platform.platform(),
              "peak_rss_mib_corrected": audit["peak_process_rss_mib"] / (1024 if platform.system() == "Darwin" else 1),
              "rss_note": "Original worker assumes Linux KiB. macOS ru_maxrss is bytes; "
                          "corrected here without modifying its archived audit.",
              "parser_load_seconds": audit["model_load_seconds"],
              "parse_seconds": sum(audit["parse_seconds"])}
    write(ROOT / "race_cpu_v1/hc3_parse_coverage.json", report)
    print(json.dumps(report, indent=2))
    assert not bad


if __name__ == "__main__":
    main()
