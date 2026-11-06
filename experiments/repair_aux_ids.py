"""Repair HC3 identifiers without resampling or changing any evaluation text."""

from collections import Counter, defaultdict
import json

from experiments.common import DATA_ROOT, file_hash, read_jsonl, write_json, write_jsonl
from experiments.neural import PROJECT_ROOT


def main():
    original = DATA_ROOT.parent / "hc3_aux_v1"
    target = DATA_ROOT.parent / "hc3_aux_v1_1"
    if target.exists():
        raise FileExistsError(f"Preserving existing repaired protocol {target}")
    protocol = json.loads((original / "protocol.json").read_text())
    assert file_hash(original / "test.jsonl") == protocol["answer_sha256"]
    rows = read_jsonl(original / "test.jsonl")
    old_ids = Counter(row["item_id"] for row in rows)
    fixed, groups = [], defaultdict(list)
    for row in rows:
        group = f"hc3_{row['domain']}:{row['question_sha256']}"
        kind = "ai" if row["binary_label"] else "human"
        changed = {**row, "item_id": f"{group}_{kind}", "group_id": group,
                   "legacy_item_id": row["item_id"], "legacy_group_id": row["group_id"]}
        assert all(changed[key] == value for key, value in row.items() if key not in ("item_id", "group_id"))
        fixed.append(changed)
        groups[group].append(changed["binary_label"])
    assert len({row["item_id"] for row in fixed}) == 1000
    assert len(groups) == 500 and all(sorted(values) == [0, 1] for values in groups.values())
    target.mkdir(parents=True)
    write_jsonl(target / "test.jsonl", fixed)
    report = {**protocol, "protocol_id": "hc3_aux_v1_1", "data_root": str(target),
              "answer_sha256": file_hash(target / "test.jsonl"),
              "identifier_repair": {
                  "reason": "Source index may be null or repeated; source:index was not a unique key",
                  "source_protocol_sha256": file_hash(original / "protocol.json"),
                  "source_answer_sha256": protocol["answer_sha256"],
                  "original_unique_item_ids": len(old_ids), "repaired_unique_item_ids": len(fixed),
                  "unique_question_groups": len(groups),
                  "fields_changed": ["item_id", "group_id"],
                  "all_text_labels_sources_and_order_unchanged": True,
                  "selection_or_tuning_repeated": False,
                  "repair_script_sha256": file_hash(__file__),
              }}
    write_json(target / "protocol.json", report)
    write_json(PROJECT_ROOT / "experiments/reports/hc3_aux_v1_1.json", report)
    print(json.dumps(report["identifier_repair"], indent=2))


if __name__ == "__main__":
    main()
