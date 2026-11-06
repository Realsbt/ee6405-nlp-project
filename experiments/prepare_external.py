"""Freeze a small independent HC3 question-paired binary auxiliary evaluation.

Selection is based on source, a fixed seed, and text-overlap checks only. No model
prediction or test performance participates in selecting these records.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from experiments.common import ASSETS_ROOT, DATA_ROOT, file_hash, read_jsonl, write_json, write_jsonl
from experiments.neural import PROJECT_ROOT, load_records, tokenizer_local
from experiments.prepare_data import text_hash


def main():
    source = ASSETS_ROOT / "hc3"
    out = DATA_ROOT.parent / "hc3_aux_v1"
    if out.exists():
        raise FileExistsError(f"Preserving frozen auxiliary data {out}")
    source_manifest = json.loads((source / "source_manifest.json").read_text())
    assert file_hash(source / "all.jsonl") == next(r["sha256"] for r in source_manifest["files"] if r["name"] == "all.jsonl")
    hart_hashes = {row["normalized_text_sha256"] for split in ("train", "val", "test") for row in load_records(split)}
    hart_hashes.update(row["normalized_text_sha256"] for row in read_jsonl(DATA_ROOT / "exclusions.jsonl"))
    candidates = []
    exclusions = Counter()
    seen_questions = set()
    raw_count = 0
    with (source / "all.jsonl").open() as stream:
        for line in stream:
            raw_count += 1
            row = json.loads(line)
            human = next((x.strip() for x in row["human_answers"] if isinstance(x, str) and x.strip()), None)
            ai = next((x.strip() for x in row["chatgpt_answers"] if isinstance(x, str) and x.strip()), None)
            if not human or not ai:
                exclusions["missing_one_answer_type"] += 1
                continue
            h, a = text_hash(human), text_hash(ai)
            q = text_hash(row["question"])
            if h == a:
                exclusions["identical_human_ai_answers"] += 1
            elif h in hart_hashes or a in hart_hashes:
                exclusions["normalized_answer_overlap_with_hart"] += 1
            elif q in seen_questions:
                exclusions["duplicate_question"] += 1
            else:
                seen_questions.add(q)
                candidates.append({"question_id": f"{row['source']}:{row['index']}", "question_sha256": q,
                                   "source": row["source"], "human": human, "ai": ai,
                                   "human_sha256": h, "ai_sha256": a})
    labels_by_hash = defaultdict(set)
    for row in candidates:
        labels_by_hash[row["human_sha256"]].add(0)
        labels_by_hash[row["ai_sha256"]].add(1)
    ambiguous_hashes = {key for key, labels in labels_by_hash.items() if len(labels) > 1}
    eligible = []
    for row in candidates:
        if row["human_sha256"] in ambiguous_hashes or row["ai_sha256"] in ambiguous_hashes:
            exclusions["ambiguous_answer_across_questions"] += 1
        else:
            eligible.append(row)
    rng = np.random.default_rng(2026)
    chosen, answer_hashes = [], set()
    sources = sorted({row["source"] for row in eligible})
    assert len(sources) == 5
    skipped_during_sampling = 0
    for domain in sources:
        pool = sorted((row for row in eligible if row["source"] == domain), key=lambda row: row["question_id"])
        count = 0
        for index in rng.permutation(len(pool)):
            row = pool[index]
            if row["human_sha256"] in answer_hashes or row["ai_sha256"] in answer_hashes:
                skipped_during_sampling += 1
                continue
            chosen.append(row)
            answer_hashes.update((row["human_sha256"], row["ai_sha256"]))
            count += 1
            if count == 100:
                break
        assert count == 100, f"Insufficient unique pairs in {domain}"
    records = []
    for row in chosen:
        for kind, binary_label, label_id, label in (("human", 0, 0, "human_written"), ("ai", 1, 2, "ai_generated")):
            records.append({"item_id": f"hc3_{row['question_id']}_{kind}", "group_id": f"hc3_{row['question_id']}",
                            "domain": row["source"], "question_sha256": row["question_sha256"],
                            "article": row[kind], "label": label, "label_id": label_id, "binary_label": binary_label,
                            "normalized_text_sha256": row[f"{kind}_sha256"]})
    tokenizer = tokenizer_local()
    marker = re.compile(r"\b(?:as an? (?:ai|(?:large )?language model)|chatgpt)\b", re.I)
    for start in range(0, len(records), 128):
        chunk = records[start:start+128]
        tokens = tokenizer([r["article"] for r in chunk], truncation=True, max_length=512,
                           return_offsets_mapping=True, return_attention_mask=False)
        for row, offsets in zip(chunk, tokens["offset_mapping"]):
            end = max(end for begin, end in offsets if end > begin)
            row["model_text"] = row["article"][:end]
            row["input_truncated"] = end < len(row["article"].rstrip())
            row["ai_marker_present"] = bool(marker.search(row["article"]))
    assert len(records) == 1000 and len(answer_hashes) == 1000
    assert not (answer_hashes & hart_hashes)
    out.mkdir(parents=True)
    write_jsonl(out / "test.jsonl", records)
    report = {
        "protocol_id": "hc3_aux_v1", "source": source_manifest, "original_question_rows": raw_count,
        "question_pairs": len(chosen), "answer_records": len(records), "seed": 2026,
        "source_question_pairs": dict(Counter(r["source"] for r in chosen)),
        "selection": "100 questions per source, first nonempty human and ChatGPT answer; fixed random order; no predictions used",
        "exclusions_before_sampling": dict(exclusions), "same_label_duplicate_pairs_skipped_while_sampling": skipped_during_sampling,
        "overlap_with_hart_normalized_text": 0, "duplicate_selected_answers": 0,
        "binary_mapping": {"human": 0, "ChatGPT": 1},
        "four_class_score_mapping": "P(AI creator) = P(ai_generated) + P(ai_humanized); threshold fixed at 0.5",
        "four_class_label_ids_are_proxy_pure_classes": True,
        "statistics_by_binary_class": {str(label): {
            "count": sum(r["binary_label"] == label for r in records),
            "median_words": float(np.median([len(r["article"].split()) for r in records if r["binary_label"] == label])),
            "truncated_at_512": sum(r["input_truncated"] for r in records if r["binary_label"] == label),
            "approximate_ai_marker_present": sum(r["ai_marker_present"] for r in records if r["binary_label"] == label),
        } for label in (0, 1)},
        "answer_sha256": file_hash(out / "test.jsonl"), "prepare_script_sha256": file_hash(__file__),
        "evidence_boundary": "Auxiliary binary OOD evaluation only. No labels distinguish polishing/humanization; it cannot validate four-class generalization. Unfiltered answer-length and explicit AI-marker differences remain.",
        "license": "HC3 card lists CC-BY-SA-4.0 with any stricter source-dataset terms preserved; see downloaded original README",
        "data_root": str(out),
    }
    write_json(out / "protocol.json", report)
    write_json(PROJECT_ROOT / "experiments/reports/hc3_aux_v1.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
