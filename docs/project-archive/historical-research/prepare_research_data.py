"""Audit original HART provenance and freeze a conservative MixSet transfer subset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parent
DOMAINS = ("arxiv", "essay", "news", "writing")
POLISH = {f"{model}_polish_{level}" for model in ("2llama", "gpt4")
          for level in ("token", "sentence")}
ADAPT = {"MGT_polish_token", "MGT_polish_sentence"}


def text_hash(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def originals():
    directory = ROOT / "resources/truth-mirror/benchmark/hart"
    for domain in DOMAINS:
        for split in ("dev", "test"):
            path = directory / f"{domain}.{split}.json"
            for row in json.loads(path.read_text()):
                yield path, row


def derive_label(row):
    content, language, identifier = (row.get(k, "") for k in
                                     ("content_source", "language_source", "id"))
    if content == "human" and language == "human" and "/" not in identifier:
        return "human_written"
    if content == "human" and language.startswith("rephrase:") and identifier.startswith("rep/"):
        return "human_ai_polished"
    if content.startswith("machine:"):
        if identifier.startswith("gen/"):
            return "ai_generated"
        if identifier.startswith("hum/gen/") and language.startswith("humanize:"):
            return "ai_humanized" if language in ("humanize:human", "humanize:tool") else "ai_generated"
    return None


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def audit_hart(output):
    index = defaultdict(list)
    input_hashes = {}
    for path, row in originals():
        index[(row["id"].split("/")[-1], text_hash(row["generation"]))].append(
            (path, row, derive_label(row)))
        relative = str(path.relative_to(ROOT))
        if relative not in input_hashes:
            input_hashes[relative] = digest(path)
    counts, actors, labels, entries = Counter(), Counter(), Counter(), []
    for path in sorted((ROOT / "resources/assets/hart_official").glob("*_graph.jsonl")):
        input_hashes[str(path.relative_to(ROOT))] = digest(path)
        for line in path.open():
            row = json.loads(line)
            matches = index.get((row["group_id"], text_hash(row["article"])), [])
            candidate_labels = {label for _, _, label in matches}
            if not matches:
                status = "unmatched_text"
            elif None in candidate_labels or len(candidate_labels) != 1:
                status = "ambiguous_metadata"
            elif next(iter(candidate_labels)) != row["label"]:
                status = "label_disagreement"
            else:
                status = "matched_label"
            counts[status] += 1
            labels[row["label"]] += 1
            if status == "matched_label" and row["label"] == "ai_humanized":
                actor_set = {r["language_source"] for _, r, _ in matches}
                actors[next(iter(actor_set)) if len(actor_set) == 1 else "ambiguous_actor"] += 1
            entries.append({
                "item_id": row["item_id"], "group_id": row["group_id"],
                "published_label": row["label"], "status": status,
                "normalized_text_sha256": text_hash(row["article"]),
                "candidates": [{"source": str(p.relative_to(ROOT)), "source_id": r["id"],
                                "derived_label": label, "content_source": r.get("content_source"),
                                "language_source": r.get("language_source"),
                                "process_records": r.get("process_records", [])}
                               for p, r, label in matches],
            })
    if not entries:
        raise ValueError("No released HART rows; fetch pinned graph data first")
    output.mkdir()
    with (output / "provenance.jsonl").open("x") as stream:
        for row in entries:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"scope": "Normalized-text AND source-group match against original metadata; "
                       "dataset provenance, not independent historical authorship verification.",
              "raw_commit": revision(ROOT / "resources/truth-mirror"),
              "script_sha256": digest(Path(__file__)), "input_sha256": input_hashes,
              "n_released": len(entries), "status_counts": dict(counts),
              "published_labels": dict(labels), "matched_humanized_actors": dict(actors)}
    write_new(output / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("n_released", "status_counts", "matched_humanized_actors")}, indent=2))


def prepare_mixset(output):
    source = ROOT / "resources/MixSet/data/MixSet/Mixset_test.json"
    rows = json.loads(source.read_text())
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    # Prevent original or edited siblings from surviving an exact overlap exclusion.
    disallowed = {text_hash(r["generation"]) for _, r in originals()}
    hc3 = ROOT / "resources/experiments/hc3_aux_v1_1/test.jsonl"
    disallowed.update(text_hash(json.loads(line)["article"]) for line in hc3.open())
    blocked_groups = {text_hash(row["original"]) for row in rows
                      if text_hash(row["original"]) in disallowed or text_hash(row["revised"]) in disallowed}
    candidates, excluded = [], Counter()
    for row in rows:
        operation = row["mixset_category"]
        if operation not in POLISH | ADAPT:
            excluded[f"operation_outside_primary_mapping:{operation}"] += 1
            continue
        group = text_hash(row["original"])
        if group in blocked_groups:
            excluded["source_family_exact_overlap"] += 1
            continue
        human_origin = operation in POLISH
        for field, label in (("original", "human_written" if human_origin else "ai_generated"),
                             ("revised", "human_ai_polished" if human_origin else "ai_humanized")):
            text = row[field].strip()
            if not text:
                raise ValueError("Empty source text")
            candidates.append({
                "item_id": f"mixset_{row['id']}_{field}",
                "group_id": f"mixset_{group}", "domain": row["category"],
                "operation": "original" if field == "original" else operation,
                "source_id": row["id"], "source_field": field,
                "label": label, "article": text, "normalized_text_sha256": text_hash(text),
            })
    by_text = defaultdict(list)
    for row in candidates:
        by_text[row["normalized_text_sha256"]].append(row)
    conflicting_groups = {r["group_id"] for matches in by_text.values()
                          if len({r["label"] for r in matches}) > 1 for r in matches}
    retained = []
    for matches in by_text.values():
        matches = [r for r in matches if r["group_id"] not in conflicting_groups]
        if not matches:
            continue
        matches.sort(key=lambda r: r["item_id"])
        kept = dict(matches[0])
        kept["source_item_ids"] = [r["item_id"] for r in matches]
        if len({r["group_id"] for r in matches}) > 1:
            # Do not pretend a cross-family duplicate has a unique bootstrap cluster.
            raise ValueError("Duplicate text links distinct families; implement union-find before evaluation")
        retained.append(kept)
        excluded["same_label_exact_duplicate"] += len(matches) - 1
    retained.sort(key=lambda r: r["item_id"])
    if set(r["label"] for r in retained) != {
        "human_written", "human_ai_polished", "ai_generated", "ai_humanized"
    }:
        raise ValueError("All four mapped labels must be present")
    output.mkdir()
    data_path = output / "test.jsonl"
    with data_path.open("x") as stream:
        for row in retained:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "protocol_id": "mixset_conservative_transfer_v1",
        "status": "frozen_before_model_predictions",
        "scope": "Author-defined test split, conservative four-class operational mapping; "
                 "not the original MixSet binary benchmark or complete four-class generalization.",
        "source_commit": revision(ROOT / "resources/MixSet"),
        "source_sha256": digest(source), "data_sha256": digest(data_path),
        "script_sha256": digest(Path(__file__)),
        "mapping_evidence": "MixSet paper section 3.2, Table 2/3; official dataset README. "
                            "AI polish -> human_ai_polished; manual MGT token/sentence adapt -> "
                            "ai_humanized. Original source texts retained once.",
        "excluded_operations": "complete/rewrite/humanize: different content semantics or "
                               "multi-stage LLM-assisted/human-review process",
        "license": "No repository-root LICENSE found at inspected revision; local research "
                   "evaluation only, public redistribution permission not established.",
        "raw_test_rows": len(rows), "n": len(retained),
        "groups": len({r["group_id"] for r in retained}),
        "labels": dict(Counter(r["label"] for r in retained)),
        "operations": dict(Counter(r["operation"] for r in retained)),
        "exclusions": dict(excluded),
        "conflicting_groups_removed": len(conflicting_groups),
        "exact_overlap_blocked_families": len(blocked_groups),
        "overlap_scope": "All original English HART records plus selected HC3 text. "
                         "No semantic near-duplicate or base-pretraining exposure guarantee.",
        "models": ["tfidf_lr_v1", "roberta_s3407"],
        "thresholds": "Fixed 0.5 and HART-validation negative-only empirical 1% FPR; "
                      "no MixSet training, tuning or model selection.",
    }
    write_new(output / "protocol.json", report)
    print(json.dumps({k: report[k] for k in ("n", "groups", "labels", "exclusions",
                                             "conflicting_groups_removed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["audit-hart", "mixset"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving the existing audit/frozen protocol")
    (audit_hart if args.task == "audit-hart" else prepare_mixset)(args.output)
