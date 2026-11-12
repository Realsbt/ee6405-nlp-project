"""Opt-in RST offset adapter for verified Unicode compatibility equivalence.

Does not normalize the model input. Rejects changed nonspace content and EDU
boundaries inside the compatibility expansion of a single original character.
"""

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

from replay_hc3_cpu import ROOT
from experiments import rst_alignment, rst_worker
from screen_structure_cpu import digest

STRICT_REMAP = rst_alignment.remap_exact
FALLBACKS = []
PROJECTED_IGNORED = frozenset("\u200b\ufeff\u200d\u2581")


def compact(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())


def projected_compact(text):
    retained = "".join(c for c in text if c not in PROJECTED_IGNORED)
    return compact(retained)


def remap_projected(unit, original_text):
    units = rst_alignment.tree_leaves(unit)
    contents = [projected_compact(leaf.text) for leaf in units]
    target = projected_compact(original_text)
    if not all(contents) or "".join(contents) != target:
        raise ValueError("RST output differs after audited character projection")
    prefix_lengths = [len(projected_compact(original_text[:i])) for i in range(len(original_text) + 1)]
    if any(a > b for a, b in zip(prefix_lengths, prefix_lengths[1:])):
        raise ValueError("Projected normalization is not monotonic")
    boundaries = {value: i for i, value in enumerate(prefix_lengths)}
    cursor, mapped = 0, []
    for content in contents:
        stop = cursor + len(content)
        if stop not in boundaries:
            raise ValueError("EDU boundary splits a Unicode compatibility expansion")
        start = 0 if not mapped else mapped[-1][1]
        end = len(original_text) if stop == len(target) else boundaries[stop]
        if start >= end or projected_compact(original_text[start:end]) != content:
            raise ValueError("Projected EDU span does not preserve parser content")
        mapped.append((start, end))
        cursor = stop
    for leaf, (start, end) in zip(units, mapped):
        leaf.start, leaf.end, leaf.text = start, end, original_text[start:end]

    def update(node):
        children = [c for c in (getattr(node, "left", None), getattr(node, "right", None)) if c is not None]
        for child in children:
            update(child)
        if children:
            node.start, node.end = children[0].start, children[-1].end
            node.text = original_text[node.start:node.end]

    update(unit)
    return sum(c in PROJECTED_IGNORED for c in original_text)


def remap_verified(self, unit, positions, originals, original_text):
    try:
        return STRICT_REMAP(self, unit, positions, originals, original_text)
    except ValueError:
        pass
    units = rst_alignment.tree_leaves(unit)
    normalized, offsets = [], []
    for index, char in enumerate(original_text):
        for c in unicodedata.normalize("NFKC", char):
            if not c.isspace():
                normalized.append(c)
                offsets.append(index)
    target = "".join(normalized)
    # Cross-character composition cannot be represented by this per-char map.
    if target != compact(original_text):
        ignored = remap_projected(unit, original_text)
        FALLBACKS.append({"original_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                          "mode": "audited_projection", "ignored_characters": ignored})
        return
    contents = [compact(leaf.text) for leaf in units]
    if not all(contents) or "".join(contents) != target:
        ignored = remap_projected(unit, original_text)
        FALLBACKS.append({"original_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                          "mode": "audited_projection", "ignored_characters": ignored})
        return
    cursor, mapped = 0, []
    for content in contents:
        stop = cursor + len(content)
        if cursor and offsets[cursor] == offsets[cursor - 1]:
            raise ValueError("EDU boundary splits a Unicode compatibility expansion")
        mapped.append((offsets[cursor], offsets[stop - 1] + 1))
        cursor = stop
    for leaf, (start, end) in zip(units, mapped):
        leaf.start, leaf.end = start, end
        leaf.text = original_text[start:end]

    def update(node):
        children = [c for c in (getattr(node, "left", None), getattr(node, "right", None)) if c is not None]
        for child in children:
            update(child)
        if children:
            node.start, node.end = children[0].start, children[-1].end
            node.text = original_text[node.start:node.end]

    update(unit)
    FALLBACKS.append({"original_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                      "mode": "per_character_nfkc", "normalized_nonspace_characters": len(target)})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    rst_alignment.remap_exact = remap_verified
    try:
        rst_worker.main()
    finally:
        rst_alignment.remap_exact = STRICT_REMAP
    audit_path = args.output.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    audit.update({
        "character_offset_adapter": "Strict exact match first; opt-in NFKC-equivalent offset map only on failure. "
                                    "Original text unchanged; split expansions/content mismatches rejected.",
        "compatibility_fallbacks": FALLBACKS,
        "compatibility_adapter_sha256": digest(__file__),
    })
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"verified_nfkc_fallbacks": len(FALLBACKS)}), flush=True)


if __name__ == "__main__":
    main()
