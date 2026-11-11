"""Audited RST offset recovery without changing model input text."""

import hashlib
import unicodedata

from experiments import rst_alignment


STRICT_REMAP = rst_alignment.remap_exact
PROJECTED_IGNORED = frozenset("\u200b\ufeff\u200d\u2581")


def compact(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text) if not char.isspace())


def projected_compact(text):
    return compact("".join(char for char in text if char not in PROJECTED_IGNORED))


def _update_parents(unit, original_text):
    children = [child for child in (getattr(unit, "left", None), getattr(unit, "right", None))
                if child is not None]
    for child in children:
        _update_parents(child, original_text)
    if children:
        unit.start, unit.end = children[0].start, children[-1].end
        unit.text = original_text[unit.start:unit.end]


def _remap_projection(unit, original_text):
    leaves = rst_alignment.tree_leaves(unit)
    contents = [projected_compact(leaf.text) for leaf in leaves]
    target = projected_compact(original_text)
    if not all(contents) or "".join(contents) != target:
        raise ValueError("RST output differs after audited character projection")
    prefix = [len(projected_compact(original_text[:index])) for index in range(len(original_text) + 1)]
    boundaries = {value: index for index, value in enumerate(prefix)}
    cursor = 0
    mapped = []
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
    for leaf, (start, end) in zip(leaves, mapped):
        leaf.start, leaf.end, leaf.text = start, end, original_text[start:end]
    _update_parents(unit, original_text)
    return {"mode": "audited_projection", "ignored_characters":
            sum(char in PROJECTED_IGNORED for char in original_text)}


def remap_verified(parser, unit, positions, originals, original_text):
    """Try exact alignment, then only equivalence-preserving offset maps."""
    try:
        STRICT_REMAP(parser, unit, positions, originals, original_text)
        return {"mode": "strict_exact"}
    except ValueError:
        pass
    leaves = rst_alignment.tree_leaves(unit)
    normalized = []
    offsets = []
    for index, char in enumerate(original_text):
        for normalized_char in unicodedata.normalize("NFKC", char):
            if not normalized_char.isspace():
                normalized.append(normalized_char)
                offsets.append(index)
    target = "".join(normalized)
    if target != compact(original_text):
        result = _remap_projection(unit, original_text)
    else:
        contents = [compact(leaf.text) for leaf in leaves]
        if not all(contents) or "".join(contents) != target:
            result = _remap_projection(unit, original_text)
        else:
            cursor = 0
            mapped = []
            for content in contents:
                stop = cursor + len(content)
                if cursor and offsets[cursor] == offsets[cursor - 1]:
                    raise ValueError("EDU boundary splits a Unicode compatibility expansion")
                mapped.append((offsets[cursor], offsets[stop - 1] + 1))
                cursor = stop
            for leaf, (start, end) in zip(leaves, mapped):
                leaf.start, leaf.end, leaf.text = start, end, original_text[start:end]
            _update_parents(unit, original_text)
            result = {"mode": "per_character_nfkc", "normalized_characters": len(target)}
    return {**result, "original_sha256": hashlib.sha256(original_text.encode()).hexdigest()}
