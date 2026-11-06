"""Exactly align decoded EDU text to original non-whitespace characters.

The RST-DT converter constructs EDU offsets in decoded-subword text, while its
default remapper assumes a space-joined word-token coordinate system. These can
differ. This adapter preserves predicted EDU contents/order/tree relations and
only restores original character offsets. Any non-whitespace mismatch is an
error, never a guessed boundary or silently dropped character.
"""

from contextlib import contextmanager


def tree_leaves(node):
    if getattr(node, "left", None) is None and getattr(node, "right", None) is None:
        return [node]
    return (tree_leaves(node.left) if getattr(node, "left", None) else []) + (tree_leaves(node.right) if getattr(node, "right", None) else [])


def remap_exact(self, unit, positions, originals, original_text):
    units = tree_leaves(unit)
    indices = [index for index, char in enumerate(original_text) if not char.isspace()]
    target = "".join(original_text[index] for index in indices)
    contents = ["".join(char for char in leaf.text if not char.isspace()) for leaf in units]
    if not all(contents) or "".join(contents) != target:
        raise ValueError("Decoded RST EDUs differ from original non-whitespace text; exact alignment refused")
    cursor = 0
    for leaf, content in zip(units, contents):
        leaf.start = indices[cursor]
        cursor += len(content)
        leaf.end = indices[cursor - 1] + 1
        leaf.text = original_text[leaf.start:leaf.end]
    assert cursor == len(indices)
    def update(node):
        left, right = getattr(node, "left", None), getattr(node, "right", None)
        if left is not None:
            update(left)
        if right is not None:
            update(right)
        if left is not None or right is not None:
            node.start = (left or right).start
            node.end = (right or left).end
            node.text = original_text[node.start:node.end]
    update(unit)


@contextmanager
def exact_tree_alignment():
    from isanlp_rst.base_predictor import BasePredictor
    original = BasePredictor.remap_tree_offsets
    try:
        BasePredictor.remap_tree_offsets = remap_exact
        yield
    finally:
        BasePredictor.remap_tree_offsets = original
