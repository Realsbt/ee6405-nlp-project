from types import SimpleNamespace
import unittest

from parse_nfkc_cpu import remap_verified


def node(text, left=None, right=None):
    return SimpleNamespace(text=text, start=0, end=len(text), left=left, right=right)


class NFKCAlignmentTests(unittest.TestCase):
    def test_compatibility_expansion_preserves_original(self):
        tree = node("A... B", node("A..."), node("B"))
        remap_verified(None, tree, None, None, "A\u2026 B")
        self.assertEqual((tree.left.start, tree.left.end, tree.left.text), (0, 2, "A\u2026"))
        self.assertEqual((tree.right.start, tree.right.end), (3, 4))
        self.assertEqual(tree.text, "A\u2026 B")

    def test_split_expansion_rejected(self):
        tree = node("A...B", node("A."), node("..B"))
        with self.assertRaisesRegex(ValueError, "boundary"):
            remap_verified(None, tree, None, None, "A\u2026B")

    def test_content_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            remap_verified(None, node("AXB"), None, None, "A\u2026B")

    def test_plain_text_exact_path(self):
        tree = node("Hello world", node("Hello"), node("world"))
        remap_verified(None, tree, None, None, "Hello world")
        self.assertEqual(tree.right.start, 6)

    def test_cross_character_composition_preserves_original(self):
        tree = node("\u00e9")
        remap_verified(None, tree, None, None, "e\u0301")
        self.assertEqual((tree.start, tree.end, tree.text), (0, 2, "e\u0301"))

    def test_audited_ignored_character_preserves_original(self):
        tree = node("AB", node("A"), node("B"))
        remap_verified(None, tree, None, None, "A\u200bB")
        self.assertEqual(tree.left.text, "A\u200b")
        self.assertEqual(tree.right.text, "B")

    def test_unapproved_deleted_character_rejected(self):
        with self.assertRaisesRegex(ValueError, "projection"):
            remap_verified(None, node("AB"), None, None, "A#B")


if __name__ == "__main__":
    unittest.main()
