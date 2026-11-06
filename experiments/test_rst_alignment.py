"""Regression checks for the observed decoded-space/original-space mismatch."""

from types import SimpleNamespace
import unittest

from experiments.rst_alignment import remap_exact


def leaf(text):
    return SimpleNamespace(text=text, left=None, right=None)


class AlignmentTests(unittest.TestCase):
    def test_missing_inter_sentence_space_keeps_first_letters(self):
        a, b = leaf("One sentence ."), leaf("Next sentence .")
        root = SimpleNamespace(left=a, right=b, text="unused")
        text = "One sentence.Next sentence."
        remap_exact(None, root, [], [], text)
        self.assertEqual(a.text, "One sentence.")
        self.assertEqual(b.text, "Next sentence.")
        self.assertEqual(a.end, b.start)
        self.assertEqual(root.text, text)

    def test_repeated_tokens_and_irregular_whitespace_are_unambiguous(self):
        a, b = leaf("foo foo"), leaf("foo")
        root = SimpleNamespace(left=a, right=b, text="unused")
        text = "  foo\nfoofoo  "
        remap_exact(None, root, [], [], text)
        self.assertEqual(a.text, "foo\nfoo")
        self.assertEqual(b.text, "foo")
        self.assertEqual((a.start, b.end), (2, len(text) - 2))

    def test_nonwhitespace_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact alignment refused"):
            remap_exact(None, leaf("wrong"), [], [], "original")


if __name__ == "__main__":
    unittest.main()
