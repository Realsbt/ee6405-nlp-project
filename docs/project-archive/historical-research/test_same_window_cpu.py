import unittest

from prepare_same_window_cpu import select_cells, visible_prefix


class TinyTokenizer:
    def __call__(self, text, max_length=None, truncation=False, return_offsets_mapping=False):
        n = min(len(text), max_length - 2) if truncation else len(text)
        result = {"input_ids": [0] + list(range(3, 3 + n)) + [2]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(0, 0)] + [(i, i + 1) for i in range(n)] + [(0, 0)]
        return result


class SameWindowTests(unittest.TestCase):
    def test_bounded_prefix(self):
        visible, ids = visible_prefix("abcdefghijk", TinyTokenizer(), limit=8)
        self.assertEqual(visible, "abcdef")
        self.assertEqual(len(ids), 8)

    def test_retokenized_boundary_overflow_shrinks_prefix(self):
        class BoundaryTokenizer(TinyTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if not kwargs.get("truncation") and len(text) == 6:
                    result["input_ids"].append(99)
                return result
        visible, ids = visible_prefix("abcdefghijk", BoundaryTokenizer(), limit=8)
        self.assertEqual(visible, "abcde")
        self.assertEqual(len(ids), 7)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            visible_prefix("", TinyTokenizer())

    def test_cell_selection_deterministic_and_complete(self):
        records = [{"item_id": f"{d}_{c}_{i}", "domain": d, "label_id": c}
                   for d in ("a", "b", "c", "d") for c in range(4) for i in (1, 0)]
        chosen = select_cells(records, 1)
        self.assertEqual(len(chosen), 16)
        self.assertTrue(all(r["item_id"].endswith("_0") for r in chosen))
        with self.assertRaises(ValueError):
            select_cells(records[:2], 1)


if __name__ == "__main__":
    unittest.main()
