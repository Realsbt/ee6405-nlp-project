import unittest

from complete_external_cpu import check_parsed


class ParseCoverageTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"item_id": "one", "article": "Hello world", "label": "human_written"}]
        self.parsed = [{**self.source[0], "rst_structure": {
            "text": "Hello world", "start": 0, "end": 11,
            "left": {"text": "Hello", "start": 0, "end": 5, "left": None, "right": None},
            "right": {"text": "world", "start": 6, "end": 11, "left": None, "right": None},
        }}]

    def test_whitespace_gap_accepted(self):
        self.assertEqual(check_parsed(self.source, self.parsed)["records"], 1)

    def test_nonspace_gap_rejected(self):
        self.parsed[0]["rst_structure"]["right"].update(start=7, text="orld")
        with self.assertRaisesRegex(ValueError, "Uncovered"):
            check_parsed(self.source, self.parsed)

    def test_changed_source_rejected(self):
        self.parsed[0]["label"] = "ai_generated"
        with self.assertRaisesRegex(ValueError, "metadata"):
            check_parsed(self.source, self.parsed)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            check_parsed(self.source * 2, self.parsed * 2)


if __name__ == "__main__":
    unittest.main()
