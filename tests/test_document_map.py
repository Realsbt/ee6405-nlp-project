"""Contract tests for the v5 navigation layer.

These exercise the real explanation payloads (synthetic checkpoints) and the
pure helpers that the Document Map, focus graph and version switching rely on.
They do not claim trained-checkpoint accuracy.
"""
import unittest

from experiments.document_explanations import LABELS, display_segments, explain_document

from tests.test_document_explanations import Service, fake_data_interfaces


def paragraphs(text):
    """Mirror of makeParagraphs() in document_explorer.html."""
    out, start = [], 0
    i = 0
    while i < len(text):
        if text[i] == '\n':
            j = i
            while j < len(text) and text[j] in '\n\r\t ':
                j += 1
            if j > i + 1 or (j < len(text) and text[i:j].count('\n') >= 2):
                if text[start:i].strip():
                    out.append((start, i))
                start = j
                i = j
                continue
        i += 1
    if text[start:].strip():
        out.append((start, len(text)))
    return out


def overlaps(unit, para):
    return unit['start'] < para[1] and unit['end'] > para[0]


class DocumentMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = Service()

    def test_paragraphs_cover_text_without_overlap(self):
        text = 'First block here.\n\nSecond block here.\n\nThird block here.'
        spans = paragraphs(text)
        self.assertEqual(len(spans), 3)
        for a, b in spans:
            self.assertTrue(text[a:b].strip())
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            self.assertLessEqual(b1, a2)

    def test_single_block_text(self):
        self.assertEqual(paragraphs('One paragraph only.'), [(0, 19)])

    def test_blank_text_has_no_paragraphs(self):
        self.assertEqual(paragraphs('   \n\n  '), [])

    def test_edu_offsets_match_source_text(self):
        text = 'Human writers compose a personal story. Software can polish the final wording.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'race', max_units=2)
        for unit in out['units']:
            self.assertEqual(text[unit['start']:unit['end']], text[unit['start_utf16']:unit['end_utf16']])
            self.assertGreater(unit['end'], unit['start'])

    def test_units_are_ordered_and_disjoint(self):
        text = 'Human writers compose a personal story. Software can polish the final wording.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'race', max_units=2)
        last = 0
        for unit in out['units']:
            self.assertGreaterEqual(unit['start'], last)
            last = unit['end']

    def test_cross_paragraph_edu_keeps_real_offsets(self):
        # An EDU that spans a paragraph break must not be split into two fake units.
        text = 'Human writers compose a personal story.\n\nSoftware can polish the final wording.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'race', max_units=2)
        spans = paragraphs(text)
        crossing = [u for u in out['units'] if any(overlaps(u, p) for p in spans)]
        self.assertTrue(crossing)
        for unit in out['units']:
            self.assertEqual(text[unit['start']:unit['end']], text[unit['start_utf16']:unit['end_utf16']])

    def test_every_paragraph_maps_to_at_least_one_unit(self):
        text = 'Human writers compose a personal story.\n\nSoftware can polish the final wording.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'race', max_units=2)
        for para in paragraphs(text):
            self.assertTrue(any(overlaps(u, para) for u in out['units']), para)


class OnDemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = Service()

    def setUp(self):
        self.text = 'Human writers compose a personal story. Software can polish the final wording.'

    def test_unrequested_units_have_no_influence(self):
        with fake_data_interfaces():
            out = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
        pending = [u for u in out['units'] if u['status'] != 'computed']
        self.assertTrue(pending)
        for unit in pending:
            # The key may exist, but it must never carry a fabricated number.
            self.assertIsNone(unit.get('delta_pp'))
            self.assertIsNone(unit.get('without_scores'))

    def test_batch_limit_is_enforced(self):
        with fake_data_interfaces():
            out = explain_document(self.service, self.text, 'race', max_units=1)
        self.assertEqual(out['checked_units'], 1)

    def test_second_request_can_target_a_different_unit(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            second = explain_document(self.service, self.text, 'race', unit_ids=['u2'], max_units=1)
        self.assertEqual(first['checked_units'], 1)
        self.assertEqual(second['checked_units'], 1)
        self.assertEqual(first['text_sha256'], second['text_sha256'])
        self.assertEqual(first['method'], second['method'])

    def test_document_prediction_is_stable_across_requests(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            second = explain_document(self.service, self.text, 'race', unit_ids=['u2'], max_units=1)
        for c in LABELS:
            self.assertAlmostEqual(first['document']['scores'][c], second['document']['scores'][c], places=6)

    def test_cache_key_changes_with_text(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            other = explain_document(self.service, self.text + ' Extra sentence.', 'race', unit_ids=['u1'], max_units=1)
        self.assertNotEqual(first['text_sha256'], other['text_sha256'])


class VersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = Service()

    def test_revised_text_gets_an_independent_prediction(self):
        original = 'Human writers compose a personal story. Software can polish the final wording.'
        revised = 'Software generates automated output. Machines write content.'
        with fake_data_interfaces():
            a = explain_document(self.service, original, 'race', unit_ids=['u1'], max_units=1)
            b = explain_document(self.service, revised, 'race', unit_ids=['u1'], max_units=1)
        self.assertNotEqual(a['text_sha256'], b['text_sha256'])
        self.assertNotEqual(a['document']['scores'], b['document']['scores'])

    def test_original_prediction_is_not_mutated_by_revision(self):
        original = 'Human writers compose a personal story. Software can polish the final wording.'
        revised = 'Software generates automated output. Machines write content.'
        with fake_data_interfaces():
            a = explain_document(self.service, original, 'race', unit_ids=['u1'], max_units=1)
            frozen = dict(a['document']['scores'])
            explain_document(self.service, revised, 'race', unit_ids=['u1'], max_units=1)
        self.assertEqual(a['document']['scores'], frozen)

    def test_display_segments_are_not_claimed_as_rst(self):
        text = 'Human writers compose a personal story. Software can polish the final wording.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'roberta', max_units=1)
        self.assertEqual(out['units_kind'], 'display_segments')
        self.assertIsNone(out['graph'])
        self.assertEqual(len(display_segments(text)), len(out['units']))


if __name__ == '__main__':
    unittest.main()
