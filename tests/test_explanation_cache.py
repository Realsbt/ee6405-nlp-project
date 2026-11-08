"""Preparation-cache contract tests for the v5.1 on-demand explanation path.

These use the synthetic fixtures from test_document_explanations; they verify
reuse and invalidation behaviour, not trained-checkpoint accuracy.
"""
import unittest

import torch

from experiments.document_explanations import (
    PREP_CACHE_ENTRIES, cache_stats, clear_preparation_cache, explain_document,
    preparation_key, remember_document, reset_cache_stats,
)
from tests.test_document_explanations import Service, fake_data_interfaces


class PreparationCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        clear_preparation_cache()
        self.service = Service()
        self.text = 'Human writers compose a personal story. Software can polish the final wording.'

    def test_prepare_only_returns_metadata_without_influence(self):
        with fake_data_interfaces():
            out = explain_document(self.service, self.text, 'race', prepare_only=True)
        self.assertEqual(out['status'], 'prepared')
        self.assertEqual(out['checked_units'], 0)
        self.assertEqual(out['selected_unit_ids'], [])
        self.assertIsNone(out['restored_max_abs_diff'])
        self.assertTrue(out['units'])
        self.assertTrue(out['graph']['nodes'])
        for unit in out['units']:
            self.assertIn(unit['status'], ('not_computed', 'outside_lexical_window'))
            self.assertIsNone(unit['delta_pp'])
            self.assertIsNone(unit['without_scores'])

    def test_prepare_only_still_reports_coverage(self):
        text = 'word ' * 520 + '. Final sentence.'
        with fake_data_interfaces():
            out = explain_document(self.service, text, 'roberta', prepare_only=True)
        self.assertEqual(out['units'][0]['coverage'], 'partial')
        self.assertEqual(out['units'][-1]['status'], 'outside_lexical_window')
        self.assertGreater(out['document']['input']['lexical_covered_characters'], 0)

    def test_second_batch_reuses_preparation(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            calls_after_first = self.service.predict_calls
            second = explain_document(self.service, self.text, 'race', unit_ids=['u2'], max_units=1)
        self.assertEqual(first['cache']['preparation'], 'miss')
        self.assertEqual(second['cache']['preparation'], 'hit')
        self.assertEqual(self.service.predict_calls, calls_after_first)
        self.assertEqual(second['checked_units'], 1)
        self.assertEqual(second['units'][1]['status'], 'computed')
        self.assertEqual(second['units'][0]['status'], 'not_computed')

    def test_reused_preparation_keeps_identical_baseline(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            second = explain_document(self.service, self.text, 'race', unit_ids=['u1', 'u2'], max_units=2)
        self.assertEqual(first['document']['scores'], second['document']['scores'])
        self.assertLessEqual(second['restored_max_abs_diff'], 2e-5)
        self.assertEqual(first['units'][0]['delta_pp'], second['units'][0]['delta_pp'])
        self.assertEqual(second['units'][1]['status'], 'computed')

    def test_refresh_rebuilds_preparation(self):
        with fake_data_interfaces():
            explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
            refreshed = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1, refresh=True)
        self.assertEqual(refreshed['cache']['preparation'], 'miss')
        self.assertEqual(refreshed['checked_units'], 1)

    def test_key_separates_text_model_and_service(self):
        other = Service()
        base = preparation_key(self.text, 'race', self.service)
        self.assertNotEqual(base, preparation_key(self.text + ' ', 'race', self.service))
        self.assertNotEqual(base, preparation_key(self.text, 'roberta', self.service))
        self.assertNotEqual(base, preparation_key(self.text, 'race', other))

    def test_remember_document_avoids_a_second_prediction(self):
        document = self.service.predict(self.text, 'race')
        calls = self.service.predict_calls
        remember_document(self.text, 'race', document, self.service)
        with fake_data_interfaces():
            out = explain_document(self.service, self.text, 'race', unit_ids=['u1'], max_units=1)
        self.assertEqual(self.service.predict_calls, calls)
        self.assertEqual(out['document']['scores'], document['scores'])

    def test_remembered_document_is_not_shared_across_services(self):
        document = self.service.predict(self.text, 'race')
        remember_document(self.text, 'race', document, self.service)
        other = Service()
        with fake_data_interfaces():
            explain_document(other, self.text, 'race', unit_ids=['u1'], max_units=1)
        self.assertEqual(other.predict_calls, 1)

    def test_cache_is_bounded(self):
        with fake_data_interfaces():
            for i in range(PREP_CACHE_ENTRIES + 2):
                explain_document(self.service, self.text + ' ' * i, 'roberta', prepare_only=True)
        stats = cache_stats()
        self.assertLessEqual(stats['prepared_entries'], PREP_CACHE_ENTRIES)
        self.assertGreaterEqual(stats['evictions'], 2)

    def test_stats_count_hits_and_misses(self):
        with fake_data_interfaces():
            explain_document(self.service, self.text, 'roberta', prepare_only=True)
            reset_cache_stats()
            explain_document(self.service, self.text, 'roberta', prepare_only=True)
            explain_document(self.service, self.text, 'roberta', prepare_only=True)
        stats = cache_stats()
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['preprocessing_version'], 'ee6405.explain-prep/v1')

    def test_clear_cache_forces_rebuild(self):
        with fake_data_interfaces():
            explain_document(self.service, self.text, 'roberta', prepare_only=True)
            clear_preparation_cache()
            out = explain_document(self.service, self.text, 'roberta', prepare_only=True)
        self.assertEqual(out['cache']['preparation'], 'miss')

    def test_prepare_only_does_not_leak_hooks(self):
        with fake_data_interfaces():
            explain_document(self.service, self.text, 'race', prepare_only=True)
        for module in self.service.models['race'].modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_tfidf_preparation_is_reused(self):
        with fake_data_interfaces():
            first = explain_document(self.service, self.text, 'tfidf', unit_ids=['u1'], max_units=1)
            second = explain_document(self.service, self.text, 'tfidf', unit_ids=['u2'], max_units=1)
        self.assertEqual(first['cache']['preparation'], 'miss')
        self.assertEqual(second['cache']['preparation'], 'hit')
        self.assertEqual(second['units'][1]['status'], 'computed')


if __name__ == '__main__':
    unittest.main()
