"""Contract tests for the TF-IDF-only server used on a small VPS.

These use a synthetic pipeline so they run without the real checkpoint; the
real model is exercised by scripts/verify_document_explanations.py.
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from experiments import tfidf_server


def synthetic_model():
    texts = [
        'Human people compose essays. Personal stories make sense.',
        'Software generates automated output. Machines write content.',
        'Writers polish their essay wording. The revision sounds formal.',
        'Humanized machine content is rewritten. Stories seem personal.',
    ] * 3
    labels = [0, 2, 1, 3] * 3
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=200)),
    ]).fit(texts, labels)


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.model = synthetic_model()

    def test_scores_are_a_probability_vector(self):
        out = tfidf_server.predict(self.model, 'Human people compose essays.')
        self.assertEqual(set(out['scores']), set(tfidf_server.LABELS))
        self.assertAlmostEqual(sum(out['scores'].values()), 1.0, places=6)
        self.assertIn(out['prediction'], tfidf_server.LABELS)

    def test_full_text_window_is_reported(self):
        text = 'Human people compose essays. ' * 200
        out = tfidf_server.predict(self.model, text)
        self.assertEqual(out['input']['lexical_window'], 'full_text')
        self.assertFalse(out['input']['lexical_truncated'])
        self.assertEqual(out['input']['lexical_covered_characters'], len(text))

    def test_empty_and_oversized_text_rejected(self):
        for text in ('', '   ', 'x' * (tfidf_server.MAX_CHARACTERS + 1)):
            with self.assertRaises(ValueError):
                tfidf_server.predict(self.model, text)

    def test_non_string_rejected(self):
        with self.assertRaises(ValueError):
            tfidf_server.predict(self.model, None)

    def test_registry_marks_other_models_unavailable(self):
        registry = tfidf_server.registry()
        by_id = {row['id']: row for row in registry['models']}
        self.assertTrue(by_id['tfidf']['available'])
        for name in ('roberta', 'race', 'race_no_supcon'):
            self.assertFalse(by_id[name]['available'])
            self.assertTrue(by_id[name]['availability_reason'])
        self.assertIsNone(registry['explanation_api'])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = synthetic_model()
        cls.page = b'<!doctype html><title>Text Lab</title>'
        handler = tfidf_server.make_handler(cls.model, cls.page, set())
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def url(self, path):
        return f'http://127.0.0.1:{self.port}{path}'

    def post(self, payload):
        request = urllib.request.Request(
            self.url('/api/predict'),
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST')
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def test_health(self):
        with urllib.request.urlopen(self.url('/api/health'), timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['status'], 'ok')

    def test_index_serves_the_page(self):
        with urllib.request.urlopen(self.url('/'), timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'Text Lab', response.read())

    def test_models_endpoint(self):
        with urllib.request.urlopen(self.url('/api/models'), timeout=10) as response:
            body = json.loads(response.read())
        self.assertEqual(len(body['models']), 4)

    def test_predict_contract(self):
        status, body = self.post({'model': 'tfidf', 'text': 'Human people compose essays.'})
        self.assertEqual(status, 200)
        self.assertEqual(body['model'], 'tfidf')
        self.assertAlmostEqual(sum(body['scores'].values()), 1.0, places=6)

    def test_other_models_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'model': 'race', 'text': 'Human people compose essays.'})
        self.assertEqual(caught.exception.code, 400)

    def test_empty_text_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'model': 'tfidf', 'text': ''})
        self.assertEqual(caught.exception.code, 400)

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.url('/api/explain'), timeout=10)
        self.assertEqual(caught.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
