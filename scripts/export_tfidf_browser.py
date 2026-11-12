"""Export the TF-IDF + logistic-regression model for in-browser inference.

The model is a fixed vocabulary, IDF vector and linear classifier, so it can
run in JavaScript without a backend. This script writes a compact payload and
verifies that a pure-Python reimplementation of the browser math reproduces
scikit-learn's probabilities exactly.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b')


def load_model():
    import joblib
    from experiments.inference import resolve_recorded_path

    report = ROOT / 'experiments/reports/tfidf_lr_v1/result.json'
    result = json.loads(report.read_text(encoding='utf-8'))
    return joblib.load(resolve_recorded_path(result['model_path']))


def quantize(values: np.ndarray, dtype: str) -> tuple[str, float]:
    """Store a float array as base64 of a fixed-width type."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if dtype == 'float32':
        q = flat.astype(np.float32)
        return base64.b64encode(q.tobytes()).decode('ascii'), 1.0
    if dtype == 'int16':
        scale = float(np.max(np.abs(flat))) or 1.0
        q = np.round(flat / scale * 32767).astype(np.int16)
        return base64.b64encode(q.tobytes()).decode('ascii'), scale
    raise ValueError(dtype)


def export(model) -> dict:
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    vocabulary = vectorizer.vocabulary_
    # Index order must match the coefficient columns.
    terms = [None] * len(vocabulary)
    for term, index in vocabulary.items():
        terms[index] = term
    if any(t is None for t in terms):
        raise SystemExit('Vocabulary indices are not contiguous')

    coef_b64, coef_scale = quantize(classifier.coef_, 'float32')
    idf_b64, idf_scale = quantize(vectorizer.idf_, 'float32')
    return {
        'schema': 'ee6405.tfidf-browser/v1',
        'terms': '\n'.join(terms),
        'ngram_range': list(vectorizer.ngram_range),
        'lowercase': bool(vectorizer.lowercase),
        'sublinear_tf': bool(vectorizer.sublinear_tf),
        'smooth_idf': bool(vectorizer.smooth_idf),
        'norm': vectorizer.norm,
        'token_pattern': vectorizer.token_pattern,
        'idf': {'encoding': 'float32', 'data': idf_b64},
        'coef': {'encoding': 'float32', 'data': coef_b64,
                 'shape': list(classifier.coef_.shape)},
        'intercept': [float(x) for x in classifier.intercept_],
        'classes': [int(x) for x in classifier.classes_],
        'n_features': int(len(terms)),
        'n_documents': int(len(vectorizer.idf_)),
    }


def decode(payload: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    terms = payload['terms'].split('\n')
    idf = np.frombuffer(base64.b64decode(payload['idf']['data']), dtype=np.float32).astype(np.float64)
    coef = np.frombuffer(base64.b64decode(payload['coef']['data']), dtype=np.float32).astype(np.float64)
    coef = coef.reshape(payload['coef']['shape'])
    intercept = np.asarray(payload['intercept'], dtype=np.float64)
    return terms, idf, coef, intercept


def browser_transform(text: str, payload: dict, terms: list[str], idf: np.ndarray) -> np.ndarray:
    """Mirror of the JavaScript transform, used to prove parity."""
    index = {term: i for i, term in enumerate(terms)}
    source = text.lower() if payload['lowercase'] else text
    words = [m.group() for m in TOKEN_PATTERN.finditer(source)]
    lo, hi = payload['ngram_range']
    counts: dict[int, int] = {}
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            j = index.get(' '.join(words[i:i + n]))
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
    vector = np.zeros(len(terms), dtype=np.float64)
    for j, count in counts.items():
        vector[j] = (1.0 + math.log(count)) if payload['sublinear_tf'] else float(count)
    vector *= idf
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def verify(model, payload: dict, texts: list[str]) -> float:
    terms, idf, coef, intercept = decode(payload)
    worst = 0.0
    for text in texts:
        expected = model.predict_proba([text])[0]
        vector = browser_transform(text, payload, terms, idf)
        actual = softmax(coef @ vector + intercept)
        worst = max(worst, float(np.max(np.abs(expected - actual))))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'docs/demo/tfidf-model.json'))
    args = parser.parse_args()

    model = load_model()
    payload = export(model)
    texts = [
        'The library extended its opening hours during the examination period.',
        'However, the booking system has caused confusion among first-year students.',
        'The university plans to publish a short guide next week.',
        'Human writers compose a personal story about their own experience.',
        'Software generates automated output and machines write content quickly.',
        'Writers polish their essay wording and the revision sounds formal.',
        'Humanized machine content is rewritten so that stories seem personal.',
        'A party of Orcs sent by Saruman and Sauron attack the Fellowship.',
    ]
    worst = verify(model, payload, texts)
    print(f'parity max abs diff: {worst:.3e}')
    if worst > 1e-6:
        raise SystemExit('Browser math does not match scikit-learn; refusing to export')

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    print(f'Wrote {out} ({out.stat().st_size/1024:.0f} KiB)')


if __name__ == '__main__':
    main()
