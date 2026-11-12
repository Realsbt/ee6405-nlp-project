"""Three information-theoretic and stylometric detectors for HART four-class.

M1  Entropy and compressibility profile
    Shannon entropy of characters, conditional entropy of character bigrams,
    zlib compression ratio, and sentence-length burstiness. These measure how
    predictable the surface form is, which differs between human and generated
    text without any learned vocabulary.

M2  Stylometric feature vector
    Function-word frequencies (Burrows-style), punctuation profile, lexical
    richness (type-token ratio, hapax ratio), and sentence-length moments.
    This is authorship-attribution style analysis applied to provenance.

M3  Character n-gram cross-entropy divergence
    A smoothed character 4-gram model per class; classify by the class whose
    model assigns the text the lowest cross-entropy. This is a generative
    information-theoretic baseline, not a discriminative classifier.

All three train only on the group-disjoint pool and evaluate on the campaign's
S partition, so no HART group appears in both training and evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import zlib
from collections import Counter, defaultdict

import numpy as np

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'experiments/campaign12h/ee6405_scorefirst_36_12h'

SENTENCE = re.compile(r'[.!?]+(?:\s|$)')
WORD = re.compile(r"[A-Za-z']+")

# Burrows-style function words: high frequency, low semantic content.
FUNCTION_WORDS = (
    'the of and to a in that is was he for it with as his on be at by i this had not are but from or '
    'have an they which one you were her all she there would their we him been has when who will more no '
    'if out so said what up its about into than them can only other new some could time these two may then '
    'do first any my now such like our over man me even most made after also did many before must through '
    'back years where much your way well down should because each just those people mr how too little state '
    'good very make world still own see men work long get here between both life being under never day same '
    'another know while last might us great old year off come since against go came right used take three'
).split()

PUNCTUATION = '.,;:!?\'"()[]-–—…/\\@#$%&*+=<>{}|~`^_'


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE.split(text)]
    return [p for p in parts if p]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def shannon(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def conditional_entropy(text: str) -> float:
    """H(next | current) over characters, a measure of local predictability."""
    pairs = Counter(zip(text, text[1:]))
    first = Counter(text[:-1])
    total = sum(pairs.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for (a, b), count in pairs.items():
        p_ab = count / total
        p_a = first[a] / total
        h -= p_ab * math.log2(p_ab / p_a)
    return h


def burstiness(lengths: list[int]) -> float:
    """Coefficient of variation of sentence length; human text is burstier."""
    if len(lengths) < 2:
        return 0.0
    arr = np.asarray(lengths, dtype=float)
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else 0.0


def m1_features(text: str) -> list[float]:
    chars = Counter(text)
    raw = text.encode('utf-8')
    compressed = len(zlib.compress(raw, 9))
    lengths = [len(s) for s in sentences(text)]
    return [
        shannon(chars),
        conditional_entropy(text),
        compressed / max(1, len(raw)),
        burstiness(lengths),
        len(lengths) / max(1, len(words(text))),
        len(set(chars)) / max(1, len(chars)),
    ]


def m2_features(text: str) -> list[float]:
    lower = text.lower()
    tokens = words(text)
    n_tokens = max(1, len(tokens))
    counts = Counter(tokens)
    feats = [counts[w] / n_tokens for w in FUNCTION_WORDS]
    punct = Counter(ch for ch in text if ch in PUNCTUATION)
    total_punct = max(1, sum(punct.values()))
    feats += [punct[ch] / total_punct for ch in PUNCTUATION]
    types = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    feats += [
        types / n_tokens,
        hapax / n_tokens,
        math.log1p(n_tokens),
        len(text) / n_tokens,
    ]
    lengths = [len(s) for s in sentences(text)]
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        feats += [float(arr.mean()), float(arr.std()), float(np.median(arr)), float(arr.max())]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    return feats


class CharNgramModel:
    """Smoothed character n-gram model per class, scored by cross-entropy."""

    def __init__(self, n: int = 4, alpha: float = 0.1):
        self.n = n
        self.alpha = alpha
        self.counts: dict[int, Counter] = {}
        self.totals: dict[int, Counter] = {}
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[int]) -> 'CharNgramModel':
        for label in set(labels):
            self.counts[label] = Counter()
            self.totals[label] = Counter()
        for text, label in zip(texts, labels):
            padded = '\x02' * (self.n - 1) + text + '\x03'
            self.vocab.update(padded)
            for i in range(len(padded) - self.n + 1):
                gram = padded[i:i + self.n]
                self.counts[label][gram] += 1
                self.totals[label][gram[:-1]] += 1
        return self

    def cross_entropy(self, text: str, label: int) -> float:
        padded = '\x02' * (self.n - 1) + text + '\x03'
        v = len(self.vocab) + 2
        total = 0.0
        n = 0
        for i in range(len(padded) - self.n + 1):
            gram = padded[i:i + self.n]
            context = gram[:-1]
            c = self.counts[label][gram]
            t = self.totals[label][context]
            total -= math.log2((c + self.alpha) / (t + self.alpha * v))
            n += 1
        return total / max(1, n)

    def predict_proba(self, text: str) -> np.ndarray:
        scores = np.array([-self.cross_entropy(text, label) for label in range(len(LABELS))])
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train - mean) / std, (test - mean) / std


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }


def load_pool() -> dict:
    return json.loads((ROOT / '.local/clean_pool.json').read_text(encoding='utf-8'))


def load_s_partition() -> tuple[list[str], np.ndarray]:
    manifest = json.loads((CAMPAIGN / 'protocol/split_manifest.json').read_text(encoding='utf-8'))
    wanted = {r['item_id']: r for r in manifest['records'] if r['partition'] == 'S'}
    texts, labels = [], []
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        path = ROOT / 'assets/hart_official' / split
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    texts.append(record['article'])
                    labels.append(LABEL_ID[record['label']])
    order = {r['item_id']: i for i, r in enumerate(manifest['records'])}
    pairs = sorted(zip(texts, labels), key=lambda p: 0)
    return texts, np.asarray(labels)


def load_s_ordered() -> tuple[list[str], np.ndarray]:
    """Return S texts in the manifest's canonical order."""
    manifest = json.loads((CAMPAIGN / 'protocol/split_manifest.json').read_text(encoding='utf-8'))
    s_records = [r for r in manifest['records'] if r['partition'] == 'S']
    wanted = {r['item_id'] for r in s_records}
    by_id = {}
    for split in ('train_graph.jsonl', 'val_graph.jsonl', 'test_graph.jsonl'):
        path = ROOT / 'assets/hart_official' / split
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['item_id'] in wanted:
                    by_id[record['item_id']] = record['article']
    texts = [by_id[r['item_id']] for r in s_records]
    labels = np.asarray([int(r['label_id']) for r in s_records])
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / '.local/info_methods.json'))
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pool = load_pool()
    train_rows = pool['train_graph.jsonl']
    train_texts = [r['article'] for r in train_rows]
    train_labels = np.asarray([LABEL_ID[r['label']] for r in train_rows])
    print(f'train {len(train_texts)}  labels {np.bincount(train_labels)}', flush=True)

    s_texts, s_labels = load_s_ordered()
    print(f'S {len(s_texts)}  labels {np.bincount(s_labels)}', flush=True)

    results = {}

    # M1
    print('M1 entropy/compression…', flush=True)
    x_train = np.asarray([m1_features(t) for t in train_texts])
    x_test = np.asarray([m1_features(t) for t in s_texts])
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(
        scaler.transform(x_train), train_labels)
    pred = clf.predict(scaler.transform(x_test))
    results['M1_entropy_compression'] = {**metrics(s_labels, pred),
                                         'features': x_train.shape[1],
                                         'proba': clf.predict_proba(scaler.transform(x_test)).tolist()}
    print('  ', results['M1_entropy_compression']['accuracy'],
          results['M1_entropy_compression']['macro_f1'], flush=True)

    # M2
    print('M2 stylometric…', flush=True)
    x_train = np.asarray([m2_features(t) for t in train_texts])
    x_test = np.asarray([m2_features(t) for t in s_texts])
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced').fit(
        scaler.transform(x_train), train_labels)
    pred = clf.predict(scaler.transform(x_test))
    results['M2_stylometric'] = {**metrics(s_labels, pred),
                                 'features': x_train.shape[1],
                                 'proba': clf.predict_proba(scaler.transform(x_test)).tolist()}
    print('  ', results['M2_stylometric']['accuracy'],
          results['M2_stylometric']['macro_f1'], flush=True)

    # M3
    print('M3 char n-gram cross-entropy…', flush=True)
    model = CharNgramModel(n=4, alpha=0.1).fit(train_texts, train_labels.tolist())
    proba = np.asarray([model.predict_proba(t) for t in s_texts])
    pred = proba.argmax(axis=1)
    results['M3_char_ngram_cross_entropy'] = {**metrics(s_labels, pred),
                                              'features': len(model.vocab),
                                              'proba': proba.tolist()}
    print('  ', results['M3_char_ngram_cross_entropy']['accuracy'],
          results['M3_char_ngram_cross_entropy']['macro_f1'], flush=True)

    # Fusion with the campaign's best baseline (E26) to test complementarity.
    e26 = json.loads((CAMPAIGN / 'top5_revision_01/E26_prediction_with_ids.json').read_text(encoding='utf-8'))
    e26_proba = np.asarray([r['probabilities'] for r in e26['prediction_rows']])
    e26_pred = e26_proba.argmax(axis=1)
    results['E26_reference'] = metrics(s_labels, e26_pred)

    for name in ('M1_entropy_compression', 'M2_stylometric', 'M3_char_ngram_cross_entropy'):
        fused = 0.5 * e26_proba + 0.5 * np.asarray(results[name]['proba'])
        results[f'{name}+E26'] = metrics(s_labels, fused.argmax(axis=1))

    pathlib.Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nwrote {args.out}')
    for name, row in results.items():
        print(f"{name:34} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
