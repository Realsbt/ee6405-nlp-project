"""Is `ai_humanized` written by a human separable from `ai_generated` at all?

Trains a dedicated binary classifier on the training split for each of the two
humanized sub-populations and scores it on the published test items. If a
purpose-built model with the full training set cannot beat the four-class
model's own recall, the confusion is a property of the data, not of the
architecture.
"""
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HART = ROOT / 'assets/hart_official'
LAB = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def load(split):
    out = []
    with (HART / f'{split}_graph.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            out.append(json.loads(line))
    return out


def stylometry(text):
    words = text.split()
    n = max(1, len(words))
    lens = np.array([len(w) for w in words]) if words else np.array([0.0])
    punct = sum(1 for c in text if c in ',.;:!?')
    return np.array([
        len(text), n, len(text) / n, lens.mean(), lens.std(),
        punct / max(1, len(text)), text.count('\n') / max(1, len(text)),
        sum(1 for c in text if c.isupper()) / max(1, len(text)),
        sum(1 for c in text if c.isdigit()) / max(1, len(text)),
        len(set(words)) / n,
    ])


def build(train, test, positive_kind):
    """Binary task: ai_humanized(kind) vs ai_generated."""
    def keep(record):
        if record['label'] == 'ai_generated':
            return True
        return record['label'] == 'ai_humanized' and record['model_name'] == positive_kind

    tr = [r for r in train if keep(r)]
    te = [r for r in test if keep(r)]
    y_tr = np.array([1 if r['label'] == 'ai_humanized' else 0 for r in tr])
    y_te = np.array([1 if r['label'] == 'ai_humanized' else 0 for r in te])
    return tr, y_tr, te, y_te


def run(train, test, positive_kind):
    tr, y_tr, te, y_te = build(train, test, positive_kind)
    print(f'\n=== ai_humanized[{positive_kind}] vs ai_generated ===')
    print(f'train n={len(tr)} (pos {int(y_tr.sum())})  test n={len(te)} (pos {int(y_te.sum())})')

    text_tr = [r['article'] for r in tr]
    text_te = [r['article'] for r in te]

    # 1. lexical only
    lex = make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                        min_df=3, sublinear_tf=True, max_features=200000),
                        LogisticRegression(max_iter=2000, C=4.0, class_weight='balanced'))
    lex.fit(text_tr, y_tr)
    p_lex = lex.predict_proba(text_te)[:, 1]

    # 2. stylometry only
    sty_tr = np.vstack([stylometry(t) for t in text_tr])
    sty_te = np.vstack([stylometry(t) for t in text_te])
    sty = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                             class_weight='balanced'))
    sty.fit(sty_tr, y_tr)
    p_sty = sty.predict_proba(sty_te)[:, 1]

    # 3. combined
    from scipy.sparse import hstack, csr_matrix
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3,
                          sublinear_tf=True, max_features=200000)
    X_tr = hstack([vec.fit_transform(text_tr), csr_matrix(sty_tr)]).tocsr()
    X_te = hstack([vec.transform(text_te), csr_matrix(sty_te)]).tocsr()
    both = LogisticRegression(max_iter=3000, C=4.0, class_weight='balanced')
    both.fit(X_tr, y_tr)
    p_both = both.predict_proba(X_te)[:, 1]

    base = max(y_te.mean(), 1 - y_te.mean())
    print(f'  majority-class baseline      {100*base:6.2f}%')
    for name, p in (('char n-gram TF-IDF', p_lex), ('stylometry only', p_sty),
                    ('TF-IDF + stylometry', p_both)):
        pred = (p >= 0.5).astype(int)
        acc = (pred == y_te).mean()
        rec = (pred[y_te == 1] == 1).mean()
        prec = (y_te[pred == 1] == 1).mean() if (pred == 1).any() else 0.0
        print(f'  {name:28} acc {100*acc:6.2f}%  recall(pos) {100*rec:6.2f}%  precision {100*prec:6.2f}%')
    return {'kind': positive_kind, 'n_train': len(tr), 'n_test': len(te),
            'pos_test': int(y_te.sum()), 'majority': float(base),
            'lexical_acc': float((p_lex >= .5).astype(int).__eq__(y_te).mean()),
            'stylometry_acc': float((p_sty >= .5).astype(int).__eq__(y_te).mean()),
            'combined_acc': float((p_both >= .5).astype(int).__eq__(y_te).mean()),
            'combined_recall': float((p_both[y_te == 1] >= .5).mean()),
            'combined_precision': float((y_te[p_both >= .5] == 1).mean()) if (p_both >= .5).any() else 0.0}


def main():
    train, test = load('train'), load('test')
    results = [run(train, test, kind) for kind in ('tool', 'human')]
    (ROOT / '.local/separability.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('\nwrote .local/separability.json')


if __name__ == '__main__':
    main()
