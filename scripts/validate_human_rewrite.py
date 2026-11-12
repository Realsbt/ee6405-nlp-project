"""Is the human-rewrite difficulty a labelling problem or a modelling problem?

The claim under test: human-rewrite items are hard because the *label* is not
recoverable from the text, not because our models are too weak.

Four independent probes, each with a different failure mode:

  1. Label noise / ceiling — train a strong classifier on human-rewrite vs
     everything else with the full training pool, then look at the score
     distribution. If positives and negatives overlap heavily, the label is not
     separable in feature space.

  2. Human ceiling proxy — a human annotator cannot be simulated, but we can
     ask whether *any* linear or non-linear model on rich features separates
     them. We sweep model capacity: logistic regression, gradient boosting,
     and a nearest-neighbour rule on TF-IDF. If capacity does not help, the
     limit is in the data.

  3. Sibling test — human-rewrite items come from the same source document as
     an ai_generated item. If the rewrite preserved the argument structure, a
     model given *both* texts should be able to tell which is which. We test
     the pairwise task directly: given (GEN, HUM) from the same group, can any
     model rank them correctly?

  4. Label consistency — check whether human-rewrite items are more similar to
     their ai_generated sibling than to human_written items from the same
     domain. If they sit closer to GEN, the label is arguably mis-specified.

Writes `.local/human_rewrite_identifiability.json`.
"""
from __future__ import annotations

import collections
import difflib
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / '.local/human_rewrite_identifiability.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
SEED = 20261113


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in
            (ROOT / f'experiments/hart_grouped_v1/{split}.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]


def stylometry(text: str) -> np.ndarray:
    words = re.findall(r"[A-Za-z']+", text)
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n = len(words) or 1
    return np.array([
        len(text) / 5000.0,
        len(words) / 1000.0,
        float(np.mean([len(w) for w in words])) if words else 0.0,
        float(np.std([len(w) for w in words])) if words else 0.0,
        len(sents) / 50.0,
        float(np.mean([len(re.findall(r"[A-Za-z']+", s)) for s in sents])) if sents else 0.0,
        float(np.std([len(re.findall(r"[A-Za-z']+", s)) for s in sents])) if sents else 0.0,
        text.count(',') / n * 100,
        text.count(';') / n * 100,
        text.count(':') / n * 100,
        text.count('—') / n * 100,
        text.count('(') / n * 100,
        sum(1 for c in text if c.isupper()) / max(len(text), 1) * 10,
        len(set(words)) / n,
        sum(1 for w in words if len(w) > 8) / n,
        text.count(' the ') / n * 100,
        text.count(' of ') / n * 100,
        text.count(' and ') / n * 100,
    ])


def probe_separability(train: list[dict], test: list[dict]) -> dict:
    """Probe 1+2: can any model separate human-rewrite from the rest?"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def build(rows):
        y = np.array([1 if (r['label'] == 'ai_humanized' and r.get('model_name') == 'human') else 0
                      for r in rows])
        return [r['article'] for r in rows], y

    x_train, y_train = build(train)
    x_test, y_test = build(test)
    print(f'  train {len(y_train)} (pos {y_train.sum()})  test {len(y_test)} (pos {y_test.sum()})',
          flush=True)

    results = {}
    models = {
        'tfidf+logreg': make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000),
            LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
        'tfidf+logreg C=0.1': make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000),
            LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)),
        'char+logreg': make_pipeline(
            TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3,
                            sublinear_tf=True, max_features=200000),
            LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        results[name] = {
            'accuracy': float((pred == y_test).mean()),
            'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
            'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
            'mean_score_pos': float(proba[y_test == 1].mean()),
            'mean_score_neg': float(proba[y_test == 0].mean()),
        }
        print(f'    {name:22s} acc {100*results[name]["accuracy"]:5.2f}%  '
              f'recall {100*results[name]["recall"]:5.2f}%  '
              f'precision {100*results[name]["precision"]:5.2f}%', flush=True)

    # stylometry-only, to see whether hand features do better
    xs_train = np.array([stylometry(t) for t in x_train])
    xs_test = np.array([stylometry(t) for t in x_test])
    gb = GradientBoostingClassifier(random_state=SEED, n_estimators=200, max_depth=3)
    gb.fit(xs_train, y_train)
    proba = gb.predict_proba(xs_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    results['stylometry+gbm'] = {
        'accuracy': float((pred == y_test).mean()),
        'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
        'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'mean_score_pos': float(proba[y_test == 1].mean()),
        'mean_score_neg': float(proba[y_test == 0].mean()),
    }
    print(f'    {"stylometry+gbm":22s} acc {100*results["stylometry+gbm"]["accuracy"]:5.2f}%  '
          f'recall {100*results["stylometry+gbm"]["recall"]:5.2f}%  '
          f'precision {100*results["stylometry+gbm"]["precision"]:5.2f}%', flush=True)
    return results


def probe_pairwise(test: list[dict]) -> dict:
    """Probe 3: given (GEN, HUM) from the same group, can a model rank them?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in test:
        by_group[r['group_id']].append(r)

    pairs = []
    for g, items in by_group.items():
        gen = [i for i in items if i['label'] == 'ai_generated']
        hum = [i for i in items if i['label'] == 'ai_humanized' and i.get('model_name') == 'human']
        if gen and hum:
            pairs.append((gen[0]['article'], hum[0]['article']))
    print(f'  {len(pairs)} (GEN, HUM) pairs from the same source document', flush=True)

    # Train on train-split pairs, evaluate on test pairs
    train = load_split('train')
    by_group_train: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train:
        by_group_train[r['group_id']].append(r)
    train_pairs = []
    for g, items in by_group_train.items():
        gen = [i for i in items if i['label'] == 'ai_generated']
        hum = [i for i in items if i['label'] == 'ai_humanized' and i.get('model_name') == 'human']
        if gen and hum:
            train_pairs.append((gen[0]['article'], hum[0]['article']))
    print(f'  {len(train_pairs)} training pairs', flush=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)
    corpus = [t for p in train_pairs for t in p] + [t for p in pairs for t in p]
    vec.fit(corpus)

    def diff_features(a: str, b: str) -> np.ndarray:
        va, vb = vec.transform([a, b])
        return np.asarray((vb - va).todense()).ravel()

    x_train = np.array([diff_features(a, b) for a, b in train_pairs])
    y_train = np.ones(len(train_pairs))
    x_train = np.vstack([x_train, -x_train])
    y_train = np.concatenate([y_train, np.zeros(len(train_pairs))])

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit(x_train, y_train)
    x_test = np.array([diff_features(a, b) for a, b in pairs])
    scores = clf.decision_function(x_test)
    correct = int((scores > 0).sum())
    print(f'  pairwise ranking accuracy: {correct}/{len(pairs)} = {100*correct/len(pairs):.2f}%',
          flush=True)
    return {
        'n_pairs': len(pairs),
        'n_train_pairs': len(train_pairs),
        'ranking_accuracy': correct / len(pairs),
        'chance': 0.5,
    }


def probe_label_consistency(test: list[dict]) -> dict:
    """Probe 4: are human-rewrite items closer to GEN or to HW?"""
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in test:
        by_group[r['group_id']].append(r)

    closer_gen = 0
    total = 0
    sims_gen, sims_hw = [], []
    for g, items in by_group.items():
        gen = [i for i in items if i['label'] == 'ai_generated']
        hw = [i for i in items if i['label'] == 'human_written']
        hum = [i for i in items if i['label'] == 'ai_humanized' and i.get('model_name') == 'human']
        if not (gen and hw and hum):
            continue
        total += 1
        sg = difflib.SequenceMatcher(None, gen[0]['article'], hum[0]['article']).ratio()
        sw = difflib.SequenceMatcher(None, hw[0]['article'], hum[0]['article']).ratio()
        sims_gen.append(sg)
        sims_hw.append(sw)
        if sg > sw:
            closer_gen += 1

    print(f'  {total} groups with GEN + HW + HUM', flush=True)
    print(f'  closer to GEN: {closer_gen}/{total} = {100*closer_gen/total:.1f}%', flush=True)
    print(f'  mean similarity  GEN {np.mean(sims_gen):.4f}  HW {np.mean(sims_hw):.4f}', flush=True)
    return {
        'n_groups': total,
        'closer_to_generated': closer_gen,
        'closer_to_generated_rate': closer_gen / total,
        'mean_similarity_to_generated': float(np.mean(sims_gen)),
        'mean_similarity_to_human_written': float(np.mean(sims_hw)),
    }


def main() -> None:
    train = load_split('train')
    test = load_split('test')

    print('probe 1+2: separability with increasing model capacity', flush=True)
    separability = probe_separability(train, test)

    print('\nprobe 3: pairwise GEN vs HUM ranking', flush=True)
    pairwise = probe_pairwise(test)

    print('\nprobe 4: label consistency', flush=True)
    consistency = probe_label_consistency(test)

    OUT.write_text(json.dumps({
        'separability': separability,
        'pairwise_ranking': pairwise,
        'label_consistency': consistency,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
