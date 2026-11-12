"""Can any combination of the 13 published models beat E30?

E30 is an equal mean of E01 + E05 + E25. This asks whether a *learned*
combination does better.

Protocol (deliberately strict, after the short-document leakage audit):
  * every stacker is fit on the campaign **validation** split
  * the published **test** set is scored exactly once, at the end
  * the test set is never used to choose a method, a weight, or a feature

An earlier version of this experiment cross-fitted on the test set and
reported 95.15%. That number is not usable: the stacker had seen 4/5 of the
test labels. This version reports the honest figure and keeps the leaked
number only to quantify the bias.

Writes `.local/stacking.json`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/stacking.json'
N_BOOT = 4000
SEED = 20261113

LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']


def load_val() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payload = json.loads((ROOT / '.local/all_models_val.json').read_text(encoding='utf-8'))
    truth = np.asarray([LABELS.index(v) for v in payload['labels']])
    probs = {name: np.asarray(entry['probabilities'], dtype=np.float64)
             for name, entry in payload['models'].items()}
    return truth, probs


def load_test() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    recs = [json.loads(line) for line in
            (ROOT / 'experiments/hart_grouped_v1/test.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    truth = np.asarray([LABELS.index(r['label']) for r in recs])
    groups = np.asarray([r['group_id'] for r in recs])

    local = json.loads((ROOT / '.local/local_reinference.json').read_text(encoding='utf-8'))
    campaign = json.loads((ROOT / '.local/campaign_top3_published_test.json').read_text(encoding='utf-8'))
    probs: dict[str, np.ndarray] = {}
    for name, entry in local.items():
        probs[name] = np.asarray(entry['probabilities'], dtype=np.float64)
    for name, entry in campaign.items():
        probs[name] = np.asarray(entry['probabilities'], dtype=np.float64)
    for name, p in probs.items():
        assert p.shape == (len(truth), 4), f'{name} shape {p.shape}'
    return truth, groups, probs


def accuracy(p: np.ndarray, truth: np.ndarray) -> float:
    return float((p.argmax(1) == truth).mean())


def macro_f1(p: np.ndarray, truth: np.ndarray) -> float:
    pred = p.argmax(1)
    scores = []
    for c in range(4):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores))


def log_loss(p: np.ndarray, truth: np.ndarray) -> float:
    q = np.clip(p, 1e-12, 1.0)
    return float(-np.log(q[np.arange(len(truth)), truth]).mean())


def fit_nonneg_weights(stack: np.ndarray, truth: np.ndarray, rounds: int = 80) -> np.ndarray:
    """Coordinate ascent on log-loss over a simplex of model weights."""
    n_models = stack.shape[0]
    w = np.full(n_models, 1.0 / n_models)
    best = log_loss(np.tensordot(w, stack, axes=1), truth)
    step = 0.25
    for _ in range(rounds):
        improved = False
        for i in range(n_models):
            for delta in (step, -step):
                trial = w.copy()
                trial[i] = max(0.0, trial[i] + delta)
                total = trial.sum()
                if total <= 0:
                    continue
                trial /= total
                score = log_loss(np.tensordot(trial, stack, axes=1), truth)
                if score < best - 1e-9:
                    best, w, improved = score, trial, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return w


def fit_logreg(stack: np.ndarray, truth: np.ndarray, epochs: int = 600,
               lr: float = 0.35, l2: float = 1e-3) -> np.ndarray:
    """Multinomial logistic regression on concatenated probabilities."""
    x = np.concatenate([s for s in stack], axis=1)
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros((x.shape[1], 4))
    onehot = np.eye(4)[truth]
    for _ in range(epochs):
        logits = x @ w
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        w -= lr * (x.T @ (p - onehot) / len(x) + l2 * w)
    return w


def apply_logreg(w: np.ndarray, stack: np.ndarray) -> np.ndarray:
    x = np.concatenate([s for s in stack], axis=1)
    x = np.hstack([x, np.ones((len(x), 1))])
    logits = x @ w
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def paired_bootstrap(base: np.ndarray, cand: np.ndarray, truth: np.ndarray,
                     groups: np.ndarray, seed: int = SEED) -> dict:
    """Resample whole document groups, not individual items."""
    unique = np.array(sorted(set(groups)))
    index = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    base_ok = base.argmax(1) == truth
    cand_ok = cand.argmax(1) == truth
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        deltas[b] = 100 * (cand_ok[idx].mean() - base_ok[idx].mean())
    return {
        'mean_delta_pp': float(deltas.mean()),
        'ci95_pp': [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        'p_better': float((deltas > 0).mean()),
    }


def main() -> None:
    val_truth, val_probs = load_val()
    test_truth, test_groups, test_probs = load_test()

    names = sorted(set(val_probs) & set(test_probs))
    assert len(names) == 13, f'expected 13 shared models, got {len(names)}'
    val_stack = np.stack([val_probs[n] for n in names])
    test_stack = np.stack([test_probs[n] for n in names])

    baseline = test_probs['E30']
    results: dict[str, dict] = {}

    def record(label: str, p: np.ndarray, note: str = '') -> None:
        results[label] = {
            'accuracy': accuracy(p, test_truth),
            'macro_f1': macro_f1(p, test_truth),
            'log_loss': log_loss(p, test_truth),
            'bootstrap_vs_E30': paired_bootstrap(baseline, p, test_truth, test_groups),
            'note': note,
        }

    record('E30 (baseline)', baseline, 'equal mean of E01+E05+E25')

    # --- stackers fit on validation, applied to test once -----------------
    w = fit_nonneg_weights(val_stack, val_truth)
    record('nonneg weights (val-fit)', np.tensordot(w, test_stack, axes=1),
           'weights: ' + ', '.join(f'{n}={x:.3f}' for n, x in zip(names, w) if x > 0.01))

    record('logreg stack (val-fit)', apply_logreg(fit_logreg(val_stack, val_truth), test_stack))
    record('mean of all 13', test_stack.mean(0))

    # strong subset, chosen by validation accuracy only
    val_acc = {n: accuracy(val_probs[n], val_truth) for n in names}
    strong = [n for n in names if val_acc[n] > 0.92]
    strong_val = np.stack([val_probs[n] for n in strong])
    strong_test = np.stack([test_probs[n] for n in strong])
    record(f'logreg on {len(strong)} strong (val-fit)',
           apply_logreg(fit_logreg(strong_val, val_truth), strong_test),
           'strong = ' + ', '.join(strong))
    record(f'nonneg on {len(strong)} strong (val-fit)',
           np.tensordot(fit_nonneg_weights(strong_val, val_truth), strong_test, axes=1))

    # --- what the leaked protocol would have claimed ----------------------
    rng = np.random.default_rng(SEED)
    unique = np.array(sorted(set(test_groups)))
    rng.shuffle(unique)
    fold_of = {g: i % 5 for i, g in enumerate(unique)}
    folds = np.asarray([fold_of[g] for g in test_groups])
    leaked = np.zeros((len(test_truth), 4))
    for f in range(5):
        train, hold = folds != f, folds == f
        leaked[hold] = apply_logreg(fit_logreg(test_stack[:, train], test_truth[train]),
                                    test_stack[:, hold])
    record('logreg stack (test crossfit)', leaked,
           'LEAKED: stacker saw 4/5 of the test labels — kept only to quantify the bias')

    OUT.write_text(json.dumps({
        'protocol': {
            'stacker_fit_on': 'campaign validation split (1,600 items)',
            'test_used_for': 'single final evaluation',
            'bootstrap_unit': 'document group',
            'bootstrap_rounds': N_BOOT,
        },
        'n_val': int(len(val_truth)),
        'n_test': int(len(test_truth)),
        'n_groups': int(len(set(test_groups))),
        'models': names,
        'strong_models': strong,
        'val_accuracy': val_acc,
        'results': results,
    }, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'{"method":36s} {"acc":>7s} {"f1":>7s} {"logloss":>8s} {"d vs E30":>10s} {"95% CI":>18s} {"P":>6s}')
    print('-' * 100)
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]['accuracy']):
        b = r['bootstrap_vs_E30']
        ci = f'[{b["ci95_pp"][0]:+.2f}, {b["ci95_pp"][1]:+.2f}]'
        print(f'{label:36s} {100*r["accuracy"]:6.2f}% {100*r["macro_f1"]:6.2f}% '
              f'{r["log_loss"]:8.4f} {b["mean_delta_pp"]:+9.2f} {ci:>18s} {b["p_better"]:6.3f}')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
