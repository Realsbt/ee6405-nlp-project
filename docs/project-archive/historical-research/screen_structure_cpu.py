"""Train-only structural models and development-only fusion screening.

Uses published FULL-TEXT RST: exploratory, not a same-window comparison.
Never loads the grouped test partition or evaluates test predictions.
"""
import hashlib
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def features(record):
    """Whitelist text-derived features; no labels, domains, IDs or generator names."""
    text = record['article']
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\S+', text)
    length = {'log_chars': np.log1p(len(text)), 'log_words': np.log1p(len(words)),
              'log_sentences': np.log1p(len(sentences))}
    tree = record['rst_structure']
    if not isinstance(tree, dict):
        raise ValueError('Missing RST tree')
    stack = [(tree, 0)]
    depths, sizes, balance = [], [], []
    relations, nuclearities = Counter(), Counter()
    internal = 0
    while stack:
        node, depth = stack.pop()
        children = [node[k] for k in ('left', 'right') if node.get(k) is not None]
        if children:
            internal += 1
            relations[str(node.get('relation', 'unknown'))] += 1
            nuclearities[str(node.get('nuclearity', 'unknown'))] += 1
            if len(children) == 2:
                a, b = [max(0, c['end'] - c['start']) for c in children]
                balance.append(abs(a - b) / max(a + b, 1))
            stack.extend((c, depth + 1) for c in children)
        else:
            depths.append(depth)
            sizes.append(len(re.findall(r'\S+', node.get('text', ''))))
    if not depths:
        raise ValueError('No EDU leaves')
    structure = {
        'log_edu_count': np.log1p(len(depths)),
        'leaf_depth_mean': float(np.mean(depths)),
        'leaf_depth_max': float(max(depths)),
        'leaf_depth_sd': float(np.std(depths)),
        'edu_words_mean': float(np.mean(sizes)),
        'edu_words_sd': float(np.std(sizes)),
        'span_imbalance_mean': float(np.mean(balance)) if balance else 0.0,
    }
    structure.update({f'relation:{k}': v / max(internal, 1) for k, v in relations.items()})
    structure.update({f'nuclearity:{k}': v / max(internal, 1) for k, v in nuclearities.items()})
    return {'length': length, 'structure': structure, 'combined': {**length, **structure}}


def align_predictions(records, predictions):
    mapping = {r['item_id']: r for r in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r['item_id'] for r in records}:
        raise ValueError('Prediction ID mismatch or duplicates')
    result = []
    for r in records:
        p = mapping[r['item_id']]
        if p['true_label'] != r['label'] or p['group_id'] != r['group_id']:
            raise ValueError('Prediction label/group mismatch')
        result.append([p['probabilities'][k] for k in LABELS])
    result = np.asarray(result)
    if not np.isfinite(result).all() or (result < 0).any() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    return result / result.sum(1, keepdims=True)


def metrics(y, p, records):
    predicted = p.argmax(1)
    domain = {}
    for name in sorted({r['domain'] for r in records}):
        mask = np.array([r['domain'] == name for r in records])
        domain[name] = float(f1_score(y[mask], predicted[mask], labels=range(4), average='macro', zero_division=0))
    return {'macro_f1': float(f1_score(y, predicted, labels=range(4), average='macro', zero_division=0)),
            'class_recall': dict(zip(LABELS, recall_score(y, predicted, labels=range(4), average=None, zero_division=0).tolist())),
            'domain_macro_f1': domain, 'worst_domain_macro_f1': min(domain.values())}


def main():
    out = ROOT / 'structure_screen_v1'
    out.mkdir()  # Refuse to overwrite previous evidence.
    started = time.perf_counter()
    data = ROOT / 'resources/experiments/hart_grouped_v1'
    protocol = json.loads((data / 'protocol.json').read_text())
    split_rows = {}
    hashes = {}
    for split in ('train', 'val'):
        p = data / f'{split}.jsonl'
        hashes[str(p.relative_to(ROOT))] = digest(p)
        if hashes[str(p.relative_to(ROOT))] != protocol['partitions'][split]['sha256']:
            raise ValueError('Partition hash mismatch')
        split_rows[split] = rows(p)
    train, val = split_rows['train'], split_rows['val']
    if {r['split_block_id'] for r in train} & {r['split_block_id'] for r in val}:
        raise ValueError('Source blocks overlap')
    if {r['normalized_text_sha256'] for r in train} & {r['normalized_text_sha256'] for r in val}:
        raise ValueError('Exact text overlaps')
    wanted = {r['item_id']: r for r in train + val}
    if len(wanted) != len(train) + len(val):
        raise ValueError('Duplicate item IDs')
    write(out / 'protocol.json', {
        'status': 'development_screen_only', 'C_grid': C_GRID, 'fusion_alpha_grid': ALPHAS,
        'fit_on': 'train only', 'select_on': 'development macro-F1; smaller C/alpha breaks ties',
        'fusion': '(1-alpha)*RoBERTa_CE_s3407 + alpha*structural_LR',
        'graph_scope': 'published full-text RST; not same-window, not a causal structure ablation',
        'test_evaluation': False, 'script_sha256': digest(__file__), 'input_sha256': hashes,
        'limitations': 'Development also selected the neural checkpoint. Screening estimates are optimistic. '
                       'Strongest archived configuration RoBERTa+SupCon is not available here as development predictions. '
                       'Structure features include length proxies. Parsing cost is excluded from training timing.'})
    extracted = {}
    # Sources may be old raw partitions; select ONLY frozen current train/dev IDs.
    for path in sorted((ROOT / 'resources/assets/hart_official').glob('*_graph.jsonl')):
        hashes[str(path.relative_to(ROOT))] = digest(path)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                identifier = r['item_id']
                if identifier not in wanted:
                    continue
                expected = wanted[identifier]
                if r['article'] != expected['article'] or r['label'] != expected['label']:
                    raise ValueError('Raw/frozen record mismatch')
                if identifier in extracted:
                    raise ValueError('Duplicate raw record')
                extracted[identifier] = features(r)
    if set(extracted) != set(wanted):
        raise ValueError('Missing graph records')
    print(f'Extracted {len(extracted)} train/dev feature records', flush=True)
    y_train = np.array([r['label_id'] for r in train])
    y_val = np.array([r['label_id'] for r in val])
    prediction_path = ROOT / 'transfer_roberta_v1/validation_predictions.jsonl'
    neural = align_predictions(val, rows(prediction_path))
    hashes[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
    baseline = metrics(y_val, neural, val)
    report = {'baseline_roberta_ce_s3407': baseline, 'models': {}, 'input_sha256': hashes}
    with threadpool_limits(limits=2):
        for kind in ('length', 'structure', 'combined'):
            x_train = [extracted[r['item_id']][kind] for r in train]
            x_val = [extracted[r['item_id']][kind] for r in val]
            candidates, best = [], None
            for c in C_GRID:
                model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(),
                                      LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=3407))
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(x_train, y_train)
                if model[-1].classes_.tolist() != list(range(4)):
                    raise ValueError('Wrong class order')
                p = model.predict_proba(x_val)
                m = metrics(y_val, p, val)
                candidates.append({'C': c, 'development': m})
                if best is None or m['macro_f1'] > best[0]:
                    best = (m['macro_f1'], c, model, p, m)
            _, c, model, p, m = best
            joblib.dump(model, out / f'{kind}.joblib')
            fusion = [{'alpha': a, 'development': metrics(y_val, (1-a)*neural+a*p, val)} for a in ALPHAS]
            selected = max(fusion, key=lambda r: (r['development']['macro_f1'], -r['alpha']))
            report['models'][kind] = {'selected_C': c, 'candidates': candidates, 'development': m,
                                      'fusion_grid': fusion, 'selected_fusion': selected,
                                      'feature_names': model[0].get_feature_names_out().tolist()}
            with (out / f'{kind}_development_predictions.jsonl').open('x') as f:
                for r, probs in zip(val, p):
                    f.write(json.dumps({'item_id': r['item_id'], 'group_id': r['group_id'],
                                        'true_label': r['label'], 'probabilities': dict(zip(LABELS, probs.tolist()))})+'\n')
            print(f'{kind}: dev F1 {m["macro_f1"]:.4f}, fusion {selected["development"]["macro_f1"]:.4f}, alpha {selected["alpha"]}', flush=True)
    report['elapsed_seconds'] = time.perf_counter() - started
    write(out / 'development_results.json', report)
    lines = ['# CPU 结构特征与融合：开发集筛选', '',
             '本轮只训练轻量模型和筛选候选，未评估测试集，不能视为已确认提升。', '',
             f'训练 {len(train)} 条；开发 {len(val)} 条。RoBERTa CE seed3407 开发 Macro-F1：{baseline["macro_f1"]:.2%}。', '',
             '| 特征 | 单独模型开发 F1 | 最佳融合开发 F1 | 相对 CE 基线（百分点） | 结构权重 |',
             '|---|---:|---:|---:|---:|']
    for kind, result in report['models'].items():
        f = result['selected_fusion']
        lines.append(f'| {kind} | {result["development"]["macro_f1"]:.2%} | {f["development"]["macro_f1"]:.2%} | {(f["development"]["macro_f1"]-baseline["macro_f1"])*100:+.2f} | {f["alpha"]} |')
    lines += ['', '解释边界：', '',
              '- 使用发布的全文 RST，而神经文本模型只看首512 tokens；不是同窗口对照。',
              '- 不读取来源、领域、生成器和标签作为特征；结构统计仍可能含长度及领域信号。',
              '- 正则强度及融合权重均按开发集选择；开发集还参与原神经模型选 checkpoint，分数有选择乐观偏差。',
              '- 对照为 CE 单种子，尚未对照最强 RoBERTa+SupCon；融合改善不能称为超过当前最佳方案。',
              '- 原始图文件流式扫描只提取冻结 train/dev ID，不提取或评估当前 test 特征。',
              '- 没有新增解析；CPU耗时不包括原解析成本，不能当作完整部署成本。',
              '- 若继续验证，须先冻结候选，再做独立评测、多种子及最强基线对照。']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
