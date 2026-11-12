"""Bounded CPU improvement search; freeze development selections before evaluation.

All neural encoders/checkpoints remain frozen. No test-based candidate selection.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from screen_structure_cpu import ROOT, LABELS, digest, rows, write, metrics
from evaluation_metrics import evaluate_creator, negative_only_threshold

OUT = ROOT / 'cpu_improvement_v1'
DATA = ROOT / 'resources/experiments/hart_grouped_v1'
ARCHIVE = ROOT / 'repository/experiments/reports'
CS = [0.1, 1.0, 10.0]
WEIGHTS = [None, 'balanced']
BLENDS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
BIAS = [0.75, 1.0, 1.25, 1.5, 2.0]


def probability_array(records, path, key='probabilities'):
    source = rows(path)
    by_id = {r['item_id']: r for r in source}
    if len(by_id) != len(source) or set(by_id) != {r['item_id'] for r in records}:
        raise ValueError(f'Prediction ID mismatch: {path}')
    p = []
    for r in records:
        s = by_id[r['item_id']]
        if 'true_label' in s and s['true_label'] != r['label']:
            raise ValueError('Prediction label mismatch')
        if 'group_id' in s and s['group_id'] != r['group_id']:
            raise ValueError('Prediction group mismatch')
        p.append([s[key][label] for label in LABELS])
    return validate_probabilities(p)


def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid four-class probabilities')
    return p / p.sum(1, keepdims=True)


def specialize(p, q, beta):
    """Preserve creator probability mass; adjust only AI-generated/humanized split."""
    p = validate_probabilities(p)
    q = np.asarray(q)
    if q.shape != (len(p), 2) or not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1):
        raise ValueError('Invalid specialist probabilities')
    if not 0 <= beta <= 1:
        raise ValueError('Invalid blend')
    result = p.copy()
    result[:, 2:] = (1-beta)*p[:, 2:] + beta*p[:, 2:].sum(1, keepdims=True)*q
    return validate_probabilities(result)


def bias_humanized(p, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid class multiplier')
    q = p.copy()
    q[:, 3] *= factor
    return q / q.sum(1, keepdims=True)


def score(y, p):
    cm = confusion_matrix(y, p.argmax(1), labels=range(4))
    den = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm), den, out=np.zeros(4), where=den > 0).mean())


def bootstrap_deltas(records, y, candidates, references, n=2000):
    """Source-block paired percentile bootstrap, fixed fitted models only."""
    groups = [r.get('split_block_id', r['group_id']) for r in records]
    names = sorted(set(groups))
    group_index = dict(zip(names, range(len(names))))
    gi = np.array([group_index[g] for g in groups])
    all_p = {**candidates, **references}
    tensors = {}
    for name, p in all_p.items():
        cm = np.zeros((len(names), 4, 4), dtype=int)
        np.add.at(cm, (gi, y, p.argmax(1)), 1)
        tensors[name] = cm
    rng = np.random.default_rng(20261109)
    weights = rng.multinomial(len(names), np.full(len(names), 1/len(names)), size=n)
    scores = {}
    for name, tensor in tensors.items():
        cm = np.einsum('bg,gij->bij', weights, tensor)
        den = cm.sum(1)+cm.sum(2)
        scores[name] = np.divide(2*np.diagonal(cm, axis1=1, axis2=2), den,
                                 out=np.zeros((n,4)), where=den > 0).mean(1)
    return {name: {ref: {'delta': score(y, all_p[name])-score(y, all_p[ref]),
                         'paired_95_percentile': np.quantile(scores[name]-scores[ref], [.025,.975]).tolist()}
                   for ref in references} for name in candidates}


def save_predictions(path, records, p):
    with path.open('x') as f:
        for row, probs in zip(records, p):
            f.write(json.dumps({'item_id': row['item_id'], 'group_id': row['group_id'],
                                'true_label': row['label'], 'probabilities': dict(zip(LABELS, probs.tolist()))})+'\n')


def selected_candidates(base, race, old_tfidf, lexical, specialist, specs):
    result = {}
    for name, s in specs.items():
        a = s.get('alpha', 0)
        if name == 'race_blend':
            result[name] = (1-a)*base+a*race
        elif name == 'old_tfidf_blend':
            result[name] = (1-a)*base+a*old_tfidf
        elif name == 'lexical_blend':
            result[name] = (1-a)*base+a*lexical
        elif name == 'lexical_only':
            result[name] = lexical
        elif name == 'specialist':
            result[name] = specialize(base, specialist, s['beta'])
        elif name == 'class_bias':
            result[name] = bias_humanized(base, s['factor'])
        else:
            raise ValueError(name)
    return result


def main():
    OUT.mkdir()
    start = time.perf_counter()
    protocol = json.loads((DATA / 'protocol.json').read_text())
    loaded = {}
    hashes = {}
    for split in ('train', 'val'):
        path = DATA / f'{split}.jsonl'
        hashes[str(path.relative_to(ROOT))] = digest(path)
        if digest(path) != protocol['partitions'][split]['sha256']:
            raise ValueError('Partition changed')
        loaded[split] = rows(path)
    train, val = loaded['train'], loaded['val']
    if {r['split_block_id'] for r in train} & {r['split_block_id'] for r in val}:
        raise ValueError('Source overlap')
    if {r['normalized_text_sha256'] for r in train} & {r['normalized_text_sha256'] for r in val}:
        raise ValueError('Text overlap')
    write(OUT / 'protocol.json', {
        'date': '2026-11-09', 'device': 'CPU', 'train_count': len(train), 'development_count': len(val),
        'candidate_families': ['race_blend', 'old_tfidf_blend', 'lexical_only', 'lexical_blend', 'specialist', 'class_bias'],
        'C_grid': CS, 'class_weight_grid': WEIGHTS, 'blend_grid': BLENDS, 'humanized_factor_grid': BIAS,
        'text_budget': 'Existing model_text: first 512 RoBERTa tokens',
        'vectorizer': 'word(1,2), max50k + char_wb(3,5), max80k; min_df3; sublinear TF; train-only fit; row L2 normalize',
        'selection': 'Development Macro-F1 only; simpler/first candidate wins ties; freeze before test evaluation',
        'specialist': 'Train only on ai_generated/ai_humanized; preserve creator mass at inference',
        'neural_base': 'Frozen RoBERTa CE seed3407; RACE SupCon seed3407 optional second inference',
        'known_limitations': ['Development already used for original checkpoint selection and previous research.',
                              'HART test and external datasets have been inspected in previous work; evaluation is post-hoc.',
                              'No new independent test set or training seed replication.',
                              'RACE branch uses original full-text graph; not equal information budget.',
                              'Strong SupCon baseline lacks local weights/dev probabilities; use archived test only as reference.'],
        'input_sha256': hashes, 'script_sha256': digest(__file__)})
    y_train = np.array([r['label_id'] for r in train])
    y_val = np.array([r['label_id'] for r in val])
    neural_paths = {'base': ROOT/'transfer_roberta_v1/validation_predictions.jsonl',
                    'race': ROOT/'race_cpu_v1/validation_predictions.jsonl',
                    'tfidf': ROOT/'transfer_tfidf_v1/validation_predictions.jsonl'}
    p_val = {k: probability_array(val, p) for k,p in neural_paths.items()}
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in neural_paths.values()})
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=3, sublinear_tf=True)),
        ('character', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=80000, min_df=3, sublinear_tf=True))])
    xt = normalize(vectorizer.fit_transform([r['model_text'] for r in train]))
    xv = normalize(vectorizer.transform([r['model_text'] for r in val]))
    joblib.dump(vectorizer, OUT/'vectorizer.joblib')
    print(f'Train matrix {xt.shape}, {xt.nnz} nonzeros', flush=True)
    model_specs, models, probability = {}, {}, {}
    with threadpool_limits(limits=2):
        for name in ('lexical', 'specialist'):
            mask = np.ones(len(train), dtype=bool) if name == 'lexical' else y_train >= 2
            target = y_train[mask] if name == 'lexical' else y_train[mask]-2
            grid, best = [], None
            for weight in WEIGHTS:
                for c in CS:
                    t = time.perf_counter()
                    model = LogisticRegression(C=c, class_weight=weight, solver='lbfgs', max_iter=1500, random_state=3407)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        model.fit(xt[mask], target)
                    expected = list(range(4 if name == 'lexical' else 2))
                    if model.classes_.tolist() != expected:
                        raise ValueError('Class order differs')
                    p = model.predict_proba(xv)
                    blends = [{'alpha': a, 'macro_f1': score(y_val, (1-a)*p_val['base']+a*p)} for a in BLENDS] if name == 'lexical' else [
                        {'beta': b, 'macro_f1': score(y_val, specialize(p_val['base'], p, b))} for b in BLENDS]
                    # Lexical model C/weight selected on standalone F1; specialist selected on final four-class F1.
                    model_score = score(y_val, p) if name == 'lexical' else max(x['macro_f1'] for x in blends)
                    row = {'C':c, 'class_weight':weight, 'selection_score':model_score, 'blend_grid':blends,
                           'fit_seconds':time.perf_counter()-t, 'iterations': model.n_iter_.tolist()}
                    grid.append(row)
                    print(f'{name} C={c} weight={weight}: dev {model_score:.4f} ({row["fit_seconds"]:.1f}s)', flush=True)
                    if best is None or model_score > best[0]:
                        best = (model_score, model, p, row)
            _, model, p, selected = best
            model_specs[name] = {'selected':selected, 'grid':grid}
            models[name], probability[name] = model, p
            joblib.dump(model, OUT/f'{name}.joblib')
            write(OUT/f'{name}_selection.json', model_specs[name])
    grids = {}
    for name, partner in (('race_blend',p_val['race']), ('old_tfidf_blend',p_val['tfidf']), ('lexical_blend',probability['lexical'])):
        grids[name] = [{'alpha':a, 'macro_f1':score(y_val,(1-a)*p_val['base']+a*partner)} for a in BLENDS]
    grids['lexical_only'] = [{'macro_f1':score(y_val, probability['lexical'])}]
    grids['specialist'] = [{'beta':b, 'macro_f1':score(y_val,specialize(p_val['base'], probability['specialist'], b))} for b in BLENDS]
    grids['class_bias'] = [{'factor':f, 'macro_f1':score(y_val,bias_humanized(p_val['base'],f))} for f in BIAS]
    specs = {name:max(grid, key=lambda r:r['macro_f1']) for name,grid in grids.items()}
    champion = max(specs, key=lambda name:specs[name]['macro_f1'])
    development = selected_candidates(p_val['base'],p_val['race'],p_val['tfidf'],probability['lexical'],probability['specialist'],specs)
    thresholds = {name:negative_only_threshold(p[y_val<2,2:].sum(1)) for name,p in development.items()}
    thresholds['roberta_ce_s3407'] = negative_only_threshold(p_val['base'][y_val<2,2:].sum(1))
    selection = {'models':model_specs, 'candidate_grids':grids, 'selected':specs,
                 'development_champion':champion, 'development':{name:metrics(y_val,p,val) for name,p in development.items()},
                 'baseline_development':metrics(y_val,p_val['base'],val), 'thresholds':thresholds,
                 'input_sha256':hashes, 'test_used_for_selection':False,
                 'model_sha256':{p.name:digest(p) for p in OUT.glob('*.joblib')}}
    write(OUT/'selection.json',selection)  # Freeze all candidates and operating points BEFORE opening test.
    print(f'Frozen development champion: {champion}, {specs[champion]}', flush=True)
    for name,p in development.items():
        save_predictions(OUT/f'{name}_development_predictions.jsonl',val,p)
    path = DATA/'test.jsonl'
    if digest(path) != protocol['partitions']['test']['sha256']:
        raise ValueError('Test partition changed')
    test = rows(path)
    if {r['split_block_id'] for r in test} & {r['split_block_id'] for r in train+val}:
        raise ValueError('Test source overlap')
    hashes[str(path.relative_to(ROOT))] = digest(path)
    test_paths = {'base':ARCHIVE/'roberta_s3407/test_predictions.jsonl',
                  'race':ARCHIVE/'race_s3407/test_predictions.jsonl',
                  'tfidf':ARCHIVE/'tfidf_lr_v1/test_predictions.jsonl',
                  'strong':ARCHIVE/'roberta_supcon_s3407/test_predictions.jsonl'}
    p_test = {name:probability_array(test,path) for name,path in test_paths.items()}
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in test_paths.values()})
    x_test = normalize(vectorizer.transform([r['model_text'] for r in test]))
    candidates = selected_candidates(p_test['base'],p_test['race'],p_test['tfidf'],
                                     models['lexical'].predict_proba(x_test),models['specialist'].predict_proba(x_test),specs)
    y_test = np.array([r['label_id'] for r in test])
    reference = {'roberta_ce_s3407':p_test['base'], 'roberta_supcon_s3407':p_test['strong']}
    uncertainty = bootstrap_deltas(test,y_test,candidates,reference)
    result = {'development_champion':champion, 'selection_sha256':digest(OUT/'selection.json'),
              'hart':{name:metrics(y_test,p,test) for name,p in {**reference,**candidates}.items()},
              'hart_creator':{name:evaluate_creator(y_test>=2,p[:,2:].sum(1),thresholds[name]['threshold'])
                              for name,p in {**{'roberta_ce_s3407':p_test['base']},**candidates}.items()},
              'hart_uncertainty':uncertainty,
              'uncertainty_scope':'2000 source-block paired bootstrap replicates of fixed predictions; no retraining/selection uncertainty, no multiplicity correction.'}
    for name,p in candidates.items():
        save_predictions(OUT/f'{name}_test_predictions.jsonl',test,p)
    # Existing conservative external set: fixed mapping and no external tuning.
    mix_path = ROOT/'resources/mixset_conservative_transfer_v1/test.jsonl'
    mix_protocol = json.loads((mix_path.parent/'protocol.json').read_text())
    if digest(mix_path) != mix_protocol['data_sha256']:
        raise ValueError('External subset changed')
    mix = rows(mix_path)
    from replay_hc3_cpu import tokenizer_local
    tokenizer = tokenizer_local()
    texts = []
    for r in mix:
        offsets = tokenizer(r['article'], max_length=512, truncation=True, return_offsets_mapping=True)['offset_mapping']
        texts.append(r['article'][:max(end for start,end in offsets)])
    xm = normalize(vectorizer.transform(texts))
    pm = probability_array(mix,ROOT/'transfer_roberta_v1/mixset_predictions.jsonl')
    ptm = probability_array(mix,ROOT/'transfer_tfidf_v1/mixset_predictions.jsonl')
    mix_specs = {k:v for k,v in specs.items() if k != 'race_blend'}
    mix_candidates = selected_candidates(pm,None,ptm,models['lexical'].predict_proba(xm),models['specialist'].predict_proba(xm),mix_specs)
    ym = np.array([LABELS.index(r['label']) for r in mix])
    result['mixset'] = {name:metrics(ym,p,mix) for name,p in {'roberta_ce_s3407':pm,**mix_candidates}.items()}
    result['mixset_creator'] = {name:evaluate_creator(ym>=2,p[:,2:].sum(1),thresholds[name]['threshold']) for name,p in {'roberta_ce_s3407':pm,**mix_candidates}.items()}
    result['mixset_uncertainty'] = bootstrap_deltas(mix,ym,mix_candidates,{'roberta_ce_s3407':pm})
    result['mixset_race_blend_status'] = 'not_evaluated: no frozen RACE external predictions; do not substitute other models'
    hashes[str(mix_path.relative_to(ROOT))] = digest(mix_path)
    for name,p in mix_candidates.items():
        save_predictions(OUT/f'{name}_mixset_predictions.jsonl',mix,p)
    result['elapsed_seconds'] = time.perf_counter()-start
    result['input_sha256'] = hashes
    write(OUT/'results.json', result)
    lines = ['# CPU 改进实验：开发集冻结后评测', '',
             f'开发集选出的候选：**{champion}**。不得按下面测试分数重新选择候选。', '',
             '| 方法 | 开发 Macro-F1 | HART 测试 Macro-F1 | MixSet 子集 Macro-F1 |',
             '|---|---:|---:|---:|',
             f'| RoBERTa CE s3407 | {selection["baseline_development"]["macro_f1"]:.2%} | {result["hart"]["roberta_ce_s3407"]["macro_f1"]:.2%} | {result["mixset"]["roberta_ce_s3407"]["macro_f1"]:.2%} |',
             f'| RoBERTa SupCon s3407（归档强对照） | 未取得预测 | {result["hart"]["roberta_supcon_s3407"]["macro_f1"]:.2%} | 未评估 |']
    for name in specs:
        external = f'{result["mixset"][name]["macro_f1"]:.2%}' if name in result['mixset'] else '未评估'
        lines.append(f'| {name} | {specs[name]["macro_f1"]:.2%} | {result["hart"][name]["macro_f1"]:.2%} | {external} |')
    lines += ['', '## 条件置信区间', '', '| 方法 | 相对 CE 差值/95%区间（百分点） | 相对 SupCon 差值/95%区间（百分点） |', '|---|---|---|']
    for name,u in uncertainty.items():
        cells = []
        for ref in reference:
            v=u[ref]; lo,hi=v['paired_95_percentile']
            cells.append(f'{100*v["delta"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]')
        lines.append(f'| {name} | '+ ' | '.join(cells)+' |')
    lines += ['', '## 协议与边界', '',
              '- 词/字符 TF-IDF 只在训练集拟合；正则、类别权重、融合系数及难分类别乘数只按开发集选择。',
              '- specialist 只重分配 AI 两类概率，保持 AI 创作者总概率，因此不会改善该二分类分数的排序。',
              '- 所有候选及低误报阈值先写入 selection.json，再读取当前测试和外部数据。',
              '- HART 测试及 MixSet 曾在前期分析中使用；这是后验探索，不是全新独立确认。',
              '- 区间仅反映固定拟合模型在来源组重采样下的不确定性，不能代替多种子训练或纠正模型选择偏差。',
              '- SupCon 对照来自同一种子3407的归档预测；不能把单次结果与三种子均值当作等条件比较。',
              '- RACE 融合需要两个神经模型及全文 parser，额外成本未计入本轮 CPU 小模型拟合时间。',
              '- MixSet 是359条保守映射子集；RACE融合未做此子集推理，其余方法未在此集调参。',
              '- 原项目代码及权重未修改；输出包含可重放模型、配置、逐样本预测和来源组区间。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]), flush=True)


if __name__ == '__main__':
    main()
