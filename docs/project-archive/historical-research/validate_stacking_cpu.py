"""Fixed-recipe controls and seed-transfer checks; no test-driven tuning."""
import json
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from stack_cpu import ROOT, DATA, ARCHIVE, rows, write, digest, probability_array, stack_features, model_for, metrics, bootstrap_deltas, save_predictions

OUT=ROOT/'cpu_stacking_controls_v1'
VARIANTS={'ce_only':['ce'], 'ce_tfidf':['ce','tfidf'], 'ce_newlex':['ce','newlex'],
          'ce_race':['ce','race'], 'all_original':['ce','race','tfidf']}


def main():
    OUT.mkdir()
    original_selection=json.loads((ROOT/'cpu_stacking_v1/selection.json').read_text())
    spec=original_selection['selected']
    val=rows(DATA/'val.jsonl')
    y=np.array([r['label_id'] for r in val])
    groups=np.array([r['split_block_id'] for r in val])
    val_paths={'ce':ROOT/'transfer_roberta_v1/validation_predictions.jsonl',
               'race':ROOT/'race_cpu_v1/validation_predictions.jsonl',
               'tfidf':ROOT/'transfer_tfidf_v1/validation_predictions.jsonl',
               'newlex':ROOT/'cpu_improvement_v1/lexical_only_development_predictions.jsonl'}
    pv={k:probability_array(val,p) for k,p in val_paths.items()}
    write(OUT/'protocol.json',{'fixed_spec':spec,'variants':VARIANTS,'seeds':[3407,42,2026],
        'meta_fit':'seed3407 development only, same fixed C for every variant',
        'other_seed_evaluation':'Apply UNCHANGED seed3407 meta-model to other seed base probabilities; not independently trained stacking replicates',
        'purpose':'Post-hoc controls after first stacking result; compare extra development-label fitting and base-model complementarity',
        'test_selection':False,'original_selection_sha256':digest(ROOT/'cpu_stacking_v1/selection.json'),
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [DATA/'val.jsonl',*val_paths.values()]},
        'script_sha256':digest(__file__)})
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=20261109).split(np.zeros((len(y),1)),y,groups))
    models,dev={},{}
    with threadpool_limits(limits=2):
        for name,inputs in VARIANTS.items():
            x=stack_features([pv[k] for k in inputs])
            oof=np.zeros((len(y),4))
            for train,held in folds:
                if set(groups[train]) & set(groups[held]):
                    raise ValueError('Crossfit source leakage')
                m=model_for(spec).fit(x[train],y[train])
                oof[held]=m.predict_proba(x[held])
            models[name]=model_for(spec).fit(x,y)
            dev[name]=metrics(y,oof,val)
            joblib.dump(models[name],OUT/f'{name}.joblib')
            save_predictions(OUT/f'{name}_oof_predictions.jsonl',val,oof)
            print(name,dev[name]['macro_f1'],flush=True)
    write(OUT/'frozen_models.json',{'development_oof':dev,'fixed_spec':spec,
        'model_sha256':{p.name:digest(p) for p in OUT.glob('*.joblib')},'test_used_for_selection':False})
    test=rows(DATA/'test.jsonl')
    if set(groups)&{r['split_block_id'] for r in test}:
        raise ValueError('Test source overlap')
    yt=np.array([r['label_id'] for r in test])
    shared={'tfidf':probability_array(test,ARCHIVE/'tfidf_lr_v1/test_predictions.jsonl'),
            'newlex':probability_array(test,ROOT/'cpu_improvement_v1/lexical_only_test_predictions.jsonl')}
    result={'development_oof':dev,'by_base_seed':{},'reload_max_difference':{}}
    primary_predictions={}
    for seed in (3407,42,2026):
        ps={**shared,'ce':probability_array(test,ARCHIVE/f'roberta_s{seed}/test_predictions.jsonl'),
                    'race':probability_array(test,ARCHIVE/f'race_s{seed}/test_predictions.jsonl')}
        strong=probability_array(test,ARCHIVE/f'roberta_supcon_s{seed}/test_predictions.jsonl')
        predictions={name:models[name].predict_proba(stack_features([ps[k] for k in inputs])) for name,inputs in VARIANTS.items()}
        references={'ce':ps['ce'],'supcon':strong}
        result['by_base_seed'][str(seed)]={'metrics':{name:metrics(yt,p,test) for name,p in {**references,**predictions}.items()},
            'paired_uncertainty':bootstrap_deltas(test,yt,predictions,references)}
        for name,p in predictions.items():
            save_predictions(OUT/f'{name}_s{seed}_test_predictions.jsonl',test,p)
            reloaded=joblib.load(OUT/f'{name}.joblib').predict_proba(stack_features([ps[k] for k in VARIANTS[name]]))
            delta=float(np.abs(p-reloaded).max())
            if delta>1e-12:
                raise ValueError('Reload failed')
            result['reload_max_difference'][f'{name}_s{seed}']=delta
        if seed==3407:
            primary_predictions=predictions
    original=probability_array(test,ROOT/'cpu_stacking_v1/test_predictions.jsonl')
    if not np.allclose(original,primary_predictions['all_original'],rtol=0,atol=1e-12):
        raise ValueError('Original stacking not reproduced')
    result['primary_controls_uncertainty']=bootstrap_deltas(test,yt,
        {k:v for k,v in primary_predictions.items() if k!='ce_only'}, {'ce_only_meta':primary_predictions['ce_only']})
    mix=rows(ROOT/'resources/mixset_conservative_transfer_v1/test.jsonl')
    ym=np.array([['human_written','human_ai_polished','ai_generated','ai_humanized'].index(r['label']) for r in mix])
    pm={'ce':probability_array(mix,ROOT/'transfer_roberta_v1/mixset_predictions.jsonl'),
        'tfidf':probability_array(mix,ROOT/'transfer_tfidf_v1/mixset_predictions.jsonl'),
        'newlex':probability_array(mix,ROOT/'cpu_improvement_v1/lexical_only_mixset_predictions.jsonl')}
    result['mixset']={'ce':metrics(ym,pm['ce'],mix)}
    for name,inputs in VARIANTS.items():
        if 'race' in inputs:
            continue
        p=models[name].predict_proba(stack_features([pm[k] for k in inputs]))
        result['mixset'][name]=metrics(ym,p,mix)
        save_predictions(OUT/f'{name}_mixset_predictions.jsonl',mix,p)
    write(OUT/'results.json',result)
    lines=['# Stacking同标签预算对照与固定元模型跨种子检查','',
           '所有变体使用相同开发标签、同一来源组4折、相同已冻结正则 C；没有根据测试分数重调参数。','',
           '| 方法 | 开发OOF F1 | 测试3407 | 底层种子42 | 底层种子2026 | MixSet子集 |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ['ce','supcon',*VARIANTS]:
        d=f'{dev[name]["macro_f1"]:.2%}' if name in dev else '—'
        scores=[f'{result["by_base_seed"][str(s)]["metrics"][name]["macro_f1"]:.2%}' for s in (3407,42,2026)]
        ext=f'{result["mixset"][name]["macro_f1"]:.2%}' if name in result['mixset'] else '未评'
        lines.append(f'| {name} | {d} | '+ ' | '.join(scores)+f' | {ext} |')
    lines+=['','- ce_only 是相同开发标签预算的决策层重拟合对照，可检查增益是否只来自额外拟合。',
            '- ce_tfidf/ce_newlex 不用RACE或parser；all_original需两个神经模型及parser。',
            '- 42/2026列只替换底层神经模型，保持3407上拟合的元模型不变；不能称为三次独立stacking训练。',
            '- newlex本身已用同一开发集选正则，neural checkpoint也用过开发标签；OOF仍有选择乐观偏差。',
            '- 这些是首轮结果后的固定配置验证，测试集已见，不声称全新独立确认。',
            '- 原始stacking完全重放一致，所有元模型保存重载的概率最大差为0。',
            '- MixSet只评无需RACE的变体，没有因此选择或修改配置。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]),flush=True)


if __name__=='__main__':
    main()
