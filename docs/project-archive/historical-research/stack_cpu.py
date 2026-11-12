"""Source-group cross-fitted stacking of frozen model predictions on CPU.

Meta-models train on development folds, NOT in-sample neural training predictions.
Underlying neural checkpoints were already selected using this development set.
"""
import time
import json
import warnings
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from improve_cpu import ROOT, DATA, ARCHIVE, rows, write, digest, probability_array, score, metrics, bootstrap_deltas, save_predictions
from evaluation_metrics import evaluate_creator, negative_only_threshold

OUT = ROOT/'cpu_stacking_v1'
SPECS = [{'kind':'logistic','C':c} for c in (.01,.1,1,10)] + [
    {'kind':'histogram','max_leaf_nodes':leaves,'l2_regularization':10.0} for leaves in (3,7)]


def stack_features(probabilities):
    blocks=[]
    for p in probabilities:
        z=np.log(np.clip(p,1e-12,1))
        blocks.extend([z-z.mean(1,keepdims=True),p])
    x=np.concatenate(blocks,axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Invalid stacking features')
    return x


def model_for(spec):
    if spec['kind']=='logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(C=spec['C'],max_iter=2000,random_state=3407))
    return HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=spec['max_leaf_nodes'],
                                          min_samples_leaf=30,l2_regularization=spec['l2_regularization'],
                                          early_stopping=False,random_state=3407)


def main():
    OUT.mkdir()
    started=time.perf_counter()
    protocol=json.loads((DATA/'protocol.json').read_text())
    path=DATA/'val.jsonl'
    if digest(path)!=protocol['partitions']['val']['sha256']:
        raise ValueError('Development partition changed')
    val=rows(path)
    y=np.array([r['label_id'] for r in val])
    groups=np.array([r['split_block_id'] for r in val])
    paths=[ROOT/'transfer_roberta_v1/validation_predictions.jsonl',ROOT/'race_cpu_v1/validation_predictions.jsonl',
           ROOT/'transfer_tfidf_v1/validation_predictions.jsonl']
    pv=[probability_array(val,p) for p in paths]
    x=stack_features(pv)
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=20261109).split(x,y,groups))
    fold_ids=np.full(len(val),-1)
    for i,(fit,held) in enumerate(folds):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError('Meta-fold source overlap')
        fold_ids[held]=i
    write(OUT/'protocol.json',{
        'date':'2026-11-09','candidate_grid':SPECS,'features':'centered log probabilities + probabilities from CE, RACE SupCon, TF-IDF, all seed3407',
        'fit':'4-fold source-group cross-fitting on existing development set; final meta-model fit on full development',
        'select':'OOF Macro-F1; first candidate breaks ties; select and save before opening test',
        'threshold':'1% empirical FPR from OOF development probabilities; not guaranteed calibrated for final all-dev fit',
        'folds':{r['item_id']:int(f) for r,f in zip(val,fold_ids)},
        'limitations':['Neural checkpoints already used all development labels for checkpoint selection: OOF is not independent confirmation.',
                       'Prior HART/HC3 test inspection makes this post-hoc exploratory research.',
                       'Inference requires two neural models plus TF-IDF and RST parser; not a cheaper single-model replacement.'],
        'script_sha256':digest(__file__),'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [path,*paths]}})
    candidates=[]
    best=None
    with threadpool_limits(limits=2):
        for spec in SPECS:
            p=np.zeros((len(val),4))
            for fit,held in folds:
                model=model_for(spec)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model.fit(x[fit],y[fit])
                if model.classes_.tolist()!=list(range(4)):
                    raise ValueError('Missing class')
                p[held]=model.predict_proba(x[held])
            result={'spec':spec,'oof_development':metrics(y,p,val)}
            candidates.append(result)
            print(f'{spec}: OOF dev {score(y,p):.4f}',flush=True)
            if best is None or score(y,p)>best[0]:
                best=(score(y,p),spec,p)
        _,spec,oof=best
        model=model_for(spec)
        model.fit(x,y)
    joblib.dump(model,OUT/'meta_model.joblib')
    threshold=negative_only_threshold(oof[y<2,2:].sum(1))
    selection={'selected':spec,'candidates':candidates,'baseline_development':metrics(y,pv[0],val),
               'selected_oof_development':metrics(y,oof,val),'threshold':threshold,
               'test_used_for_selection':False,'model_sha256':digest(OUT/'meta_model.joblib')}
    write(OUT/'selection.json',selection)
    save_predictions(OUT/'development_oof_predictions.jsonl',val,oof)
    if digest(DATA/'test.jsonl')!=protocol['partitions']['test']['sha256']:
        raise ValueError('Test partition changed')
    test=rows(DATA/'test.jsonl')
    if set(groups)&{r['split_block_id'] for r in test}:
        raise ValueError('Dev/test group overlap')
    runs=['roberta_s3407','race_s3407','tfidf_lr_v1']
    pt=[probability_array(test,ARCHIVE/run/'test_predictions.jsonl') for run in runs]
    p=model.predict_proba(stack_features(pt))
    strong=probability_array(test,ARCHIVE/'roberta_supcon_s3407/test_predictions.jsonl')
    yt=np.array([r['label_id'] for r in test])
    result={'selection_sha256':digest(OUT/'selection.json'),
            'hart':{'stacking':metrics(yt,p,test),'roberta_ce':metrics(yt,pt[0],test),'roberta_supcon':metrics(yt,strong,test)},
            'uncertainty':bootstrap_deltas(test,yt,{'stacking':p},{'roberta_ce':pt[0],'roberta_supcon':strong}),
            'hart_creator':evaluate_creator(yt>=2,p[:,2:].sum(1),threshold['threshold'])}
    save_predictions(OUT/'test_predictions.jsonl',test,p)
    hc3=rows(ROOT/'resources/experiments/hc3_aux_v1_1/test.jsonl')
    ph=[probability_array(hc3,ARCHIVE/run/'hc3_aux_v1_1_predictions.jsonl','four_class_scores') for run in runs]
    hp=model.predict_proba(stack_features(ph))
    hy=np.array([r['binary_label'] for r in hc3])
    result['hc3_creator']=evaluate_creator(hy,hp[:,2:].sum(1),threshold['threshold'])
    base_threshold=negative_only_threshold(pv[0][y<2,2:].sum(1))['threshold']
    result['hc3_base_creator']=evaluate_creator(hy,ph[0][:,2:].sum(1),base_threshold)
    with (OUT/'hc3_predictions.jsonl').open('x') as f:
        for row,probs in zip(hc3,hp):
            f.write(json.dumps({'item_id':row['item_id'],'true_binary_label':row['binary_label'],
                                'four_class_scores':dict(zip(['human_written','human_ai_polished','ai_generated','ai_humanized'],probs.tolist()))})+'\n')
    result['elapsed_seconds']=time.perf_counter()-started
    write(OUT/'results.json',result)
    lines=['# CPU 来源组交叉拟合 Stacking', '',
           f'开发集 OOF 选中：`{spec}`；OOF Macro-F1={best[0]:.2%}。', '',
           '| 方法 | HART测试 Macro-F1 |', '|---|---:|']
    lines.extend(f'| {name} | {m["macro_f1"]:.2%} |' for name,m in result['hart'].items())
    lines+=['','| 对照 | Stacking差值（百分点） | 来源组95%条件区间 |','|---|---:|---|']
    for name,u in result['uncertainty']['stacking'].items():
        lo,hi=u['paired_95_percentile']
        lines.append(f'| {name} | {100*u["delta"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines+=['','## HC3二分类辅助迁移（固定阈值0.5）','','| 方法 | Macro-F1 | AUROC |','|---|---:|---:|']
    for name,key in [('RoBERTa CE','hc3_base_creator'),('Stacking','hc3_creator')]:
        m=result[key]
        lines.append(f'| {name} | {m["at_threshold"]["fixed_0_5"]["macro_f1"]:.2%} | {m["auroc"]:.2%} |')
    lines+=['','## 限制','',
            '- 元模型在开发集4折来源组交叉拟合，以OOF指标选配置，再用全部开发集拟合最终模型；测试不用于选配置。',
            '- 底层神经模型选checkpoint已看过全部开发标签，因此OOF不是完全独立验证；这只是工程候选筛选。',
            '- 仍是单种子、已见测试集的后验实验；来源组区间不包含重训和选模不确定性，也未作多重比较校正。',
            '- 需要RoBERTa、RACE、TF-IDF及parser；改善若成立也不能宣称单模型更轻或提出新篇章架构。',
            '- HC3只有二分类真值，不报告四分类准确率；MixSet缺RACE预测，本轮未评此融合。',
            '- 低误报阈值来自OOF概率，最终全开发集拟合可能改变校准；测试实际误报/召回已另行记录。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:11]),flush=True)


if __name__=='__main__':
    main()
