"""Chinese summaries, comparable same-GPU timings and a beginner reading guide."""
import json
from pathlib import Path
import numpy as np
from experiments.common import DATA_ROOT, LABELS, write_json
from experiments.research_report import INITIAL_COMMIT

BASELINE = DATA_ROOT.parent/'neural_runs/a4000_baseline_roberta_supcon_s3407'


def baseline_reference():
    row=json.loads((BASELINE/'result.json').read_text())
    events=[json.loads(l) for l in (BASELINE/'events.jsonl').read_text().splitlines()]
    epochs=[e for e in events if e['event']=='epoch_complete']
    return {'run_id':row['run_id'],'validation_macro_f1':row['validation']['macro_f1'],
            'test_macro_f1':row['test']['macro_f1'],
            'fit_seconds_median':float(np.median([e['fit_seconds'] for e in epochs])),
            'validation_seconds_median':float(np.median([e['validation_seconds'] for e in epochs])),
            'test_forward_seconds':row['efficiency']['test_forward_seconds_excluding_tokenization'],
            'test_samples':row['test']['samples'],'seed':3407}


def method(config):
    return (f"解冻{config['architecture']['unfreeze_bert_layers']}层；顶层LR={config['learning_rate']:g}，"
            f"层间衰减={config.get('layer_lr_decay',1)}；SupCon={config['use_supcon']}，"
            f"投影头={config.get('supcon_projection',False)}；内容/后处理辅助权重="
            f"{config.get('aux_content_weight',0)}/{config.get('aux_edit_weight',0)}；"
            f"真实batch={config['batch_size']}，累积={config.get('gradient_accumulation_steps',1)}")


def summarize_run(batch, run_id):
    batch=Path(batch); state=json.loads((batch/'pipeline.json').read_text())
    root=Path(state['run_root'])/run_id; row=json.loads((root/'result.json').read_text())
    base=baseline_reference(); write_json(batch/'baseline_reference.json',base)
    delta=100*(row['validation']['macro_f1']-base['validation_macro_f1'])
    fits=row['efficiency']['fit_seconds_per_epoch']; fit=float(np.median(fits))
    infer=row['efficiency']['validation_forward_seconds']
    result={'run_id':run_id,'method':method(row['config']),'seed':row['config']['seed'],
            'validation_delta_pp':delta,'training_speed_ratio':base['fit_seconds_median']/fit,
            'inference_speed_ratio':base['validation_seconds_median']/infer,
            'comparison':'Same A4000 baseline seed 3407, validation only; other seeds are not paired gains'}
    write_json(root/'comparison.json',result)
    label='涨' if delta>0 else '跌' if delta<0 else '持平'
    (root/'SUMMARY.md').write_text(f"# {run_id}\n\n- 方法：{result['method']}。\n"
        f"- Validation Macro-F1：{row['validation']['macro_f1']:.4%}；相对本机原版基线{label} {abs(delta):.3f} 个百分点。\n"
        f"- 训练速度：每epoch纯训练中位数 {fit:.2f} 秒，为原版的 {result['training_speed_ratio']:.2f} 倍吞吐。\n"
        f"- 推理速度：同1600条validation的FP32前向 {infer:.2f} 秒，为原版的 {result['inference_speed_ratio']:.2f} 倍吞吐。\n"
        f"- 速度比大于1表示更快；不含分词/加载，不能理解成完整部署延迟。\n"
        f"- 原版seed3407；本实验seed{row['config']['seed']}。非同种子比较仅是参考，不视为因果证据。\n"
        f"- 本次筛选未读取test，不能拿这个分数直接对比历史test分数。\n")


def summarize_batch(batch):
    batch=Path(batch); state=json.loads((batch/'pipeline.json').read_text())
    base=baseline_reference(); final=json.loads((batch/'final_evaluation/result.json').read_text())
    snapshot=json.loads((batch/'initial_git_results.json').read_text())['results']
    seeds=(3407,42,2026)
    old=np.array([snapshot[f'experiments/reports/roberta_supcon_s{s}/result.json']['test']['macro_f1'] for s in seeds])
    byseed={m['seed']:m for m in final['members']}; new=np.array([byseed[s]['test']['macro_f1'] for s in seeds])
    gain=float((new-old).mean()*100); sd=float(new.std(ddof=1)*100)
    selected=byseed[3407]['run_id']; root=Path(state['run_root'])/selected
    cfg=json.loads((root/'config.json').read_text())
    runtime=[m['seconds']/base['test_samples'] for m in final['members']]
    ratio=(base['test_forward_seconds']/base['test_samples'])/float(np.mean(runtime))
    train=json.loads((root/'comparison.json').read_text())
    brief=(f"# 本批次简报\n\n- 方法：{method(cfg)}。\n"
           f"- 最终三种子test Macro-F1：{new.mean():.4%} ± {sd:.3f}个百分点（样本SD）；"
           f"原仓库三种子为{old.mean():.4%}；{'上涨' if gain>0 else '下降' if gain<0 else '持平'} {abs(gain):.3f}个百分点。\n"
           f"- 相对本机原版seed3407的训练吞吐：{train['training_speed_ratio']:.2f}倍；"
           f"同3194条test的平均单模型FP32推理吞吐：{ratio:.2f}倍。\n"
           f"- 三模型集成需要三次前向，不能用单模型速度当作集成速度。\n"
           f"- 以validation选定配置后才运行test；未按test分数回头改配置。"
           f"跨历史环境差异仍存在，不能将全部差异归因于方法本身。\n"
           f"- 条件推送门槛：最终三种子test Macro-F1均值高于原仓库三种子均值即推送；小幅增益不等于统计显著。\n"
           f"- [详细数字报告](REPORT.md) · [中文指标与图解](详细报告与图解.md) · [过程及错误](PROCESS_LOG.md)。\n")
    (batch/'SUMMARY.md').write_text(brief)
    lines=['# 详细报告与可视化阅读指南','',brief,'## 四个标签是什么','',
           '`human_written`：人工原文；`human_ai_polished`：人工内容经AI润色；`ai_generated`：AI内容；`ai_humanized`：AI内容经人或工具改写。','',
           '## 分数该怎么看','',
           '- Accuracy：全部文本中分对的比例，容易被数量多的类别主导。',
           '- Precision：被模型判为某类的文本中，有多少真属于该类。Recall：真正属于某类的文本，有多少被找出来。',
           '- F1综合precision与recall；Macro-F1把四类F1同等平均，因此少数类ai_humanized也同样重要。本项目主要比较这个指标。',
           '- Macro-AUROC衡量把每类正例排在负例前面的能力；接近1通常更好，但不是“准确率”。',
           '- TPR@1%FPR：误报率不超过1%时找出的正例比例；这是本评估集上的描述性分析，不是经过独立校准的上线阈值。',
           '- 均值±SD：三个随机种子的平均和波动，不是置信区间。0.30个百分点是91.15%到91.45%，不是乘1.003。','',
           '## 每张图表达什么','',
           '### 1. 验证集配置比较','![验证比较](validation_comparison.png)','横向条越长表示该配置在validation的Macro-F1越高。所有筛选只用这张图对应的验证指标，不参考最终test。','',
           '### 2. 学习曲线','![学习曲线](learning_curves.png)','横轴是遍历训练数据的轮数epoch，纵轴是validation Macro-F1。升高说明模型在改善；训练更多却不升可能已收敛或过拟合。每条线是不同实验，不要只挑某轮尖峰去声称稳定提升。','',
           '### 3. 与原始仓库的测试结果比较','![原版比较](historical_test_comparison.png)',
           f'原版读取固定Git提交 `{INITIAL_COMMIT}`；candidate是本批次选中的配置。柱高是三个种子的平均，误差棒是样本SD。Macro-F1和AUROC是不同指标，不能直接把两种柱高当同一分数。','',
           '### 4. 逐类别F1比较','![类别比较](class_f1_comparison.png)','每组是一个类别。重点看ai_humanized是否改善，以及是否牺牲其他类别。整体涨分不等于每类都涨。','',
           '### 5. 混淆矩阵','![混淆矩阵](confusion_matrix.png)','行是真实类别，列是模型预测。对角线代表分对，其他格代表错分。比如ai_humanized这一行、ai_generated这一列的数字，就是“改写AI文本被误判为直接AI生成”的数量。不同类样本数不同，不宜只按颜色深浅比较难度。此图是预先固定的等权集成结果。','',
           '## 本批次详细测试指标','',
           '| Seed | Accuracy | Macro-F1 | AUROC | TPR@1%FPR |','| --- | ---: | ---: | ---: | ---: |']
    for seed in seeds:
        r=byseed[seed]['test']; lines.append(f"| {seed} | {r['accuracy']:.4%} | {r['macro_f1']:.4%} | {r['macro_auroc_ovr']:.4%} | {r['macro_tpr_at_1pct_fpr_roc']:.4%} |")
    lines+=['','| 类别 | Precision均值 | Recall均值 | F1均值 |','| --- | ---: | ---: | ---: |']
    for label in LABELS:
        v=[float(np.mean([byseed[s]['test']['per_class'][label][k] for s in seeds])) for k in ('precision','recall','f1')]
        lines.append(f'| {label} | {v[0]:.4%} | {v[1]:.4%} | {v[2]:.4%} |')
    lines+=['','## 适用边界','','使用同一固定test的持续改进，不能叫全新独立测试。速度基线来自相同A4000，历史分数基线来自原仓库，二者分工不同。完整配置、哈希、逐条预测见批次文件，失败记录见PROCESS_LOG.md和logs/。']
    (batch/'详细报告与图解.md').write_text('\n'.join(lines)+'\n')
    write_json(batch/'score_change.json',{'paired_gain_pp':gain,'new_mean_macro_f1':float(new.mean()),
               'old_mean_macro_f1':float(old.mean()),'paired_differences_pp':((new-old)*100).tolist(),
               'publish_eligible':bool(gain>0), 'meets_research_target':bool(gain>=.30 and np.sum(new<old)<2),'single_model_inference_speed_ratio':ratio})
