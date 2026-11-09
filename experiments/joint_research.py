"""Fixed factorial experiments on the six-layer recipe; validation-only selection."""
import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from experiments.common import DATA_ROOT, file_hash, write_json
from experiments.neural import PROJECT_ROOT
from experiments.research import validate

SEEDS = (3407, 42, 2026)
AXES = ('projection', 'weak_supcon', 'auxiliary')


def anchor_path(config):
    path = Path(config['joint_plan']['anchor_run'])
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def base_config(config):
    base = copy.deepcopy(config)
    base.pop('joint_plan')
    validate(base)
    if base['seed'] != 3407 or base['architecture']['unfreeze_bert_layers'] != 6:
        raise ValueError('Joint experiments require the six-layer seed3407 anchor')
    if not base['use_supcon'] or base.get('supcon_projection') or base.get('supcon_weight') != 1.0:
        raise ValueError('Unexpected anchor contrastive recipe')
    if base.get('aux_content_weight', 0) or base.get('aux_edit_weight', 0):
        raise ValueError('The anchor must not already have auxiliary heads')
    return base


def candidates(config):
    base = base_config(config)
    spec = config['joint_plan']; axes = spec['factorial_axes']
    weights = axes['supcon_weight']; auxiliary = axes['aux_content_and_edit_weights']
    if (axes['projection'] != [False, True] or len(weights) != 2 or weights[0] != 1.0
            or not 0 < weights[1] < 1 or len(auxiliary) != 2 or auxiliary[0] != 0 or not 0 < auxiliary[1] <= 1):
        raise ValueError('Invalid fixed factorial axes')
    if (spec['screening_seed'] != 3407 or spec['confirmation_seeds'] != [42, 2026]
            or spec['max_new_training_runs'] != 13):
        raise ValueError('Unexpected seeds or experiment budget')
    for key in ('min_f1_gain', 'auroc_slack', 'minority_f1_slack'):
        if not 0 <= spec[key] < 1 or not np.isfinite(spec[key]):
            raise ValueError('Invalid validation threshold')
    if not spec['min_f1_gain'] > 0:
        raise ValueError('Require a strictly positive minimum F1 gain')
    rows = []
    for p, r, h in itertools.product((0, 1), repeat=3):
        c = copy.deepcopy(base)
        c.update(supcon_projection=bool(p), supcon_weight=weights[r],
                 aux_content_weight=auxiliary[h], aux_edit_weight=auxiliary[h])
        rows.append({'tag': f'f_p{p}_r{r}_h{h}', 'bits': [p, r, h], 'config': c})
    flat = copy.deepcopy(base); flat['layer_lr_decay'] = 1.0
    ce = copy.deepcopy(base); ce.update(use_supcon=False, supcon_weight=0.0)
    rows += [{'tag': 'control_flat_decay', 'bits': None, 'config': flat},
             {'tag': 'control_ce_only', 'bits': None, 'config': ce}]
    for row in rows:
        validate(row['config'])
    return rows


def verify_anchor(config):
    root = anchor_path(config)
    row = json.loads((root/'result.json').read_text())
    saved = json.loads((root/'config.json').read_text())
    if saved != base_config(config) or row['config'] != saved:
        raise ValueError('Anchor configuration differs from the declared baseline')
    if row['status'] != 'completed' or row.get('test_partition_loaded') is not False:
        raise ValueError('Expected a completed validation-only training artifact')
    if row['data_protocol_sha256'] != file_hash(DATA_ROOT/'protocol.json'):
        raise ValueError('Anchor protocol differs')
    if row['checkpoint_sha256'] != file_hash(root/'best.pt') or not (root/'last.pt').is_file():
        raise ValueError('Anchor checkpoint missing or changed')
    for name, digest in row['code_sha256'].items():
        if file_hash(root/'source'/name) != digest:
            raise ValueError('Anchor training source changed')
    return row


def preflight_plan(config, output):
    candidates(config)
    anchor = verify_anchor(config)
    folder = Path(output).parent
    while not folder.exists(): folder = folder.parent
    if shutil.disk_usage(folder).free < storage_required(config):
        raise RuntimeError('Insufficient disk for the complete joint plan plus 12 GiB reserve')
    return anchor


def storage_required(config):
    root = anchor_path(config)
    return 26*((root/'best.pt').stat().st_size + (root/'last.pt').stat().st_size) + 12*1024**3


def snapshot_sources(batch):
    """Preserve actual source, including authorized edits not yet committed."""
    batch = Path(batch)
    hashes = {}
    for p in sorted((PROJECT_ROOT/'experiments').rglob('*')):
        relative = p.relative_to(PROJECT_ROOT)
        if not p.is_file() or any(x in relative.parts for x in ('reports', 'research_results', '__pycache__')):
            continue
        if p.suffix not in ('.py', '.sh', '.json', '.md', '.txt'):
            continue
        target = batch/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(relative)] = file_hash(target)
    (batch/'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=PROJECT_ROOT))
    write_json(batch/'source_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': hashes,
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        'patch_sha256': file_hash(batch/'source.patch')})


def eligible(row, baseline, spec):
    a, b = row['validation'], baseline['validation']
    return (a['macro_f1'] >= b['macro_f1'] + spec['min_f1_gain']
            and a['macro_auroc_ovr'] >= b['macro_auroc_ovr'] - spec['auroc_slack']
            and a['per_class']['ai_humanized']['f1'] >= b['per_class']['ai_humanized']['f1'] - spec['minority_f1_slack'])


def choose(rows, baseline, spec):
    acceptable = [r for r in rows if eligible(r, baseline, spec)]
    return max(acceptable, key=lambda r: (r['validation']['macro_f1'], r['validation']['macro_auroc_ovr'])) if acceptable else baseline


def paired_validation(baseline, candidate, spec):
    old = {r['config']['seed']: r['validation'] for r in baseline}
    new = {r['config']['seed']: r['validation'] for r in candidate}
    if set(old) != set(SEEDS) or set(new) != set(SEEDS):
        raise ValueError('Both recipes need all three matched seeds')
    diffs = {metric: [new[s][metric] - old[s][metric] for s in SEEDS]
             for metric in ('macro_f1', 'macro_auroc_ovr')}
    minority = [new[s]['per_class']['ai_humanized']['f1'] - old[s]['per_class']['ai_humanized']['f1'] for s in SEEDS]
    passed = (np.mean(diffs['macro_f1']) >= spec['min_f1_gain']
              and sum(d > 0 for d in diffs['macro_f1']) >= 2
              and np.mean(diffs['macro_auroc_ovr']) >= -spec['auroc_slack']
              and np.mean(minority) >= -spec['minority_f1_slack'])
    return {'passed': bool(passed), 'seeds': list(SEEDS),
            'paired_differences_pp': {k: [100*x for x in v] for k, v in diffs.items()},
            'minority_f1_differences_pp': [100*x for x in minority], 'selection_uses_test': False}


def factorial_effects(cells):
    if set(cells) != set(itertools.product((0, 1), repeat=3)):
        raise ValueError('A complete 2x2x2 factorial is required')
    effects = {}
    for metric in ('macro_f1', 'macro_auroc_ovr'):
        values = {k: v['validation'][metric] for k, v in cells.items()}
        main = {axis: 100*(np.mean([v for k, v in values.items() if k[i]]) -
                                 np.mean([v for k, v in values.items() if not k[i]])) for i, axis in enumerate(AXES)}
        pairs = {}
        for i, j in itertools.combinations(range(3), 2):
            pairs[f'{AXES[i]}__{AXES[j]}'] = float(100*sum(v*(1 if k[i] == k[j] else -1) for k,v in values.items())/2)
        triple = float(100*sum(v*((-1)**(3-sum(k))) for k,v in values.items()))
        joint_gain = values[1,1,1] - values[0,0,0]
        single_sum = sum(values[k]-values[0,0,0] for k in ((1,0,0),(0,1,0),(0,0,1)))
        effects[metric] = {'mean_main_effect_pp': {k: float(v) for k,v in main.items()},
                           'mean_pair_interaction_pp': pairs, 'triple_interaction_pp': triple,
                           'joint_gain_over_anchor_pp': 100*joint_gain,
                           'sum_single_gains_pp': 100*single_sum,
                           'joint_gain_minus_sum_single_gains_pp': 100*(joint_gain-single_sum)}
    return {'axes': AXES, 'effects': effects,
            'cells': [{'bits': list(k), 'run_id': r['run_id'],
                       'macro_f1': r['validation']['macro_f1'], 'macro_auroc_ovr': r['validation']['macro_auroc_ovr']}
                      for k,r in sorted(cells.items())],
            'scope': 'Validation, seed3407; descriptive contrasts, not significance tests'}


def summarize_joint_run(pipeline, run_id, anchor):
    root = Path(pipeline.state['run_root'])/run_id
    row = json.loads((root/'result.json').read_text())
    delta = {key: 100*(row['validation'][key]-anchor['validation'][key])
             for key in ('macro_f1', 'macro_auroc_ovr')}
    write_json(root/'anchor_comparison.json', {'reference_run': anchor['run_id'],
        'validation_difference_pp': delta, 'same_seed': row['config']['seed'] == 3407})
    with (root/'SUMMARY.md').open('a') as stream:
        stream.write(f"\n相对当前最优6层seed3407：validation Macro-F1 {delta['macro_f1']:+.4f}、"
                     f"Macro-AUROC {delta['macro_auroc_ovr']:+.4f} 个百分点。其他种子仅是对照参考，正式配对比较在三种子确认阶段进行。\n")


def freeze_reference(pipeline, members):
    folder = pipeline.batch/'baseline_confirmation'; folder.mkdir()
    source_names = ('final_evaluate.py', 'train_neural.py', 'research.py', 'neural.py', 'common.py', 'paths.py')
    manifest = {'data_protocol_sha256': file_hash(DATA_ROOT/'protocol.json'), 'seeds': list(SEEDS),
        'selection_rule': 'Predeclared six-layer anchor, three matched seeds; no test selection',
        'evaluation_source_sha256': {str(PROJECT_ROOT/'experiments'/n): file_hash(PROJECT_ROOT/'experiments'/n) for n in source_names},
        'members': []}
    for row in members:
        root = Path(pipeline.state['run_root'])/row['run_id']
        manifest['members'].append({'run_id': row['run_id'], 'weight': 1/3,
            'checkpoint': str(root/'best.pt'), 'checkpoint_sha256': file_hash(root/'best.pt'),
            'config': str(root/'config.json'), 'config_sha256': file_hash(root/'config.json'),
            'source_sha256': {str(root/'source'/p): h for p,h in row['code_sha256'].items()}})
    target = folder/'final_manifest.json'
    with target.open('x') as stream: json.dump(manifest, stream, indent=2)
    target.with_suffix('.sha256').write_text(file_hash(target)+'\n')
    return target


def execute_joint(pipeline, config):
    anchor = verify_anchor(config)
    base = base_config(config); spec = config['joint_plan']
    old_root = anchor_path(config)
    # Reuse immutable artifacts without duplicating or modifying the old run.
    root = Path(pipeline.state['run_root']); root.mkdir(parents=True, exist_ok=True)
    (root/anchor['run_id']).symlink_to(old_root, target_is_directory=True)
    pipeline.state.update(reused_runs=[anchor['run_id']],
        selection_rule='Fixed 2x2x2 + two mechanism controls; validation F1/AUROC/minority gates and three matched seeds; no test selection')
    pipeline.state['completed_runs'].append(anchor['run_id']); pipeline.save()
    pipeline.operation('reuse_anchor', run_id=anchor['run_id'], checkpoint_sha256=anchor['checkpoint_sha256'])
    write_json(pipeline.batch/'JOINT_PLAN.json', {'specification': spec, 'candidates': candidates(config), 'max_new_training_runs': 13})
    baseline = [anchor]
    for seed in SEEDS[1:]:
        c = copy.deepcopy(base); c['seed'] = seed
        baseline.append(pipeline.train('anchor_confirmation', c))
    cells = {(0,0,0): anchor}; screened = []
    for item in candidates(config):
        if item['bits'] == [0,0,0]: continue
        result = pipeline.train(item['tag'], item['config']); screened.append(result)
        if item['bits'] is not None: cells[tuple(item['bits'])] = result
    write_json(pipeline.batch/'factorial_effects.json', factorial_effects(cells))
    chosen = choose(screened, anchor, spec)
    members = baseline
    confirmation = {'passed': False, 'reason': 'No screened configuration passed the prespecified validation gates', 'selection_uses_test': False}
    if chosen is not anchor:
        selected = [chosen]
        for seed in SEEDS[1:]:
            c = copy.deepcopy(chosen['config']); c['seed'] = seed
            selected.append(pipeline.train('joint_selected', c))
        confirmation = paired_validation(baseline, selected, spec)
        if confirmation['passed']: members = selected
    confirmation.update(screen_selected_run=chosen['run_id'], final_selected_run=members[0]['run_id'],
                        used_anchor_fallback=members is baseline, thresholds=spec)
    write_json(pipeline.batch/'joint_validation.json', confirmation)
    pipeline.state.update(joint_anchor_selected=members is baseline, joint_auroc_slack=spec['auroc_slack'])
    pipeline.save()
    pipeline.finalize(members, baseline_members=baseline if members is not baseline else None)


def finalize_comparison(batch):
    batch = Path(batch)
    candidate = json.loads((batch/'final_evaluation/result.json').read_text())
    state = json.loads((batch/'pipeline.json').read_text())
    baseline_file = batch/'baseline_confirmation/final_evaluation/result.json'
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else candidate
    old = {r['seed']: r['test'] for r in baseline['members']}
    new = {r['seed']: r['test'] for r in candidate['members']}
    gains = {m: [100*(new[s][m]-old[s][m]) for s in SEEDS] for m in ('macro_f1','macro_auroc_ovr')}
    gain = float(np.mean(gains['macro_f1'])); auc_gain = float(np.mean(gains['macro_auroc_ovr']))
    comparison = {'reference': 'Current best six-layer recipe, matched seeds', 'seeds': SEEDS,
        'baseline_f1_mean': float(np.mean([old[s]['macro_f1'] for s in SEEDS])),
        'candidate_f1_mean': float(np.mean([new[s]['macro_f1'] for s in SEEDS])),
        'paired_differences_pp': gains, 'f1_gain_pp': gain, 'auroc_gain_pp': auc_gain,
        'anchor_fallback': state['joint_anchor_selected'], 'scope': 'Repeated existing fixed test; no independent new test'}
    write_json(batch/'joint_comparison.json', comparison)
    scores = json.loads((batch/'score_change.json').read_text())
    scores.update(gain_vs_six_layer_anchor_pp=gain, auroc_gain_vs_six_layer_anchor_pp=auc_gain,
        publish_eligible=bool(scores['publish_eligible'] and not state['joint_anchor_selected'] and gain>0
                              and auc_gain >= -100*state['joint_auroc_slack']))
    write_json(batch/'score_change.json', scores)
    note = (f"\n\n本轮以现有6层方案为主要参照：三种子平均Macro-F1变化 {gain:+.4f} 个百分点，"
            f"Macro-AUROC变化 {auc_gain:+.4f} 个百分点。是否回退原方案：{state['joint_anchor_selected']}。"
            "自动推送还要求超过6层参照且AUROC下降不超过0.1个百分点；不是只超过最初仓库就推送。\n")
    for name in ('SUMMARY.md','详细报告与图解.md'):
        with (batch/name).open('a') as stream: stream.write(note)
    lines = ['# 联合增强结果', note,
             '逐种子差值见 `joint_comparison.json`；验证筛选见 `joint_validation.json`。', '',
             '以下是seed3407验证集上的描述性对比，不是显著性检验。两两交互对第三因素两个水平取平均；三项联合超额增益则直接比较三项全开与三个单项增益之和。', '',
             '| 效应 | Macro-F1（百分点） | Macro-AUROC（百分点） |', '| --- | ---: | ---: |']
    effects_file = batch/'factorial_effects.json'
    if effects_file.exists():
        effects = json.loads(effects_file.read_text())['effects']
        for group in ('mean_main_effect_pp', 'mean_pair_interaction_pp'):
            for key, value in effects['macro_f1'][group].items():
                lines.append(f"| {key} | {value:+.4f} | {effects['macro_auroc_ovr'][group][key]:+.4f} |")
        lines.append(f"| 三项交互 | {effects['macro_f1']['triple_interaction_pp']:+.4f} | {effects['macro_auroc_ovr']['triple_interaction_pp']:+.4f} |")
        lines.append(f"| 三项联合超额增益 | {effects['macro_f1']['joint_gain_minus_sum_single_gains_pp']:+.4f} | {effects['macro_auroc_ovr']['joint_gain_minus_sum_single_gains_pp']:+.4f} |")
    (batch/'JOINT_REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')


def prepare(config_path, output):
    config = json.loads(config_path.read_text()); anchor = verify_anchor(config)
    output.mkdir(parents=True, exist_ok=False)
    rows = candidates(config)
    for row in rows: write_json(output/'configs'/f"{row['tag']}.json", row['config'])
    for seed in SEEDS[1:]:
        c = base_config(config); c['seed'] = seed
        write_json(output/'configs'/f'anchor_confirmation_s{seed}.json', c)
    write_json(output/'plan.json', {'config': str(config_path.resolve()), 'anchor_run': anchor['run_id'],
        'anchor_checkpoint_sha256': anchor['checkpoint_sha256'], 'factorial_cells': rows,
        'new_screening_runs': 9, 'baseline_confirmation_runs': 2, 'max_candidate_confirmation_runs': 2,
        'max_new_training_runs': 13, 'selection_uses_test': False, 'specification': config['joint_plan']})
    estimated = storage_required(config)
    write_json(output/'preparation.json', {'status': 'prepared', 'required_free_bytes_upper_estimate': estimated,
        'available_free_bytes': shutil.disk_usage(output).free, 'storage_sufficient': shutil.disk_usage(output).free >= estimated})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); prepare(args.config, args.output)
    print(f'Prepared fixed factorial plan: {args.output}')
