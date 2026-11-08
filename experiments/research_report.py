"""Produce artifacts against immutable results from the initial Git commit."""
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common import LABELS, file_hash, write_json
from experiments.neural import PROJECT_ROOT

INITIAL_COMMIT = '877f12bb5b83a4825780f654309ecd9959f74f5a'


def report(batch):
    batch = Path(batch)
    snapshot = {}
    historical = {}
    for kind in ('roberta', 'roberta_supcon', 'race_no_supcon', 'race'):
        historical[kind] = []
        for seed in (3407, 42, 2026):
            path = f'experiments/reports/{kind}_s{seed}/result.json'
            raw = subprocess.check_output(['git', 'show', f'{INITIAL_COMMIT}:{path}'], cwd=PROJECT_ROOT)
            row = json.loads(raw); snapshot[path] = row
            historical[kind].append(row['test'])
    write_json(batch / 'initial_git_results.json', {'commit': INITIAL_COMMIT, 'results': snapshot})
    states = json.loads((batch / 'pipeline.json').read_text())
    runs = []
    for run_id in states.get('completed_runs', []):
        path = Path(states['run_root']) / run_id
        row = json.loads((path / 'result.json').read_text())
        if file_hash(path / 'best.pt') != row['checkpoint_sha256']:
            raise ValueError('Checkpoint mismatch while reporting')
        if not (path / 'last.pt').is_file():
            raise ValueError('Missing recovery checkpoint')
        runs.append(row)
    if not runs:
        raise ValueError('No completed validation runs')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh([r['run_id'] for r in runs], [100*r['validation']['macro_f1'] for r in runs])
    ax.set_xlabel('Validation Macro-F1 (%)'); ax.set_xlim(0,100); fig.tight_layout()
    fig.savefig(batch / 'validation_comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    for row in runs:
        events = [json.loads(s) for s in (Path(states['run_root']) / row['run_id'] / 'events.jsonl').read_text().splitlines()]
        epochs = [e for e in events if e['event']=='epoch_complete']
        ax.plot([e['epoch'] for e in epochs], [100*e['validation']['macro_f1'] for e in epochs], label=row['run_id'])
    ax.set(xlabel='Epoch',ylabel='Validation Macro-F1 (%)'); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(batch / 'learning_curves.png', dpi=160); plt.close(fig)
    lines = ['# A4000 research batch report', '', f'Historical results pinned to `{INITIAL_COMMIT}`.',
             'Selection uses validation only. Repeated fixed test, not an independent new test.', '',
             '| Run | Validation F1 | ai_humanized F1 | Train+val seconds | Peak MiB |', '| --- | ---: | ---: | ---: | ---: |']
    for r in runs:
        lines.append(f"| {r['run_id']} | {r['validation']['macro_f1']:.4%} | {r['validation']['per_class']['ai_humanized']['f1']:.4%} | {r['efficiency']['training_validation_checkpoint_seconds']:.1f} | {r['efficiency']['peak_cuda_allocated_mib']:.1f} |")
    final_path = batch / 'final_evaluation/result.json'
    if final_path.exists():
        final = json.loads(final_path.read_text())
        groups = dict(historical, improved=[x['test'] for x in final['members']])
        names = list(groups)
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        for ax, metric in zip(axes, ('macro_f1','macro_auroc_ovr')):
            means = [100*np.mean([r[metric] for r in groups[n]]) for n in names]
            sd = [100*np.std([r[metric] for r in groups[n]],ddof=1) if len(groups[n])>1 else 0 for n in names]
            ax.bar(names, means, yerr=sd, capsize=4)
            ax.set_ylabel(metric+' (%)'); ax.tick_params(axis='x', rotation=25)
        fig.tight_layout(); fig.savefig(batch/'historical_test_comparison.png',dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10,5)); x=np.arange(4); width=.36
        for i, name in enumerate(('roberta_supcon','improved')):
            mean=[100*np.mean([r['per_class'][c]['f1'] for r in groups[name]]) for c in LABELS]
            ax.bar(x+(i-.5)*width,mean,width,label=name)
        ax.set_xticks(x,LABELS); ax.set_ylabel('Test class F1 (%)'); ax.legend(); fig.tight_layout()
        fig.savefig(batch/'class_f1_comparison.png',dpi=160); plt.close(fig)
        fig, ax=plt.subplots(figsize=(6,5)); cm=np.asarray(final['ensemble']['confusion_matrix'])
        im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
        ax.set_xticks(range(4),LABELS,rotation=30,ha='right'); ax.set_yticks(range(4),LABELS)
        ax.set(xlabel='Predicted',ylabel='True',title='Fixed equal-weight ensemble')
        fig.tight_layout(); fig.savefig(batch/'confusion_matrix.png',dpi=160); plt.close(fig)
        lines += ['', '| Test method | Macro-F1 mean ± sample SD | Macro-AUROC |', '| --- | ---: | ---: |']
        for n, rows in groups.items():
            mean=np.mean([r['macro_f1'] for r in rows]); sd=np.std([r['macro_f1'] for r in rows],ddof=1) if len(rows)>1 else 0
            lines.append(f"| {n} | {mean:.4%} ± {sd:.4%} | {np.mean([r['macro_auroc_ovr'] for r in rows]):.4%} |")
        old=np.array([x['macro_f1'] for x in historical['roberta_supcon']])
        new=np.array([x['test']['macro_f1'] for x in sorted(final['members'],key=lambda x:(3407,42,2026).index(x['seed']))])
        if len(new)==3:
            lines += ['', f'Paired seed F1 differences (percentage points): {((new-old)*100).tolist()}.',
                      'Hardware/runtime also differ from historical runs; these differences do not isolate architecture effects.']
        lines += ['', f"Equal-weight ensemble test Macro-F1: {final['ensemble']['macro_f1']:.4%}."]
    else:
        lines += ['', 'Final test not completed. Validation metrics must not be compared numerically to historical test as a gain.']
    lines += ['', '## Figures', '', '![Validation](validation_comparison.png)', '![Learning curves](learning_curves.png)']
    if final_path.exists():
        lines += ['![Historical test](historical_test_comparison.png)', '![Class F1](class_f1_comparison.png)', '![Confusion](confusion_matrix.png)']
    (batch/'REPORT.md').write_text('\n'.join(lines)+'\n')
    expected=['REPORT.md','initial_git_results.json','validation_comparison.png','learning_curves.png']
    if final_path.exists(): expected += ['historical_test_comparison.png','class_f1_comparison.png','confusion_matrix.png','final_evaluation/result.json']
    write_json(batch/'artifacts.json', {name: file_hash(batch/name) for name in expected})
    return bool(final_path.exists())
