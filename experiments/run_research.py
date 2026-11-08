"""Bounded validation-only ablations, frozen final test, reports, then hibernate."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from experiments.common import DATA_ROOT, file_hash, write_json
from experiments.neural import RUN_ROOT, PROJECT_ROOT
from experiments.hyperstack import Hyperstack
from experiments.hibernate_watchdog import atomic, alive
from experiments.research_report import report, INITIAL_COMMIT


class Pipeline:
    def __init__(self, args):
        self.args=args; self.batch=args.output.resolve()
        if self.batch.exists():
            raise FileExistsError('Choose a new batch output directory')
        # Never start any experiments without authenticated self-hibernation preflight.
        preflight=Hyperstack().preflight()
        self.batch.mkdir(parents=True)
        write_json(self.batch/'hyperstack-preflight.json',preflight)
        self.state={'phase':'setup','run_root':str(RUN_ROOT),'completed_runs':[], 'created':time.time(),
                    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
                    'initial_commit':INITIAL_COMMIT, 'ready_to_hibernate':False}
        self.save()
        self.watchlog=(self.batch/'watchdog.log').open('a')
        self.watch=subprocess.Popen([sys.executable,'-u','-m','experiments.hibernate_watchdog',
            '--batch',str(self.batch),'--grace-seconds',str(args.grace_seconds),'--max-hours',str(args.max_hours),'--progress',str(args.progress.resolve())],
            cwd=PROJECT_ROOT,stdout=self.watchlog,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(60):
            if (self.batch/'watchdog.json').exists():
                break
            if self.watch.poll() is not None:
                raise RuntimeError('Watchdog failed to arm; no training started')
            time.sleep(.5)
        else:
            raise RuntimeError('Watchdog arming timed out; no training started')
        self.progress('休眠 watchdog 已装好；训练结束/失败后最多 10 分钟进行收尾，然后调用 Hyperstack API。')

    def save(self):
        atomic(self.batch/'pipeline.json',self.state)

    def progress(self, message):
        now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        with self.args.progress.open('a') as stream:
            stream.write(f'\n### 研究队列 CKPT — {now}\n\n- {message}\n- 批次目录：`{self.batch}`。\n')

    def command(self, command, log, phase, timeout=7200):
        if not alive(self.watch.pid):
            raise RuntimeError('Watchdog not alive; refusing more compute')
        with (self.batch/log).open('w') as stream:
            process=subprocess.Popen(command,cwd=PROJECT_ROOT,stdout=stream,stderr=subprocess.STDOUT)
            self.state.update(phase=phase,training_pid=process.pid if phase=='training' else None,
                              training_deadline=time.time()+timeout if phase=='training' else None)
            self.save()
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise RuntimeError(f'{phase} exceeded time limit')
            except BaseException:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
            finally:
                if phase == 'training':
                    self.state['last_training_exit'] = time.time()
                self.state.update(phase='postprocessing',training_pid=None,training_deadline=None)
                self.save()
        if code:
            raise RuntimeError(f'{phase} failed with exit {code}; see {log}')

    def train(self, tag, config):
        run_id=f'{self.args.name}_{tag}_s{config["seed"]}'
        path=self.batch/'configs'/f'{run_id}.json'; write_json(path,config)
        if shutil.disk_usage(self.batch).free < 12 * 1024**3:
            raise RuntimeError('Less than 12 GiB free; preserving existing experiments')
        self.state['current_run']=run_id; self.save()
        self.progress(f'开始 `{run_id}`：解冻 {config["architecture"]["unfreeze_bert_layers"]} 层，batch {config["batch_size"]}，只读取 train/validation。')
        cmd=[sys.executable,'-u','-m','experiments.train_neural','--config',str(path),'--run-name',run_id]
        self.command(cmd+['--smoke'],f'{run_id}_smoke.log','training')
        smoke=json.loads((RUN_ROOT/f'smoke_{run_id}'/'smoke.json').read_text())
        if smoke['status']!='passed' or smoke['test_partition_loaded']:
            raise RuntimeError('Invalid smoke result')
        self.command(cmd,f'{run_id}.log','training')
        result=json.loads((RUN_ROOT/run_id/'result.json').read_text())
        if result['test_partition_loaded'] or file_hash(RUN_ROOT/run_id/'best.pt')!=result['checkpoint_sha256']:
            raise RuntimeError('Screening leaked test or checkpoint hash mismatch')
        self.state['completed_runs'].append(run_id); self.save()
        self.progress(f'完成 `{run_id}`，validation Macro-F1={result["validation"]["macro_f1"]:.4%}；best/last checkpoint 已保存。')
        return result

    def execute(self):
        base=json.loads(self.args.config.read_text())
        self.progress('队列已启动；每个运行完成后写入验证指标，训练期间独立监控约每 10 分钟记录。')
        if base["seed"] != 3407 or not base["use_supcon"]:
            raise ValueError("Research queue requires seed 3407 and SupCon enabled")
        if self.args.single_config:
            best = self.train('single_config', base)
        else:
            results=[]
            for depth in (1,2,4,6,12):
                c=copy.deepcopy(base); c['architecture']['unfreeze_bert_layers']=depth
                c['batch_size']=8 if depth==12 else 16; c['gradient_checkpointing']=depth==12
                tag=f'u{depth}_b{c["batch_size"]}'
                try:
                    results.append(self.train(tag,c))
                except RuntimeError:
                    status=RUN_ROOT/self.state['current_run']/'status.json'
                    smoke_status=RUN_ROOT/f'smoke_{self.state["current_run"]}'/'status.json'
                    failed=json.loads((status if status.exists() else smoke_status).read_text()) if status.exists() or smoke_status.exists() else {}
                    if depth==6 and failed.get('oom'):
                        c['batch_size']=8
                        results.append(self.train('u6_b8_after_oom',c))
                    else: raise
            best=max(results,key=lambda x:x['validation']['macro_f1'])
            c=copy.deepcopy(best['config']); c['supcon_projection']=True
            projection=self.train('projection',c)
            # Projection is retained only when validation improves; no test used.
            if projection['validation']['macro_f1']>best['validation']['macro_f1']: best=projection
            for tag,alpha,beta in (('aux_content',.1,0),('aux_edit',0,.1),('aux_both',.1,.1)):
                c=copy.deepcopy(best['config']); c.update(aux_content_weight=alpha,aux_edit_weight=beta)
                # Every auxiliary ablation has the same fixed selected backbone/projection.
                result=self.train(tag,c); results.append(result)
            best=max([best]+results,key=lambda x:x['validation']['macro_f1'])
        members=[best]
        for seed in (42,2026):
            c=copy.deepcopy(best['config']); c['seed']=seed
            members.append(self.train('selected',c))
        manifest={'data_protocol_sha256':file_hash(DATA_ROOT/'protocol.json'),
                  'git_commit':self.state['git_commit'],'initial_git_commit':INITIAL_COMMIT,
                  'seeds':[3407,42,2026], 'selection_rule':('Predeclared single recipe; no structure search' if self.args.single_config else 'Max validation Macro-F1; depth -> projection -> fixed auxiliary ablations; no test selection'),
                  'ensemble_rule':'Three selected-config seeds, fixed equal probability average',
                  'test_command':f'{sys.executable} -m experiments.final_evaluate --manifest {self.batch / "final_manifest.json"}',
                  'evaluation_source_sha256':{str(PROJECT_ROOT/'experiments'/n): file_hash(PROJECT_ROOT/'experiments'/n)
                      for n in ('final_evaluate.py','train_neural.py','research.py','neural.py','common.py','paths.py')},
                  'members':[]}
        for row in members:
            root=RUN_ROOT/row['run_id']
            manifest['members'].append({'run_id':row['run_id'],'weight':1/3,
                'checkpoint':str(root/'best.pt'),'checkpoint_sha256':file_hash(root/'best.pt'),
                'config':str(root/'config.json'),'config_sha256':file_hash(root/'config.json'),
                'source_sha256':{str(root/'source'/p):h for p,h in row['code_sha256'].items()}})
        # Exclusive create; never rewrite a manifest after observing test.
        with (self.batch/'final_manifest.json').open('x') as stream:
            json.dump(manifest,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
        (self.batch/'final_manifest.sha256').write_text(file_hash(self.batch/'final_manifest.json')+'\n')
        self.command([sys.executable,'-u','-m','experiments.final_evaluate','--manifest',str(self.batch/'final_manifest.json')],
                     'final-evaluation.log','evaluation',timeout=self.args.grace_seconds-120)
        # The idle watchdog's timer covers both evaluation and report generation.
        if not report(self.batch): raise RuntimeError('Final evaluation incomplete')
        checks=json.loads((self.batch/'artifacts.json').read_text())
        if not all(file_hash(self.batch/p)==h for p,h in checks.items()):
            raise RuntimeError('Report/visualization integrity failure')
        self.progress('训练结束、三种子 final evaluation 完成；checkpoint 哈希、报告、混淆矩阵和初始 Git 结果对比图均已校验。即将休眠。')
        self.state.update(phase='complete',ready_to_hibernate=True,completed=time.time()); self.save()
        os.sync()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'experiments/configs/a4000_research.json')
    p.add_argument('--name',required=True)
    p.add_argument('--single-config', action='store_true', help='Skip search; train this recipe for all three seeds, then evaluate/report/hibernate')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--progress',type=Path,default=PROJECT_ROOT.parent/'PROGRESS.md')
    p.add_argument('--grace-seconds',type=int,default=600)
    p.add_argument('--max-hours',type=float,default=8)
    args=p.parse_args()
    from experiments.train_neural import run_name
    run_name(args.name)
    if len(args.name)>32: p.error('Batch name must be <=32 chars')
    if not 180<=args.grace_seconds<=600 or not 0<args.max_hours<=24: p.error('grace=180..600; max-hours=(0,24]')
    if subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_ROOT,text=True).strip():
        p.error('Commit reviewed changes before running a research batch')
    lock=(RUN_ROOT.parent/'research.lock'); lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as locked:
        fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipeline=Pipeline(args)
        try: pipeline.execute()
        except BaseException as error:
            pipeline.state.update(phase='failed',error_type=type(error).__name__,error=str(error),training_pid=None)
            pipeline.save()
            pipeline.progress(f'队列失败：{type(error).__name__}；保留配置/日志，watchdog 将在训练停止后的期限内休眠，不无限重试训练。')
            try: report(pipeline.batch)
            except Exception: pass
            raise


if __name__=='__main__':
    main()
