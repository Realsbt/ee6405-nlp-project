"""Detailed W&B telemetry with durable local logs and bounded cloud shutdown."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from experiments.common import LABELS, write_json


class Tracker:
    def __init__(self, output, config, resume=False, job_type='train'):
        import wandb
        self.wandb=wandb; self.output=Path(output); self.run=None
        self.started=time.perf_counter(); self.log_interval=config.get('wandb_log_interval',10)
        self.run=wandb.init(project=config.get('wandb_project','ee6405-a4000-research'),
            entity=config.get('wandb_entity') or None, group=os.environ.get('EE6405_BATCH_NAME'),
            id=hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:16],
            name=self.output.name, job_type=job_type, config=config,
            resume='allow' if resume else 'never', dir=str(self.output),
            tags=['hart','a4000',job_type],
            settings=wandb.Settings(init_timeout=45,finish_timeout=30,console='off',save_code=False,disable_git=True))
        self.run.define_metric('batch_step')
        self.run.define_metric('epoch')
        for prefix in ('train/*','lr/*','memory/*'):
            self.run.define_metric(prefix,step_metric='batch_step')
        self.run.define_metric('val/*',step_metric='epoch')
        self.run.define_metric('epoch/*',step_metric='epoch')
        self.run.define_metric('val/macro_f1',summary='max',step_metric='epoch')
        write_json(self.output/'wandb_run.json',{'id':self.run.id,'url':self.run.url,
                   'project':self.run.project,'entity':self.run.entity,'mode':self.run.settings.mode})
        self.run.summary['test_partition_loaded']=False

    def error(self,error):
        with (self.output/'tracking_errors.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':time.time(),'type':type(error).__name__,'error':str(error)})+'\n')

    def log(self,values):
        try: self.run.log(values)
        except Exception as error: self.error(error)

    def watch(self,model,config):
        try:
            self.run.watch(model,log=config.get('wandb_watch','all'),
                           log_freq=config.get('wandb_watch_frequency',500),log_graph=False)
        except Exception as error: self.error(error)

    def event(self,kind,record):
        if kind=='epoch_complete':
            values={'epoch':record['epoch'],'val/best_epoch':record['best_epoch'],
                    'val/stale_epochs':record['stale_epochs'],'epoch/fit_seconds':record['fit_seconds'],
                    'epoch/validation_seconds':record['validation_seconds'],
                    'epoch/mean_train_loss':record['mean_train_loss'],
                    'epoch/mean_ce':record['mean_train_ce'],'epoch/mean_supcon':record['mean_train_supcon']}
            self.metrics(values,'val',record['validation']); self.log(values)
        elif kind=='initialized':
            for k,v in record.items():
                if k not in ('event','time_utc'): self.run.summary[k]=v
        elif kind in ('completed','smoke_passed','interrupted'):
            self.run.summary['status']=kind

    @staticmethod
    def metrics(values,prefix,metrics):
        for k,v in metrics.items():
            if isinstance(v,(int,float)): values[f'{prefix}/{k}']=v
        for label,items in metrics['per_class'].items():
            for k,v in items.items():
                if v is not None: values[f'{prefix}/{label}/{k}']=v
        for i,row in enumerate(metrics['confusion_matrix']):
            for j,v in enumerate(row): values[f'{prefix}/confusion/{LABELS[i]}/{LABELS[j]}']=v

    def evaluation(self,metrics,probs,records,prefix='best_val'):
        try:
            values={};self.metrics(values,prefix,metrics)
            labels=[r['label_id'] for r in records]
            values[f'{prefix}/confusion_matrix']=self.wandb.plot.confusion_matrix(
                probs=probs,y_true=labels,class_names=LABELS)
            values[f'{prefix}/roc']=self.wandb.plot.roc_curve(labels,probs,labels=LABELS)
            values[f'{prefix}/precision_recall']=self.wandb.plot.pr_curve(labels,probs,labels=LABELS)
            values[f'{prefix}/confidence']=self.wandb.Histogram(np.max(probs,axis=1))
            values[f'{prefix}/predictions']=self.wandb.Table(
                columns=['item_id','true_label','prediction']+LABELS,
                data=[[r['item_id'],r['label'],LABELS[int(p.argmax())]]+[float(x) for x in p]
                      for r,p in zip(records,probs)])
            self.log(values)
        except Exception as error: self.error(error)

    def finish(self,status,summary=None):
        if self.run is None:return
        try:
            self.run.summary['status']=status
            self.run.summary['wall_seconds']=time.perf_counter()-self.started
            artifact=self.wandb.Artifact(self.output.name+'-metadata',type='experiment-metadata')
            for name in ('config.json','parameters.json','result.json','events.jsonl','smoke.json','wandb_run.json'):
                path=self.output/name
                if path.exists():artifact.add_file(str(path),name=name)
            self.run.log_artifact(artifact)
            if summary:
                for k in ('checkpoint_sha256','best_epoch','test_partition_loaded','code_sha256','efficiency'):
                    if k in summary:self.run.summary[k]=summary[k]
            self.run.finish(exit_code=0 if status in ('completed','smoke_passed') else 1)
        except Exception as error:self.error(error)
        finally:self.run=None


def upload_batch(batch,config):
    """Upload report plots/metadata, never raw text, credentials or model weights."""
    batch=Path(batch);tracker=Tracker(batch,config,job_type='batch-report')
    try:
        for p in batch.glob('*.png'):tracker.log({f'report/{p.stem}':tracker.wandb.Image(str(p))})
        artifact=tracker.wandb.Artifact(batch.name+'-reports',type='evaluation-report')
        for name in ('SUMMARY.md','REPORT.md','详细报告与图解.md','score_change.json','artifacts.json','final_manifest.json','PROCESS_LOG.md'):
            p=batch/name
            if p.exists():artifact.add_file(str(p),name=name)
        for path in (batch/'runs').glob('*/SUMMARY.md'):
            artifact.add_file(str(path),name=str(path.relative_to(batch)))
        tracker.run.log_artifact(artifact)
        if (batch/'score_change.json').exists():
            tracker.run.summary.update(json.loads((batch/'score_change.json').read_text()))
        final=json.loads((batch/'final_evaluation/result.json').read_text())
        tracker.run.summary['test_partition_loaded']=True
        for member in final['members']:
            values={};tracker.metrics(values,f'final_test/seed_{member["seed"]}',member['test']);tracker.log(values)
        values={};tracker.metrics(values,'final_test/ensemble',final['ensemble']);tracker.log(values)
    finally:tracker.finish('completed')
