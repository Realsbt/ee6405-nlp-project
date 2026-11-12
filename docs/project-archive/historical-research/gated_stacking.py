import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path('EE6405_paper_research').resolve()))
from stack_cpu import ROOT, DATA, ARCHIVE, rows, probability_array, stack_features, model_for

fusion = ROOT/'repository/outputs/kaggle_fusion_continuation_v1/edu_fusion_experiment_v1/evaluation'
base = ('roberta','race','tfidf')
gates = ('gate3407','gate42','gate2026')
variants = {
 'original': base,
 'add_gate3407': (*base,gates[0]),
 'replace_race_with_gate3407': ('roberta','tfidf',gates[0]),
 'add_mean_gate': (*base,'mean_gate'),
 'replace_race_with_mean_gate': ('roberta','tfidf','mean_gate'),
 'add_matched_cls3407': (*base,'cls3407'),
}
def read(split):
 records=rows(DATA/f'{split}.jsonl')
 if split=='val':
  sources=[ROOT/x/'validation_predictions.jsonl' for x in ('transfer_roberta_v1','race_cpu_v1','transfer_tfidf_v1')]
 else:
  sources=[ARCHIVE/x/'test_predictions.jsonl' for x in ('roberta_s3407','race_s3407','tfidf_lr_v1')]
 names=(*base,*gates,'cls3407')
 sources += [fusion/x/f'{split}_predictions.jsonl' for x in ('fusion_gated_s3407','fusion_gated_s42','fusion_gated_s2026','fusion_cls_s3407')]
 values={key:probability_array(records,path) for key,path in zip(names,sources,strict=True)}
 values['mean_gate']=np.mean([values[g] for g in gates],axis=0)
 return records,np.array([r['label_id'] for r in records]),values

def metrics(y,p):
 return {'micro':float(np.mean(y==p)), 'macro':float(f1_score(y,p,average='macro')),
         'class_f1':f1_score(y,p,average=None,labels=range(4)).tolist()}

val,y,prob=read('val')
folds=json.loads((ROOT/'cpu_stacking_v1/protocol.json').read_text())['folds']
fold=np.array([folds[r['item_id']] for r in val])
dev={}
with threadpool_limits(limits=2):
 for name,keys in variants.items():
  x=stack_features([prob[k] for k in keys])
  pred=np.empty_like(y)
  for f in sorted(set(fold)):
   fit=fold!=f
   pred[~fit]=model_for({'kind':'logistic','C':1}).fit(x[fit],y[fit]).predict(x[~fit])
  dev[name]=metrics(y,pred)
 selected=max(variants,key=lambda n:(dev[n]['micro'],dev[n]['macro']))
 report={'development':dev,'selected':selected,'test':{}}
 path=Path('.ee6405-slides-build/gated_stacking.json')
 path.write_text(json.dumps(report,indent=2))
 _,ty,tp=read('test')
 for name in dict.fromkeys(('original',selected)):
  keys=variants[name]
  clf=model_for({'kind':'logistic','C':1}).fit(stack_features([prob[k] for k in keys]),y)
  report['test'][name]=metrics(ty,clf.predict(stack_features([tp[k] for k in keys])))
 path.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
