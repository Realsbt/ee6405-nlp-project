"""Run REAL checkpoint explanation checks locally, without training/test data.

Not run in the packaging environment. Output is a new user-chosen JSON report.
Uses one unlabeled synthetic English paragraph solely as a functionality test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def tensor_digest(model):
    digest=hashlib.sha256()
    if hasattr(model,'state_dict'):
        for key,value in model.state_dict().items():
            digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    else:
        # This model is a verified scikit-learn pipeline already loaded by main.
        import pickle
        digest.update(pickle.dumps(model,protocol=4))
    return digest.hexdigest()


def hook_counts(model):
    if not hasattr(model,'modules'):return []
    return [(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--models',nargs='+',choices=['tfidf','roberta','race','race_no_supcon'],default=['tfidf','roberta','race','race_no_supcon'])
    p.add_argument('--units',type=int,choices=range(1,9),default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Preserving existing report: '+str(args.output))
    from experiments.inference import DetectorService
    from experiments.document_explanations import explain_document,check_parity
    service=DetectorService(args.device)
    text=('The library extended its opening hours during the examination period. '
          'However, students reported confusion about the booking system. '
          'The university will publish a guide explaining how to reserve a room.')
    report={'schema':'ee6405.explanation-real-check/v1','scope':'REAL checkpoint functionality; not accuracy or authorship validation',
            'models':{},'training':False,'test_or_hc3_loaded':False,'python':sys.version,'repo':str(ROOT)}
    try:
        for name in args.models:
            started=time.perf_counter()
            try:
                before=service.predict(text,name)
                model,_=service._load_model(name);weights=tensor_digest(model);hooks=hook_counts(model)
                result=explain_document(service,text,name,max_units=args.units)
                after=service.predict(text,name)
                check_parity(before['scores'],result['document']['scores']);check_parity(before['scores'],after['scores'])
                assert tensor_digest(model)==weights,'Model state changed'
                assert hook_counts(model)==hooks,'Hook leak'
                assert result['checked_units']>0,'No unit was measured'
                if name.startswith('race'):
                    assert result['graph'] and result['document'].get('rst_tree'),'Real RST/graph missing'
                report['models'][name]={'status':'PASS','run_id':before['run_id'],'document_scores':before['scores'],
                    'method':result['method'],'checked_units':result['checked_units'],'restored_max_abs_diff':result['restored_max_abs_diff'],
                    'weight_digest_before_after':weights,'hook_counts_unchanged':True,'graph_nodes':len(result['graph']['nodes']) if result['graph'] else None,
                    'units':[{'id':u['id'],'status':u['status'],'delta_pp':u['delta_pp']} for u in result['units']],
                    'seconds':time.perf_counter()-started}
                print(name,'PASS',flush=True)
            except Exception as e:
                report['models'][name]={'status':'FAIL','error':str(e),'type':type(e).__name__,'seconds':time.perf_counter()-started}
                print(name,'FAIL',str(e),file=sys.stderr,flush=True)
    finally:service.close()
    report['status']='PASS' if all(r['status']=='PASS' for r in report['models'].values()) else 'PARTIAL_OR_FAILED'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
