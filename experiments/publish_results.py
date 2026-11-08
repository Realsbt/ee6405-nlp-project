"""Only publish verified score gains, to the one explicitly named branch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from experiments.neural import PROJECT_ROOT
from experiments.common import write_json


def publish_if_improved(batch, branch, timeout=90):
    batch=Path(batch)
    gain=json.loads((batch/'score_change.json').read_text())
    record={'branch':branch,'eligible':gain['publish_eligible'],'gain_pp':gain['paired_gain_pp']}
    if not gain['publish_eligible']:
        record['status']='skipped_no_confirmed_gain'; write_json(batch/'publication.json',record); return record
    try:
        current=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT_ROOT,text=True).strip()
        if branch!='yuzhe' or current!=branch:
            raise RuntimeError('Refusing to push outside the single authorized branch yuzhe')
        if subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_ROOT,text=True).strip():
            raise RuntimeError('Refusing to mix unrelated uncommitted changes')
        target=PROJECT_ROOT/'experiments/research_results'/batch.name
        if target.exists(): raise FileExistsError('Preserving previous published artifacts')
        files=[]
        for p in batch.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.jsonl','.md','.png','.sha256','.log') and 'source' not in p.relative_to(batch).parts and 'wandb' not in p.relative_to(batch).parts:
                # Never publish weights, source text partitions, or a secret file.
                files.append(p)
        key=Path(os.environ['HYPERSTACK_API_KEY_FILE']).read_bytes().strip()
        if any(key and key in p.read_bytes() for p in files):
            raise RuntimeError('Secret scan rejected publication')
        if sum(p.stat().st_size for p in files)>50*1024**2:
            raise RuntimeError('Publication exceeds 50 MiB; review artifacts before pushing')
        for p in files:
            out=target/p.relative_to(batch); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,out)
        relative=str(target.relative_to(PROJECT_ROOT))
        subprocess.run(['git','add','--',relative],cwd=PROJECT_ROOT,check=True,timeout=10)
        subprocess.run(['git','-c','user.name=Codex Agent','-c','user.email=codex@localhost','commit','-m',
                        f'Report verified research gain {gain["paired_gain_pp"]:.3f}pp ({batch.name})'],cwd=PROJECT_ROOT,check=True,timeout=15)
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
        push=subprocess.run(['git','push','--set-upstream','origin',f'HEAD:refs/heads/{branch}'],cwd=PROJECT_ROOT,
                            env=env,capture_output=True,text=True,timeout=timeout)
        record.update(status='pushed' if push.returncode==0 else 'push_failed',exit_code=push.returncode,
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip())
        (batch/'logs/git-push.log').write_text((push.stdout+push.stderr).replace(key.decode(),'[REDACTED]'))
    except Exception as error:
        record.update(status='publication_failed',error_type=type(error).__name__,error=str(error))
    write_json(batch/'publication.json',record)
    return record
