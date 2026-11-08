"""Independent watchdog: hibernate after readiness or a bounded idle deadline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from experiments.hyperstack import Hyperstack


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def alive(pid):
    try:
        # Zombies are finished even while waiting for their parent to reap them.
        return Path(f'/proc/{int(pid)}/stat').read_text().split()[2] != 'Z'
    except (FileNotFoundError, TypeError, ValueError):
        return False


def decision(state, now, idle_since, grace, absolute_deadline):
    active = alive(state.get('training_pid')) if state.get('phase') == 'training' else False
    if active:
        idle_since = None
    elif idle_since is None:
        idle_since = min(now, state.get("last_training_exit", now))
    deadline = absolute_deadline
    if idle_since is not None:
        deadline = min(deadline, idle_since + grace)
    if active and state.get('training_deadline'):
        deadline = min(deadline, state['training_deadline'])
    return bool(state.get('ready_to_hibernate') or now >= deadline), idle_since, deadline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--grace-seconds', type=int, default=600)
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--progress', type=Path)
    args = p.parse_args()
    if not 30 <= args.grace_seconds <= 600 or not 0 < args.max_hours <= 24:
        p.error('grace must be 30..600 seconds; max-hours in (0,24]')
    api = Hyperstack(); verified = api.preflight()
    args.batch.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    idle_since = None
    last_progress = 0.0
    atomic(args.batch / 'watchdog.json', {'status': 'armed', 'pid': os.getpid(),
           'grace_seconds': args.grace_seconds, 'absolute_deadline': deadline, **verified})
    while True:
        try:
            state = json.loads((args.batch / 'pipeline.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        due, idle_since, current_deadline = decision(state, time.time(), idle_since, args.grace_seconds, deadline)
        if args.progress and time.monotonic() - last_progress >= 600:
            now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            run = state.get('current_run')
            status = {}
            if run and state.get('run_root'):
                try: status = json.loads((Path(state['run_root'])/run/'status.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError): pass
            with args.progress.open('a') as stream:
                stream.write(f'\n### Watchdog CKPT — {now}\n\n- 阶段：{state.get("phase")}；运行：{run}；完成 epoch：{status.get("last_completed_epoch")}。\n- 当前休眠兜底期限 Unix time：{current_deadline:.0f}；批次：`{args.batch}`。\n')
            last_progress = time.monotonic()
        if due:
            reason = 'artifacts_verified' if state.get('ready_to_hibernate') else 'deadline_fallback'
            atomic(args.batch / 'hibernate-request.json', {'status': 'requesting', 'reason': reason,
                   'time': time.time(), 'vm_id': api.vm_id, 'phase': state.get('phase'),
                   'artifacts_complete': bool(state.get('ready_to_hibernate'))})
            os.sync()
            # Keep retrying on provider/network failures; never fall back to SHUTOFF.
            try:
                response = api.hibernate()
                atomic(args.batch / 'hibernate-request.json', {'status': 'accepted', 'reason': reason,
                       'vm_id': api.vm_id, 'time': time.time(), 'response_status': response.get('status'),
                       'artifacts_complete': bool(state.get('ready_to_hibernate'))})
                os.sync()
                return
            except RuntimeError as error:
                atomic(args.batch / 'hibernate-error.json', {'error': str(error), 'time': time.time()})
                time.sleep(5)
        else:
            time.sleep(2)


if __name__ == '__main__':
    main()
