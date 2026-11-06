"""Create an isolated Python 3.11 environment for Linux / WSL2."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paths import STORAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='python3.11', help='Python 3.11 executable')
    parser.add_argument('--torch', choices=['cpu', 'cu128'], default='cpu')
    args = parser.parse_args()
    version = json.loads(subprocess.check_output([
        args.python, '-c', 'import sys,json; print(json.dumps(list(sys.version_info[:2])))'
    ], text=True))
    if version != [3, 11]:
        raise SystemExit('Use Python 3.11 via --python. See docs/REPRODUCING.md.')
    env = STORAGE / '.venv'
    if env.exists():
        raise SystemExit(f'Environment already exists; preserving it: {env}')
    env.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.python, '-m', 'venv', str(env)], check=True)
    python = str(env / 'bin/python')
    commands = [
        [python, '-m', 'pip', 'install', 'torch==2.7.0', '--index-url',
         f'https://download.pytorch.org/whl/{args.torch}'],
        [python, '-m', 'pip', 'install', '-r', str(ROOT / 'experiments/requirements-core.txt')],
        [python, '-m', 'pip', 'install', '--no-deps', '-r', str(ROOT / 'experiments/requirements-rst.txt')],
        [python, '-m', 'pip', 'install', '-r', str(ROOT / 'experiments/requirements-fetch.txt')],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    subprocess.run([python, '-c', 'import torch, torch_geometric, transformers; from isanlp_rst.parser import Parser; print("Core and parser imports passed")'], check=True)
    print(f'Environment ready: {python}')
    print('Next: use this Python to run scripts/fetch_resources.py.')


if __name__ == '__main__':
    main()
