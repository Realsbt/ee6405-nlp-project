"""Check a teammate installation; optionally run all four demo models on CPU."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paths import STORAGE, UPSTREAM, parser_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', action='store_true', help='Load checkpoints and parse a short English text')
    parser.add_argument('--training-only', action='store_true',
                        help='Verify base training resources without requiring historical demo checkpoints')
    parser.add_argument('--output', type=Path, help='Save a new JSON record outside the curated experiment reports')
    args = parser.parse_args()
    if args.training_only and args.predict:
        parser.error('--training-only cannot be combined with --predict')
    if args.output and args.output.exists():
        raise SystemExit(f'Preserving existing report: {args.output}')
    manifest = json.loads((ROOT / 'resources/manifest.json').read_text())
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != manifest['upstream']['commit']:
        raise SystemExit('Upstream source revision differs from resources/manifest.json')
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Upstream checkout has local changes; use the clean pinned source for reproduction')
    if args.training_only:
        subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_resources.py'),
                        '--profile', 'full', '--skip-checkpoints', '--verify-only'], check=True)
        import torch
        from experiments.neural import import_upstream, load_records, tokenizer_local
        from isanlp_rst.parser import Parser
        import_upstream()
        tokenizer_local()
        counts = {split: len(load_records(split)) for split in ('train', 'val', 'test')}
        if counts != {'train': 11190, 'val': 1600, 'test': 3194}:
            raise RuntimeError(f'Unexpected partition sizes: {counts}')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Training requires CUDA and BF16')
        report = {'status': 'passed', 'scope': 'Training installation and frozen data integrity; no model evaluation.',
                  'resource_directory': str(STORAGE), 'python': sys.version.split()[0],
                  'upstream_commit': actual, 'partition_sizes': counts,
                  'gpu': torch.cuda.get_device_name(0), 'cuda': True, 'bf16': True,
                  'packages': {name: importlib.metadata.version(name) for name in
                               ('torch', 'torch-geometric', 'transformers', 'scikit-learn', 'isanlp-rst')}}
        payload = json.dumps(report, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                stream.write(payload)
        print(payload)
        return
    from experiments.inference import DetectorService
    service = DetectorService(device='cpu')
    report = {
        'resource_directory': str(STORAGE), 'python': sys.version.split()[0],
        'parser_python': parser_python(), 'upstream_commit': actual,
        'packages': {name: importlib.metadata.version(name) for name in (
            'torch', 'torch-geometric', 'transformers', 'scikit-learn', 'isanlp-rst')},
        'registry': service.registry(), 'predictions': {},
        'scope': 'CPU functionality check, not a full held-out evaluation or training reproduction.',
    }
    try:
        missing = [row['id'] for row in report['registry']['models'] if not row['available']]
        if missing:
            raise RuntimeError('Run scripts/fetch_resources.py --profile demo first; unavailable: ' + ', '.join(missing))
        if args.predict:
            text = ('The students compared several text classifiers using the same training and test sets. '
                    'Although the graph model used discourse relations, the simpler baseline remained competitive. '
                    'They reported both successful cases and errors to explain the limits of the experiment.')
            for row in report['registry']['models']:
                prediction = service.predict(text, row['id'])
                report['predictions'][row['id']] = prediction
                print(f"PASS {row['id']}: {prediction['prediction']}", file=sys.stderr, flush=True)
        report['status'] = 'passed'
    finally:
        service.close()
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload)


if __name__ == '__main__':
    main()
