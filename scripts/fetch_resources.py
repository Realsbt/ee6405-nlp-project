"""Fetch pinned external assets and private course checkpoints, verifying SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paths import STORAGE, ASSETS, UPSTREAM


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_files(manifest, profile, skip_checkpoints=False):
    groups = {'backbone', 'parser', 'checkpoint-demo'}
    if profile == 'full':
        groups |= {'datasets', 'checkpoint-extra'}
    return [row for row in manifest['files'] if row['group'] in groups
            and not (skip_checkpoints and row['source'] == 'github-release')]


def configure_parser():
    source, local = ASSETS / 'isanlp_rst_rstdt', ASSETS / 'isanlp_rst_local'
    config = json.loads((source / 'config.json').read_text())
    config['model']['transformer']['model_name'] = str(ASSETS / 'xlm-roberta-large-config')
    local.mkdir(parents=True, exist_ok=True)
    content = json.dumps(config, indent=2) + '\n'
    target = local / 'config.json'
    if target.exists() and json.loads(target.read_text()) != config:
        raise RuntimeError(f'Existing parser configuration differs; preserving it: {target}')
    if not target.exists():
        target.write_text(content)
    for name in ('best_weights.pt', 'relation_table.txt'):
        target = local / name
        if target.exists() or target.is_symlink():
            if target.resolve() != (source / name).resolve():
                raise RuntimeError(f'Existing parser link differs: {target}')
        else:
            target.symlink_to(Path('..') / 'isanlp_rst_rstdt' / name)


def prepare_upstream(spec):
    if not UPSTREAM.exists():
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', spec['repository'], str(UPSTREAM)], check=True)
        subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', '--detach', spec['commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip()
    if actual != spec['commit'] or dirty:
        raise RuntimeError('Existing upstream checkout differs; preserve it and use a clean pinned checkout via EE6405_UPSTREAM.')


def fetch(row, destination):
    if row['source'] == 'huggingface':
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=row['repo'], repo_type=row['repo_type'], revision=row['revision'],
                        filename=row['filename'], local_dir=destination.parent)
    elif row['source'] == 'google-drive':
        try:
            import gdown
        except ImportError:
            raise RuntimeError('Install requirements-fetch.txt, or obtain the HART file from its official link: ' + row['url'])
        temporary = destination.with_suffix(destination.suffix + '.part')
        result = gdown.download(id=row['file_id'], output=str(temporary), quiet=False, use_cookies=False)
        if result is None:
            raise RuntimeError('Google Drive could not provide this file. Download it manually: ' + row['url'])
        if digest(temporary) != row['sha256']:
            raise RuntimeError(f'Download hash mismatch: {temporary}')
        temporary.replace(destination)
    else:
        if not shutil.which('gh'):
            raise RuntimeError('Install GitHub CLI and run gh auth login with an account that can read the private group repository.')
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='download-') as folder:
            subprocess.run(['gh', 'release', 'download', row['tag'], '--repo', row['repository'],
                            '--pattern', row['asset'], '--dir', folder], check=True)
            downloaded = Path(folder) / row['asset']
            if digest(downloaded) != row['sha256']:
                raise RuntimeError(f'Checkpoint hash mismatch: {row["asset"]}')
            downloaded.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['demo', 'full'], default='demo')
    parser.add_argument('--skip-checkpoints', action='store_true', help='For training from base weights only')
    parser.add_argument('--verify-only', action='store_true', help='Read-only verification; no downloads or setup')
    parser.add_argument('--local-source', type=Path, help='Copy verified assets from an existing resource directory instead of downloading')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'resources/manifest.json').read_text())
    rows = selected_files(manifest, args.profile, args.skip_checkpoints)
    print(f'Resource directory: {STORAGE}', flush=True)
    print(f'Selected {len(rows)} files, {sum(r["bytes"] for r in rows)/1024**3:.2f} GiB (excluding environment and generated data).', flush=True)
    missing = []
    for row in rows:
        path = STORAGE / row['path']
        if path.is_file():
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise RuntimeError(f'Existing file differs; preserving it: {path}')
        elif args.verify_only:
            missing.append(row['path'])
            continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.local_source:
                source = args.local_source / row['path']
                if not source.is_file() or digest(source) != row['sha256']:
                    raise RuntimeError(f'Local source missing or corrupt: {source}')
                shutil.copyfile(source, path)
            else:
                fetch(row, path)
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise RuntimeError(f'Resource hash mismatch: {path}')
        print(f'OK {row["path"]}', flush=True)
    if missing:
        raise SystemExit('Missing resources:\n' + '\n'.join(missing))
    if not args.verify_only:
        prepare_upstream(manifest['upstream'])
        configure_parser()
    print('Selected resources match the pinned manifest.', flush=True)


if __name__ == '__main__':
    main()
