"""Fetch pinned external assets and private course checkpoints, verifying SHA-256."""

import argparse
import hashlib
import json
import os
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


def _same_content(path, source, expected):
    return path.is_file() and path.stat().st_size == source.stat().st_size and digest(path) == expected


def materialize_resource(source, target, expected, mode='auto'):
    """Materialize an immutable parser file with portable link fallbacks."""
    if not source.is_file() or source.stat().st_size != expected['bytes'] or digest(source) != expected['sha256']:
        raise RuntimeError(f'Source resource is missing or corrupt: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _same_content(target, source, expected['sha256']):
        return 'existing'
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise RuntimeError(f'Existing managed target differs; preserving it: {target}')
    attempts = ('symlink', 'hardlink', 'copy') if mode == 'auto' else (mode,)
    errors = []
    for strategy in attempts:
        try:
            if strategy == 'symlink':
                target.symlink_to(os.path.relpath(source, target.parent))
            elif strategy == 'hardlink':
                target.hardlink_to(source)
            else:
                temporary = target.with_name(target.name + '.part')
                with source.open('rb') as input_stream, temporary.open('wb') as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, 4 * 1024 * 1024)
                if not _same_content(temporary, source, expected['sha256']):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f'Copied resource hash mismatch: {temporary}')
                temporary.replace(target)
            if not _same_content(target, source, expected['sha256']):
                raise RuntimeError(f'{strategy} resource verification failed: {target}')
            return strategy
        except (OSError, RuntimeError) as error:
            errors.append(f'{strategy}: {error}')
            if target.is_symlink() or target.exists():
                target.unlink()
    raise RuntimeError('Could not materialize resource: ' + '; '.join(errors))


def selected_files(manifest, profile, skip_checkpoints=False):
    groups = {'backbone', 'parser', 'checkpoint-demo'}
    if profile == 'full':
        groups |= {'datasets', 'checkpoint-extra'}
    return [row for row in manifest['files'] if row['group'] in groups
            and not (skip_checkpoints and row['source'] == 'github-release')]


def configure_parser(manifest, link_mode='auto'):
    source, local = ASSETS / 'isanlp_rst_rstdt', ASSETS / 'isanlp_rst_local'
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    config['model']['transformer']['model_name'] = str(ASSETS / 'xlm-roberta-large-config')
    local.mkdir(parents=True, exist_ok=True)
    content = json.dumps(config, indent=2) + '\n'
    target = local / 'config.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != config:
        state = local / 'installation.json'
        if not state.is_file():
            raise RuntimeError(f'Existing parser configuration differs; preserving it: {target}')
        temporary = target.with_suffix('.json.part')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(target)
    elif not target.exists():
        temporary = target.with_suffix('.json.part')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(target)
    rows = {row['path']: row for row in manifest['files']}
    link_state = {}
    for name in ('best_weights.pt', 'relation_table.txt'):
        target = local / name
        link_state[name] = materialize_resource(source / name, target, rows[f'assets/isanlp_rst_rstdt/{name}'], link_mode)
    (local / 'installation.json').write_text(
        json.dumps({'source_config_sha256': digest(source / 'config.json'),
                    'links': link_state}, indent=2) + '\n', encoding='utf-8')


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
    parser.add_argument('--link-mode', choices=['auto', 'symlink', 'hardlink', 'copy'], default='auto')
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
    if args.verify_only:
        upstream = UPSTREAM
        if not upstream.is_dir():
            missing.append(f'upstream checkout: {upstream}')
        else:
            actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
            if actual != manifest['upstream']['commit'] or dirty:
                missing.append(f'upstream version/dirty: {upstream}')
        if not (ASSETS / 'isanlp_rst_local' / 'config.json').is_file():
            missing.append(str(ASSETS / 'isanlp_rst_local' / 'config.json'))
        else:
            for name in ('best_weights.pt', 'relation_table.txt'):
                target = ASSETS / 'isanlp_rst_local' / name
                source = ASSETS / 'isanlp_rst_rstdt' / name
                row = next(item for item in manifest['files']
                           if item['path'] == f'assets/isanlp_rst_rstdt/{name}')
                if not _same_content(target, source, row['sha256']):
                    missing.append(f'parser derived resource: {target}')
    else:
        prepare_upstream(manifest['upstream'])
        configure_parser(manifest, args.link_mode)
    print('Selected resources match the pinned manifest.', flush=True)


if __name__ == '__main__':
    main()
