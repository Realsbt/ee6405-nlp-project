"""Relocate resources while preserving paths recorded in historical evidence."""

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_STORAGE = Path('/data/botao/ee6405-race-validation')
ORIGINAL_PROJECT = Path('/home/botao/文档/ChatGPT/小组作业')
STORAGE = Path(os.environ.get('EE6405_STORAGE') or (
    ORIGINAL_STORAGE if ORIGINAL_STORAGE.is_dir() else PROJECT_ROOT / '.local/resources'
)).expanduser().resolve()
ASSETS = STORAGE / 'assets'
DATA_ROOT = STORAGE / 'experiments/hart_grouped_v1'
BACKBONE = ASSETS / 'roberta-base'
UPSTREAM = Path(os.environ.get('EE6405_UPSTREAM', PROJECT_ROOT / 'local_validation/upstream/RACE')).expanduser().resolve()


def resolve_recorded_path(value):
    """Map only the two known historical prefixes; never rewrite saved JSON."""
    path = Path(value).expanduser()
    for original, current in ((ORIGINAL_STORAGE, STORAGE), (ORIGINAL_PROJECT, PROJECT_ROOT)):
        if path.is_relative_to(original):
            return current / path.relative_to(original)
    return path


def parser_python():
    override = os.environ.get('EE6405_RST_PYTHON')
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f'EE6405_RST_PYTHON does not exist: {candidate}')
        return str(candidate)
    candidates = (
        STORAGE / '.venv-rst/Scripts/python.exe',
        STORAGE / '.venv-rst/bin/python',
        STORAGE / '.venv/Scripts/python.exe',
        STORAGE / '.venv/bin/python',
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable
