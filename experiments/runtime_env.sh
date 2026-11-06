#!/usr/bin/env bash
# Sourced by entry points after they change to the repository root.
if [[ -z "${EE6405_STORAGE:-}" ]]; then
    if [[ -d /data/botao/ee6405-race-validation ]]; then
        export EE6405_STORAGE=/data/botao/ee6405-race-validation
    else
        export EE6405_STORAGE="$PWD/.local/resources"
    fi
fi
export EE6405_PYTHON="${EE6405_PYTHON:-$EE6405_STORAGE/.venv/bin/python}"
if [[ ! -x "$EE6405_PYTHON" ]]; then
    printf 'Missing environment: %s\nFollow docs/REPRODUCING.md to install it.\n' "$EE6405_PYTHON" >&2
    exit 2
fi
export HF_HOME="$EE6405_STORAGE/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$EE6405_STORAGE/cache/matplotlib"
