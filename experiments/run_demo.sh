#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$DEMO_DIR/.."
export HF_HOME=/data/botao/ee6405-race-validation/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec /data/botao/ee6405-race-validation/.venv/bin/python -u -m experiments.demo_server "$@"
