#!/usr/bin/env bash
set -euo pipefail
REVIEW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REVIEW_DIR/.."
export HF_HOME=/data/botao/ee6405-race-validation/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/data/botao/ee6405-race-validation/cache/matplotlib
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REVIEW_PYTHON=/data/botao/ee6405-race-validation/.venv/bin/python
"$REVIEW_PYTHON" -m experiments.analyze_results --require-complete
exec "$REVIEW_PYTHON" -m experiments.verify_project
