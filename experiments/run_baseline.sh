#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$EXPERIMENT_DIR/.."
BASELINE_ID="${1:-tfidf_lr_v1}"
if [[ ! "$BASELINE_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Run name may contain only letters, digits, underscores and hyphens.\n' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
source experiments/runtime_env.sh
BASELINE_PYTHON="$EE6405_PYTHON"
BASELINE_OUTPUT="experiments/reports/$BASELINE_ID"
if [[ ! -f "$BASELINE_OUTPUT/result.json" ]]; then
    "$BASELINE_PYTHON" -u -m experiments.train_tfidf \
        --output "$BASELINE_OUTPUT" \
        --model "$EE6405_STORAGE/experiments/models/$BASELINE_ID.joblib"
fi
"$BASELINE_PYTHON" -m experiments.verify_tfidf --output "$BASELINE_OUTPUT"
