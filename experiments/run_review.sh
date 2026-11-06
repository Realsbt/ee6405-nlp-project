#!/usr/bin/env bash
set -euo pipefail
REVIEW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REVIEW_DIR/.."
source experiments/runtime_env.sh
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REVIEW_PYTHON="$EE6405_PYTHON"
"$REVIEW_PYTHON" -m experiments.analyze_results --require-complete
exec "$REVIEW_PYTHON" -m experiments.verify_project
