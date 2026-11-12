#!/usr/bin/env bash
set -euo pipefail
NEURAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$NEURAL_DIR/.."
source experiments/runtime_env.sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$EE6405_PYTHON" -u -m experiments.train_neural "$@"
