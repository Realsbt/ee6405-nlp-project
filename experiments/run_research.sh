#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source experiments/runtime_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$EE6405_PYTHON" -u -m experiments.run_research "$@"
