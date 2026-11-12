#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$DEMO_DIR/.."
source experiments/runtime_env.sh
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec "$EE6405_PYTHON" -u -m experiments.demo_server "$@"
