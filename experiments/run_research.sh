#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source experiments/runtime_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
stamp="$(date -u +%Y-%m-%d_%H-%M-%S)"
has_name=false
has_output=false
for arg in "$@"; do
    case "$arg" in
        --name|--name=*) has_name=true ;;
        --output|--output=*) has_output=true ;;
    esac
done
extra=()
if ! "$has_name"; then extra+=(--name "$stamp"); fi
if ! "$has_output"; then extra+=(--output "$EE6405_STORAGE/experiments/research_batches/$stamp"); fi
exec "$EE6405_PYTHON" -u -m experiments.run_research "$@" "${extra[@]}"
