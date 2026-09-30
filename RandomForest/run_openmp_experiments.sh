#!/usr/bin/env bash
# Launches the OpenMP scaling sweep. Energy is collected by
# run_openmp_experiments.py (Windows RAPL via python/measure_energy.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if command -v python >/dev/null 2>&1; then
    exec python run_openmp_experiments.py "$@"
fi
exec python3 run_openmp_experiments.py "$@"
