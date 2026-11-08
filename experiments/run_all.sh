#!/usr/bin/env bash
# Full pipeline: audit, Arm T (KuaiRec big), Arm T replication (ML-1M),
# Arm F v2 (KuaiRec small, corrected) with the exposure arm, random-split
# arm, verdicts. The coverage gate runs inside the pipeline and can stop
# it (exit 3) with results/coverage.json as the output.
# Usage: experiments/run_all.sh [stage]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
exec "$PY" experiments/run_all.py "${1:-all}"
