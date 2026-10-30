#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

workloads=(
    job_light
    stats_ceb
    ergastf1
    job_m_ours
    job_large_ours
)

cd "$ROOT_DIR"

for workload in "${workloads[@]}"; do
    for extended in false true; do
        echo "=== workload=$workload extended=$extended ==="
        "$PYTHON_BIN" scripts/py/measure_postgresql_statistics.py \
            --workload "$workload" \
            --extended "$extended"
    done
done

echo "All PostgreSQL statistics experiments completed."
