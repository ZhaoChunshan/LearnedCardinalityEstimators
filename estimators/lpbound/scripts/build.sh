#!/usr/bin/env bash
set -euo pipefail
lpbound_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
jobs="${LPBOUND_BUILD_JOBS:-8}"
if [[ "${1:-}" == "--jobs" ]]; then
  jobs="${2:?--jobs requires a positive number}"
elif [[ $# -gt 0 ]]; then
  echo 'Usage: build.sh [--jobs N]' >&2
  exit 2
fi
cmake -S "$lpbound_root/src/lpbound/cpp_solver" \
  -B "$lpbound_root/work/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$lpbound_root/work/build" --target lpbound_parallel --parallel "$jobs"
"$lpbound_root/work/build/lpbound_parallel" --help
