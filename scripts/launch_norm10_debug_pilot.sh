#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p diagnostics/norm10_debug_pilot
if [[ -f diagnostics/norm10_debug_pilot/pid ]] && kill -0 "$(cat diagnostics/norm10_debug_pilot/pid)" 2>/dev/null; then
  echo 'Existing process is still running' >&2
  exit 1
fi
rm -f diagnostics/norm10_debug_pilot/DONE diagnostics/norm10_debug_pilot/FAILED
printf '{"state":"RUNNING","stage":"INITIALIZING"}\n' > diagnostics/norm10_debug_pilot/status.json
PYTHONPYCACHEPREFIX=/tmp/norm10-debug OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
nohup .venv-calibration/bin/python -u scripts/run_norm10_debug_pilot.py \
  > diagnostics/norm10_debug_pilot/run.log 2>&1 < /dev/null &
norm_pid=$!
echo "$norm_pid" > diagnostics/norm10_debug_pilot/pid
disown "$norm_pid"
kill -0 "$norm_pid"
test -f diagnostics/norm10_debug_pilot/pid
.venv-calibration/bin/python -c 'import json; assert json.load(open("diagnostics/norm10_debug_pilot/status.json"))["state"] == "RUNNING"'
echo "started pid=$norm_pid"
