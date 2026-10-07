#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p diagnostics/vpu10_e2e_pilot
rm -f diagnostics/vpu10_e2e_pilot/DONE diagnostics/vpu10_e2e_pilot/FAILED \
  diagnostics/vpu10_e2e_pilot/status.json diagnostics/vpu10_e2e_pilot/pid
PYTHONPYCACHEPREFIX=/tmp/vpu10-e2e OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  nohup .venv-calibration/bin/python scripts/run_vpu10_e2e_pilot.py \
  > diagnostics/vpu10_e2e_pilot/run.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > diagnostics/vpu10_e2e_pilot/pid
i=0
while [ "$i" -lt 50 ] && [ ! -f diagnostics/vpu10_e2e_pilot/status.json ]; do
  sleep 0.1
  i=$((i + 1))
done
kill -0 "$pid"
test -f diagnostics/vpu10_e2e_pilot/status.json
.venv-calibration/bin/python -c 'import json; assert json.load(open("diagnostics/vpu10_e2e_pilot/status.json"))["state"] == "RUNNING"'
