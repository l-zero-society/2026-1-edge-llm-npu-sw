#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
output_dir="$PWD/diagnostics/norm10_scale_probe"
mkdir -p "$output_dir"
if [[ -s "$output_dir/pid" ]] && kill -0 "$(cat "$output_dir/pid")" 2>/dev/null; then
  echo 'Norm10 scale probe already has a live process' >&2
  exit 1
fi
if [[ -e "$output_dir/DONE" ]]; then
  echo 'Scientific results already complete; refusing to rerun for publication' >&2
  exit 1
fi
# Retain failed-attempt evidence. No historical experiment directory is touched.
for name in FAILED PUSH_FAILED status.json run.log; do
  if [[ -e "$output_dir/$name" ]]; then
    mkdir -p "$output_dir/work/previous_attempt"
    mv "$output_dir/$name" "$output_dir/work/previous_attempt/$name.$(date +%s)"
  fi
done
PYTHONPYCACHEPREFIX=/tmp/norm10-scale-probe \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
nohup .venv-calibration/bin/python -u scripts/run_norm10_scale_probe.py \
  > "$output_dir/run.log" 2>&1 < /dev/null &
probe_pid=$!
echo "$probe_pid" > "$output_dir/pid"
disown "$probe_pid" 2>/dev/null || true
kill -0 "$probe_pid"
echo "started pid=$probe_pid"
