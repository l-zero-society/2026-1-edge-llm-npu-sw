#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="$repo_root/diagnostics/norm10_debug_pilot"
label="com.lzero.norm10-debug-pilot"
domain="gui/$(id -u)"

if [[ "${1:-}" == "--run" ]]; then
  cd "$repo_root"
  echo "$$" > "$output_dir/pid"
  exec env \
    PYTHONPYCACHEPREFIX=/tmp/norm10-debug \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KMP_USE_SHM=0 \
    PYTORCH_ENABLE_MPS_FALLBACK=0 \
    .venv-calibration/bin/python -u scripts/run_norm10_debug_pilot.py \
    >> "$output_dir/run.log" 2>&1
fi

cd "$repo_root"
mkdir -p "$output_dir"
if launchctl print "$domain/$label" >/dev/null 2>&1; then
  echo "launchd job already exists: $domain/$label" >&2
  exit 1
fi

rm -f "$output_dir/DONE" "$output_dir/FAILED" "$output_dir/pid"
: > "$output_dir/run.log"
launchctl submit \
  -l "$label" \
  -- "$repo_root/scripts/launch_norm10_debug_pilot.sh" --run

# Confirm launchd owns the job and the Python runner, rather than this launcher,
# has written the first RUNNING heartbeat.
for _ in {1..200}; do
  if [[ -f "$output_dir/status.json" ]] && \
     .venv-calibration/bin/python -c \
       'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="RUNNING" and "elapsed_seconds" in s' \
       "$output_dir/status.json" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

launchctl print "$domain/$label" >/dev/null
test -s "$output_dir/pid"
.venv-calibration/bin/python -c \
  'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="RUNNING" and "elapsed_seconds" in s' \
  "$output_dir/status.json"
echo "started launchd_job=$domain/$label pid=$(cat "$output_dir/pid")"
