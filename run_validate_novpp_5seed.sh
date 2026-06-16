#!/usr/bin/env bash
# 5-seed No-VPP validation with heartbeat wrapper
set -e

LOG="logs/validate_novpp_5seed.log"
mkdir -p logs

echo "=== No-VPP 5-seed validation (200k steps) ==="
python scripts/run_no_vpp_baseline.py \
  --config config/experiment/train_no_vpp_ppo.yaml \
  --seeds 5 \
  --output-root outputs/experiments \
  --exp-name no_vpp_validate_5seed \
  --device cpu >> "$LOG" 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] No-VPP 5-seed still running (PID $PID)"
done
wait "$PID"
EXIT=$?
echo "=== No-VPP 5-seed validation complete (exit=$EXIT) ==="
exit $EXIT
