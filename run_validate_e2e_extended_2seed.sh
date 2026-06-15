#!/usr/bin/env bash
# E2E extended training with heartbeat wrapper to keep background task alive.
set -e

LOG="logs/validate_e2e_extended_2seed.log"
mkdir -p logs

echo "=== E2E extended training with aligned reward (2 seeds) ==="
python scripts/run_e2e_extended_training.py \
  --config config/experiment/train_end_to_end_ppo.yaml \
  --seeds 2 \
  --steps 200000 500000 1000000 2000000 \
  --align-reward \
  --output-root outputs/validate_e2e_extended \
  --device cpu \
  --skip-existing >> "$LOG" 2>&1 &
PID=$!

# Keep background task heartbeat alive by printing periodic status
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] E2E extended still running (PID $PID)"
done

wait "$PID"
EXIT=$?
echo "=== E2E extended validation complete (exit=$EXIT) ==="
exit $EXIT
