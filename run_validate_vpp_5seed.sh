#!/usr/bin/env bash
# 5-seed VPP full validation with heartbeat wrapper
set -e

LOG="logs/validate_vpp_5seed.log"
mkdir -p logs

echo "=== VPP full 5-seed validation (200k steps) ==="
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seed 0 --output-dir outputs/validate_vpp_5seed/seed_0 --device cpu >> "$LOG" 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] VPP seed 0 still running (PID $PID)"
done
wait "$PID"

python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seed 1 --output-dir outputs/validate_vpp_5seed/seed_1 --device cpu >> "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] VPP seed 1 still running (PID $PID)"
done
wait "$PID"

python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seed 2 --output-dir outputs/validate_vpp_5seed/seed_2 --device cpu >> "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] VPP seed 2 still running (PID $PID)"
done
wait "$PID"

python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seed 3 --output-dir outputs/validate_vpp_5seed/seed_3 --device cpu >> "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] VPP seed 3 still running (PID $PID)"
done
wait "$PID"

python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seed 4 --output-dir outputs/validate_vpp_5seed/seed_4 --device cpu >> "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[heartbeat $(date '+%H:%M:%S')] VPP seed 4 still running (PID $PID)"
done
wait "$PID"

echo "=== VPP 5-seed validation complete ==="
