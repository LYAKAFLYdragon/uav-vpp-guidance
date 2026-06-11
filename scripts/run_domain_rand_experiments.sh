#!/bin/bash
# Run domain randomization experiments (3 seeds DR + 3 seeds control)
set -e
cd "$(dirname "$0")/.."

LOGDIR=logs/domain_rand_train
mkdir -p "$LOGDIR"

echo "Starting domain randomization training..."
python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml \
  --device cpu --seed 0 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_domain_rand_s0 \
  > "$LOGDIR/s0_domain_rand.log" 2>&1 &
PID1=$!

python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml \
  --device cpu --seed 1 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_domain_rand_s1 \
  > "$LOGDIR/s1_domain_rand.log" 2>&1 &
PID2=$!

python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml \
  --device cpu --seed 2 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_domain_rand_s2 \
  > "$LOGDIR/s2_domain_rand.log" 2>&1 &
PID3=$!

echo "Starting control group training..."
python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --device cpu --seed 0 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_control_s0 \
  > "$LOGDIR/s0_control.log" 2>&1 &
PID4=$!

python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --device cpu --seed 1 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_control_s1 \
  > "$LOGDIR/s1_control.log" 2>&1 &
PID5=$!

python -u -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --device cpu --seed 2 \
  --output-dir outputs/experiments/no_prediction_vpp_ppo_control_s2 \
  > "$LOGDIR/s2_control.log" 2>&1 &
PID6=$!

echo "All 6 training jobs launched:"
echo "  Domain Rand: $PID1 $PID2 $PID3"
echo "  Control:     $PID4 $PID5 $PID6"

wait $PID1
echo "Domain rand seed 0 done"
wait $PID2
echo "Domain rand seed 1 done"
wait $PID3
echo "Domain rand seed 2 done"
wait $PID4
echo "Control seed 0 done"
wait $PID5
echo "Control seed 1 done"
wait $PID6
echo "Control seed 2 done"

echo "All training complete!"
