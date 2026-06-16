#!/usr/bin/env bash
# Small-batch validation: 2 seeds, core architectures + clamped VPP
set -e
DEVICE=cpu

echo "=== VPP full (2 seeds, 200k) ==="
for seed in 0 1; do
  python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --seed $seed \
    --output-dir outputs/validate_vpp_2seed/seed_$seed \
    --device $DEVICE
done

echo "=== No-VPP baseline (2 seeds, 200k) ==="
python scripts/run_no_vpp_baseline.py \
  --config config/experiment/train_no_vpp_ppo.yaml \
  --seeds 2 \
  --output-root outputs/experiments \
  --exp-name no_vpp_validate_2seed \
  --device $DEVICE

echo "=== VPP clamped +/-100m (2 seeds, 200k) ==="
python scripts/run_vpp_clamped_ablation.py \
  --base-config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --seeds 2 \
  --long-limit 100 --lat-limit 100 --vert-limit 100 \
  --output-root outputs/validate_vpp_clamped_2seed \
  --device $DEVICE

echo "=== Core validation complete ==="
