#!/bin/bash
# Crossing v2 finetune training script
# Run this from the repo root.

set -e

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
CONFIG="config/experiment/train_prediction_vpp_ppo_crossing_v2.yaml"

echo "=================================================="
echo "Crossing v2 Finetune"
echo "Config:   ${CONFIG}"
echo "=================================================="

${PYTHON} -m uav_vpp_guidance.training.train_prediction_vpp_combat_finetune \
  --config "${CONFIG}" \
  --device cpu \
  --attack-zone-close-range-max-aoa-deg 60

echo ""
echo "Done."
