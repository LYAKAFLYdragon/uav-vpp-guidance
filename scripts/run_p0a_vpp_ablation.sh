#!/usr/bin/env bash
# P0-A: VPP Ablation Experiment
# Compare VPP-enabled policy vs direct-command (no VPP) policy
#
# Output paths are resolved from config/checkpoint_registry.yaml.

set -e

PYTHON="python"
REGISTRY="config/checkpoint_registry.yaml"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

# Resolve output paths from registry
OUT_VPP=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key p0a_vpp --field output_dir)
OUT_NO_VPP=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key p0a_no_vpp --field output_dir)

echo "========================================"
echo "P0-A: VPP Ablation"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

# --- VPP Group (baseline) ---
CONFIG_VPP="config/experiment/train_no_prediction_vpp_ppo.yaml"

echo ""
echo "[1/2] Training VPP group..."
echo "Config: $CONFIG_VPP"
echo "Output: $OUT_VPP"

if [ -f "$OUT_VPP/checkpoints/best.pt" ]; then
    echo "Checkpoint already exists, skipping training."
else
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        sed -i 's/device: cuda/device: cpu/g' "$CONFIG_VPP"
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$CONFIG_VPP" \
            --seed 0 \
            --output-dir "$OUT_VPP"
        sed -i 's/device: cpu/device: cuda/g' "$CONFIG_VPP"
    else
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$CONFIG_VPP" \
            --seed 0 \
            --output-dir "$OUT_VPP"
    fi
fi

# --- No-VPP Group (zero-offset baseline) ---
CONFIG_NO_VPP="config/experiment/train_no_vpp_direct_command.yaml"

echo ""
echo "[2/2] Training No-VPP group (zero-offset baseline)..."
echo "Config: $CONFIG_NO_VPP"
echo "Output: $OUT_NO_VPP"

if [ -f "$OUT_NO_VPP/checkpoints/best.pt" ]; then
    echo "Checkpoint already exists, skipping training."
else
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        sed -i 's/device: cuda/device: cpu/g' "$CONFIG_NO_VPP"
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$CONFIG_NO_VPP" \
            --seed 0 \
            --output-dir "$OUT_NO_VPP"
        sed -i 's/device: cpu/device: cuda/g' "$CONFIG_NO_VPP"
    else
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$CONFIG_NO_VPP" \
            --seed 0 \
            --output-dir "$OUT_NO_VPP"
    fi
fi

echo ""
echo "========================================"
echo "P0-A VPP Ablation complete!"
echo "VPP:      $OUT_VPP"
echo "No-VPP:   $OUT_NO_VPP"
echo "========================================"
