#!/usr/bin/env bash
# P0-B: Bilevel Optimization Ablation Experiment
# Compare single-layer PPO vs bilevel strategy-gain optimization
#
# Checkpoint and output paths are resolved from config/checkpoint_registry.yaml.

set -e

PYTHON="python"
REGISTRY="config/checkpoint_registry.yaml"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

# Resolve paths from registry
SINGLE_CKPT=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key no_prediction_vpp_ppo --field checkpoint)
OUT_BILEVEL=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key p0b_bilevel --field output_dir)

echo "========================================"
echo "P0-B: Bilevel Ablation"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

# --- Single-layer baseline (already trained) ---
if [ ! -f "$SINGLE_CKPT" ]; then
    echo "ERROR: Single-layer checkpoint not found: $SINGLE_CKPT"
    echo "Please run P0-A VPP training first or ensure the canonical model exists."
    exit 1
fi
echo "Single-layer checkpoint: $SINGLE_CKPT"

# --- Bilevel group ---
CONFIG_BILEVEL="config/experiment/proposed_bilevel.yaml"

echo ""
echo "[1/1] Training bilevel optimization..."
echo "Config: $CONFIG_BILEVEL"
echo "Output: $OUT_BILEVEL"

if [ -f "$OUT_BILEVEL/bilevel_results.json" ]; then
    echo "Bilevel results already exist, skipping training."
else
    $PYTHON -m uav_vpp_guidance.training.train_bilevel \
        --config "$CONFIG_BILEVEL" \
        --checkpoint "$SINGLE_CKPT" \
        --n-episodes 200 \
        --outer-every 10 \
        --inner-iter 20 \
        --output-dir "$OUT_BILEVEL"
fi

echo ""
echo "========================================"
echo "P0-B Bilevel Ablation complete!"
echo "Single-layer: $SINGLE_CKPT"
echo "Bilevel:      $OUT_BILEVEL"
echo "========================================"
