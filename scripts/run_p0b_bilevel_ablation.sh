#!/usr/bin/env bash
# P0-B: Bilevel Optimization Ablation Experiment
# Compare single-layer PPO vs bilevel strategy-gain optimization

set -e

PYTHON="python"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "========================================"
echo "P0-B: Bilevel Ablation"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

# --- Single-layer baseline (already trained) ---
SINGLE_CKPT="outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt"
if [ ! -f "$SINGLE_CKPT" ]; then
    echo "ERROR: Single-layer checkpoint not found: $SINGLE_CKPT"
    echo "Please run P0-A VPP training first."
    exit 1
fi
echo "Single-layer checkpoint: $SINGLE_CKPT"

# --- Bilevel group ---
CONFIG_BILEVEL="config/experiment/proposed_bilevel.yaml"
OUT_BILEVEL="outputs/experiments/p0b_bilevel_s0"

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
