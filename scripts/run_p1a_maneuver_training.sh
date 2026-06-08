#!/usr/bin/env bash
# P1-A: Maneuvering Target Training
# Train No-Pred, CV, CA under sinusoidal target motion

set -e

PYTHON="python"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "========================================"
echo "P1-A: Maneuvering Target Training"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

METHODS=(
    "train_no_prediction_vpp_ppo_maneuver:maneuver_no_pred"
    "train_vpp_ppo_cv_maneuver:maneuver_cv"
    "train_vpp_ppo_ca_maneuver:maneuver_ca"
)

TOTAL=${#METHODS[@]}
DONE=0

for method_pair in "${METHODS[@]}"; do
    IFS=: read -r config_prefix exp_prefix <<< "$method_pair"
    exp_name="${exp_prefix}_s0"
    config_path="config/experiment/${config_prefix}.yaml"
    output_dir="outputs/experiments/$exp_name"

    DONE=$((DONE + 1))
    echo ""
    echo "========================================"
    echo "RUN $DONE/$TOTAL: $exp_name"
    echo "Config: $config_path | Seed: 0"
    echo "========================================"

    if [ -f "$output_dir/checkpoints/best.pt" ]; then
        echo "Checkpoint already exists, skipping training."
        continue
    fi

    if [ "$CUDA_AVAILABLE" != "True" ]; then
        sed -i 's/device: cuda/device: cpu/g' "$config_path"
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$config_path" \
            --seed 0 \
            --output-dir "$output_dir"
        sed -i 's/device: cpu/device: cuda/g' "$config_path"
    else
        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$config_path" \
            --seed 0 \
            --output-dir "$output_dir"
    fi

    if [ ! -f "$output_dir/checkpoints/best.pt" ]; then
        echo "WARNING: Checkpoint missing for $exp_name"
    else
        echo "OK: Checkpoint saved for $exp_name"
    fi
done

echo ""
echo "========================================"
echo "P1-A Maneuvering Target Training complete!"
echo "========================================"
