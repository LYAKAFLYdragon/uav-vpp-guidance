#!/usr/bin/env bash
# P1-A: Maneuvering Target Training
# Train No-Pred, CV, CA under sinusoidal target motion
#
# Output paths are resolved from config/checkpoint_registry.yaml.

set -e

PYTHON="python"
REGISTRY="config/checkpoint_registry.yaml"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "========================================"
echo "P1-A: Maneuvering Target Training"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

# Config file prefix → registry key mapping
METHODS=(
    "train_no_prediction_vpp_ppo_maneuver:p1a_no_pred"
    "train_vpp_ppo_cv_maneuver:p1a_cv"
    "train_vpp_ppo_ca_maneuver:p1a_ca"
)

TOTAL=${#METHODS[@]}
DONE=0

for method_pair in "${METHODS[@]}"; do
    IFS=: read -r config_prefix registry_key <<< "$method_pair"
    output_dir=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key "$registry_key" --field output_dir)
    config_path="config/experiment/${config_prefix}.yaml"

    DONE=$((DONE + 1))
    echo ""
    echo "========================================"
    echo "RUN $DONE/$TOTAL: $registry_key"
    echo "Config: $config_path | Output: $output_dir"
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
        echo "WARNING: Checkpoint missing for $registry_key"
    else
        echo "OK: Checkpoint saved for $registry_key"
    fi
done

echo ""
echo "========================================"
echo "P1-A Maneuvering Target Training complete!"
echo "========================================"
