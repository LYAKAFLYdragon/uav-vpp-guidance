#!/usr/bin/env bash
# P1-B: Neural Predictor (LSTM/GRU) Training on Maneuvering Targets
#
# Output paths are resolved from config/checkpoint_registry.yaml.

set -e

PYTHON="python"
REGISTRY="config/checkpoint_registry.yaml"
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "========================================"
echo "P1-B: Neural Predictor Maneuver Training"
echo "CUDA available: $CUDA_AVAILABLE"
echo "========================================"

# Config file prefix → registry key mapping
METHODS=(
    "train_vpp_ppo_lstm_frozen_maneuver:p1b_lstm"
    "train_vpp_ppo_gru_frozen_maneuver:p1b_gru"
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

    # LSTM/GRU configs already use device: cpu, but just in case
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
echo "P1-B Neural Predictor Training complete!"
echo "========================================"
