#!/usr/bin/env bash
# Maneuvering target training: 3 methods x 1 seed
# Temporarily modifies target_mode, then restores original config
#
# Output paths are resolved from config/checkpoint_registry.yaml.

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
REGISTRY="config/checkpoint_registry.yaml"

# Config file prefix → registry key mapping
METHODS=(
    "train_no_prediction_vpp_ppo:p1a_no_pred"
    "train_vpp_ppo_cv:p1a_cv"
    "train_vpp_ppo_ca:p1a_ca"
)

TOTAL=3
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
    echo "Start: $(date +%H:%M:%S)"
    echo "========================================"

    # Temporarily change target_mode to sinusoidal
    sed -i 's/target_mode: constant_velocity/target_mode: sinusoidal/g' "$config_path"

    $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
        --config "$config_path" \
        --seed 0 \
        --output-dir "$output_dir"

    EXIT_CODE=$?

    # Restore original target_mode
    sed -i 's/target_mode: sinusoidal/target_mode: constant_velocity/g' "$config_path"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for $registry_key (exit $EXIT_CODE)"
        continue
    fi

    # Verify checkpoint
    if [ ! -f "$output_dir/checkpoints/best.pt" ]; then
        echo "WARNING: Checkpoint missing for $registry_key"
    else
        echo "OK: Checkpoint saved for $registry_key"
    fi

    echo "End: $(date +%H:%M:%S)"
done

echo ""
echo "========================================"
echo "All maneuvering target training complete!"
echo "Restoring configs..."
echo "========================================"

# Double-check restoration
for method_pair in "${METHODS[@]}"; do
    IFS=: read -r config_prefix _ <<< "$method_pair"
    config_path="config/experiment/${config_prefix}.yaml"
    current_mode=$(grep "target_mode:" "$config_path" | head -1 | awk '{print $2}')
    if [ "$current_mode" = "constant_velocity" ]; then
        echo "  $config_path: OK (constant_velocity)"
    else
        echo "  $config_path: WARNING (current: $current_mode)"
        sed -i 's/target_mode: sinusoidal/target_mode: constant_velocity/g' "$config_path"
    fi
done
