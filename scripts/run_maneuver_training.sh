#!/usr/bin/env bash
# Maneuvering target training: 3 methods x 1 seed
# Temporarily modifies target_mode, then restores original config

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
METHODS=(
    "train_no_prediction_vpp_ppo:maneuver_no_pred"
    "train_vpp_ppo_cv:maneuver_cv"
    "train_vpp_ppo_ca:maneuver_ca"
)

TOTAL=3
DONE=0

for method_pair in "${METHODS[@]}"; do
    IFS=: read -r config_prefix exp_prefix <<< "$method_pair"
    exp_name="${exp_prefix}_s0"
    config_path="config/experiment/${config_prefix}.yaml"

    DONE=$((DONE + 1))
    echo ""
    echo "========================================"
    echo "RUN $DONE/$TOTAL: $exp_name"
    echo "Config: $config_path | Seed: 0"
    echo "Start: $(date +%H:%M:%S)"
    echo "========================================"

    # Temporarily change target_mode to sinusoidal
    sed -i 's/target_mode: constant_velocity/target_mode: sinusoidal/g' "$config_path"

    $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
        --config "$config_path" \
        --seed 0 \
        --output-dir "outputs/experiments/$exp_name"

    EXIT_CODE=$?

    # Restore original target_mode
    sed -i 's/target_mode: sinusoidal/target_mode: constant_velocity/g' "$config_path"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for $exp_name (exit $EXIT_CODE)"
        continue
    fi

    # Verify checkpoint
    if [ ! -f "outputs/experiments/$exp_name/checkpoints/best.pt" ]; then
        echo "WARNING: Checkpoint missing for $exp_name"
    else
        echo "OK: Checkpoint saved for $exp_name"
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
