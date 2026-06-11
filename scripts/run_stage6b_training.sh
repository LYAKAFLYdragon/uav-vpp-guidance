#!/usr/bin/env bash
# Stage 6B full training: 3 methods x 3 seeds
# Run after smoke test passes
#
# Output paths are resolved from config/checkpoint_registry.yaml.

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
REGISTRY="config/checkpoint_registry.yaml"
SEEDS=(0 1 2)

# Config file prefix → registry key mapping
METHODS=(
    "train_no_prediction_vpp_ppo:stage6b_no_pred"
    "train_vpp_ppo_cv:stage6b_cv"
    "train_vpp_ppo_ca:stage6b_ca"
)

TOTAL=9
DONE=1  # no_pred_s0 already completed

for seed in "${SEEDS[@]}"; do
    for method_pair in "${METHODS[@]}"; do
        IFS=: read -r config_prefix registry_key <<< "$method_pair"
        output_dir=$($PYTHON scripts/get_registry_path.py --registry "$REGISTRY" --key "$registry_key" --field output_dir --seed "$seed")
        config_path="config/experiment/${config_prefix}.yaml"

        # Skip already completed run
        if [ "$registry_key" = "stage6b_no_pred" ] && [ "$seed" = "0" ]; then
            echo "SKIP: $registry_key seed=$seed (already done)"
            continue
        fi

        DONE=$((DONE + 1))
        echo ""
        echo "========================================"
        echo "RUN $DONE/$TOTAL: $registry_key seed=$seed"
        echo "Config: $config_path | Output: $output_dir"
        echo "Start: $(date +%H:%M:%S)"
        echo "========================================"

        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$config_path" \
            --seed "$seed" \
            --output-dir "$output_dir"

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: Training failed for $registry_key seed=$seed (exit $EXIT_CODE)"
            continue
        fi

        # Verify checkpoint
        if [ ! -f "$output_dir/checkpoints/best.pt" ]; then
            echo "WARNING: Checkpoint missing for $registry_key seed=$seed"
        else
            echo "OK: Checkpoint saved for $registry_key seed=$seed"
        fi

        echo "End: $(date +%H:%M:%S)"
    done
done

echo ""
echo "========================================"
echo "All Stage 6B training complete!"
echo "========================================"
