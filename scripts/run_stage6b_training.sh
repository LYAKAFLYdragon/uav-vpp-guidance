#!/usr/bin/env bash
# Stage 6B full training: 3 methods x 3 seeds
# Run after smoke test passes

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
SEEDS=(0 1 2)
METHODS=(
    "train_no_prediction_vpp_ppo:stage6b_no_pred"
    "train_vpp_ppo_cv:stage6b_cv"
    "train_vpp_ppo_ca:stage6b_ca"
)

TOTAL=9
DONE=1  # no_pred_s0 already completed

for seed in "${SEEDS[@]}"; do
    for method_pair in "${METHODS[@]}"; do
        IFS=: read -r config_prefix exp_prefix <<< "$method_pair"
        exp_name="${exp_prefix}_s${seed}"
        config_path="config/experiment/${config_prefix}.yaml"

        # Skip already completed run
        if [ "$exp_name" = "stage6b_no_pred_s0" ]; then
            echo "SKIP: $exp_name (already done)"
            continue
        fi

        DONE=$((DONE + 1))
        echo ""
        echo "========================================"
        echo "RUN $DONE/$TOTAL: $exp_name"
        echo "Config: $config_path | Seed: $seed"
        echo "Start: $(date +%H:%M:%S)"
        echo "========================================"

        $PYTHON -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
            --config "$config_path" \
            --seed "$seed" \
            --output-dir "outputs/experiments/$exp_name"

        EXIT_CODE=$?

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
done

echo ""
echo "========================================"
echo "All Stage 6B training complete!"
echo "========================================"
