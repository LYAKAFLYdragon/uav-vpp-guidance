#!/usr/bin/env bash
# Run all evaluations after core experiments complete.
# Usage: bash scripts/run_all_evaluations.sh

set -e

PYTHON="python"
EVAL_SEEDS="0 1 2 3 4 5 6 7 8 9"

echo "========================================"
echo "Running all evaluations"
echo "========================================"

# --- P0-A: VPP Ablation ---
echo ""
echo ">>> P0-A: VPP Ablation Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0a_vpp_ablation \
    --checkpoint-map no_prediction=outputs/experiments/p0a_vpp_s0/checkpoints/best.pt \
    --allow-missing-methods

$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0a_no_vpp_ablation \
    --checkpoint-map no_prediction=outputs/experiments/p0a_no_vpp_s0/checkpoints/best.pt \
    --allow-missing-methods

# --- P0-B: Bilevel Ablation ---
echo ""
echo ">>> P0-B: Bilevel Ablation Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods bilevel \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0b_bilevel_ablation \
    --checkpoint-map bilevel=outputs/experiments/p0b_bilevel_s0/checkpoints/best.pt \
    --allow-missing-methods

# --- P1-A: Maneuvering Target (No-Pred / CV / CA) ---
echo ""
echo ">>> P1-A: Maneuvering Target Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_maneuvering_target.yaml \
    --methods no_prediction cv_prediction ca_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p1a_maneuver_target \
    --checkpoint-map \
        no_prediction=outputs/experiments/maneuver_no_pred_s0/checkpoints/best.pt,\
        cv_prediction=outputs/experiments/maneuver_cv_s0/checkpoints/best.pt,\
        ca_prediction=outputs/experiments/maneuver_ca_s0/checkpoints/best.pt \
    --allow-missing-methods

# --- P1-B: Neural Predictors on Maneuvering Target ---
echo ""
echo ">>> P1-B: Neural Predictor Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_maneuvering_target.yaml \
    --methods lstm_frozen gru_frozen \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p1b_neural_maneuver \
    --checkpoint-map \
        lstm_frozen=outputs/experiments/maneuver_lstm_s0/checkpoints/best.pt,\
        gru_frozen=outputs/experiments/maneuver_gru_s0/checkpoints/best.pt \
    --allow-missing-methods

# --- Stage 6B: Constant velocity baseline (existing results) ---
echo ""
echo ">>> Stage 6B: Constant Velocity Baseline"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction cv_prediction ca_prediction lstm_frozen gru_frozen \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/stage6b_constant_velocity \
    --allow-missing-methods

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
