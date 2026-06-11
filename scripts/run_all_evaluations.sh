#!/usr/bin/env bash
# Run all evaluations after core experiments complete.
# Usage: bash scripts/run_all_evaluations.sh
#
# Checkpoint paths are resolved from config/checkpoint_registry.yaml via
# --registry-stage. Do NOT hard-code checkpoint paths here.

set -e

PYTHON="python"
EVAL_SEEDS="0 1 2 3 4 5 6 7 8 9"
REGISTRY="config/checkpoint_registry.yaml"

echo "========================================"
echo "Running all evaluations"
echo "========================================"

# Verify registry integrity first
echo ""
echo ">>> Verifying checkpoint registry..."
$PYTHON scripts/verify_checkpoint_registry.py --registry "$REGISTRY" --check-existence
echo "Registry OK."

# --- P0-A: VPP Ablation ---
echo ""
echo ">>> P0-A: VPP Ablation Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0a_vpp_ablation \
    --registry "$REGISTRY" \
    --registry-stage p0a \
    --allow-missing-methods

$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0a_no_vpp_ablation \
    --registry "$REGISTRY" \
    --registry-stage p0a_no_vpp \
    --allow-missing-methods

# --- P0-B: Bilevel Ablation ---
echo ""
echo ">>> P0-B: Bilevel Ablation Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods bilevel \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p0b_bilevel_ablation \
    --registry "$REGISTRY" \
    --registry-stage p0b \
    --allow-missing-methods

# --- P1-A: Maneuvering Target (No-Pred / CV / CA) ---
echo ""
echo ">>> P1-A: Maneuvering Target Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_maneuvering_target.yaml \
    --methods no_prediction cv_prediction ca_prediction \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p1a_maneuver_target \
    --registry "$REGISTRY" \
    --registry-stage p1a \
    --allow-missing-methods

# --- P1-B: Neural Predictors on Maneuvering Target ---
echo ""
echo ">>> P1-B: Neural Predictor Evaluation"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_maneuvering_target.yaml \
    --methods lstm_frozen gru_frozen \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/p1b_neural_maneuver \
    --registry "$REGISTRY" \
    --registry-stage p1b \
    --allow-missing-methods

# --- Stage 6B: Constant velocity baseline (existing results) ---
echo ""
echo ">>> Stage 6B: Constant Velocity Baseline"
$PYTHON scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --methods no_prediction cv_prediction ca_prediction lstm_frozen gru_frozen \
    --seeds $EVAL_SEEDS \
    --output-dir docs/results/stage6b_constant_velocity \
    --registry "$REGISTRY" \
    --registry-stage stage6f5 \
    --allow-missing-methods

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
