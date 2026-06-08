#!/usr/bin/env bash
# Run all core experiments according to task_vpp_bilevel_experiment_plan.md
# Priority: P0-A > P0-B > P1-A > P1-B

set -e

PYTHON="python"

echo "========================================"
echo "Core Experiments Runner"
echo "========================================"
echo "This script executes P0-A, P0-B, P1-A, P1-B in sequence."
echo ""

# P0-A: VPP Ablation (must complete)
echo ">>> Starting P0-A: VPP Ablation"
bash scripts/run_p0a_vpp_ablation.sh
echo ""

# P0-B: Bilevel Ablation (depends on P0-A checkpoint)
echo ">>> Starting P0-B: Bilevel Ablation"
bash scripts/run_p0b_bilevel_ablation.sh
echo ""

# P1-A: Maneuvering Target (parallelizable, but run sequentially here)
echo ">>> Starting P1-A: Maneuvering Target Training"
bash scripts/run_p1a_maneuver_training.sh
echo ""

# P1-B: Neural Predictor on Maneuvering Targets
echo ">>> Starting P1-B: Neural Predictor Training"
bash scripts/run_p1b_neural_predictor_training.sh
echo ""

echo "========================================"
echo "All core experiments complete!"
echo "Next step: run scripts/run_all_evaluations.sh"
echo "========================================"
