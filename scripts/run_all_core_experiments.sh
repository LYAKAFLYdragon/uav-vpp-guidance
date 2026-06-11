#!/usr/bin/env bash
# Run all core experiments according to task_vpp_bilevel_experiment_plan.md
# Priority: P0-A > P0-B > P1-A > P1-B

set -e

PYTHON="python"
REGISTRY="config/checkpoint_registry.yaml"

echo "========================================"
echo "Core Experiments Runner"
echo "========================================"
echo "This script executes P0-A, P0-B, P1-A, P1-B in sequence."
echo ""

# Pre-flight: verify checkpoint registry integrity
echo ">>> Pre-flight: verifying checkpoint registry..."
$PYTHON scripts/verify_checkpoint_registry.py --registry "$REGISTRY"
if [ $? -ne 0 ]; then
    echo "ERROR: Checkpoint registry validation failed. Aborting."
    echo "Run 'python scripts/verify_checkpoint_registry.py' for details."
    exit 1
fi
echo "Registry verification passed."
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
