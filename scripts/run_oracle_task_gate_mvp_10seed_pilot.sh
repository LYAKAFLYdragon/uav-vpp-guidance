#!/bin/bash
# Oracle Task Gate MVP 10-seed pilot (canonical AoA 60 combat-only scope)
# Run this from the repo root.

set -e

PYTHON="/d/Anaconda3/envs/jsbenv/python.exe"
CONFIG="config/experiment/jsbsim_hrl_oracle_task_gate_mvp.yaml"
RUN_STATUS="formal"
N_EPISODES=1
METHODS="oracle_task_gate"
TASKS="head_on crossing_feasible"

# Parse arguments
OPPONENT_STAGE="${1:-expert}"
RUN_ID_SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"

RUN_ID="oracle_task_gate_mvp_${OPPONENT_STAGE}_10seed_${RUN_ID_SUFFIX}"

echo "=================================================="
echo "Oracle Task Gate MVP Pilot (AoA 60 scope)"
echo "Config:   ${CONFIG}"
echo "Opponent: ${OPPONENT_STAGE}"
echo "Run ID:   ${RUN_ID}"
echo "Methods:  ${METHODS}"
echo "Tasks:    ${TASKS}"
echo "Seeds:    0-9"
echo "=================================================="

${PYTHON} scripts/run_jsbsim_hrl_comparison.py \
  --config "${CONFIG}" \
  --run-id "${RUN_ID}" \
  --methods ${METHODS} \
  --tasks ${TASKS} \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --opponent-stage "${OPPONENT_STAGE}" \
  --run-status "${RUN_STATUS}" \
  --n-episodes ${N_EPISODES} \
  --attack-zone-close-range-max-aoa-deg 60 \
  --device cpu

echo ""
echo "Done. Output: outputs/jsbsim_hrl_comparison/${RUN_ID}/"
