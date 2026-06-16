#!/bin/bash
# Wave 1: expand VPP JSBSim mild seeds to 15 total (existing 0-4 + new 5-16).
set -e

mkdir -p outputs/logs/wave1 outputs/experiments

CONFIG="config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml"
STEPS=50000
BACKEND="jsbsim"
DEVICE="cpu"

# Limit per-process threads to avoid oversubscription on the 88-core machine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Batch 1: seeds 5-12 (8 parallel)
for seed in $(seq 5 12); do
    echo "[Wave1] Launch VPP seed $seed"
    python scripts/train_curriculum_ppo.py \
        --config "$CONFIG" \
        --output-dir "outputs/experiments/vpp_50k_s${seed}" \
        --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" \
        --seed "$seed" \
        > "outputs/logs/wave1/vpp_s${seed}.log" 2>&1 &
done
wait
echo "[Wave1] Batch 1 (seeds 5-12) complete."

# Batch 2: seeds 13-16 (4 parallel)
for seed in $(seq 13 16); do
    echo "[Wave1] Launch VPP seed $seed"
    python scripts/train_curriculum_ppo.py \
        --config "$CONFIG" \
        --output-dir "outputs/experiments/vpp_50k_s${seed}" \
        --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" \
        --seed "$seed" \
        > "outputs/logs/wave1/vpp_s${seed}.log" 2>&1 &
done
wait
echo "[Wave1] Batch 2 (seeds 13-16) complete."

# Evaluate all new seeds
for seed in $(seq 5 16); do
    CKPT="outputs/experiments/vpp_50k_s${seed}/checkpoints/best.pt"
    if [ ! -f "$CKPT" ]; then
        CKPT="outputs/experiments/vpp_50k_s${seed}/checkpoints/last.pt"
    fi
    python scripts/eval_checkpoint_scenarios.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --scenarios all --episodes 20 --seed-base $((3000 + seed * 100)) \
        --output "outputs/logs/wave1/eval_s${seed}.json" \
        > "outputs/logs/wave1/eval_s${seed}.log" 2>&1 &
done
wait
echo "[Wave1] Evaluation complete."

# Aggregate
python - <<'PY'
import json, glob, os
rows = []
for path in sorted(glob.glob("outputs/logs/wave1/eval_s*.json")):
    data = json.load(open(path))
    seed = int(os.path.basename(path).replace("eval_s", "").replace(".json", ""))
    for r in data:
        rows.append({"seed": seed, **r})
out = "outputs/wave1_15seed_stats.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Aggregated {len(rows)} scenario-seed records to {out}")
PY
