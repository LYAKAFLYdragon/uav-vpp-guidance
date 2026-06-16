#!/bin/bash
# Wave 2: curriculum/reward ablations for VPP JSBSim mild.
set -e

mkdir -p outputs/logs/wave2 outputs/experiments

STEPS=50000
BACKEND="jsbsim"
DEVICE="cpu"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Launch 4 reward configs in parallel
python scripts/train_curriculum_ppo.py \
    --config config/experiment/wave2_vpp_jsbsim_maneuver_base.yaml \
    --output-dir outputs/experiments/wave2_vpp_jsbsim_maneuver_base \
    --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" --seed 0 \
    > outputs/logs/wave2/base.log 2>&1 &

python scripts/train_curriculum_ppo.py \
    --config config/experiment/wave2_vpp_jsbsim_maneuver_closing.yaml \
    --output-dir outputs/experiments/wave2_vpp_jsbsim_maneuver_closing \
    --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" --seed 0 \
    > outputs/logs/wave2/closing.log 2>&1 &

python scripts/train_curriculum_ppo.py \
    --config config/experiment/wave2_vpp_jsbsim_maneuver_angle.yaml \
    --output-dir outputs/experiments/wave2_vpp_jsbsim_maneuver_angle \
    --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" --seed 0 \
    > outputs/logs/wave2/angle.log 2>&1 &

python scripts/train_curriculum_ppo.py \
    --config config/experiment/wave2_vpp_jsbsim_maneuver_both.yaml \
    --output-dir outputs/experiments/wave2_vpp_jsbsim_maneuver_both \
    --backend "$BACKEND" --device "$DEVICE" --total-timesteps "$STEPS" --seed 0 \
    > outputs/logs/wave2/both.log 2>&1 &

wait
echo "[Wave2] Training complete."

# Evaluate
for cfg in base closing angle both; do
    CKPT="outputs/experiments/wave2_vpp_jsbsim_maneuver_${cfg}/checkpoints/best.pt"
    if [ ! -f "$CKPT" ]; then
        CKPT="outputs/experiments/wave2_vpp_jsbsim_maneuver_${cfg}/checkpoints/last.pt"
    fi
    python scripts/eval_checkpoint_scenarios.py \
        --config "config/experiment/wave2_vpp_jsbsim_maneuver_${cfg}.yaml" \
        --checkpoint "$CKPT" \
        --scenarios all --episodes 20 --seed-base 4000 \
        --output "outputs/logs/wave2/eval_${cfg}.json" \
        > "outputs/logs/wave2/eval_${cfg}.log" 2>&1 &
done
wait
echo "[Wave2] Evaluation complete."

# Aggregate
python - <<'PY'
import json, glob, os
rows = []
for path in sorted(glob.glob("outputs/logs/wave2/eval_*.json")):
    cfg = os.path.basename(path).replace("eval_", "").replace(".json", "")
    for r in json.load(open(path)):
        rows.append({"config": cfg, **r})
out = "outputs/wave2_curriculum_stats.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Aggregated {len(rows)} scenario-config records to {out}")
PY
