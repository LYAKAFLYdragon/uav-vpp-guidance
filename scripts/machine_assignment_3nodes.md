# 3-Machine Task Assignment

## Machine Inventory

| Machine | CPU | RAM | GPU | Role |
|---------|-----|-----|-----|------|
| Machine 1 (local) | 12 cores | 15.4 GB | None | Evaluation + light CPU tasks |
| Machine 2 (desktop) | 16 cores / 24 threads | — | RTX 3090 (CUDA capable) | Predictor training + GPU-friendly tasks |
| Machine 3 (server) | 44 cores / 88 threads | — | GTX 1650 (no CUDA env) | Bulk CPU training |

## Assignment Rationale

- **Machine 3 (server)** runs the largest number of sequential CPU training seeds. High core count ensures OS and training are responsive, but seeds still run one-at-a-time due to Windows memory constraints.
- **Machine 2 (desktop)** runs predictor experiments (LSTM/GRU) where the RTX 3090 can accelerate recurrent networks if CUDA is enabled.
- **Machine 1 (local)** runs evaluation-only tasks and CEM optimization, which are less CPU-intensive.

---

## Machine 1 — Local (Evaluation + CEM)

### Setup
```bash
cd e:/uav-vpp-guidance
python scripts/verify_machine_setup.py
```

### Tasks (no heavy training)
```bash
# F1: Capture region high-res grid (~90 min)
python scripts/compute_capture_region.py \
    --checkpoint outputs/experiments/baseline_10seed_s0/checkpoints/best.pt

# F2: Inference timing benchmark (~10 min)
python scripts/benchmark_inference_time.py \
    --checkpoint outputs/experiments/baseline_10seed_s0/checkpoints/best.pt

# C1: CEM gain optimization (~60 min)
python scripts/run_bilevel_audit.py \
    --config config/experiment/stage6f5_feasible_geometry_constrained.yaml

# E3: Domain randomization robustness evaluation (~60 min)
# Requires E1/E2 checkpoints from Machine 3 to be copied first
python scripts/evaluate_domain_rand_v2.py \
    --dr-checkpoints \
        outputs/experiments/no_prediction_vpp_ppo_domain_rand_s0/checkpoints/best.pt \
        outputs/experiments/no_prediction_vpp_ppo_domain_rand_s1/checkpoints/best.pt \
        outputs/experiments/no_prediction_vpp_ppo_domain_rand_s2/checkpoints/best.pt \
    --control-checkpoints \
        outputs/experiments/no_prediction_vpp_ppo_control_s0/checkpoints/best.pt \
        outputs/experiments/no_prediction_vpp_ppo_control_s1/checkpoints/best.pt \
        outputs/experiments/no_prediction_vpp_ppo_control_s2/checkpoints/best.pt \
    --scales 0.0 0.5 1.0 1.5 2.0 \
    --num-episodes 30

# B4: Crossing generalization grid (~120 min)
python scripts/evaluate_crossing_generalization.py
```

**Estimated Total**: ~6 hours  
**Deliverables**: `docs/results/capture_region_highres/`, `docs/results/inference_time/`, `docs/results/bilevel_audit/`, `docs/results/domain_randomization/`, `docs/results/crossing_generalization/`

---

## Machine 2 — Desktop (Predictors + GPU)

### Setup
```bash
cd <project_root>
git pull origin main

# Optional: enable CUDA for LSTM/GRU if PyTorch with CUDA is installed
python -c "import torch; print(torch.cuda.is_available())"

python scripts/verify_machine_setup.py
```

### Tasks
```bash
# D1-D5: Predictor stratification (50 seeds, ~8h)
# LSTM/GRU will use GPU if torch.cuda.is_available()
python scripts/run_machine_subset.py --experiment-ids D1 D2 D3 D4 D5

# C2: Default gains ablation (10 seeds, ~1h)
python scripts/run_machine_subset.py --experiment-ids C2
```

**Estimated Total**: ~8-9 hours  
**Deliverables**: `outputs/experiments/maneuver_*_s{0-9}/`, `outputs/experiments/default_gains_seed{0-9}/`

### GPU Note
If PyTorch CUDA is installed, add `--device cuda` to training commands for D4/D5. The current `run_machine_subset.py` uses the script's default device (CPU). To force GPU, either:
1. Set environment variable: `export CUDA_VISIBLE_DEVICES=0` and modify configs to use `"device": "cuda"`
2. Or manually run D4/D5 with `--device cuda`

---

## Machine 3 — Server (Bulk CPU Training)

### Setup
```bash
cd <project_root>
git pull origin main
python scripts/verify_machine_setup.py
```

### Tasks
```bash
# A1-A2: Core architecture necessity (20 seeds, ~2.5h)
python scripts/run_machine_subset.py --experiment-ids A1 A2

# B1-B3: Crossing breakthrough (30 seeds, ~3h)
python scripts/run_machine_subset.py --experiment-ids B1 B2 B3

# E1-E2: Domain randomization (20 seeds, ~2.5h)
python scripts/run_machine_subset.py --experiment-ids E1 E2
```

**Estimated Total**: ~8 hours  
**Deliverables**: `outputs/experiments/baseline_10seed_s{0-9}/`, `outputs/experiments/end_to_end_ppo_seed{0-9}/`, `outputs/experiments/constrained_10seed_s{0-9}/`, `outputs/experiments/curriculum_ppo_s{0-9}/`, `outputs/experiments/hybrid_mode_switch_s{0-9}/`, `outputs/experiments/no_prediction_vpp_ppo_domain_rand_s{0-9}/`, `outputs/experiments/no_prediction_vpp_ppo_control_s{0-9}/`

---

## Suggested Execution Order

1. **Start Machine 3 first** — it produces checkpoints needed by Machines 1 and 2 for some evaluations.
2. **Start Machine 2** — predictor training is independent but longest.
3. **Start Machine 1** — some tasks need checkpoints from Machine 3 (domain rand eval, capture region).

## Result Collection

After all machines finish, on the central aggregation machine:

```bash
python scripts/collect_distributed_results.py \
    --machine-dirs /path/to/machine1 /path/to/machine2 /path/to/machine3 \
    --output-dir outputs/aggregated

python scripts/aggregate_10seed_results.py \
    --raw-files outputs/aggregated/results/10seed_evaluation/raw_results*.json \
    --output-dir docs/results/10seed_evaluation

python scripts/aggregate_crossing_generalization.py \
    --raw-files outputs/aggregated/results/crossing_generalization/raw_results*.json \
    --output-dir docs/results/crossing_generalization

python scripts/aggregate_domain_rand.py \
    --raw-files outputs/aggregated/results/domain_randomization/raw_results*.json \
    --output-dir docs/results/domain_randomization

python scripts/aggregate_predictor_stratification.py \
    --raw-files outputs/aggregated/results/predictor_stratification/raw_results*.json \
    --output-dir docs/results/predictor_stratification
```

## Fallback / Rebalancing

If Machine 3 finishes early, move some seeds from Machine 2 (e.g., C2 default gains) to Machine 3.  
If Machine 2 has CUDA issues, run D4/D5 on CPU — they will take longer but still complete.
