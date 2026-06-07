# Paper Results

This directory contains all reproducible artifacts for the paper.

## Directory Structure

```
docs/results/
├── paper_benchmark/          # Task 3: Paper benchmark figures and tables
│   ├── figures/
│   │   └── figure1_method_comparison.png
│   ├── tables/
│   │   └── comparison_table.md
│   ├── summary.md            # Human-readable benchmark report
│   ├── results.csv           # Aggregated per-method results
│   ├── raw_episodes.csv      # Episode-level raw data
│   └── run_manifest.json     # Provenance (git hash, CLI, config SHA)
├── stage6b/                  # Task 1: Stage 6B core comparison
│   ├── prediction_metrics.csv
│   ├── summary.md
│   └── cross_seed_summary.json
└── discussion_crossing_paragraph.md  # Task 2: Discussion text
```

## Reproduction Instructions

### Paper Benchmark (Task 3)

```bash
python scripts/run_paper_benchmark.py \
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
    --backend simple \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --scenarios regression \
    --methods no_prediction cv_prediction ca_prediction gain_only \
    --output-dir docs/results/paper_benchmark
```

### Stage 6B Core Comparison (Task 1)

```bash
# 1. Evaluate each training seed (checkpoints already trained)
for seed in 0 1 2; do
    python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
        --config config/experiment/stage6b_comparison.yaml \
        --backend simple \
        --episodes-per-scenario 50 \
        --seeds 0 1 2 \
        --scenarios favorable neutral disadvantage challenging \
        --method-checkpoint no_prediction=outputs/experiments/no_prediction_vpp_ppo_seed${seed}/checkpoints/best.pt \
        --method-checkpoint cv_prediction=outputs/experiments/vpp_ppo_cv_prediction_seed${seed}/checkpoints/best.pt \
        --method-checkpoint ca_prediction=outputs/experiments/vpp_ppo_ca_prediction_seed${seed}/checkpoints/best.pt \
        --output-dir outputs/stage6b/eval_seed${seed}
done

# 2. Aggregate cross-seed results
python scripts/aggregate_stage6b_results.py \
    --input-root outputs/stage6b \
    --output-dir docs/results/stage6b
```

## Git Commit

All results in this directory were generated from commit `fa9dbb2`.
