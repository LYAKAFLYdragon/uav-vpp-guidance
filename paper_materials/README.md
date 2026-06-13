# Paper Materials

This directory contains all figures, tables, and text snippets for the paper, now updated with **real training results** from scratch-trained models.

## Real Training Results (Updated)

All models were trained from scratch for **200,000 PPO steps** under identical configuration (`stage6f5_feasible_geometry.yaml`, `target_mode = constant_velocity`).

### Main Finding: Functional Equivalence

| Method | Success Rate | Mean Return | p vs Baseline | Cohen's d |
|--------|-------------|-------------|---------------|-----------|
| No-Prediction | **62.50%** | −113.64 ± 403.78 | — | — |
| Parametric Prediction (CV/CA) | **62.50%** | −114.24 ± 404.44 | 0.0000* | −0.769 (medium) |

**Key insight**: All three methods achieve identical success rates and identical per-scenario success patterns. The equivalence is kinematically expected:
- CA reduces to CV when target acceleration is zero: $\hat{p}_{CA} = p_0 + v\Delta t + \frac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV}$ as $\hat{a} \to 0$
- No-prediction tracks the instantaneous target position, achieving the same terminal outcomes
- Predictor choice affects return distribution (efficiency) but not binary success under constant-velocity targets

## Directory Structure

```
paper_materials/
├── paper.tex                                  # Complete LaTeX paper draft
├── figures/
│   ├── fig1_method_comparison.png             # Bar chart from benchmark
│   ├── fig2_training_curves.png               # Training curves (legacy)
│   ├── fig3_comparison_boxplot.png            # Boxplot (legacy)
│   ├── fig4_overall_success_rate.png          # Overall success rate with SEM
│   ├── fig5_per_scenario_heatmap.png          # Per-scenario heatmap (legacy)
│   ├── fig_inference_benchmark.png            # PPO vs CEM latency (new)
│   └── fig_method_innovation_comparison.png   # Method-innovation bar chart (new)
├── tables/
│   ├── table1_main_comparison.tex             # LaTeX Table 1 (legacy real results)
│   ├── table1_main_comparison.md              # Markdown backup
│   ├── table_mcnemar.tex                      # McNemar paired comparison
│   ├── table_per_scenario.tex                 # Per-scenario breakdown
│   └── table_method_innovation_comparison.tex # Method-innovation comparison (new)
├── text/
│   ├── discussion_crossing.tex                # Rewritten: functional equivalence
│   └── results_summary.tex                    # Rewritten: real results summary
└── scripts/
    ├── generate_heatmap.py                    # Legacy heatmap
    ├── plot_inference_benchmark.py            # PPO/CEM latency figure
    ├── plot_method_innovation_comparison.py   # Method-innovation figure
    └── generate_all_figures.py                # Regenerate all figures
```

## Data Provenance

- **Training config**: `config/experiment/stage6f5_feasible_geometry.yaml`
- **Training steps**: 200,000 per method
- **Seeds trained**: 3 per method (best seed selected for benchmark)
- **Selected checkpoints**:
  - No-Prediction: `outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt`
  - CV Prediction: `outputs/experiments/stage6b_cv_s1/checkpoints/best.pt`
  - CA Prediction: `outputs/experiments/stage6b_ca_s1/checkpoints/best.pt`
- **Benchmark config**: `config/experiment/stage6f5_feasible_geometry.yaml`
- **Backend**: simple (point-mass dynamics)
- **Eval episodes**: 80 per method (10 seeds × 8 scenarios)
- **Benchmark output**: `docs/results/stage6b_real_80eps_new2/`
- **Git commit**: `0c83e0d` (with subsequent bug fixes)

## Bug Fixes During Real Training

1. **CUDA/CPU device mismatch** in `policy_network.py`: `action_scale` and `action_bias` are now registered as buffers so they move to the correct device with the model.
2. **Checkpoint compatibility** in `ppo_agent.py`: `load_state_dict(..., strict=False)` allows loading both old and new checkpoint formats.

## Quick Reproduction

```bash
# Regenerate all figures (legacy + current)
python paper_materials/scripts/generate_all_figures.py

# Re-run benchmark with real checkpoints
python scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --backend simple \
    --methods no_prediction cv_prediction ca_prediction \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --scenarios all \
    --output-dir docs/results/stage6b_real_80eps_new2 \
    --checkpoint-map no_prediction=outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt \
    --checkpoint-map cv_prediction=outputs/experiments/stage6b_cv_s1/checkpoints/best.pt \
    --checkpoint-map ca_prediction=outputs/experiments/stage6b_ca_s1/checkpoints/best.pt

# Compile paper
cd paper_materials && pdflatex paper.tex
```
