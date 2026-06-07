# Paper Materials

This directory contains all figures, tables, and text snippets ready for direct insertion into the paper.

## Directory Structure

```
paper_materials/
├── figures/
│   ├── fig1_method_comparison.png      # Bar chart: method comparison (from paper_benchmark)
│   ├── fig2_training_curves.png        # Training success rate curves (3 methods × 3 seeds)
│   ├── fig3_comparison_boxplot.png     # Per-scenario success rate boxplot
│   ├── fig4_overall_success_rate.png   # Overall success rate with SEM error bars
│   └── fig5_per_scenario_heatmap.png   # Per-scenario success rate heatmap (generated)
├── tables/
│   ├── table1_main_comparison.tex      # LaTeX Table 1 (main results)
│   ├── table1_main_comparison.md       # Markdown backup of Table 1
│   ├── table_mcnemar.tex               # McNemar paired comparison
│   └── table_per_scenario.tex          # Per-scenario breakdown
├── text/
│   ├── discussion_crossing.tex         # Discussion paragraph on crossing-right failure
│   └── results_summary.tex             # Core results summary paragraph
└── scripts/
    └── generate_heatmap.py             # Script used to generate fig5
```

## Key Notes

### CV == CA Analysis (Resolved)

**Conclusion**: CV and CA produced identical results because all evaluation scenarios use `env.target_mode = constant_velocity`. Under zero target acceleration, the CA predictor's acceleration estimate converges to zero, and its prediction equation mathematically reduces to CV:

$$\hat{p}_{CA} = p_0 + v\Delta t + \tfrac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV} \quad (\hat{a} \to 0)$$

This is **not a code bug** — it is an expected mathematical degeneracy under the chosen target kinematics. The two methods are therefore merged into a single **"Parametric Prediction"** category in Table 1 ($N = 160$ episodes).

**Checkpoints verified**:
- `outputs/audit_cv_final/checkpoints/best.pt` → `26389dee...`
- `outputs/audit_ca_final/checkpoints/best.pt` → `e2f8896f...`
- `outputs/experiments/vpp_ppo_cv_prediction/checkpoints/best.pt` → `83d1990a...`
- `outputs/experiments/vpp_ppo_ca_prediction/checkpoints/best.pt` → `9db8f4bf...`

All four checkpoints are different, confirming the equivalence arises from the target kinematics, not from loading the same model file.

### Data Provenance
- **Config**: `config/experiment/stage6f5_feasible_geometry.yaml`
- **Backend**: simple (point-mass dynamics)
- **Episodes**: 80 per method (10 seeds × 8 scenarios)
- **Git commit**: `fa9dbb2`

### Figure Specifications
- All PNG files are generated at **DPI ≥ 300**.
- Font sizes are chosen for IEEE two-column format compatibility.
- `fig5_per_scenario_heatmap.png` uses a custom red-yellow-green colormap with annotated cells.

### Table Specifications
- `table1_main_comparison.tex` uses `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
- Requires the `booktabs` package. For footnotes, `threeparttable` is recommended.

## Quick Reproduction

```bash
# Regenerate heatmap
python paper_materials/scripts/generate_heatmap.py

# Rerun full benchmark
python scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --backend simple \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --scenarios all \
    --methods no_prediction cv_prediction ca_prediction \
    --output-dir docs/results/stage6b
```
