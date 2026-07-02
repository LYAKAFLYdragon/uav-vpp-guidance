@echo off
echo Formal Held-Out Evaluation: Expert Stage
echo ==========================================
echo.
echo This runs the FULL formal held-out evaluation with opponent_stage=expert.
echo 36 head_on + 4 crossing scenarios = 40 episodes per method.
echo Expected runtime: ~15-20 minutes.
echo.
echo Output will be written to:
echo   outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_expert_20260701\
echo.

cd /d E:\uav-vpp-guidance

D:\Anaconda3\envs\jsbenv\python.exe scripts\run_jsbsim_hrl_comparison.py ^
  --config config\experiment\jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout.yaml ^
  --output-root outputs\jsbsim_hrl_comparison ^
  --backend jsbsim --run-status formal --device cpu ^
  --attack-zone-close-range-max-aoa-deg 60 ^
  --run-id oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_expert_20260701 ^
  --opponent-stage expert

echo.
echo Expert stage complete. Check output dir:
echo   outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_expert_20260701\
echo.
pause
