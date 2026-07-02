@echo off
echo Formal Held-Out Evaluation: End-to-End Stage
echo ================================================
echo.
echo This runs the FULL formal held-out evaluation with opponent_stage=end_to_end.
echo 36 head_on + 4 crossing scenarios = 40 episodes per method.
echo Expected runtime: ~30-40 minutes (requires manual batch execution).
echo.
echo Output will be written to:
echo   outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_end_to_end_20260701\
echo.

cd /d E:\uav-vpp-guidance

D:\Anaconda3\envs\jsbenv\python.exe scripts\run_jsbsim_hrl_comparison.py ^
  --config config\experiment\jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout.yaml ^
  --output-root outputs\jsbsim_hrl_comparison ^
  --backend jsbsim --run-status formal --device cpu ^
  --attack-zone-close-range-max-aoa-deg 60 ^
  --run-id oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_end_to_end_20260701 ^
  --opponent-stage end_to_end

echo.
echo End-to-end stage complete. Check output dir:
echo   outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_end_to_end_20260701\
echo.
pause
