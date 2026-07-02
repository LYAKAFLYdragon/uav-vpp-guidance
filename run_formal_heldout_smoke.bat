@echo off
echo Formal Held-Out Smoke Run
echo ==========================
echo.
echo This will run a MINIMAL smoke test with 2 held-out scenarios
echo to validate the runner, artifact pipeline, and provenance recording.
echo Expected runtime: ~2-3 minutes.
echo.

cd /d E:\uav-vpp-guidance

D:\Anaconda3\envs\jsbenv\python.exe scripts\run_jsbsim_hrl_comparison.py ^
  --config config\experiment\jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout.yaml ^
  --output-root outputs\jsbsim_hrl_comparison ^
  --backend jsbsim --run-status smoke --device cpu ^
  --attack-zone-close-range-max-aoa-deg 60 ^
  --run-id oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_smoke_20260701 ^
  --opponent-stage expert ^
  --allow-missing-checkpoints

echo.
echo Smoke run complete. Check output dir:
echo   outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_smoke_20260701\
echo.
pause
