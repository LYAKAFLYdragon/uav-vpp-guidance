@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Clean-Worktree Formal Runner: End-to-End Stage
REM ============================================================================
REM Purpose: Run formal held-out evaluation from a CLEAN git worktree at the
REM frozen SHA. This ensures paper-safe=true regardless of the
REM dirty state of the main working tree.
REM
REM Precondition: Expert stage must have passed the pre-registered pass/fail
REM criterion (commander win_rate >= oracle win_rate) before running end_to-end.
REM
REM External dependencies NOT in git (must be synced):
REM   - envs/JSBSim        (JSBSim aircraft data)
REM   - outputs/experiments  (training checkpoints)
REM   - outputs/trajectory_prediction (predictor checkpoints)
REM   - outputs/diagnostics/...checkpoint_snapshots (retrospective-selected commander checkpoint)
REM
REM Run-time: ~30-40 minutes (36 head-on + 4 crossing scenarios)
REM ============================================================================

SET "MAIN_REPO=E:\uav-vpp-guidance"
SET "WORKTREE=E:\uav-vpp-guidance-clean-formal"
IF NOT DEFINED FROZEN_SHA (
    FOR /F %%I IN ('git -C "%MAIN_REPO%" rev-parse HEAD') DO SET "FROZEN_SHA=%%I"
)
SET "RUN_ID=oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_end_to_end_20260701"
SET "OPPONENT_STAGE=end_to_end"
SET "CONFIG=config\experiment\jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout.yaml"
SET "COMMANDER_SNAPSHOT_REL=outputs\diagnostics\hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_checkpoint_retrospective_expert_10seed_20260701_checkpoint_snapshots"

echo.
echo ============================================================================
echo Clean-Worktree Formal Runner: End-to-End Stage
echo ============================================================================
echo Main repo:  %MAIN_REPO%
echo Worktree:   %WORKTREE%
echo Frozen SHA: %FROZEN_SHA%
echo Run ID:     %RUN_ID%
echo Opponent:   %OPPONENT_STAGE%
echo ============================================================================
echo.

:: ---------------------------------------------------------------------------
:: Step 0: Precondition check
:: ---------------------------------------------------------------------------
echo [Step 0] Checking preconditions...
SET "EXPERT_RESULT=%MAIN_REPO%\outputs\jsbsim_hrl_comparison\oracle_vs_secondary_clamp_fresh_geometry_formal_heldout_expert_20260701"
IF NOT EXIST "%EXPERT_RESULT%" (
    echo WARNING: Expert stage results not found at:
    echo   %EXPERT_RESULT%
    echo It is recommended to run the expert stage first and verify pass/fail.
    echo.
    choice /C YN /M "Continue anyway?"
    IF ERRORLEVEL 2 exit /b 1
)

:: ---------------------------------------------------------------------------
:: Step 1: Ensure worktree exists at the frozen SHA
:: ---------------------------------------------------------------------------
IF NOT EXIST "%WORKTREE%" (
    echo [Step 1] Creating clean worktree at %FROZEN_SHA%...
    cd /d "%MAIN_REPO%"
    git worktree add --detach "%WORKTREE%" %FROZEN_SHA%
    IF ERRORLEVEL 1 (
        echo FAILED: git worktree add failed.
        pause
        exit /b 1
    )
    echo [Step 1] Worktree created.
) ELSE (
    echo [Step 1] Reusing existing worktree.
    cd /d "%WORKTREE%"
    git checkout --detach %FROZEN_SHA% 2>nul
    echo [Step 1] Checked out %FROZEN_SHA%.
)

:: Verify cleanliness
cd /d "%WORKTREE%"
git status --short >nul 2>nul
IF ERRORLEVEL 1 (
    echo WARNING: Could not verify worktree cleanliness.
) ELSE (
    git status --short > _git_status.tmp
    FOR /F "usebackq" %%A IN (`type _git_status.tmp ^| find /c /v ""`) DO SET "STATUS_LINES=%%A"
    del _git_status.tmp 2>nul
    IF %STATUS_LINES% GTR 0 (
        echo ERROR: Worktree is not clean. Aborting.
        pause
        exit /b 1
    )
)

:: ---------------------------------------------------------------------------
:: Step 2: Sync external dependencies (NOT in git)
:: ---------------------------------------------------------------------------
echo.
echo [Step 2] Syncing external dependencies...

:: 2a: envs/JSBSim (aircraft data, not tracked by git)
IF NOT EXIST "%WORKTREE%\envs\JSBSim" (
    echo   - Copying envs/JSBSim...
    xcopy /E /I /Q "%MAIN_REPO%\envs\JSBSim" "%WORKTREE%\envs\JSBSim"
) ELSE (
    echo   - envs/JSBSim already exists.
)

:: 2b: outputs/experiments (training checkpoints, not tracked by git)
IF EXIST "%WORKTREE%\outputs\experiments" (
    rmdir /S /Q "%WORKTREE%\outputs\experiments" 2>nul
)
echo   - Linking outputs/experiments (junction)...
mklink /J "%WORKTREE%\outputs\experiments" "%MAIN_REPO%\outputs\experiments" >nul 2>&1
IF ERRORLEVEL 1 (
    echo   - Junction failed, falling back to xcopy (this may take a while)...
    xcopy /E /I /Q "%MAIN_REPO%\outputs\experiments" "%WORKTREE%\outputs\experiments"
)

:: 2c: outputs/trajectory_prediction (predictor checkpoints)
IF EXIST "%WORKTREE%\outputs\trajectory_prediction" (
    rmdir /S /Q "%WORKTREE%\outputs\trajectory_prediction" 2>nul
)
echo   - Linking outputs/trajectory_prediction (junction)...
mklink /J "%WORKTREE%\outputs\trajectory_prediction" "%MAIN_REPO%\outputs\trajectory_prediction" >nul 2>&1
IF ERRORLEVEL 1 (
    echo   - Junction failed, falling back to xcopy...
    xcopy /E /I /Q "%MAIN_REPO%\outputs\trajectory_prediction" "%WORKTREE%\outputs\trajectory_prediction"
)

:: 2d: retrospective-selected commander checkpoint snapshot
IF NOT EXIST "%WORKTREE%\outputs\diagnostics" (
    mkdir "%WORKTREE%\outputs\diagnostics"
)
IF EXIST "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" (
    rmdir /S /Q "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" 2>nul
)
echo   - Linking %COMMANDER_SNAPSHOT_REL%...
mklink /J "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" "%MAIN_REPO%\%COMMANDER_SNAPSHOT_REL%" >nul 2>&1
IF ERRORLEVEL 1 (
    echo   - Junction failed, falling back to xcopy...
    xcopy /E /I /Q "%MAIN_REPO%\%COMMANDER_SNAPSHOT_REL%" "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%"
)

:: ---------------------------------------------------------------------------
:: Step 2.5: Preflight reproducibility validation
:: ---------------------------------------------------------------------------
echo.
echo [Step 2.5] Validating formal config reproducibility...
D:\Anaconda3\envs\jsbenv\python.exe "%MAIN_REPO%\scripts\preflight_formal_reproducibility.py" ^
  --repo-root "%WORKTREE%" ^
  --config "%CONFIG%"
IF ERRORLEVEL 1 (
    echo ERROR: Formal reproducibility preflight failed.
    echo The frozen worktree is missing committed dependencies or has an incompatible checkpoint.
    pause
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Step 3: Run formal evaluation from the clean worktree
:: ---------------------------------------------------------------------------
echo.
echo [Step 3] Running formal evaluation from clean worktree...
echo   This will take ~30-40 minutes.
echo.

cd /d "%WORKTREE%"
D:\Anaconda3\envs\jsbenv\python.exe scripts\run_jsbsim_hrl_comparison.py ^
  --config %CONFIG% ^
  --output-root outputs\jsbsim_hrl_comparison ^
  --backend jsbsim ^
  --run-status formal ^
  --device cpu ^
  --attack-zone-close-range-max-aoa-deg 60 ^
  --run-id %RUN_ID% ^
  --opponent-stage %OPPONENT_STAGE%

SET "RUN_EXIT=%ERRORLEVEL%"

:: ---------------------------------------------------------------------------
:: Step 4: Copy results back to main repo
:: ---------------------------------------------------------------------------
echo.
echo [Step 4] Copying results back to main repo...
SET "RESULT_SRC=%WORKTREE%\outputs\jsbsim_hrl_comparison\%RUN_ID%"
SET "RESULT_DST=%MAIN_REPO%\outputs\jsbsim_hrl_comparison\%RUN_ID%"

IF EXIST "%RESULT_SRC%" (
    xcopy /E /I /Q "%RESULT_SRC%" "%RESULT_DST%"
    echo   Results copied to: %RESULT_DST%
) ELSE (
    echo   WARNING: Result source not found: %RESULT_SRC%
)

:: ---------------------------------------------------------------------------
:: Summary
:: ---------------------------------------------------------------------------
echo.
echo ============================================================================
echo Clean-Worktree Formal Runner: End-to-End Stage — Complete
echo ============================================================================
IF %RUN_EXIT% EQU 0 (
    echo Status: SUCCESS
    echo Check result directory: %RESULT_DST%
) ELSE (
    echo Status: FAILED (exit code %RUN_EXIT%)
    echo Check worktree logs: %WORKTREE%\outputs\jsbsim_hrl_comparison\%RUN_ID%
)
echo ============================================================================
echo.

pause
endlocal
