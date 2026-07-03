@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Clean-Worktree Formal Runner: End-to-End Stage
REM ============================================================================

set "MAIN_REPO=E:\uav-vpp-guidance"
set "WORKTREE=E:\uav-vpp-guidance-clean-formal"
if not defined FROZEN_SHA (
    for /f %%I in ('git -C "%MAIN_REPO%" rev-parse HEAD') do set "FROZEN_SHA=%%I"
)
set "RUN_ID=oracle_vs_commander_post_merge_recovery_formal_heldout_end_to_end_20260703"
set "OPPONENT_STAGE=end_to_end"
set "CONFIG=config\experiment\jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_heldout.yaml"
set "COMMANDER_SNAPSHOT_REL=outputs\diagnostics\hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_checkpoint_retrospective_expert_10seed_20260701_checkpoint_snapshots"

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

REM ---------------------------------------------------------------------------
REM Step 0: Precondition check
REM ---------------------------------------------------------------------------
echo [Step 0] Checking preconditions...
set "EXPERT_RESULT=%MAIN_REPO%\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_formal_heldout_expert_20260703"
if not exist "%EXPERT_RESULT%" (
    echo WARNING: Expert stage results not found at:
    echo   %EXPERT_RESULT%
    echo It is recommended to run the expert stage first and verify pass/fail.
    echo.
    choice /C YN /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

REM ---------------------------------------------------------------------------
REM Step 1: Ensure worktree exists at the frozen SHA
REM ---------------------------------------------------------------------------
if not exist "%WORKTREE%\.git" (
    echo [Step 1] Creating clean worktree at %FROZEN_SHA%...
    cd /d "%MAIN_REPO%"
    git worktree add --detach "%WORKTREE%" %FROZEN_SHA%
    if errorlevel 1 (
        echo FAILED: git worktree add failed.
        if not defined CODEX_NONINTERACTIVE pause
        exit /b 1
    )
    echo [Step 1] Worktree created.
) else (
    echo [Step 1] Reusing existing worktree.
    cd /d "%WORKTREE%"
    git checkout --detach %FROZEN_SHA% 2>nul
    echo [Step 1] Checked out %FROZEN_SHA%.
)

REM Verify cleanliness
cd /d "%WORKTREE%"
git status --short >nul 2>nul
if errorlevel 1 (
    echo WARNING: Could not verify worktree cleanliness.
) else (
    git status --short | findstr . >nul
    if not errorlevel 1 (
        echo ERROR: Worktree is not clean. Aborting.
        if not defined CODEX_NONINTERACTIVE pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------------------
REM Step 2: Sync external dependencies
REM ---------------------------------------------------------------------------
echo.
echo [Step 2] Syncing external dependencies...

if not exist "%WORKTREE%\envs\JSBSim" (
    echo   - Copying envs/JSBSim...
    xcopy /E /I /Q /Y "%MAIN_REPO%\envs\JSBSim" "%WORKTREE%\envs\JSBSim"
) else (
    echo   - envs/JSBSim already exists.
)

if exist "%WORKTREE%\outputs\experiments" (
    rmdir /S /Q "%WORKTREE%\outputs\experiments" 2>nul
)
echo   - Linking outputs/experiments (junction)...
mklink /J "%WORKTREE%\outputs\experiments" "%MAIN_REPO%\outputs\experiments" >nul 2>&1
if errorlevel 1 (
    echo   - Junction failed, falling back to xcopy; this may take a while...
    xcopy /E /I /Q /Y "%MAIN_REPO%\outputs\experiments" "%WORKTREE%\outputs\experiments"
)

if exist "%WORKTREE%\outputs\trajectory_prediction" (
    rmdir /S /Q "%WORKTREE%\outputs\trajectory_prediction" 2>nul
)
echo   - Linking outputs/trajectory_prediction (junction)...
mklink /J "%WORKTREE%\outputs\trajectory_prediction" "%MAIN_REPO%\outputs\trajectory_prediction" >nul 2>&1
if errorlevel 1 (
    echo   - Junction failed, falling back to xcopy...
    xcopy /E /I /Q /Y "%MAIN_REPO%\outputs\trajectory_prediction" "%WORKTREE%\outputs\trajectory_prediction"
)

if not exist "%WORKTREE%\outputs\diagnostics" (
    mkdir "%WORKTREE%\outputs\diagnostics"
)
if exist "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" (
    rmdir /S /Q "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" 2>nul
)
echo   - Linking %COMMANDER_SNAPSHOT_REL%...
mklink /J "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%" "%MAIN_REPO%\%COMMANDER_SNAPSHOT_REL%" >nul 2>&1
if errorlevel 1 (
    echo   - Junction failed, falling back to xcopy...
    xcopy /E /I /Q /Y "%MAIN_REPO%\%COMMANDER_SNAPSHOT_REL%" "%WORKTREE%\%COMMANDER_SNAPSHOT_REL%"
)

REM ---------------------------------------------------------------------------
REM Step 2.5: Preflight reproducibility validation
REM ---------------------------------------------------------------------------
echo.
echo [Step 2.5] Validating formal config reproducibility...
D:\Anaconda3\envs\jsbenv\python.exe "%MAIN_REPO%\scripts\preflight_formal_reproducibility.py" ^
  --repo-root "%WORKTREE%" ^
  --config "%CONFIG%"
if errorlevel 1 (
    echo ERROR: Formal reproducibility preflight failed.
    echo The frozen worktree is missing committed dependencies or has an incompatible checkpoint.
    if not defined CODEX_NONINTERACTIVE pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Step 3: Run formal evaluation from the clean worktree
REM ---------------------------------------------------------------------------
echo.
echo [Step 3] Running formal evaluation from clean worktree...
echo   This will take about 20-30 minutes.
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

set "RUN_EXIT=%ERRORLEVEL%"

REM ---------------------------------------------------------------------------
REM Step 4: Copy results back to main repo
REM ---------------------------------------------------------------------------
echo.
echo [Step 4] Copying results back to main repo...
set "RESULT_SRC=%WORKTREE%\outputs\jsbsim_hrl_comparison\%RUN_ID%"
set "RESULT_DST=%MAIN_REPO%\outputs\jsbsim_hrl_comparison\%RUN_ID%"

if exist "%RESULT_SRC%" (
    xcopy /E /I /Q /Y "%RESULT_SRC%" "%RESULT_DST%"
    echo   Results copied to: %RESULT_DST%
) else (
    echo   WARNING: Result source not found: %RESULT_SRC%
)

REM ---------------------------------------------------------------------------
REM Summary
REM ---------------------------------------------------------------------------
echo.
echo ============================================================================
echo Clean-Worktree Formal Runner: End-to-End Stage Complete
echo ============================================================================
if %RUN_EXIT% EQU 0 (
    echo Status: SUCCESS
    echo Check result directory: %RESULT_DST%
) else (
    echo Status: FAILED, exit code %RUN_EXIT%
    echo Check worktree logs: %WORKTREE%\outputs\jsbsim_hrl_comparison\%RUN_ID%
)
echo ============================================================================
echo.

if not defined CODEX_NONINTERACTIVE pause
endlocal
