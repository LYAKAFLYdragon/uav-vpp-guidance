@echo off
setlocal EnableExtensions
REM ============================================================================
REM Thesis Group H (history-feature control group): one-click train + evaluate
REM
REM   Group H = 16-D base + 3-D target history-motion features, no prediction.
REM   Training  : scripts\run_thesis_h_group_multiseed.py  (seeds 42,43,44)
REM   Evaluation: scripts\eval_thesis_h_group.py           (A0/B protocol)
REM
REM Usage:
REM   run_thesis_h_group.bat                     full run on the GPU workstation
REM   run_thesis_h_group.bat --smoke             smoke run (tiny budget)
REM   run_thesis_h_group.bat <python.exe> [opts] custom interpreter
REM
REM Optional environment overrides (set before calling):
REM   H_GROUP_DEVICE     training device        (default: cuda)
REM   H_GROUP_SEEDS      comma-separated seeds  (default: 42,43,44)
REM   H_GROUP_OUT_ROOT   output root            (default: outputs\experiments\thesis_h_group)
REM   H_GROUP_OPPONENTS  eval opponents         (default: expert = rule-based; thesis protocol)
REM ============================================================================
cd /d "%~dp0"

set "PY_EXE=D:\Anaconda3\envs\py3.9_gpu\python.exe"
set "SMOKE="
if not "%~1"=="" (
  if /i "%~1"=="--smoke" (set "SMOKE=--smoke") else set "PY_EXE=%~1"
)
if /i "%~2"=="--smoke" set "SMOKE=--smoke"

set "DEVICE=cuda"
if defined H_GROUP_DEVICE set "DEVICE=%H_GROUP_DEVICE%"
set "SEEDS=42,43,44"
if defined H_GROUP_SEEDS set "SEEDS=%H_GROUP_SEEDS%"
set "OUT_ROOT=outputs\experiments\thesis_h_group"
if defined H_GROUP_OUT_ROOT set "OUT_ROOT=%H_GROUP_OUT_ROOT%"
set "OPPONENTS=expert"
if defined H_GROUP_OPPONENTS set "OPPONENTS=%H_GROUP_OPPONENTS%"

if not exist "%PY_EXE%" (
  echo [ERROR] Python not found: %PY_EXE%
  exit /b 1
)

set "LOG_DIR=%OUT_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "RUN_TAG=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "RUN_TAG=%RUN_TAG: =0%"
set "LOG=%LOG_DIR%\run_%RUN_TAG%.log"

echo [H-bat] python  = %PY_EXE%
echo [H-bat] device  = %DEVICE%
echo [H-bat] seeds   = %SEEDS%
echo [H-bat] outputs = %OUT_ROOT%
echo [H-bat] smoke   = "%SMOKE%"
echo [H-bat] log     = %LOG%

echo. >> "%LOG%"
echo ================================================================ >> "%LOG%"
echo [H-bat] run_tag=%RUN_TAG% python="%PY_EXE%" device=%DEVICE% seeds=%SEEDS% smoke="%SMOKE%" >> "%LOG%"
echo ================================================================ >> "%LOG%"

echo [H-bat] === step 1/2: training (seeds %SEEDS%) ===
"%PY_EXE%" scripts\run_thesis_h_group_multiseed.py --seeds %SEEDS% --device %DEVICE% --output-root "%OUT_ROOT%" %SMOKE% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [H-bat] TRAINING FAILED - see %LOG%
  exit /b 1
)

echo [H-bat] === step 2/2: evaluation (opponents %OPPONENTS%) ===
"%PY_EXE%" scripts\eval_thesis_h_group.py --seeds %SEEDS% --training-root "%OUT_ROOT%" --opponents %OPPONENTS% %SMOKE% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [H-bat] EVALUATION FAILED - see %LOG%
  exit /b 1
)

echo [H-bat] ALL DONE. Checkpoints + eval metrics under %OUT_ROOT%
exit /b 0
