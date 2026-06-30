param(
    [ValidateSet("expert", "end_to_end")]
    [string]$OpponentStage = "expert",
    [string]$RunId = "",
    [string]$PythonExe = "D:\Anaconda3\envs\jsbenv\python.exe",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$config = "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_${OpponentStage}_3seed_smoke_20260630"
}

$args = @(
    "-u",
    "scripts/run_jsbsim_hrl_comparison.py",
    "--config", $config,
    "--run-id", $RunId,
    "--run-status", "smoke",
    "--tasks", "head_on", "crossing_feasible",
    "--seeds", "480", "481", "482",
    "--backend", "jsbsim",
    "--opponent-stage", $OpponentStage,
    "--attack-zone-close-range-max-aoa-deg", "60",
    "--output-root", $OutputRoot
)

Write-Host "Running legacy hierarchical bridge 3-seed smoke for opponent stage '$OpponentStage'..."
& $PythonExe @args
