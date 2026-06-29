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

$config = "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_best_release5.yaml"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_best_release5_${OpponentStage}_10seed_20260629"
}

$args = @(
    "-u",
    "scripts/run_jsbsim_hrl_comparison.py",
    "--config", $config,
    "--run-id", $RunId,
    "--run-status", "formal-small",
    "--tasks", "head_on", "crossing_feasible",
    "--seeds", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489",
    "--backend", "jsbsim",
    "--opponent-stage", $OpponentStage,
    "--attack-zone-close-range-max-aoa-deg", "60",
    "--output-root", $OutputRoot
)

Write-Host "Running release5 inference ablation 10-seed pilot for opponent stage '$OpponentStage'..."
& $PythonExe @args
