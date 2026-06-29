param(
    [ValidateSet("expert", "end_to_end")]
    [string]$OpponentStage = "expert",
    [string]$RunId = "",
    [string]$PythonExe = "D:\Anaconda3\envs\jsbenv\python.exe",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$templateConfig = "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best.yaml"
$retrospectiveMethod = "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best"
$checkpointDir = "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents/checkpoints"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "tactical_basis_headon_mvp_combat_finetune_narrow_extents_checkpoint_retrospective_${OpponentStage}_10seed_20260629"
}

$args = @(
    "-u",
    "scripts/run_combat_finetune_checkpoint_retrospective.py",
    "--template-config", $templateConfig,
    "--retrospective-method", $retrospectiveMethod,
    "--checkpoint-dir", $checkpointDir,
    "--run-id", $RunId,
    "--run-status", "formal-small",
    "--backend", "jsbsim",
    "--opponent-stage", $OpponentStage,
    "--tasks", "head_on", "crossing_feasible",
    "--seeds", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489",
    "--attack-zone-close-range-max-aoa-deg", "60",
    "--output-root", $OutputRoot,
    "--python-exe", $PythonExe
)

if ($DryRun.IsPresent) {
    $args += "--dry-run"
}

Write-Host "Running narrow-extents checkpoint retrospective for opponent stage '$OpponentStage'..."
& $PythonExe @args
