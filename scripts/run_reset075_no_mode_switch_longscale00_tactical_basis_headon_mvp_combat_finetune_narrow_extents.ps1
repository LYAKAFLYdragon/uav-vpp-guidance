param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Seed = 0,
    [string]$PythonExe = "D:\Anaconda3\envs\jsbenv\python.exe",
    [string]$OutputDir = "",
    [switch]$Smoke,
    [string]$ResumeCheckpoint = "",
    [int]$ResumeStep = 0,
    [switch]$ResumeOptimizer
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$config = "config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml"

$args = @(
    "-m", "uav_vpp_guidance.training.train_prediction_vpp_combat_finetune",
    "--config", $config,
    "--seed", "$Seed",
    "--device", $Device,
    "--backend", "jsbsim",
    "--attack-zone-close-range-max-aoa-deg", "60"
)

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $args += @("--output-dir", $OutputDir)
}

if ($Smoke.IsPresent) {
    $args += "--smoke"
}

if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $args += @(
        "--resume", $ResumeCheckpoint,
        "--resume-step", "$ResumeStep"
    )
    if ($ResumeOptimizer.IsPresent) {
        $args += "--resume-optimizer"
    }
}

Write-Host "Running reset075 no-mode-switch longscale00 tactical-basis narrow-extents combat-only finetune..."
& $PythonExe @args
