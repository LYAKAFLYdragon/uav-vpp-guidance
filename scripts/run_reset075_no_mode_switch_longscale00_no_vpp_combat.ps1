param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Seed = 0,
    [string]$PythonExe = "D:\Anaconda3\envs\jsbenv\python.exe",
    [string]$OutputDir = "",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$config = "config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml"

$args = @(
    "-m", "uav_vpp_guidance.training.train_prediction_vpp_ppo",
    "--config", $config,
    "--seed", "$Seed",
    "--device", $Device,
    "--backend", "jsbsim"
)

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $args += @("--output-dir", $OutputDir)
}

if ($Smoke.IsPresent) {
    $args += "--smoke"
}

Write-Host "Running reset075 no-mode-switch longscale00 no-VPP combat baseline training..."
& $PythonExe @args
