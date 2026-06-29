param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison",
    [int]$HeldOutSeedStart = 480,
    [int]$HeldOutSeedCount = 480,
    [switch]$DryRun,
    [switch]$AllowMissingCheckpoints
)

$ErrorActionPreference = "Stop"

if ($HeldOutSeedCount -lt 1) {
    throw "HeldOutSeedCount must be >= 1."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$config = "config/experiment/jsbsim_hrl_reset075_heldout.yaml"
$methods = @(
    "no_prediction_vpp",
    "prediction_vpp_jsbsim_compare_long_lh1p0_reset075"
)
$tasks = @(
    "head_on",
    "crossing_feasible"
)
$heldOutSeeds = $HeldOutSeedStart..($HeldOutSeedStart + $HeldOutSeedCount - 1)

# Keep full trajectories so the formal runs can emit combat geometry diagnostics.
$commonArgs = @(
    "scripts/run_jsbsim_hrl_comparison.py",
    "--config", $config,
    "--output-root", $OutputRoot,
    "--methods"
) + $methods + @(
    "--tasks"
) + $tasks + @(
    "--seeds"
) + ($heldOutSeeds | ForEach-Object { "$_" }) + @(
    "--n-episodes", "1",
    "--backend", "jsbsim",
    "--run-status", "formal",
    "--device", $Device,
    "--attack-zone-close-range-max-aoa-deg", "60"
)

if ($DryRun) {
    $commonArgs += "--dry-run"
}

if ($AllowMissingCheckpoints) {
    $commonArgs += "--allow-missing-checkpoints"
}

$expertArgs = $commonArgs + @(
    "--run-id", "reset075_formal_heldout_expert",
    "--opponent-stage", "expert"
)

$endToEndArgs = $commonArgs + @(
    "--run-id", "reset075_formal_heldout_end_to_end",
    "--opponent-stage", "end_to_end"
)

Write-Host "Running reset075 held-out formal evaluation against expert..."
& python @expertArgs

Write-Host "Running reset075 held-out formal evaluation against end_to_end..."
& python @endToEndArgs
