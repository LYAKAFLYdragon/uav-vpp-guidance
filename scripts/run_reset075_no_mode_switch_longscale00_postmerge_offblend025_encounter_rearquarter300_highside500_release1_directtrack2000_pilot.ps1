param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison",
    [int]$SeedStart = 480,
    [int]$SeedCount = 3,
    [int]$HeadOnVerticalM = 500,
    [switch]$DryRun,
    [switch]$AllowMissingCheckpoints
)

$ErrorActionPreference = "Stop"

if ($SeedCount -lt 1) {
    throw "SeedCount must be >= 1."
}
if ($HeadOnVerticalM -lt 0) {
    throw "HeadOnVerticalM must be >= 0."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$config = "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_heldout.yaml"
$methods = @(
    "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"
)
$tasks = @(
    "head_on",
    "crossing_feasible"
)
$seeds = $SeedStart..($SeedStart + $SeedCount - 1)
$vppArgs = @(
    "--vpp-post-merge-offensive-anchor-blend-by-task",
    "head_on=0.25,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
    "head_on=170",
    "--vpp-offensive-anchor-lateral-m-by-task",
    "head_on=300,crossing_feasible=0",
    "--vpp-offensive-anchor-vertical-m-by-task",
    "head_on=$HeadOnVerticalM,crossing_feasible=0",
    "--vpp-offensive-anchor-frame-by-task",
    "head_on=encounter,crossing_feasible=target_velocity",
    "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
    "head_on=1,crossing_feasible=0",
    "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task",
    "head_on=2000"
)

$commonArgs = @(
    "scripts/run_jsbsim_hrl_comparison.py",
    "--config", $config,
    "--output-root", $OutputRoot,
    "--methods"
) + $methods + @(
    "--tasks"
) + $tasks + @(
    "--seeds"
) + ($seeds | ForEach-Object { "$_" }) + @(
    "--n-episodes", "1",
    "--backend", "jsbsim",
    "--run-status", "formal-small",
    "--device", $Device,
    "--attack-zone-close-range-max-aoa-deg", "60"
) + $vppArgs

if ($DryRun) {
    $commonArgs += "--dry-run"
}

if ($AllowMissingCheckpoints) {
    $commonArgs += "--allow-missing-checkpoints"
}

$expertArgs = $commonArgs + @(
    "--run-id",
    "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_highside${HeadOnVerticalM}_release1_directtrack2000_pilot_expert",
    "--opponent-stage", "expert"
)

$endToEndArgs = $commonArgs + @(
    "--run-id",
    "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_highside${HeadOnVerticalM}_release1_directtrack2000_pilot_end_to_end",
    "--opponent-stage", "end_to_end"
)

Write-Host "Running high-side rear-quarter pilot against expert..."
& python @expertArgs

Write-Host "Running high-side rear-quarter pilot against end_to_end..."
& python @endToEndArgs
