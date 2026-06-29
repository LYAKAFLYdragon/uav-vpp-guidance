param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison",
    [int]$SeedStart = 480,
    [int]$SeedCount = 3,
    [switch]$DryRun,
    [switch]$AllowMissingCheckpoints,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if ($SeedCount -lt 1) {
    throw "SeedCount must be >= 1."
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
    "--vpp-post-merge-anchor-mode-by-task",
    "head_on=offensive_position",
    "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps-by-task",
    "head_on=1",
    "--vpp-post-merge-anchor-mode-offensive-anchor-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend-by-task",
    "head_on=1.0,crossing_feasible=0.0",
    "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend-by-task",
    "head_on=0.25,crossing_feasible=0.0",
    "--vpp-post-merge-anchor-mode-longitudinal-scale-by-task",
    "head_on=0,crossing_feasible=1",
    "--vpp-post-merge-anchor-mode-lateral-scale-by-task",
    "head_on=0,crossing_feasible=1",
    "--vpp-post-merge-offensive-anchor-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
    "head_on=170",
    "--vpp-offensive-anchor-longitudinal-m-by-task",
    "head_on=3000,crossing_feasible=0",
    "--vpp-offensive-anchor-lateral-m-by-task",
    "head_on=300,crossing_feasible=0",
    "--vpp-offensive-anchor-frame-by-task",
    "head_on=target_velocity,crossing_feasible=target_velocity",
    "--vpp-offensive-anchor-lateral-sign-mode-by-task",
    "head_on=fixed_positive,crossing_feasible=same_side"
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

$expertRunId = "reset075_no_mode_switch_longscale00_postmerge_anchormode_longblend100_latblend025_targetvel_rearlag3000_fixedpos_geomgate_aa170_release1_pilot_expert"
$endToEndRunId = "reset075_no_mode_switch_longscale00_postmerge_anchormode_longblend100_latblend025_targetvel_rearlag3000_fixedpos_geomgate_aa170_release1_pilot_end_to_end"

$expertArgs = $commonArgs + @(
    "--run-id", $expertRunId,
    "--opponent-stage", "expert"
)
$endToEndArgs = $commonArgs + @(
    "--run-id", $endToEndRunId,
    "--opponent-stage", "end_to_end"
)

Write-Host "Running fixed-positive post-merge anchor-mode component-blend pilot against expert..."
& $PythonExe @expertArgs

Write-Host "Running fixed-positive post-merge anchor-mode component-blend pilot against end_to_end..."
& $PythonExe @endToEndArgs

$analysisScript = Join-Path $repoRoot "scripts\analyze_postmerge_anchor_mode_pilot.py"
$analysisOutputDir = "E:\uav-vpp-guidance\outputs\diagnostics\postmerge_anchor_mode_componentblend_fixedpositive_pilot"

Write-Host "Building fixed-positive post-merge anchor-mode component-blend pilot comparison report..."
& $PythonExe $analysisScript `
    --baseline-expert-run-dir "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_anchormode_takeover_encounter_rearquarter300_geomgate_aa170_release1_pilot_expert" `
    --baseline-end-to-end-run-dir "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_anchormode_takeover_encounter_rearquarter300_geomgate_aa170_release1_pilot_end_to_end" `
    --candidate-expert-run-dir "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\$expertRunId" `
    --candidate-end-to-end-run-dir "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\$endToEndRunId" `
    --method "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00" `
    --output-dir $analysisOutputDir
