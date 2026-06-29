param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
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
    "--vpp-post-merge-offensive-anchor-blend-by-task",
    "head_on=0.25,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
    "head_on=170,crossing_feasible=0",
    "--vpp-offensive-anchor-frame-by-task",
    "head_on=encounter,crossing_feasible=target_velocity",
    "--vpp-offensive-anchor-longitudinal-m-by-task",
    "head_on=800,crossing_feasible=800",
    "--vpp-offensive-anchor-lateral-m-by-task",
    "head_on=300,crossing_feasible=0",
    "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
    "head_on=1,crossing_feasible=0",
    "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task",
    "head_on=0,crossing_feasible=0",
    "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m-by-task",
    "head_on=4500,crossing_feasible=0",
    "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max-by-task",
    "head_on=-4500",
    "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend-by-task",
    "head_on=0.25,crossing_feasible=0.0"
)

$runLabel = "recoveryforwardbias_release"
$candidateRunTag = "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_recoveryforwardbias_pilot"
$candidateExpertRunDir = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\$($candidateRunTag)_expert"
$candidateEndToEndRunDir = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\$($candidateRunTag)_end_to_end"

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
    "$($candidateRunTag)_expert",
    "--opponent-stage", "expert"
)
$endToEndArgs = $commonArgs + @(
    "--run-id",
    "$($candidateRunTag)_end_to_end",
    "--opponent-stage", "end_to_end"
)

Write-Host "Running recovery-forward-bias rear-quarter pilot against expert..."
& $PythonExe @expertArgs

Write-Host "Running recovery-forward-bias rear-quarter pilot against end_to_end..."
& $PythonExe @endToEndArgs

$analysisOutputDir = "E:\uav-vpp-guidance\outputs\diagnostics\rearquarter_recoveryforwardbias_pilot"
$analysisScript = Join-Path $repoRoot "scripts\analyze_rearquarter_geometry_factor_pilot.py"

$analysisArgs = @(
    $analysisScript,
    "--baseline-label", "nodirecttrack_release",
    "--method", "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
    "--output-dir", $analysisOutputDir,
    "--run", "directtrack2000=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_expert",
    "--run", "directtrack2000=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_end_to_end",
    "--run", "nodirecttrack_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_directtrack2000_pilot_expert",
    "--run", "nodirecttrack_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_directtrack2000_pilot_end_to_end",
    "--run", "laterallatch_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_laterallatchrelease_pilot_expert",
    "--run", "laterallatch_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_laterallatchrelease_pilot_end_to_end",
    "--run", "recoveryblend_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_recoveryblend_pilot_expert",
    "--run", "recoveryblend_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_recoveryblend_pilot_end_to_end",
    "--run", "$runLabel=$candidateExpertRunDir",
    "--run", "$runLabel=$candidateEndToEndRunDir"
)

Write-Host "Building recovery-forward-bias rear-quarter pilot comparison report..."
& $PythonExe @analysisArgs
