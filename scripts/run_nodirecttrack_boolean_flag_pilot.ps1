param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [string]$OutputRoot = "outputs/jsbsim_hrl_comparison",
    [int]$SeedStart = 480,
    [int]$SeedCount = 3,
    [switch]$UseBaselineCrossingConfig,
    [double]$HeadOnPostMergePredictedTargetForwardScaleReleaseScale = [double]::NaN,
    [double]$HeadOnOffensiveAnchorLongitudinalM = 800.0,
    [double]$HeadOnOffensiveAnchorVerticalM = [double]::NaN,
    [double]$CrossingOffensiveAnchorLongitudinalM = 800.0,
    [double]$HeadOnRecoveryForwardBiasMMax = [double]::NaN,
    [double]$HeadOnRecoveryLongitudinalBlend = [double]::NaN,
    [double]$HeadOnRecoveryLateralBlend = [double]::NaN,
    [double]$HeadOnRecoveryBelowAltitudeM = [double]::NaN,
    [switch]$HeadOnRecoveryLatchOnActivation,
    [double]$HeadOnClampVpForwardBiasMMin = [double]::NaN,
    [double]$HeadOnClampNegativeLateralBelowAltitudeM = [double]::NaN,
    [string]$RunSuffix = "",
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

# Use the new clean boolean flag to reproduce the legacy nodirecttrack_release
# pilot semantics as closely as possible: only suppress head_on direct-track
# release, without bundling in extra recovery or lateral-latch behavior.
# When -UseBaselineCrossingConfig is set, crossing is left on the current
# mainline config instead of inheriting head_on experiment overrides.
$postMergeOffensiveAnchorBlendByTask = "head_on=0.25"
$postMergeOffensiveAnchorRequiresGeometryByTask = "head_on=true"
$postMergeOffensiveAnchorGeometryAaByTask = "head_on=170"
$offensiveAnchorFrameByTask = "head_on=encounter"
$offensiveAnchorLongitudinalMByTask = "head_on=$HeadOnOffensiveAnchorLongitudinalM"
$offensiveAnchorLateralMByTask = "head_on=300"
$postMergeOffensiveAnchorReleaseBlendByTask = "head_on=0.0"
$postMergeOffensiveAnchorReleaseEgoOnlyStreakByTask = "head_on=1"
$postMergeOffensiveAnchorDirectTrackEnabledByTask = "head_on=false"

if (-not $UseBaselineCrossingConfig.IsPresent) {
    $postMergeOffensiveAnchorBlendByTask += ",crossing_feasible=0.0"
    $postMergeOffensiveAnchorRequiresGeometryByTask += ",crossing_feasible=false"
    $postMergeOffensiveAnchorGeometryAaByTask += ",crossing_feasible=0"
    $offensiveAnchorFrameByTask += ",crossing_feasible=target_velocity"
    $offensiveAnchorLongitudinalMByTask += ",crossing_feasible=$CrossingOffensiveAnchorLongitudinalM"
    $offensiveAnchorLateralMByTask += ",crossing_feasible=0"
    $postMergeOffensiveAnchorReleaseBlendByTask += ",crossing_feasible=0.0"
    $postMergeOffensiveAnchorReleaseEgoOnlyStreakByTask += ",crossing_feasible=0"
    $postMergeOffensiveAnchorDirectTrackEnabledByTask += ",crossing_feasible=true"
}

$vppArgs = @(
    "--vpp-post-merge-offensive-anchor-blend-by-task",
    $postMergeOffensiveAnchorBlendByTask,
    "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
    $postMergeOffensiveAnchorRequiresGeometryByTask,
    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
    $postMergeOffensiveAnchorGeometryAaByTask,
    "--vpp-offensive-anchor-frame-by-task",
    $offensiveAnchorFrameByTask,
    "--vpp-offensive-anchor-longitudinal-m-by-task",
    $offensiveAnchorLongitudinalMByTask,
    "--vpp-offensive-anchor-lateral-m-by-task",
    $offensiveAnchorLateralMByTask,
    "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
    $postMergeOffensiveAnchorReleaseBlendByTask,
    "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
    $postMergeOffensiveAnchorReleaseEgoOnlyStreakByTask,
    # New clean boolean flag: disable direct-track release on head_on.
    # Keep the legacy altitude=0 override in place so the only intended
    # behavioral difference versus the old hack is the explicit enable switch.
    "--vpp-post-merge-offensive-anchor-blend-release-direct-track-enabled-by-task",
    $postMergeOffensiveAnchorDirectTrackEnabledByTask,
    "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task",
    "head_on=0"
)

if (-not [double]::IsNaN($HeadOnRecoveryForwardBiasMMax)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max-by-task",
        "head_on=$HeadOnRecoveryForwardBiasMMax"
    )
}

if (-not [double]::IsNaN($HeadOnPostMergePredictedTargetForwardScaleReleaseScale)) {
    $vppArgs += @(
        "--vpp-post-merge-predicted-target-forward-scale-release-scale-by-task",
        "head_on=$HeadOnPostMergePredictedTargetForwardScaleReleaseScale"
    )
}

if (-not [double]::IsNaN($HeadOnRecoveryLongitudinalBlend)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend-by-task",
        "head_on=$HeadOnRecoveryLongitudinalBlend,crossing_feasible=0.0"
    )
}

if (-not [double]::IsNaN($HeadOnRecoveryLateralBlend)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend-by-task",
        "head_on=$HeadOnRecoveryLateralBlend,crossing_feasible=0.0"
    )
}

if (-not [double]::IsNaN($HeadOnRecoveryBelowAltitudeM)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m-by-task",
        "head_on=$HeadOnRecoveryBelowAltitudeM"
    )
}

if ($HeadOnRecoveryLatchOnActivation.IsPresent) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation-by-task",
        "head_on=true,crossing_feasible=false"
    )
}

if (-not [double]::IsNaN($HeadOnClampVpForwardBiasMMin)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min-by-task",
        "head_on=$HeadOnClampVpForwardBiasMMin"
    )
}

if (-not [double]::IsNaN($HeadOnClampNegativeLateralBelowAltitudeM)) {
    $vppArgs += @(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m-by-task",
        "head_on=$HeadOnClampNegativeLateralBelowAltitudeM"
    )
}

if (-not [double]::IsNaN($HeadOnOffensiveAnchorVerticalM)) {
    $vppArgs += @(
        "--vpp-offensive-anchor-vertical-m-by-task",
        "head_on=$HeadOnOffensiveAnchorVerticalM,crossing_feasible=0.0"
    )
}

$runLabel = "nodirecttrack_boolean_flag"
$candidateRunTag = "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_boolean_flag_pilot"
if (-not [string]::IsNullOrWhiteSpace($RunSuffix)) {
    if ($RunSuffix.StartsWith("_")) {
        $candidateRunTag = "$candidateRunTag$RunSuffix"
    } else {
        $candidateRunTag = "$candidateRunTag`_$RunSuffix"
    }
}

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

Write-Host "Running nodirecttrack boolean-flag rear-quarter pilot against expert..."
& $PythonExe @expertArgs

Write-Host "Running nodirecttrack boolean-flag rear-quarter pilot against end_to_end..."
& $PythonExe @endToEndArgs

Write-Host "Pilot complete. Artifacts written to $OutputRoot/$($candidateRunTag)_*"
