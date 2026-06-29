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

$commonVppArgs = @(
    "--vpp-post-merge-offensive-anchor-blend-by-task",
    "head_on=0.25,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
    "head_on=true,crossing_feasible=false",
    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
    "head_on=170",
    "--vpp-offensive-anchor-frame-by-task",
    "head_on=encounter,crossing_feasible=target_velocity",
    "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
    "head_on=0.0,crossing_feasible=0.0",
    "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
    "head_on=1,crossing_feasible=0"
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
) + $commonVppArgs

if ($DryRun) {
    $commonArgs += "--dry-run"
}

if ($AllowMissingCheckpoints) {
    $commonArgs += "--allow-missing-checkpoints"
}

$variants = @(
    @{
        Label = "lag_only"
        RunTag = "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_lagonly800_geomgate_aa170_release1_directtrack2000_pilot"
        VppArgs = @(
            "--vpp-offensive-anchor-longitudinal-m-by-task", "head_on=800,crossing_feasible=800",
            "--vpp-offensive-anchor-lateral-m-by-task", "head_on=0,crossing_feasible=0",
            "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task", "head_on=2000"
        )
    },
    @{
        Label = "beam_only"
        RunTag = "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_beamonly300_geomgate_aa170_release1_directtrack2000_pilot"
        VppArgs = @(
            "--vpp-offensive-anchor-longitudinal-m-by-task", "head_on=0,crossing_feasible=800",
            "--vpp-offensive-anchor-lateral-m-by-task", "head_on=300,crossing_feasible=0",
            "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task", "head_on=2000"
        )
    },
    @{
        Label = "nodirecttrack_release"
        RunTag = "reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_directtrack2000_pilot"
        VppArgs = @(
            "--vpp-offensive-anchor-longitudinal-m-by-task", "head_on=800,crossing_feasible=800",
            "--vpp-offensive-anchor-lateral-m-by-task", "head_on=300,crossing_feasible=0",
            "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task", "head_on=0"
        )
    }
)

foreach ($variant in $variants) {
    $expertArgs = $commonArgs + $variant.VppArgs + @(
        "--run-id",
        "$($variant.RunTag)_expert",
        "--opponent-stage", "expert"
    )
    $endToEndArgs = $commonArgs + $variant.VppArgs + @(
        "--run-id",
        "$($variant.RunTag)_end_to_end",
        "--opponent-stage", "end_to_end"
    )

    Write-Host "Running $($variant.Label) rear-quarter factor pilot against expert..."
    & $PythonExe @expertArgs

    Write-Host "Running $($variant.Label) rear-quarter factor pilot against end_to_end..."
    & $PythonExe @endToEndArgs
}

$analysisOutputDir = "E:\uav-vpp-guidance\outputs\diagnostics\rearquarter_geometry_factor_pilot"
$analysisScript = Join-Path $repoRoot "scripts\analyze_rearquarter_geometry_factor_pilot.py"
$baselineExpert = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_expert"
$baselineEndToEnd = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_end_to_end"

$analysisArgs = @(
    $analysisScript,
    "--baseline-label", "baseline",
    "--method", "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
    "--output-dir", $analysisOutputDir,
    "--run", "baseline=$baselineExpert",
    "--run", "baseline=$baselineEndToEnd",
    "--run", "lag_only=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_lagonly800_geomgate_aa170_release1_directtrack2000_pilot_expert",
    "--run", "lag_only=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_lagonly800_geomgate_aa170_release1_directtrack2000_pilot_end_to_end",
    "--run", "beam_only=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_beamonly300_geomgate_aa170_release1_directtrack2000_pilot_expert",
    "--run", "beam_only=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_beamonly300_geomgate_aa170_release1_directtrack2000_pilot_end_to_end",
    "--run", "nodirecttrack_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_directtrack2000_pilot_expert",
    "--run", "nodirecttrack_release=E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_nodirecttrack_directtrack2000_pilot_end_to_end"
)

$optionalAnalysisRuns = @(
    @{
        Label = "scaleguard_long0"
        Expert = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_scaleguardlong0_pilot_expert"
        EndToEnd = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_scaleguardlong0_pilot_end_to_end"
    },
    @{
        Label = "stable20"
        Expert = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounterstable20_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_expert"
        EndToEnd = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounterstable20_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_end_to_end"
    }
)

foreach ($optionalRun in $optionalAnalysisRuns) {
    if ((Test-Path $optionalRun.Expert) -and (Test-Path $optionalRun.EndToEnd)) {
        $analysisArgs += @(
            "--run", "$($optionalRun.Label)=$($optionalRun.Expert)",
            "--run", "$($optionalRun.Label)=$($optionalRun.EndToEnd)"
        )
    }
}

Write-Host "Building rear-quarter geometry factor pilot comparison report..."
& $PythonExe @analysisArgs
