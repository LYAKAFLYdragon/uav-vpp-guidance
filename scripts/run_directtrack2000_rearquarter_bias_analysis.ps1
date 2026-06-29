param(
    [string]$OutputDir = "E:\uav-vpp-guidance\outputs\diagnostics\directtrack2000_rearquarter_bias_analysis",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\analyze_directtrack2000_rearquarter_bias.py"
$method = "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"

$expertPilot = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_expert_480_482"
$expertFormal = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison_directtrack2000_formal\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_formal_heldout_expert"
$endToEndPilot = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_end_to_end_480_482"
$endToEndFormal = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison_directtrack2000_formal\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_formal_heldout_end_to_end"

& $PythonExe $scriptPath `
    --expert-pilot-run-dir $expertPilot `
    --expert-formal-run-dir $expertFormal `
    --end-to-end-pilot-run-dir $endToEndPilot `
    --end-to-end-formal-run-dir $endToEndFormal `
    --task head_on `
    --method $method `
    --direct-track-altitude-threshold-m 2000 `
    --output-dir $OutputDir
