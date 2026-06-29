param(
    [string]$OutputDir = "E:\uav-vpp-guidance\outputs\diagnostics\directtrack2000_headon_failure",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\analyze_directtrack2000_headon_failure.py"
$method = "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"

$formalRunDir = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison_directtrack2000_formal\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_formal_heldout_expert"
$pilotRunDir = "E:\uav-vpp-guidance\outputs\jsbsim_hrl_comparison\reset075_no_mode_switch_longscale00_postmerge_offblend025_encounter_rearquarter300_geomgate_aa170_release1_directtrack2000_pilot_expert_480_482"

& $PythonExe $scriptPath `
    --formal-run-dir $formalRunDir `
    --pilot-run-dir $pilotRunDir `
    --task head_on `
    --method $method `
    --output-dir $OutputDir
