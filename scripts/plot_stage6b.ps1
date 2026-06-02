# Stage 6B: Plotting for No-Prediction / CV / CA Comparison
#
# Generates comparison figures from evaluation outputs.
#
# Usage:
#   .\scripts\plot_stage6b.ps1 -backend simple -scenarioSet all

param(
    [string]$backend = "simple",
    [string]$scenarioSet = "all"
)

$ErrorActionPreference = "Stop"

$baseDir = "outputs/tables/stage6b_${backend}_${scenarioSet}"

$metricsCsv = "$baseDir/prediction_metrics.csv"
$metricsJson = "$baseDir/prediction_metrics.json"
$outDir = "outputs/figures/stage6b_${backend}_${scenarioSet}"

if (-not (Test-Path $metricsCsv)) {
    throw "Metrics CSV not found: $metricsCsv. Run eval_stage6b.ps1 first."
}

Write-Host "--- Plotting Stage 6B results ---"
Write-Host "Metrics CSV: $metricsCsv"
Write-Host "Output dir: $outDir"

$jsonArg = ""
if (Test-Path $metricsJson) {
    $jsonArg = "--metrics-json $metricsJson"
}

python -m uav_vpp_guidance.visualization.plot_prediction_comparison `
    --metrics $metricsCsv `
    $jsonArg `
    --output $outDir

Write-Host ""
Write-Host "=== Stage 6B Plotting Complete ==="
Write-Host "Figures saved to: $outDir"
