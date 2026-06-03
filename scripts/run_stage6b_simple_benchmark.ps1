# Stage 6B: Simple-Backend Prediction Comparison Benchmark Runner
#
# Usage:
#   .\scripts\run_stage6b_simple_benchmark.ps1 -Smoke
#   .\scripts\run_stage6b_simple_benchmark.ps1 -Episodes 3 -Seeds 0,1 -Scenarios favorable,neutral
#   .\scripts\run_stage6b_simple_benchmark.ps1 -Episodes 20 -Seeds 0,1,2,3,4 -Scenarios favorable,neutral,disadvantage,challenging

param(
    [switch]$Smoke,
    [int]$Episodes = 20,
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [string[]]$Scenarios = @("favorable", "neutral", "disadvantage", "challenging")
)

$ConfigPath = "config/experiment/benchmark_simple_prediction_comparison.yaml"
$OutputRoot = "outputs/benchmark/stage6b_simple_prediction"

$argsList = @(
    "-m", "uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark",
    "--config", $ConfigPath,
    "--output-dir", $OutputRoot
)

if ($Smoke) {
    $argsList += "--smoke"
} else {
    $argsList += "--episodes"
    $argsList += $Episodes
    $argsList += "--seeds"
    $argsList += ($Seeds -join " ")
    $argsList += "--scenarios"
    $argsList += ($Scenarios -join " ")
}

Write-Host "Running Stage 6B benchmark..."
Write-Host "Config: $ConfigPath"
Write-Host "Output: $OutputRoot"

& python $argsList

if ($LASTEXITCODE -ne 0) {
    Write-Error "Benchmark failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host "Benchmark complete. Output: $OutputRoot"
