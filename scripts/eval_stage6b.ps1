# Stage 6B: Evaluation for No-Prediction / CV / CA Comparison
#
# Evaluates trained checkpoints on random and fixed scenarios.
# Run after train_stage6b.ps1 completes.
#
# Usage:
#   .\scripts\eval_stage6b.ps1 -backend simple -episodes 50
#   .\scripts\eval_stage6b.ps1 -backend jsbsim -episodes 10

param(
    [string]$backend = "simple",
    [int]$episodes = 50,
    [int[]]$seeds = @(0, 1, 2),
    [string]$scenarioSet = "all"   # "all" | "fixed"
)

$ErrorActionPreference = "Stop"

# Map experiment names to checkpoint paths (edit after training completes)
$checkpoints = @{
    "no_prediction" = "outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt"
    "cv_prediction" = "outputs/experiments/stage6b_cv_s0/checkpoints/best.pt"
    "ca_prediction" = "outputs/experiments/stage6b_ca_s0/checkpoints/best.pt"
}

# Override with environment variables if set
if ($env:NO_PRED_CKPT) { $checkpoints["no_prediction"] = $env:NO_PRED_CKPT }
if ($env:CV_CKPT)      { $checkpoints["cv_prediction"] = $env:CV_CKPT }
if ($env:CA_CKPT)      { $checkpoints["ca_prediction"] = $env:CA_CKPT }

# Validate checkpoints
$missing = $checkpoints.GetEnumerator() | Where-Object { -not (Test-Path $_.Value) }
if ($missing) {
    foreach ($m in $missing) {
        Write-Warning "Checkpoint not found for $($m.Key): $($m.Value)"
    }
    throw "One or more checkpoints are missing. Set env vars or edit this script."
}

# Build scenario arguments
$scenarioArg = ""
if ($scenarioSet -eq "fixed") {
    $scenarioArg = "--scenarios favorable neutral disadvantage challenging"
}

# Build per-method checkpoint arguments
$ckptArgs = @()
foreach ($method in $checkpoints.Keys) {
    $ckptArgs += "--method-checkpoint"
    $ckptArgs += "$method=$($checkpoints[$method])"
}

$outDir = "outputs/tables/stage6b_${backend}_${scenarioSet}"
Write-Host "--- Evaluating all methods on $backend backend ---"
Write-Host "Output dir: $outDir"

python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison `
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml `
    @ckptArgs `
    --backend $backend `
    --episodes $episodes `
    --seeds $seeds `
    $scenarioArg `
    --output-dir $outDir

Write-Host ""
Write-Host "=== Stage 6B Evaluation Complete ==="
Write-Host "Metrics saved to: $outDir"
