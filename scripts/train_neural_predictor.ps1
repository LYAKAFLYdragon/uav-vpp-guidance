# Neural Trajectory Predictor (LSTM/GRU) Offline Training Script
# Requires pre-generated trajectory CSVs in outputs/trajectories/

param(
    [string]$Config = "config/trajectory_prediction.yaml",
    [string]$DataDir = "outputs/trajectories",
    [string]$ModelType = "lstm",
    [string]$OutputDir = "outputs/trajectory_prediction",
    [int]$Seed = 42,
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [double]$ValRatio = 0.2,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Neural Trajectory Predictor Offline Training ===" -ForegroundColor Cyan
Write-Host "Model : $ModelType"
Write-Host "Data  : $DataDir"
Write-Host "Output: $OutputDir"
Write-Host "Seed  : $Seed"
Write-Host ""

python -m uav_vpp_guidance.trajectory_prediction.train_pipeline `
    --config $Config `
    --data-dir $DataDir `
    --model-type $ModelType `
    --output-dir $OutputDir `
    --seed $Seed `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --val-ratio $ValRatio `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "Training completed successfully." -ForegroundColor Green
