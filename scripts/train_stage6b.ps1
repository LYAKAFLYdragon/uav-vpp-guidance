# Stage 6B: Full PPO Training for No-Prediction / CV / CA Comparison
#
# Run this on a machine with sufficient compute (GPU recommended).
# Adjust $seeds and $totalTimesteps as needed.
#
# Usage:
#   .\scripts\train_stage6b.ps1

$ErrorActionPreference = "Stop"

$seeds = @(0, 1, 2)
$totalTimesteps = 200000

Write-Host "=== Stage 6B Training ==="
Write-Host "Seeds: $seeds"
Write-Host "Total timesteps per run: $totalTimesteps"
Write-Host ""

foreach ($seed in $seeds) {
    $expName = "stage6b_no_pred_s$seed"
    Write-Host "--- Training No-Prediction | seed=$seed | exp=$expName ---"
    python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
        --config config/experiment/train_no_prediction_vpp_ppo.yaml `
        --seed $seed `
        --output-dir outputs/experiments/$expName

    $expName = "stage6b_cv_s$seed"
    Write-Host "--- Training CV-Prediction | seed=$seed | exp=$expName ---"
    python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
        --config config/experiment/train_vpp_ppo_cv.yaml `
        --seed $seed `
        --output-dir outputs/experiments/$expName

    $expName = "stage6b_ca_s$seed"
    Write-Host "--- Training CA-Prediction | seed=$seed | exp=$expName ---"
    python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
        --config config/experiment/train_vpp_ppo_ca.yaml `
        --seed $seed `
        --output-dir outputs/experiments/$expName
}

Write-Host ""
Write-Host "=== Stage 6B Training Complete ==="
Write-Host "Checkpoints saved under outputs/experiments/stage6b_*/checkpoints/"
