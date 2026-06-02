# JSBSim Sanity Evaluation
#
# Loads a simple-trained checkpoint and runs a small number of episodes
# on the JSBSim high-fidelity backend to check for crashes, stalls, and saturation.
#
# Usage:
#   .\scripts\eval_jsbsim_sanity.ps1 -checkpoint outputs/experiments/stage6b_cv_s0/checkpoints/best.pt
#   .\scripts\eval_jsbsim_sanity.ps1 -checkpoint $env:CV_CKPT -episodes 10

param(
    [string]$checkpoint = "",
    [string]$config = "config/experiment/train_vpp_ppo_cv.yaml",
    [int]$episodes = 5,
    [int[]]$seeds = @(0),
    [switch]$saveTrajectories
)

$ErrorActionPreference = "Stop"

if (-not $checkpoint) {
    throw "Checkpoint path is required. Use -checkpoint path/to/best.pt"
}
if (-not (Test-Path $checkpoint)) {
    throw "Checkpoint not found: $checkpoint"
}

$trajArg = ""
if ($saveTrajectories) {
    $trajArg = "--save-trajectories"
}

$outDir = "outputs/tables/jsbsim_sanity/$(Split-Path $config -LeafBase)"
Write-Host "--- JSBSim Sanity Evaluation ---"
Write-Host "Checkpoint: $checkpoint"
Write-Host "Config: $config"
Write-Host "Output dir: $outDir"

python -m uav_vpp_guidance.evaluation.evaluate_jsbsim_sanity `
    --config $config `
    --checkpoint $checkpoint `
    --episodes $episodes `
    --seeds $seeds `
    $trajArg `
    --output-dir $outDir

Write-Host ""
Write-Host "=== JSBSim Sanity Complete ==="
