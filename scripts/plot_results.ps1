# Plot training curves from a log directory.
# Usage: .\scripts\plot_results.ps1 -LogDir <path>
param(
    [string]$LogDir = "outputs"
)

python -m uav_vpp_guidance.visualization.plot_training_curves --log-dir $LogDir
