import subprocess
import os

repo_root = r"E:\uav-vpp-guidance"
python_exe = r"D:\Anaconda3\envs\jsbenv\python.exe"

args = [
    "-m", "uav_vpp_guidance.training.train_prediction_vpp_combat_finetune",
    "--config", "config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_e2e_weighted2.yaml",
    "--seed", "0",
    "--device", "cpu",
    "--backend", "jsbsim",
    "--attack-zone-close-range-max-aoa-deg", "60"
]

proc = subprocess.Popen(
    [python_exe] + args,
    cwd=repo_root,
    stdout=open(os.path.join(repo_root, ".training_e2e_weighted2_stdout.txt"), "w"),
    stderr=open(os.path.join(repo_root, ".training_e2e_weighted2_stderr.txt"), "w"),
    creationflags=subprocess.CREATE_NO_WINDOW
)

with open(os.path.join(repo_root, ".training_e2e_weighted2_pid.txt"), "w") as f:
    f.write(str(proc.pid))

print(f"Started e2e-weighted2 training with PID {proc.pid}")
