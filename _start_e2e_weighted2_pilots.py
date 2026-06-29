import subprocess
import os

repo_root = r"E:\uav-vpp-guidance"
python_exe = r"D:\Anaconda3\envs\jsbenv\python.exe"

# Expert pilot
args_expert = [
    "-u", "scripts/run_jsbsim_hrl_comparison.py",
    "--config", "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_e2e_weighted2_best.yaml",
    "--run-id", "tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_e2e_weighted2_best_expert_10seed_20260629",
    "--run-status", "formal-small",
    "--tasks", "head_on", "crossing_feasible",
    "--seeds", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489",
    "--backend", "jsbsim",
    "--opponent-stage", "expert",
    "--attack-zone-close-range-max-aoa-deg", "60",
    "--output-root", "outputs/jsbsim_hrl_comparison"
]

proc_expert = subprocess.Popen(
    [python_exe] + args_expert,
    cwd=repo_root,
    stdout=open(os.path.join(repo_root, ".pilot_e2e_weighted2_expert_stdout.txt"), "w"),
    stderr=open(os.path.join(repo_root, ".pilot_e2e_weighted2_expert_stderr.txt"), "w"),
    creationflags=subprocess.CREATE_NO_WINDOW
)

with open(os.path.join(repo_root, ".pilot_e2e_weighted2_expert_pid.txt"), "w") as f:
    f.write(str(proc_expert.pid))

print(f"Started e2e-weighted2 expert pilot with PID {proc_expert.pid}")

# End-to-end pilot
args_e2e = [
    "-u", "scripts/run_jsbsim_hrl_comparison.py",
    "--config", "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_e2e_weighted2_best.yaml",
    "--run-id", "tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_e2e_weighted2_best_end_to_end_10seed_20260629",
    "--run-status", "formal-small",
    "--tasks", "head_on", "crossing_feasible",
    "--seeds", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489",
    "--backend", "jsbsim",
    "--opponent-stage", "end_to_end",
    "--attack-zone-close-range-max-aoa-deg", "60",
    "--output-root", "outputs/jsbsim_hrl_comparison"
]

proc_e2e = subprocess.Popen(
    [python_exe] + args_e2e,
    cwd=repo_root,
    stdout=open(os.path.join(repo_root, ".pilot_e2e_weighted2_e2e_stdout.txt"), "w"),
    stderr=open(os.path.join(repo_root, ".pilot_e2e_weighted2_e2e_stderr.txt"), "w"),
    creationflags=subprocess.CREATE_NO_WINDOW
)

with open(os.path.join(repo_root, ".pilot_e2e_weighted2_e2e_pid.txt"), "w") as f:
    f.write(str(proc_e2e.pid))

print(f"Started e2e-weighted2 end_to_end pilot with PID {proc_e2e.pid}")
