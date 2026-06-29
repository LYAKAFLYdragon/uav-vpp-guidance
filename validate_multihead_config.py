import sys
sys.path.insert(0, "src")

from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config

config = load_experiment_config("config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_multihead.yaml")

print("Config loaded successfully")
print(f"experiment.name: {config.get('experiment', {}).get('name')}")
print(f"policy.num_tasks: {config.get('policy', {}).get('num_tasks')}")
print(f"observation.include_task_type: {config.get('observation', {}).get('include_task_type')}")
print(f"combat_finetune.tasks: {config.get('combat_finetune', {}).get('tasks')}")
print(f"combat_finetune.opponent_stages: {config.get('combat_finetune', {}).get('opponent_stages')}")
