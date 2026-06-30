import sys
sys.path.insert(0, "src")

from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy

specialists_config = {
    "head_on": {
        "checkpoint": "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted/checkpoints/best.pt",
        "config_path": "config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted.yaml",
    },
    "crossing_feasible": {
        "checkpoint": "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_crossing_weighted/checkpoints/best.pt",
        "config_path": "config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_crossing_weighted.yaml",
    },
}

print("Loading OracleTaskGatePolicy with specialists...")
agent = OracleTaskGatePolicy(specialists_config=specialists_config, device="cpu")
print(f"Loaded {len(agent._specialists)} specialists: {list(agent._specialists.keys())}")

# Test set_task_name
agent.set_task_name("head_on")
print(f"Current task: {agent.current_task_name}")

# Test get_deterministic_action with dummy obs
import numpy as np
dummy_obs = np.zeros(20, dtype=np.float32)
action = agent.get_deterministic_action(dummy_obs)
print(f"Action shape: {action.shape}, dtype: {action.dtype}")

# Test crossing task
agent.set_task_name("crossing_feasible")
action2 = agent.get_deterministic_action(dummy_obs)
print(f"Crossing action shape: {action2.shape}")

print("All tests passed.")
