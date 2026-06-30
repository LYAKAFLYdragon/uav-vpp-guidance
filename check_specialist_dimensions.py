import sys
import torch

sys.path.insert(0, "src")

paths = [
    ("baseline", "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/checkpoints/last.pt"),
    ("head_on_weighted", "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted/checkpoints/best.pt"),
    ("crossing_weighted", "outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_crossing_weighted/checkpoints/best.pt"),
]

for name, path in paths:
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim")
    action_dim = ckpt.get("action_dim")
    print(f"{name}: obs_dim={obs_dim}, action_dim={action_dim}")
    # Check network shapes
    state_dict = ckpt.get("network_state_dict", {})
    for key, tensor in state_dict.items():
        if "weight" in key and "shared_net" in key and "0" in key:
            print(f"  {key}: {tensor.shape}")
            break
