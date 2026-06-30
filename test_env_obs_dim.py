import sys
sys.path.insert(0, "src")

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

config = load_yaml_config("config/experiment/jsbsim_hrl_oracle_task_gate_mvp_only.yaml")
env = CloseRangeTrackingEnv(config)
obs = env.reset(seed=0)
print(f"observation_vector shape: {obs['observation_vector'].shape}")
print(f"observation_schema dim: {obs.get('observation_schema', {}).get('dim')}")
print(f"feature_names count: {len(obs.get('observation_schema', {}).get('feature_names', []))}")
print(f"feature_names: {obs.get('observation_schema', {}).get('feature_names')}")
