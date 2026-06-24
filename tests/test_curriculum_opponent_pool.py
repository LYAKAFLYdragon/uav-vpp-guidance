import numpy as np
import torch

from scripts.train_curriculum_ppo import (
    LightweightEloOpponentPool,
    SafetyPreCurriculumGate,
    SelfPlayCheckpointOpponent,
    evaluate_scenarios,
    load_warm_start_checkpoint,
    run_evaluation,
)
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


class _FakeCombatHP:
    enabled = True


class _FakeCombatEnv:
    max_steps = 1
    combat_hp = _FakeCombatHP()

    def reset(self, scenario=None, seed=None):
        return {"observation_vector": np.zeros(16, dtype=np.float32)}

    def step(self, action):
        info = {
            "reason": "target_killed",
            "combat_success": True,
            "combat_outcome": "win",
            "combat_reason": "target_killed",
            "ego_hp": 100.0,
            "target_hp": 0.0,
            "combat_time_to_kill": 0.2,
        }
        return {"relative_state": {}}, 1.0, True, False, info


class _FakeAgent:
    def get_deterministic_action(self, obs):
        return np.zeros(3, dtype=np.float32)


def test_elo_update_direction_and_initial_value(tmp_path):
    pool = LightweightEloOpponentPool({}, tmp_path)
    entry = pool.add_checkpoint("dummy.pt", step=50000)

    assert entry.elo == 1000.0
    pool.update_from_eval(entry, ego_win_rate=1.0, num_episodes=10)

    assert entry.elo < 1000.0
    assert entry.losses == 10
    assert entry.wins == 0


def test_self_play_action_mode_inferred_from_checkpoint_config():
    assert SelfPlayCheckpointOpponent._infer_action_mode(
        {"virtual_point": {"enabled": True}}
    ) == "vpp"
    assert SelfPlayCheckpointOpponent._infer_action_mode(
        {"virtual_point": {"enabled": False}}
    ) == "direct_command"
    assert SelfPlayCheckpointOpponent._infer_action_mode(
        {"end_to_end": {"enabled": True}, "virtual_point": {"enabled": True}}
    ) == "direct_command"


def test_curriculum_evaluation_uses_combat_success_when_hp_enabled():
    env = _FakeCombatEnv()
    agent = _FakeAgent()
    cfg = {"scenarios": {"near": {}}}

    metrics = run_evaluation(env, agent, cfg, num_episodes=1, seeds=[0])
    scenario_sr = evaluate_scenarios(env, agent, cfg["scenarios"], num_episodes=1)

    assert metrics["success_rate"] == 1.0
    assert metrics["win_rate"] == 1.0
    assert scenario_sr["near"] == 1.0


def _agent_config():
    return {
        "policy": {"hidden_sizes": [16], "activation": "tanh", "action_dim": 3},
        "ppo": {"rollout_steps": 8, "minibatch_size": 4, "learning_rate": 3e-4},
    }


def test_warm_start_loads_policy_weights_without_requiring_optimizer(tmp_path):
    source = PPOAgent(obs_dim=16, action_dim=3, config=_agent_config(), device="cpu")
    with torch.no_grad():
        for param in source.network.parameters():
            param.fill_(0.123)
    checkpoint_path = tmp_path / "warm.pt"
    source.save(str(checkpoint_path))

    target = PPOAgent(obs_dim=16, action_dim=3, config=_agent_config(), device="cpu")
    meta = load_warm_start_checkpoint(
        target,
        checkpoint_path=str(checkpoint_path),
        strict_dims=True,
        load_optimizer=False,
    )

    assert meta["loaded"] is True
    assert meta["checkpoint_path"] == str(checkpoint_path)
    for param in target.network.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.123))


def test_safety_precurriculum_blocks_adversary_until_survival_gate_passes():
    gate = SafetyPreCurriculumGate(
        {
            "enabled": True,
            "min_steps": 100,
            "survival_threshold": 0.5,
            "max_crash_rate": 0.5,
            "max_out_of_bounds_rate": 0.5,
        }
    )

    assert gate.allows_adversary(global_step=0) is False
    gate.update_from_eval(
        {"survival_rate": 1.0, "crash_rate": 0.0, "out_of_bounds_rate": 0.0},
        global_step=50,
    )
    assert gate.allows_adversary(global_step=50) is False

    gate.update_from_eval(
        {"survival_rate": 1.0, "crash_rate": 0.0, "out_of_bounds_rate": 0.0},
        global_step=100,
    )
    assert gate.passed is True
    assert gate.allows_adversary(global_step=100) is True
