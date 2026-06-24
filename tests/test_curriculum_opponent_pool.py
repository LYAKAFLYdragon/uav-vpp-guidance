import numpy as np

from scripts.train_curriculum_ppo import (
    LightweightEloOpponentPool,
    SelfPlayCheckpointOpponent,
    evaluate_scenarios,
    run_evaluation,
)


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
