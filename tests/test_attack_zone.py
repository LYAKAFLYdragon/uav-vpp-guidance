import numpy as np

from uav_vpp_guidance.envs.attack_zone import (
    CombatHPManager,
    compute_attack_zone_score,
    evaluate_attack_zone,
    get_range_reward,
)


def _state(x, velocity):
    return {
        "position_m": np.array([x, 0.0, 5000.0], dtype=float),
        "velocity_vector_mps": np.array(velocity, dtype=float),
        "altitude_m": 5000.0,
    }


def test_get_range_reward_matches_legacy_attack_window():
    assert get_range_reward(4.0, 0.1) > 0.0
    assert get_range_reward(2.0, 0.1) == 0.0
    assert get_range_reward(4.0, np.deg2rad(60.0)) == 0.0


def test_close_range_gun_zone_covers_near_tasks_without_changing_legacy_function():
    assert get_range_reward(2.0, 0.1) == 0.0
    assert compute_attack_zone_score(2.0, 0.1, {"close_range_enabled": True}) > 0.0


def test_close_range_max_aoa_deg_is_explicit_attack_zone_knob():
    aoa_50_deg = np.deg2rad(50.0)
    cfg = {"legacy_range_enabled": False, "close_range_enabled": True}

    assert compute_attack_zone_score(1.0, aoa_50_deg, cfg) == 0.0
    assert (
        compute_attack_zone_score(
            1.0,
            aoa_50_deg,
            {**cfg, "close_range_max_aoa_deg": 60.0},
        )
        > 0.0
    )


def test_attack_zone_directional_geometry():
    ego = _state(0.0, [200.0, 0.0, 0.0])
    target = _state(4000.0, [200.0, 0.0, 0.0])
    ego_attack = evaluate_attack_zone(ego, target)
    target_attack = evaluate_attack_zone(target, ego)

    assert ego_attack["in_attack_zone"] is True
    assert target_attack["in_attack_zone"] is False


def test_combat_hp_manager_applies_bidirectional_damage_and_kill():
    ego = _state(0.0, [200.0, 0.0, 0.0])
    target = _state(4000.0, [200.0, 0.0, 0.0])
    hp = CombatHPManager(initial_hp=100.0, damage_per_step=100.0, enabled=True)
    info = hp.step(ego, target)

    assert info["target_hp"] == 0.0
    assert info["ego_hp"] == 100.0
    terminated, truncated, terminal_info = hp.resolve_terminal({}, current_time_s=0.2)
    assert terminated is True
    assert truncated is False
    assert terminal_info["combat_outcome"] == "win"
    assert terminal_info["combat_time_to_kill"] == 0.2
