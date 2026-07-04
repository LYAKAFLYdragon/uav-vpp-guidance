from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def _config():
    return {
        "ppo": {"rollout_steps": 8},
        "policy": {"hidden_sizes": [16, 16]},
        "commander": {
            "macro_action_repeat_steps": 2,
            "modes": [
                {
                    "id": 0,
                    "name": "head_on_specialist",
                    "specialist_key": "head_on",
                    "checkpoint": "head_on.pt",
                    "config_path": "head_on.yaml",
                },
                {
                    "id": 1,
                    "name": "crossing_specialist",
                    "specialist_key": "crossing_feasible",
                    "checkpoint": "crossing.pt",
                    "config_path": "crossing.yaml",
                },
            ],
        },
    }


class _GuardEnv:
    def __init__(
        self,
        *,
        range_m=5000.0,
        min_range_so_far_m=None,
        first_pass_complete=True,
        altitude_sequence_m=None,
        vp_lateral_bias_sequence_m=None,
        vp_forward_bias_sequence_m=None,
        vp_lateral_bias_m=0.0,
        vp_forward_bias_m=0.0,
        ego_in_attack_zone=False,
        target_in_attack_zone=False,
        own_velocity_mps=250.0,
        target_velocity_mps=250.0,
    ):
        self._first_pass_complete = first_pass_complete
        self._range_m = float(range_m)
        self._merge_min_range_so_far_m = float(
            range_m if min_range_so_far_m is None else min_range_so_far_m
        )
        self._own_velocity_mps = float(own_velocity_mps)
        self._target_velocity_mps = float(target_velocity_mps)
        self._altitude_sequence_m = list(altitude_sequence_m or [5000.0])
        self._state_call_index = 0
        self._vp_lateral_bias_sequence_m = list(
            vp_lateral_bias_sequence_m or [vp_lateral_bias_m]
        )
        self._vp_forward_bias_sequence_m = list(
            vp_forward_bias_sequence_m or [vp_forward_bias_m]
        )
        self._last_virtual_point = {
            "position_neu": np.array(
                [
                    self._range_m + float(self._vp_forward_bias_sequence_m[0]),
                    float(self._vp_lateral_bias_sequence_m[0]),
                    self._altitude_sequence_m[0],
                ],
                dtype=np.float64,
            )
        }
        self.combat_hp = MagicMock()
        self.combat_hp.last_info = {
            "ego_hp": 90.0,
            "target_hp": 80.0,
            "hp_advantage": 10.0,
            "ego_in_attack_zone": bool(ego_in_attack_zone),
            "target_in_attack_zone": bool(target_in_attack_zone),
        }

    def _get_current_states(self, noisy=False):
        del noisy
        idx = min(self._state_call_index, len(self._altitude_sequence_m) - 1)
        altitude_m = float(self._altitude_sequence_m[idx])
        vp_lateral_bias_m = float(
            self._vp_lateral_bias_sequence_m[
                min(self._state_call_index, len(self._vp_lateral_bias_sequence_m) - 1)
            ]
        )
        vp_forward_bias_m = float(
            self._vp_forward_bias_sequence_m[
                min(self._state_call_index, len(self._vp_forward_bias_sequence_m) - 1)
            ]
        )
        self._state_call_index += 1
        own = {
            "position_m": np.array([0.0, 0.0, altitude_m], dtype=np.float64),
            "velocity_vector_mps": np.array(
                [self._own_velocity_mps, 0.0, 0.0], dtype=np.float64
            ),
            "altitude_m": altitude_m,
        }
        target = {
            "position_m": np.array([self._range_m, 0.0, altitude_m], dtype=np.float64),
            "velocity_vector_mps": np.array(
                [self._target_velocity_mps, 0.0, 0.0], dtype=np.float64
            ),
            "altitude_m": altitude_m,
        }
        self._last_virtual_point = {
            "position_neu": np.array(
                [
                    self._range_m + vp_forward_bias_m,
                    vp_lateral_bias_m,
                    altitude_m,
                ],
                dtype=np.float64,
            )
        }
        return own, target


def test_head_on_snapshot_includes_min_range_so_far():
    from uav_vpp_guidance.hierarchy.commander_mode_constraints import (
        build_head_on_post_merge_reopened_crossing_snapshot,
    )

    snapshot = build_head_on_post_merge_reopened_crossing_snapshot(
        env=_GuardEnv(
            range_m=2500.0,
            min_range_so_far_m=175.0,
            first_pass_complete=True,
            vp_forward_bias_m=-3000.0,
            vp_lateral_bias_m=500.0,
        ),
        task_name="head_on",
    )

    assert snapshot["available"] is True
    assert snapshot["min_range_so_far_m"] == 175.0


def test_hierarchical_commander_policy_switches_only_on_macro_boundary():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": head_on},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": crossing},
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [0, 1]

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=_config(),
            obs_dim=6,
            device="cpu",
        )
        obs = np.zeros(6, dtype=np.float32)
        policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)

    assert mock_commander.get_deterministic_action.call_count == 2


def test_hierarchical_commander_policy_routes_to_selected_specialist():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": head_on},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": crossing},
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [0, 1]

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=_config(),
            obs_dim=6,
            device="cpu",
        )
        obs = np.zeros(6, dtype=np.float32)
        action_a = policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)
        action_c = policy.get_deterministic_action(obs)

    np.testing.assert_allclose(action_a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(action_c, np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_hierarchical_commander_policy_head_on_reopened_crossing_leash_caps_sustained_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": head_on},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": crossing},
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [1, 1, 1]
    cfg = _config()
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 2,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(_GuardEnv(range_m=5000.0))
        obs = np.zeros(6, dtype=np.float32)
        action_a = policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)
        action_c = policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)
        action_e = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action_a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(action_c, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(action_e, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is True
    assert metadata["commander_mode_constraint_reason"] == "max_consecutive_crossing_macro_steps_exceeded"
    assert metadata["commander_head_on_post_merge_reopened_crossing_leash_active"] is True


def test_hierarchical_commander_policy_secondary_clamp_forces_head_on_below_min_range():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": head_on},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": crossing},
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [1, 1]
    cfg = _config()
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_secondary_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "max_altitude_m": 5000.0,
        "min_abs_vp_lateral_to_range_ratio": 1.5,
        "altitude_drop_lookback_steps": 2,
        "min_altitude_drop_m": 150.0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=4000.0,
                altitude_sequence_m=[5200.0, 4900.0, 4700.0],
                vp_lateral_bias_m=7000.0,
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action_a = policy.get_deterministic_action(obs)
        policy.get_deterministic_action(obs)
        action_c = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action_a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(action_c, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "secondary_low_altitude_unresolved_lateral_descent"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_active"
        ]
        is True
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m"
        ]
        == 4700.0
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio"
        ]
        == 1.75
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback"
        ]
        == -500.0
    )


def test_hierarchical_commander_policy_target_threat_clamp_forces_head_on():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 4
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                first_pass_complete=True,
                target_in_attack_zone=True,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "target_attack_zone_reopened_head_on"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active"
        ]
        is True
    )


def test_hierarchical_commander_policy_target_threat_clamp_stays_narrow_without_target_threat():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                first_pass_complete=True,
                target_in_attack_zone=False,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active"
        ]
        is False
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason"
        ]
        == "target_not_in_attack_zone"
    )


def test_hierarchical_commander_policy_target_threat_clamp_promotes_head_on_to_recovery():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                first_pass_complete=True,
                target_in_attack_zone=True,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "target_attack_zone_reopened_head_on"
    )
    assert metadata["commander_selected_specialist"] == "post_merge_recovery"


def test_hierarchical_commander_policy_holds_first_forced_head_on_recovery_entry_once():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
    }
    cfg["commander"]["head_on_post_merge_first_recovery_entry_hold"] = {
        "enabled": True,
        "task_name": "head_on",
        "head_on_mode_id": 0,
        "recovery_mode_id": 2,
        "hold_window_steps": 3,
        "allowed_reasons": [
            "pre_threat_opening_overlateral_negative_forward_head_on"
        ],
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                min_range_so_far_m=120.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3200.0,
                vp_lateral_bias_m=9000.0,
                target_in_attack_zone=False,
                own_velocity_mps=250.0,
                target_velocity_mps=360.0,
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        first_action = policy.get_deterministic_action(obs)
        first_metadata = policy.get_last_step_metadata()
        second_action = policy.get_deterministic_action(obs)
        second_metadata = policy.get_last_step_metadata()
        third_action = policy.get_deterministic_action(obs)
        third_metadata = policy.get_last_step_metadata()
        fourth_action = policy.get_deterministic_action(obs)
        fourth_metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        first_action, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert first_metadata["commander_mode_id"] == 0
    assert (
        first_metadata["commander_head_on_post_merge_first_recovery_entry_hold_active"]
        is True
    )
    assert (
        first_metadata["commander_head_on_post_merge_first_recovery_entry_hold_original_reason"]
        == "pre_threat_opening_overlateral_negative_forward_head_on"
    )
    assert (
        first_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_armed_this_step"
        ]
        is True
    )
    assert (
        first_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining"
        ]
        == 3
    )
    assert first_metadata["commander_mode_constraint_reason"] == (
        "head_on_first_recovery_entry_hold"
    )

    np.testing.assert_allclose(
        second_action, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert second_metadata["commander_mode_id"] == 0
    assert (
        second_metadata["commander_head_on_post_merge_first_recovery_entry_hold_active"]
        is True
    )
    assert (
        second_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_armed_this_step"
        ]
        is False
    )
    assert (
        second_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining"
        ]
        == 2
    )
    assert second_metadata["commander_mode_constraint_reason"] == (
        "head_on_first_recovery_entry_hold"
    )
    np.testing.assert_allclose(
        third_action, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert third_metadata["commander_mode_id"] == 0
    assert (
        third_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining"
        ]
        == 1
    )
    assert third_metadata["commander_mode_constraint_reason"] == (
        "head_on_first_recovery_entry_hold"
    )
    np.testing.assert_allclose(
        fourth_action, np.array([0.2, 0.2, 0.2], dtype=np.float32)
    )
    assert fourth_metadata["commander_mode_id"] == 2
    assert (
        fourth_metadata["commander_head_on_post_merge_first_recovery_entry_hold_active"]
        is False
    )
    assert (
        fourth_metadata[
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining"
        ]
        == 0
    )
    assert fourth_metadata["commander_mode_constraint_reason"] == (
        "pre_threat_opening_overlateral_negative_forward_head_on"
    )


def test_hierarchical_commander_policy_pre_threat_entry_guard_blocks_opening_overlateral_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3200.0,
                vp_lateral_bias_m=9000.0,
                target_in_attack_zone=False,
                own_velocity_mps=250.0,
                target_velocity_mps=360.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "pre_threat_opening_overlateral_negative_forward_head_on"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active"
        ]
        is True
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps"
        ]
        >= 50.0
    )


def test_hierarchical_commander_policy_pre_threat_entry_guard_allows_non_opening_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3200.0,
                vp_lateral_bias_m=9000.0,
                target_in_attack_zone=False,
                own_velocity_mps=250.0,
                target_velocity_mps=250.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason"
        ]
        == "pre_threat_not_opening_fast_enough"
    )


def test_hierarchical_commander_policy_pre_threat_entry_guard_blocks_positive_lateral_when_forward_bias_not_negative_enough():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
        "pre_threat_positive_lateral_min_vp_lateral_bias_m": 0.0,
        "pre_threat_positive_lateral_max_vp_forward_bias_m": -5000.0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5200.0,
                min_range_so_far_m=40.0,
                first_pass_complete=True,
                vp_forward_bias_m=-4920.0,
                vp_lateral_bias_m=8200.0,
                target_in_attack_zone=False,
                own_velocity_mps=250.0,
                target_velocity_mps=360.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason"
        ]
        == "pre_threat_positive_lateral_forward_bias_not_negative_enough"
    )


def test_hierarchical_commander_policy_pre_threat_entry_guard_promotes_positive_lateral_when_forward_bias_is_strongly_negative():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_target_threat_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
        "pre_threat_positive_lateral_min_vp_lateral_bias_m": 0.0,
        "pre_threat_positive_lateral_max_vp_forward_bias_m": -5000.0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5600.0,
                min_range_so_far_m=90.0,
                first_pass_complete=True,
                vp_forward_bias_m=-6400.0,
                vp_lateral_bias_m=8900.0,
                target_in_attack_zone=False,
                own_velocity_mps=250.0,
                target_velocity_mps=360.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "pre_threat_opening_overlateral_negative_forward_head_on"
    )


def test_hierarchical_commander_policy_instantiates_double_dqn_when_configured():
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": MagicMock()},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": MagicMock()},
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0

    cfg = _config()
    cfg["commander"]["algorithm"] = "double_dqn"

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderDoubleDQNAgent",
        return_value=mock_commander,
    ) as mock_dqn_cls:
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        assert mock_dqn_cls.called is True
        assert mock_dqn_cls.call_args[1]["obs_dim"] == 6
        assert mock_dqn_cls.call_args[1]["action_dim"] == 2


def test_hierarchical_commander_policy_bootstraps_crossing_task_to_crossing_specialist_on_first_step():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("crossing_feasible")
        obs = np.zeros(6, dtype=np.float32)
        action = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert mock_commander.get_deterministic_action.call_count == 0
    assert metadata["commander_mode_id"] == 1
    assert metadata["commander_selected_specialist"] == "crossing_feasible"


def test_hierarchical_commander_policy_crossing_bootstrap_is_narrow_to_first_step():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("crossing_feasible")
        obs = np.zeros(6, dtype=np.float32)
        action_first = policy.get_deterministic_action(obs)
        action_second = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        action_first, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        action_second, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert mock_commander.get_deterministic_action.call_count == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_selected_specialist"] == "head_on"


def test_hierarchical_commander_policy_crossing_pre_merge_lock_holds_until_first_pass():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["crossing_pre_merge_mode_lock"] = {
        "enabled": True,
        "task_name": "crossing_feasible",
        "forced_specialist_key": "crossing_feasible",
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("crossing_feasible")
        env = _GuardEnv(first_pass_complete=False)
        policy.set_env(env)
        obs = np.zeros(6, dtype=np.float32)
        action_first = policy.get_deterministic_action(obs)
        action_second = policy.get_deterministic_action(obs)
        lock_metadata = policy.get_last_step_metadata()
        env._first_pass_complete = True
        action_third = policy.get_deterministic_action(obs)
        post_merge_metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        action_first, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        action_second, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        action_third, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert mock_commander.get_deterministic_action.call_count == 1
    assert lock_metadata["commander_crossing_pre_merge_mode_lock_active"] is True
    assert post_merge_metadata["commander_crossing_pre_merge_mode_lock_active"] is False
    assert post_merge_metadata["commander_selected_specialist"] == "head_on"


def test_hierarchical_commander_policy_head_on_overdeep_clamp_blocks_requested_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-10000.0,
                vp_lateral_bias_m=1500.0,
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "overdeep_low_lateral_reopened_head_on"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active"
        ]
        is True
    )


def test_hierarchical_commander_policy_head_on_overdeep_clamp_stays_narrow():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-8000.0,
                vp_lateral_bias_m=3500.0,
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active"
        ]
        is False
    )


def test_hierarchical_commander_policy_head_on_overdeep_clamp_promotes_head_on_to_recovery():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 2,
        "head_on_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-10000.0,
                vp_lateral_bias_m=1500.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "overdeep_low_lateral_reopened_head_on"
    )
    assert metadata["commander_selected_specialist"] == "post_merge_recovery"


def test_hierarchical_commander_policy_head_on_overdeep_clamp_blocks_negative_lateral_recovery():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 2,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "head_on_min_vp_lateral_bias_m": 0.0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.8,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-10000.0,
                vp_lateral_bias_m=-1500.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason"]
        == "overdeep_low_lateral_reopened_head_on"
    )


def test_hierarchical_commander_policy_head_on_close_range_reengagement_blocks_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 1
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 0,
        "head_on_min_range_so_far_m": 180.0,
        "close_range_max_range_m": 3000.0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 8000.0,
        "max_abs_vp_lateral_to_range_ratio": 1.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=2500.0,
                min_range_so_far_m=120.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3000.0,
                vp_lateral_bias_m=500.0,
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "close_range_reengagement_crossing"
    )


def test_hierarchical_commander_policy_head_on_close_range_reengagement_promotes_recovery_after_wide_merge():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "head_on_min_range_so_far_m": 180.0,
        "close_range_max_range_m": 3000.0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 8000.0,
        "max_abs_vp_lateral_to_range_ratio": 1.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=2500.0,
                min_range_so_far_m=190.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3000.0,
                vp_lateral_bias_m=500.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "close_range_reengagement_crossing"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_min_range_so_far_m"
        ]
        == 190.0
    )
    assert metadata["commander_selected_specialist"] == "post_merge_recovery"


def test_hierarchical_commander_policy_head_on_close_range_reengagement_keeps_head_on_when_merge_too_tight():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "head_on_min_range_so_far_m": 180.0,
        "close_range_max_range_m": 3000.0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 8000.0,
        "max_abs_vp_lateral_to_range_ratio": 1.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=2500.0,
                min_range_so_far_m=120.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3000.0,
                vp_lateral_bias_m=500.0,
            )
        )
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is False
    assert metadata["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active"] is True
    assert (
        metadata["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason"]
        == "close_range_reengagement_crossing"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_min_range_so_far_m"
        ]
        == 120.0
    )


def test_hierarchical_commander_policy_recovery_hold_keeps_recovery_while_geometry_stays_overlateral():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_recovery_hold"] = {
        "enabled": True,
        "task_name": "head_on",
        "recovery_mode_id": 2,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "min_abs_vp_lateral_bias_m": 6000.0,
        "min_abs_vp_lateral_to_range_ratio": 1.4,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-3500.0,
                vp_lateral_bias_m=8000.0,
            )
        )
        policy._active_mode_id = 2
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "recovery_reopened_geometry_still_overlateral"
    )
    assert metadata["commander_head_on_post_merge_recovery_hold_active"] is True
    assert metadata["commander_head_on_post_merge_recovery_hold_vp_lateral_bias_m"] == 8000.0


def test_hierarchical_commander_policy_recovery_hold_allows_release_after_geometry_improves():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.return_value = 0
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_recovery_hold"] = {
        "enabled": True,
        "task_name": "head_on",
        "recovery_mode_id": 2,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "min_abs_vp_lateral_bias_m": 6000.0,
        "min_abs_vp_lateral_to_range_ratio": 1.4,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                vp_forward_bias_m=-1500.0,
                vp_lateral_bias_m=2500.0,
            )
        )
        policy._active_mode_id = 2
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert metadata["commander_requested_mode_id"] == 0
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is False
    assert metadata["commander_head_on_post_merge_recovery_hold_active"] is False
    assert (
        metadata["commander_head_on_post_merge_recovery_hold_reason"]
        == "vp_lateral_bias_not_large_enough"
    )


def test_hierarchical_commander_policy_geometry_quality_guard_blocks_high_side_positive_forward_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [0, 1]
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_geometry_quality_guard"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "require_leash_active": True,
        "min_leash_active_steps": 2,
        "require_overdeep_seen": True,
        "min_overdeep_active_steps": 1,
        "altitude_trend_lookback_steps": 1,
        "min_altitude_drop_m": 250.0,
        "min_altitude_gain_m": 250.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "min_positive_vp_forward_bias_m": 1200.0,
        "min_abs_vp_lateral_bias_m": 2500.0,
        "min_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                altitude_sequence_m=[5000.0, 5400.0],
                vp_forward_bias_sequence_m=[-10000.0, 1500.0],
                vp_lateral_bias_sequence_m=[500.0, 3500.0],
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action_first = policy.get_deterministic_action(obs)
        action_second = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        action_first, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        action_second, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "geometry_quality_high_side_positive_forward_lateral"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active"
        ]
        is True
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps"
        ]
        == 2
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge"
        ]
        is True
    )


def test_hierarchical_commander_policy_geometry_quality_guard_blocks_low_side_negative_forward_crossing():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [0, 1]
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_geometry_quality_guard"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "require_leash_active": True,
        "min_leash_active_steps": 2,
        "require_overdeep_seen": True,
        "min_overdeep_active_steps": 1,
        "altitude_trend_lookback_steps": 1,
        "min_altitude_drop_m": 250.0,
        "min_altitude_gain_m": 250.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "min_positive_vp_forward_bias_m": 1200.0,
        "min_abs_vp_lateral_bias_m": 2500.0,
        "min_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                altitude_sequence_m=[5000.0, 4600.0],
                vp_forward_bias_sequence_m=[-10000.0, -9500.0],
                vp_lateral_bias_sequence_m=[500.0, 3500.0],
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        action_first = policy.get_deterministic_action(obs)
        action_second = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        action_first, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        action_second, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 0
    assert metadata["commander_mode_constraint_triggered"] is True
    assert (
        metadata["commander_mode_constraint_reason"]
        == "geometry_quality_low_side_negative_forward_lateral"
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback"
        ]
        == -400.0
    )


def test_hierarchical_commander_policy_geometry_quality_guard_stays_narrow_without_overdeep_history():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [0, 1]
    cfg = _config()
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_overdeep_clamp"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "max_abs_vp_lateral_to_range_ratio": 0.2,
    }
    cfg["commander"]["head_on_post_merge_reopened_crossing_geometry_quality_guard"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "require_leash_active": True,
        "min_leash_active_steps": 2,
        "require_overdeep_seen": True,
        "min_overdeep_active_steps": 1,
        "altitude_trend_lookback_steps": 1,
        "min_altitude_drop_m": 250.0,
        "min_altitude_gain_m": 250.0,
        "min_negative_vp_forward_bias_m": 9000.0,
        "min_positive_vp_forward_bias_m": 1200.0,
        "min_abs_vp_lateral_bias_m": 2500.0,
        "min_abs_vp_lateral_to_range_ratio": 0.5,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        policy.set_env(
            _GuardEnv(
                range_m=5000.0,
                first_pass_complete=True,
                altitude_sequence_m=[5000.0, 5400.0],
                vp_forward_bias_sequence_m=[200.0, 1500.0],
                vp_lateral_bias_sequence_m=[500.0, 3500.0],
            )
        )
        obs = np.zeros(6, dtype=np.float32)
        policy.get_deterministic_action(obs)
        action_second = policy.get_deterministic_action(obs)
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(
        action_second, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    assert metadata["commander_requested_mode_id"] == 1
    assert metadata["commander_mode_id"] == 1
    assert metadata["commander_mode_constraint_triggered"] is False
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active"
        ]
        is False
    )
    assert (
        metadata[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason"
        ]
        == "overdeep_not_seen"
    )


def test_hierarchical_commander_policy_uses_checkpoint_action_dim_when_mode_registry_expands():
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    mock_registry = {
        0: {"id": 0, "name": "head_on_specialist", "specialist_key": "head_on", "policy": MagicMock()},
        1: {"id": 1, "name": "crossing_specialist", "specialist_key": "crossing_feasible", "policy": MagicMock()},
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": MagicMock(),
        },
    }
    mock_commander = MagicMock()

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ) as commander_ctor, patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.torch.load",
        return_value={"action_dim": 2},
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )

    assert policy.commander_action_dim == 2
    assert commander_ctor.call_args.kwargs["action_dim"] == 2


def test_hierarchical_commander_policy_forced_recovery_mode_sets_runtime_specialist_context():
    head_on = MagicMock()
    head_on.get_deterministic_action.return_value = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    crossing = MagicMock()
    crossing.get_deterministic_action.return_value = np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    recovery = MagicMock()
    recovery.get_deterministic_action.return_value = np.array(
        [0.2, 0.2, 0.2], dtype=np.float32
    )
    mock_registry = {
        0: {
            "id": 0,
            "name": "head_on_specialist",
            "specialist_key": "head_on",
            "policy": head_on,
        },
        1: {
            "id": 1,
            "name": "crossing_specialist",
            "specialist_key": "crossing_feasible",
            "policy": crossing,
        },
        2: {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "policy": recovery,
        },
    }
    mock_commander = MagicMock()
    mock_commander.get_deterministic_action.side_effect = [1]
    cfg = _config()
    cfg["commander"]["modes"].append(
        {
            "id": 2,
            "name": "post_merge_recovery_specialist",
            "specialist_key": "post_merge_recovery",
            "specialist_profile": "post_merge_recovery",
            "source_specialist_key": "head_on",
            "checkpoint": "recovery.pt",
            "config_path": "recovery.yaml",
        }
    )
    cfg["commander"]["macro_action_repeat_steps"] = 1
    cfg["commander"]["head_on_post_merge_reopened_crossing_leash"] = {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 2,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "max_consecutive_crossing_macro_steps": 0,
    }

    with patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.load_frozen_specialist_registry",
        return_value=mock_registry,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.CommanderPPOAgent",
        return_value=mock_commander,
    ), patch(
        "uav_vpp_guidance.evaluation.hierarchical_commander_policy.torch.load",
        return_value={"action_dim": 2},
    ):
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        policy = HierarchicalCommanderPolicy(
            checkpoint_path="commander.pt",
            config=cfg,
            obs_dim=6,
            device="cpu",
        )
        policy.set_task_name("head_on")
        env = _GuardEnv(range_m=5000.0)
        env.set_runtime_specialist_context = MagicMock()
        policy.set_env(env)
        action = policy.get_deterministic_action(np.zeros(6, dtype=np.float32))
        metadata = policy.get_last_step_metadata()

    np.testing.assert_allclose(action, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    env.set_runtime_specialist_context.assert_called_once_with(
        specialist_key="post_merge_recovery",
        specialist_profile="post_merge_recovery",
        specialist_mode_name="post_merge_recovery_specialist",
    )
    assert metadata["commander_mode_id"] == 2
    assert metadata["commander_selected_specialist"] == "post_merge_recovery"
    assert metadata["commander_selected_specialist_profile"] == "post_merge_recovery"
    assert metadata["commander_selected_source_specialist"] == "head_on"
