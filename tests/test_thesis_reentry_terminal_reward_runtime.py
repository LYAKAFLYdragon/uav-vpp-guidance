from __future__ import annotations

from uav_vpp_guidance.training.thesis_reentry_terminal_reward_pilot import _terminal_reward_config


def test_runtime_extracts_only_the_four_preregistered_terminal_reward_values():
    config = {
        "reward": {
            "terminal_outcome_reward": {
                "win_bonus": 1.0,
                "loss_penalty": 1.0,
                "draw_penalty": 0.25,
                "ego_crash_oob_supplemental_penalty": 1.0,
                "outcome_mapping": {"target_crash_or_out_of_bounds": "win"},
                "applies_only_on_terminal_step": True,
            }
        }
    }

    reward = _terminal_reward_config(config)

    assert reward.win_bonus == 1.0
    assert reward.loss_penalty == 1.0
    assert reward.draw_penalty == 0.25
    assert reward.ego_crash_oob_supplemental_penalty == 1.0
