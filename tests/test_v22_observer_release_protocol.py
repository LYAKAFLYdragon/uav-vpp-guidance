"""Integration safeguards for the non-learning V22 observer-release protocol."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_v17_path_context_preflight as v17
from run_thesis_five_state_v22_observer_release_probe import Episode
from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import V21AnchorProtocolError
from uav_vpp_guidance.evaluation.v22_observer_release_protocol import V22ObserverReleaseMixin


class _FakeEnv:
    def __init__(self) -> None:
        self.modes: list[tuple[str, str]] = []

    def set_runtime_vpp_anchor_mode(self, *, mode: str, reason: str) -> None:
        self.modes.append((mode, reason))


class _FakeEpisode(V22ObserverReleaseMixin):
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.total_steps = 0
        self.history: list[object] = []
        self.last_info: dict[str, object] = {}
        self._v22_reset_observer_protocol()


def _receipt(*, valid: bool = True, fallback: bool = False, consumed: bool = False) -> dict[str, object]:
    return {
        "runtime_vpp_anchor_mode": "current_target",
        "observer_prediction_updated": True,
        "observer_prediction_valid": valid,
        "observer_prediction_fallback": fallback,
        "observer_prediction_consumed_by_control": consumed,
        "prediction_valid": valid,
        "prediction_fallback": fallback,
    }


def test_v22_releases_once_after_a_valid_ten_frame_observer_receipt() -> None:
    episode = _FakeEpisode()
    episode._v22_after_observation()
    episode.total_steps = 9
    episode.history = [object()] * 10
    episode.last_info = _receipt()

    episode._v22_after_observation()

    assert episode._v21_released is True
    assert episode.env.modes == [
        ("current_target", "v22_observer_warmup"),
        ("predicted_target", "v22_first_valid_observer_receipt"),
    ]
    assert episode._v22_observer_trace[-1]["observer_prediction_consumed_by_control"] is False


def test_v22_fails_closed_if_warmup_observer_output_enters_control() -> None:
    episode = _FakeEpisode()
    episode.total_steps = 1
    episode.history = [object()]
    episode.last_info = _receipt(consumed=True)

    with pytest.raises(V21AnchorProtocolError, match="entered control"):
        episode._v22_after_observation()

    assert episode._v22_warmup_control_consumption is True


def test_v22_fails_closed_when_prediction_falls_back_after_release() -> None:
    episode = _FakeEpisode()
    episode._v21_released = True
    episode.total_steps = 11
    episode.history = [object()] * 10
    episode.last_info = _receipt(valid=False, fallback=True)

    with pytest.raises(V21AnchorProtocolError, match="invalid after V22"):
        episode._v22_after_observation()

    assert episode._v22_post_release_prediction_fallback is True


def test_v22_episode_observe_calls_observer_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a silent base-class-only rollout from producing default-valued evidence."""
    episode = object.__new__(Episode)
    calls: list[str] = []
    monkeypatch.setattr(v17.PathContextEpisode, "_observe", lambda self, observation: calls.append("base"))
    episode._v22_after_observation = lambda: calls.append("v22")

    Episode._observe(episode, {})

    assert calls == ["base", "v22"]
