"""Deterministic offline transition-table tests for F16 evidence candidates."""
import json
from pathlib import Path

from uav_vpp_guidance.evaluation.f16_mode_state_evidence import (
    CANONICAL_ASPECT_THRESHOLD_DEG,
    ExplicitModeStateMachine,
    LegacyPermanentLatch,
    ModeEvent,
    ModeState,
    ModeSwitchParameters,
    ReleaseableLatch,
    compare_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


def _params(**overrides):
    return ModeSwitchParameters.from_config({"enabled": True, **overrides})


def test_non_trigger_and_entry_are_deterministic_and_observable():
    model = ExplicitModeStateMachine(_params())
    non_trigger = model.step(ModeEvent(2000.0, -120.0, 45.0))
    entry = model.step(ModeEvent(2000.0, -120.0, 10.0))
    assert non_trigger.state == ModeState.VPP.value
    assert non_trigger.reason == "entry_guard_not_met"
    assert entry.transition == "enter"
    assert entry.state == ModeState.DIRECT_TRACK_PN.value
    assert entry.mode_switch_effective and entry.virtual_point_source == "direct_track"
    assert entry.effective_guidance_mode == "proportional_navigation"


def test_permanent_releaseable_and_state_machine_models_differ_only_by_declared_policy():
    events = [ModeEvent(2000.0, -120.0, 10.0), ModeEvent(3600.0, -100.0, 28.0)]
    legacy = LegacyPermanentLatch(_params())
    releaseable = ReleaseableLatch(_params())
    state_machine = ExplicitModeStateMachine(_params(min_active_dwell_steps=1))
    assert [legacy.step(event).state for event in events][-1] == ModeState.DIRECT_TRACK_PN.value
    assert [releaseable.step(event).state for event in events][-1] == ModeState.VPP.value
    assert [state_machine.step(event).state for event in events][-1] == ModeState.VPP.value


def test_state_machine_hold_dwell_prevents_release_chatter_then_releases_and_reenters():
    model = ExplicitModeStateMachine(_params(min_active_dwell_steps=2))
    assert model.step(ModeEvent(2000.0, -120.0, 10.0)).transition == "enter"
    held = model.step(ModeEvent(3600.0, -100.0, 28.0))
    released = model.step(ModeEvent(3600.0, -100.0, 28.0))
    reentry = model.step(ModeEvent(2000.0, -120.0, 10.0))
    assert held.transition == "hold_dwell" and held.state == ModeState.DIRECT_TRACK_PN.value
    assert released.transition == "release" and released.state == ModeState.VPP.value
    assert reentry.transition == "enter" and reentry.state == ModeState.DIRECT_TRACK_PN.value


def test_reset_disabled_and_invalid_geometry_have_explicit_states_and_reasons():
    model = ExplicitModeStateMachine(_params(max_active_steps=1))
    model.step(ModeEvent(2000.0, -120.0, 10.0))
    timeout = model.step(ModeEvent(2000.0, -120.0, 10.0))
    reset = model.reset()
    disabled = ExplicitModeStateMachine(ModeSwitchParameters.from_config({})).step(ModeEvent(2000.0, -120.0, 10.0))
    invalid = ExplicitModeStateMachine(_params()).step(ModeEvent(float("nan"), -120.0, 10.0))
    assert timeout.state == ModeState.FAILSAFE.value and timeout.transition == "timeout_failsafe"
    assert reset.state == ModeState.VPP.value and reset.transition == "reset"
    assert disabled.state == ModeState.DISABLED.value and disabled.reason == "mode_switch_disabled"
    assert invalid.state == ModeState.FAILSAFE.value and invalid.reason == "invalid_geometry"


def test_missing_threshold_has_one_canonical_25_degree_policy_and_explicit_override_wins():
    missing = ModeSwitchParameters.from_config({"enabled": True})
    override = ModeSwitchParameters.from_config({"enabled": True, "aspect_threshold_deg": 15.0})
    assert missing.aspect_threshold_deg == CANONICAL_ASPECT_THRESHOLD_DEG
    assert override.aspect_threshold_deg == 15.0
    assert ExplicitModeStateMachine(missing).step(ModeEvent(2000.0, -120.0, 20.0)).state == ModeState.DIRECT_TRACK_PN.value
    assert ExplicitModeStateMachine(override).step(ModeEvent(2000.0, -120.0, 20.0)).state == ModeState.VPP.value


def test_crossing_requires_explicit_candidate_configuration():
    crossing_event = ModeEvent(2000.0, -120.0, 90.0)
    assert ExplicitModeStateMachine(_params()).step(crossing_event).state == ModeState.VPP.value
    crossing = ExplicitModeStateMachine(_params(crossing_aspect_threshold_deg=10.0)).step(crossing_event)
    assert crossing.state == ModeState.DIRECT_TRACK_PN.value and crossing.reason == "entry_crossing"


def test_candidate_comparison_and_gate_preserve_runtime_pending_frozen_matrix_evidence():
    evidence = compare_candidates()
    assert evidence["candidate_traces"]["legacy_permanent_latch"]["entry_hold_release_reentry"][2]["state"] == ModeState.DIRECT_TRACK_PN.value
    gate = json.loads((SPEC / "evidence/f16_mode_state_bundle/f16_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_current_runtime_latch_pending_approved_gate"
