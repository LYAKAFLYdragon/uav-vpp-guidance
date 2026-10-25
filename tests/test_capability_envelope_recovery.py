from pathlib import Path

import importlib.util
import sys
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_envelope_recovery_v1.yaml"
MODULE_PATH = REPO_ROOT / "src/uav_vpp_guidance/evaluation/capability_envelope_recovery.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capability_envelope_recovery", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state(**overrides):
    state = {
        "target_x_m": 750.0,
        "target_y_m": 150.0,
        "altitude_m": 5000.0,
        "range_m": 765.0,
        "clipping_rate": 0.0,
        "envelope_condition_valid": True,
        "backend_fallback_occurred": False,
        "terminal": False,
    }
    state.update(overrides)
    return state


def test_config_is_design_only_and_training_locked():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-V1-20260903"
    assert config["protocol"]["training_started"] is False
    assert config["protocol"]["training_permitted"] is False
    assert config["protocol"]["execution_authorized"] is False
    assert config["recovery_tactic"]["max_recovery_steps"] == 48
    assert config["recovery_tactic"]["required_dwell_steps"] == 10
    assert len(config["stable_envelope"]["stable_points_m"]) == 11


def test_inside_envelope_keeps_nominal_h0_mode():
    module = _load_module()
    policy = module.CapabilityEnvelopeRecoveryPolicy(
        [module.EnvelopePoint(750.0, 150.0)]
    )
    decision = policy.decide(_state())
    assert decision.phase == "nominal"
    assert decision.mode == "h0_current_geometry"


def test_outside_envelope_starts_bounded_recovery():
    module = _load_module()
    policy = module.CapabilityEnvelopeRecoveryPolicy(
        [module.EnvelopePoint(750.0, 150.0)], max_recovery_steps=2
    )
    decision = policy.decide(_state(target_x_m=750.0, target_y_m=250.0))
    assert decision.phase == "recover"
    assert decision.mode == "crossing"
    assert decision.target == module.EnvelopePoint(750.0, 150.0)
    assert decision.elapsed_steps == 1


def test_recovery_requires_dwell_before_returning_to_h0():
    module = _load_module()
    policy = module.CapabilityEnvelopeRecoveryPolicy(
        [module.EnvelopePoint(750.0, 150.0)], required_dwell_steps=2
    )
    policy.decide(_state(target_x_m=750.0, target_y_m=250.0))
    first = policy.decide(_state())
    second = policy.decide(_state())
    assert first.phase == "hold"
    assert first.mode == "crossing"
    assert second.phase == "complete"
    assert second.mode == "h0_current_geometry"


def test_recovery_aborts_on_safety_violation():
    module = _load_module()
    policy = module.CapabilityEnvelopeRecoveryPolicy(
        [module.EnvelopePoint(750.0, 150.0)]
    )
    decision = policy.decide(_state(target_x_m=750.0, target_y_m=250.0, clipping_rate=0.31))
    assert decision.phase == "abort"
    assert decision.reason == "clipping_above_limit"


def test_recovery_aborts_when_calibration_conditions_are_unverified():
    module = _load_module()
    policy = module.CapabilityEnvelopeRecoveryPolicy(
        [module.EnvelopePoint(750.0, 150.0)]
    )
    decision = policy.decide(_state(envelope_condition_valid=False))
    assert decision.phase == "abort"
    assert decision.reason == "envelope_condition_unverified"
