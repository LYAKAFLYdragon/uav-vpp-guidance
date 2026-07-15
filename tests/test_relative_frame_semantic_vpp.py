from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_reentry_v11_relative_frame import summarize_records
from uav_vpp_guidance.virtual_point.relative_frame_semantic import (
    RelativeFrameSemanticCompiler,
    RelativeFrameSemanticConfig,
    RelativeFrameSemanticError,
)


ROOT = Path(__file__).resolve().parent.parent
RANGES = ((-1500.0, 1500.0), (-800.0, 800.0), (-500.0, 500.0))


def _base(*, ata_deg: float = 80.0, altitude_diff_m: float = 0.0) -> np.ndarray:
    vector = np.zeros(16, dtype=np.float64)
    vector[0] = 0.30  # 1500 m; below the range-extension target.
    vector[1] = -0.30  # -60 m/s; re-entry is closing again.
    vector[2] = altitude_diff_m / 10000.0
    vector[8] = np.sin(np.deg2rad(ata_deg))
    vector[9] = np.cos(np.deg2rad(ata_deg))
    vector[10] = 0.0
    vector[11] = 1.0
    vector[12] = 0.70
    vector[13] = 0.75
    vector[14] = 0.55
    vector[15] = 0.50
    return vector


def _states(*, mirror: int = 1) -> tuple[dict, dict]:
    return (
        {"position_m": [0.0, 0.0, 5500.0], "velocity_vector_mps": [280.0, 0.0, 0.0]},
        {"position_m": [1500.0, 200.0 * mirror, 5500.0], "velocity_vector_mps": [330.0, 30.0 * mirror, 0.0]},
    )


def test_range_extension_request_moves_vpp_toward_ego_along_los():
    own, target = _states()
    receipt = RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig()).compile(
        reference_action=[0.0, 0.0, 0.0],
        base_geometry=_base(),
        own_state=own,
        target_state=target,
        raw_action_ranges=RANGES,
        profile_name="range_extension",
        enabled_axes={"range_extension": True, "signed_turn": False, "vertical_energy": False},
    )

    delta = np.asarray(receipt["actual_semantic_world_delta_m"])
    los = np.asarray(target["position_m"], dtype=float) - np.asarray(own["position_m"], dtype=float)
    los /= np.linalg.norm(los)
    assert receipt["commands"]["range_extension"] > 0.0
    assert np.dot(delta, los) < 0.0
    assert receipt["actual_semantic_delta_norm_m"] >= 100.0


def test_disabled_axes_preserve_the_reference_action_exactly():
    own, target = _states()
    receipt = RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig()).compile(
        reference_action=[0.2, -0.3, 0.1],
        base_geometry=_base(),
        own_state=own,
        target_state=target,
        raw_action_ranges=RANGES,
        profile_name="range_extension",
        enabled_axes={"range_extension": False, "signed_turn": False, "vertical_energy": False},
    )

    assert receipt["actual_semantic_delta_norm_m"] == pytest.approx(0.0, abs=1e-9)
    assert receipt["applied_action"] == pytest.approx([0.2, -0.3, 0.1], abs=1e-9)


def test_compiler_fails_closed_for_an_invalid_base_contract():
    own, target = _states()
    with pytest.raises(RelativeFrameSemanticError, match="16-D"):
        RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig()).compile(
            reference_action=[0.0, 0.0, 0.0],
            base_geometry=[0.0] * 15,
            own_state=own,
            target_state=target,
            raw_action_ranges=RANGES,
            profile_name="range_extension",
            enabled_axes={},
        )


def _record(method: str, opponent: str, pair_key: str, *, semantic: float, vpp: float, guidance: float, rate: float, loss: float) -> dict:
    step = {
        "actual_semantic_delta_norm_m": semantic,
        "env_vpp_world_offset_m": [vpp, 0.0, 0.0],
        "guidance_command_normalized": [guidance, 0.0, 0.0],
        "after_range_rate_mps": rate,
    }
    return {
        "method": method,
        "opponent": opponent,
        "pair_key": pair_key,
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "handoff_state_hash": "shared",
        "intent_loss_auc20": loss,
        "ego_failure": False,
        "window": [step] * 20,
    }


def test_summary_requires_all_three_transfer_layers_and_separate_opponents():
    records = []
    for opponent in ("expert", "end_to_end", "independent_ppo_vpp"):
        for index in range(2):
            key = f"{opponent}-{index}"
            records.append(_record("frozen_fixed_head_on", opponent, key, semantic=0.0, vpp=0.0, guidance=0.0, rate=-40.0, loss=1.0))
            for method in ("range_extension_los", "range_extension_los_turn", "range_extension_full"):
                records.append(_record(method, opponent, key, semantic=200.0, vpp=200.0, guidance=0.1, rate=-30.0, loss=0.98))
    summary = summarize_records(
        records,
        scenarios_per_opponent=2,
        min_window_steps=20,
        min_pairs=2,
        min_semantic_delta_norm_m=100.0,
        min_vpp_delta_norm_m=100.0,
        min_guidance_delta_norm=0.02,
        min_range_rate_delta_mps=5.0,
        practical_intent_loss_delta=-0.01,
        safety_delta_max=0.05,
    )

    assert summary["passes"] is True
    assert summary["semantic_transfer_candidates"] == ["range_extension_los", "range_extension_los_turn", "range_extension_full"]


def test_v11_design_is_nonlearning_and_uses_a_new_manifest(tmp_path: Path):
    path = ROOT / "scripts" / "run_thesis_five_state_reentry_v11_relative_frame_semantic.py"
    spec = importlib.util.spec_from_file_location("v11_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v11_relative_frame_semantic_design.yaml").read_text(encoding="utf-8"))

    assert config["authorization"]["execution_permitted"] is False
    assert config["authorization"]["training_permitted"] is False
    assert config["authorization"]["reward_change_permitted"] is False
    assert config["semantic_compiler"]["raw_action_offset_frame"] == "world_neu"
    # The real output root is intentionally occupied after the one-time run.
    # Validate design invariants against an otherwise identical fresh root.
    config["outputs"]["root"] = str(tmp_path / "fresh_v11_output")
    temporary_config = tmp_path / "v11_design.yaml"
    temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = module.validate_design(temporary_config)
    assert result["planned_records"] == 144


def test_v11_verifier_only_tolerates_serialization_scale_float_noise():
    path = ROOT / "scripts" / "verify_thesis_five_state_reentry_v11_relative_frame_semantic.py"
    spec = importlib.util.spec_from_file_location("v11_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module._equivalent({"value": 0.2499999999999}, {"value": 0.25})
    assert not module._equivalent({"value": 0.25}, {"value": 0.2501})
