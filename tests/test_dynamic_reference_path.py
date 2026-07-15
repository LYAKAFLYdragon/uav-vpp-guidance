import numpy as np
import pytest
import yaml
from pathlib import Path

from uav_vpp_guidance.virtual_point.dynamic_reference_path import (
    DynamicReferencePathConfig,
    DynamicReferencePathError,
    build_three_anchor_path,
    compile_path_anchor_action,
    path_progress_from_range_rate,
    semantic_profile_for_anchor,
    select_path_anchor,
)


def test_three_anchor_path_uses_only_current_positions_and_frozen_prediction():
    config = DynamicReferencePathConfig(escape_extent_m=100.0, connector_fraction=0.5)
    path = build_three_anchor_path(
        own_position_neu=[1000.0, 0.0, 5000.0],
        target_position_neu=[0.0, 0.0, 5000.0],
        predicted_target_position_neu=[200.0, 300.0, 5000.0],
        config=config,
    )
    assert np.allclose(path[0], [100.0, 0.0, 5000.0])
    assert np.allclose(path[1], [150.0, 150.0, 5000.0])
    assert np.allclose(path[2], [200.0, 300.0, 5000.0])


def test_progress_selects_all_three_anchors_without_time_indexing():
    config = DynamicReferencePathConfig(opening_rate_start_mps=0.0, opening_rate_full_mps=60.0)
    path = np.arange(9, dtype=float).reshape(3, 3)
    assert select_path_anchor(path, path_progress_from_range_rate(-10.0, config))[0] == 0
    assert select_path_anchor(path, path_progress_from_range_rate(30.0, config))[0] == 1
    assert select_path_anchor(path, path_progress_from_range_rate(90.0, config))[0] == 2
    assert semantic_profile_for_anchor(0) == "range_extension"
    assert semantic_profile_for_anchor(1) == "range_extension"
    assert semantic_profile_for_anchor(2) == "reentry_preparation"


def test_path_rejects_coincident_or_nonfinite_geometry():
    config = DynamicReferencePathConfig()
    with pytest.raises(DynamicReferencePathError, match="distinct"):
        build_three_anchor_path(
            own_position_neu=[0.0, 0.0, 0.0],
            target_position_neu=[0.0, 0.0, 0.0],
            predicted_target_position_neu=[1.0, 0.0, 0.0],
            config=config,
        )


def test_path_anchor_action_reuses_the_existing_bounded_vpp_action_head():
    receipt = compile_path_anchor_action(
        semantic_action=[0.0, 0.0, 0.0],
        selected_anchor_neu=[100.0, -50.0, 25.0],
        generator_anchor_neu=[0.0, 0.0, 0.0],
        raw_action_ranges=[[-1000.0, 1000.0], [-1000.0, 1000.0], [-1000.0, 1000.0]],
        raw_action_offset_frame="world_neu",
        own_state={"position_neu": [0.0, 0.0, 0.0]},
        target_state={"position_neu": [500.0, 0.0, 0.0]},
    )
    assert np.allclose(receipt["applied_action"], [0.1, -0.05, 0.025])
    assert np.allclose(receipt["applied_vpp_position_neu"], [100.0, -50.0, 25.0])
    assert receipt["clipped"] is False


def test_path_anchor_action_reports_clipping_instead_of_expanding_the_interface():
    receipt = compile_path_anchor_action(
        semantic_action=[0.0, 0.0, 0.0],
        selected_anchor_neu=[5000.0, 0.0, 0.0],
        generator_anchor_neu=[0.0, 0.0, 0.0],
        raw_action_ranges=[[-1000.0, 1000.0], [-1000.0, 1000.0], [-1000.0, 1000.0]],
        raw_action_offset_frame="world_neu",
        own_state={"position_neu": [0.0, 0.0, 0.0]},
        target_state={"position_neu": [500.0, 0.0, 0.0]},
    )
    assert receipt["clipped"] is True
    assert np.allclose(receipt["applied_action"], [1.0, 0.0, 0.0])
    assert receipt["actual_path_delta_norm_m"] < receipt["requested_path_delta_norm_m"]


def test_v16_design_keeps_path_progress_separate_from_training_and_terminal_semantics():
    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "config/experiment/thesis_five_state_reentry_v16_dynamic_reference_path_design.yaml").read_text(encoding="utf-8"))
    assert config["authorization"]["training_permitted"] is False
    assert config["contract"]["future_simulator_state"] == "prohibited"
    assert config["contract"]["synthetic_target"] == "prohibited"
    assert config["path"]["anchor_count"] == 3
    assert config["path"]["semantic_profiles_by_anchor"] == ["range_extension", "range_extension", "reentry_preparation"]
