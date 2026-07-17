from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.temporal_reference_compiler import (
    TemporalReferenceCompilerError,
    TemporalReferenceConfig,
    TemporalReferenceTemplate,
    c1_window,
    compile_temporal_reference_program,
    compile_temporal_reference_step,
)


CONFIG = TemporalReferenceConfig(horizon_steps=40, high_level_dt_s=0.2, max_action_delta=0.35, max_action_slew=0.10)
BASIS = [[1000.0, 0.0, 0.0], [0.0, 800.0, 0.0], [0.0, 0.0, 500.0]]


def test_compiler_emits_a_deterministic_c1_bounded_program() -> None:
    template = TemporalReferenceTemplate("combined_break_extension", (0.20, 0.24, 0.22))
    references = [[0.0, 0.0, 0.0]] * CONFIG.horizon_steps
    receipts = compile_temporal_reference_program(
        template=template,
        reference_actions=references,
        basis_vectors_by_step=[BASIS] * CONFIG.horizon_steps,
        config=CONFIG,
        lateral_sign=1,
        vertical_sign=-1,
    )
    repeat = compile_temporal_reference_program(
        template=template,
        reference_actions=references,
        basis_vectors_by_step=[BASIS] * CONFIG.horizon_steps,
        config=CONFIG,
        lateral_sign=1,
        vertical_sign=-1,
    )
    actions = np.asarray([item["applied_action"] for item in receipts])
    assert receipts == repeat
    assert receipts[0]["c1_window"] == 0.0
    assert receipts[-1]["c1_window"] == pytest.approx(0.0, abs=1e-12)
    assert all(item["uses_future_state"] is False for item in receipts)
    assert np.max(np.abs(actions)) <= 1.0
    assert np.max(np.abs(np.diff(actions, axis=0))) <= CONFIG.max_action_slew + 1e-12
    assert max(item["basis_rank"] for item in receipts) == 3
    assert max(item["projection_residual_norm_m"] for item in receipts) == pytest.approx(0.0)


def test_template_signs_change_only_the_requested_tactical_axes() -> None:
    template = TemporalReferenceTemplate("lateral_extension", (0.20, 0.32, 0.0))
    positive = compile_temporal_reference_step(
        template=template,
        step_index=20,
        reference_action=[0.0, 0.0, 0.0],
        previous_applied_action=[0.0, 0.0, 0.0],
        basis_vectors_world_m=BASIS,
        config=CONFIG,
        lateral_sign=1,
    )
    negative = compile_temporal_reference_step(
        template=template,
        step_index=20,
        reference_action=[0.0, 0.0, 0.0],
        previous_applied_action=[0.0, 0.0, 0.0],
        basis_vectors_world_m=BASIS,
        config=CONFIG,
        lateral_sign=-1,
    )
    assert positive["requested_reference_world_delta_m"][0] == pytest.approx(
        negative["requested_reference_world_delta_m"][0]
    )
    assert positive["requested_reference_world_delta_m"][1] == pytest.approx(
        -negative["requested_reference_world_delta_m"][1]
    )
    assert positive["requested_reference_world_delta_m"][2] == pytest.approx(0.0)


def test_compiler_rejects_out_of_envelope_template_and_singular_basis() -> None:
    with pytest.raises(TemporalReferenceCompilerError, match="exceeds the frozen max_action_delta"):
        compile_temporal_reference_step(
            template=TemporalReferenceTemplate("vertical_extension", (0.20, 0.0, 0.50)),
            step_index=1,
            reference_action=[0.0, 0.0, 0.0],
            previous_applied_action=[0.0, 0.0, 0.0],
            basis_vectors_world_m=BASIS,
            config=CONFIG,
        )
    with pytest.raises(TemporalReferenceCompilerError, match="rank-3"):
        compile_temporal_reference_step(
            template=TemporalReferenceTemplate("vertical_extension", (0.18, 0.0, 0.28)),
            step_index=1,
            reference_action=[0.0, 0.0, 0.0],
            previous_applied_action=[0.0, 0.0, 0.0],
            basis_vectors_world_m=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            config=CONFIG,
        )


def test_c1_window_has_zero_endpoints_and_is_symmetric() -> None:
    values = np.asarray([c1_window(step, CONFIG) for step in range(CONFIG.horizon_steps)])
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(values, values[::-1])
    assert np.max(values) <= 1.0
