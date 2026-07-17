from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.tactical_basis_motif_policy import (
    MotifPulseConfig,
    TacticalBasisMotifError,
    compile_tactical_basis_motif_action,
)


def _basis():
    return (
        np.array([1000.0, 0.0, 0.0]),
        np.array([0.0, 1600.0, 0.0]),
        np.array([0.0, 0.0, 600.0]),
    )


def _states():
    return {"position_m": [0.0, 0.0, 5000.0]}, {"position_m": [1000.0, 0.0, 5200.0]}


def test_lateral_break_is_projected_through_active_io_basis():
    own, target = _states()
    receipt = compile_tactical_basis_motif_action(
        motif="lateral_break_same_side",
        intervention_step=3,
        reference_action=[0.0, 0.0, 0.0],
        basis_vectors=_basis(),
        own_state=own,
        target_state=target,
        config=MotifPulseConfig(),
    )

    assert receipt["basis_rank"] == 3
    assert receipt["pulse_scale"] == 1.0
    assert receipt["applied_action"] == pytest.approx([0.0, 0.35, 0.0])
    assert receipt["actual_world_delta_m"] == pytest.approx([0.0, 560.0, 0.0])
    assert receipt["projection_residual_norm_m"] == pytest.approx(0.0)


def test_range_extension_respects_the_existing_action_bound():
    own, target = _states()
    receipt = compile_tactical_basis_motif_action(
        motif="range_extension_los",
        intervention_step=3,
        reference_action=[0.0, 0.0, 0.0],
        basis_vectors=_basis(),
        own_state=own,
        target_state=target,
        config=MotifPulseConfig(),
    )

    assert receipt["delta_limited"] is True
    # LOS is three-dimensional here, so the requested range-opening direction
    # retains a vertical component while the longitudinal axis reaches its cap.
    assert receipt["applied_action"][0] == pytest.approx(-0.35)
    assert receipt["applied_action"][1] == pytest.approx(0.0)
    assert receipt["actual_world_delta_m"] == pytest.approx([-350.0, 0.0, -137.28129459672883])
    assert receipt["projection_residual_norm_m"] == pytest.approx(336.406, abs=1e-2)


def test_rank_deficient_tactical_basis_is_rejected_without_fallback():
    own, target = _states()
    with pytest.raises(TacticalBasisMotifError, match="rank-3"):
        compile_tactical_basis_motif_action(
            motif="vertical_separation",
            intervention_step=1,
            reference_action=[0.0, 0.0, 0.0],
            basis_vectors=(np.array([1.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
            own_state=own,
            target_state=target,
            config=MotifPulseConfig(),
        )
