"""Static F14 observation availability and migration-gate evidence; no schema mutation."""
from __future__ import annotations


BASE_FEATURES = ["range_m", "range_rate_mps", "altitude_diff_m", "speed_diff_mps", "los_azimuth_sin", "los_azimuth_cos", "los_elevation_sin", "los_elevation_cos", "ata_sin", "ata_cos", "aa_sin", "aa_cos", "own_speed", "target_speed", "own_altitude", "target_altitude"]


def audit() -> dict:
    signals = [
        {"signal": "guidance_gains", "status": "existing_optional", "source": "current_gains", "units": "gain-specific", "update": "guidance step", "normalization": "none", "availability": "include_gains (+2)", "leakage": "none; controller parameters"},
        {"signal": "virtual_point_tracking_error", "status": "existing_optional", "source": "last_virtual_point-target_position", "units": "m", "update": "environment step", "normalization": "5000m xy, 10000m z", "availability": "include_guidance_state (+3)", "leakage": "must remain causal"},
        {"signal": "command_saturation", "status": "existing_optional", "source": "last command clipping", "units": "binary", "update": "environment step", "normalization": "binary", "availability": "include_saturation (+3)", "leakage": "prior/applied command only"},
        {"signal": "prediction_state", "status": "existing_optional", "source": "prediction integration", "units": "normalized m/mps/variance", "update": "prediction step", "normalization": "integration-defined", "availability": "prediction flags (+14)", "leakage": "valid/fallback required"},
        {"signal": "aoa_beta_roll_angular_rates", "status": "candidate_absent", "source": "own aircraft state", "units": "rad, rad/s", "update": "sensor/backend step", "normalization": "proposal required", "availability": "not in policy vector", "leakage": "current/past measurement only; backend parity required"},
        {"signal": "command_tracking_error", "status": "candidate_absent", "source": "applied command and measured state", "units": "channel-specific", "update": "control step", "normalization": "proposal required", "availability": "not in policy vector", "leakage": "must use delayed applied command, never future target"},
        {"signal": "target_turn_rate", "status": "candidate_absent", "source": "target velocity history", "units": "rad/s", "update": "target state step", "normalization": "proposal required", "availability": "not in policy vector", "leakage": "requires causal history and fallback"},
        {"signal": "observation_history", "status": "candidate_absent", "source": "prior policy observations", "units": "mixed", "update": "decision step", "normalization": "per feature", "availability": "temporal config exists but no audited schema segment", "leakage": "causal buffer/reset semantics required"},
    ]
    return {"base_schema": {"dim": 16, "ordered_features": BASE_FEATURES, "mutation": "prohibited pending separate approved migration"}, "signals": signals, "experiment_protocol": {"baseline": "current 16D plus current optional segments", "minimal_candidates": ["single current-state flight-dynamics segment", "command tracking error", "causal history"], "metrics": ["task/safety benefit", "policy sensitivity", "feature availability", "backend parity"], "required_controls": ["fixed seeds/scenarios/checkpoints/horizon", "schema-version and checkpoint migration only after approval"]}}
