"""
Lightweight schema validator for trajectory_prediction configuration.

Usage:
    from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
    validate_tp_config(config_dict, on_unknown="warn")
"""

import warnings
from typing import Literal


_VALID_PREDICTOR_TYPES = {"constant_velocity", "constant_acceleration", "lstm", "gru"}
_VALID_FALLBACK_MODES = {"constant_velocity", "constant_acceleration", "current_target", "none"}
_VALID_ANCHOR_MODES = {"current_target", "predicted_target"}


def _is_device_str(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.lower()
    if s == "cpu":
        return True
    if s.startswith("cuda"):
        return True
    return False


def validate_tp_config(config: dict, on_unknown: Literal["warn", "raise"] = "warn") -> list:
    """Validate trajectory_prediction configuration.

    Returns:
        list of error/warning messages.
    """
    errors = []
    if not isinstance(config, dict):
        errors.append("trajectory_prediction config must be a dict")
        return errors

    predictor_type = config.get("predictor_type")
    if predictor_type is not None and predictor_type not in _VALID_PREDICTOR_TYPES:
        errors.append(
            f"Invalid predictor_type: {predictor_type!r}. "
            f"Expected one of {_VALID_PREDICTOR_TYPES}"
        )

    pred_cfg = config.get("prediction", {})
    fallback_mode = pred_cfg.get("fallback_mode")
    if fallback_mode is not None and fallback_mode not in _VALID_FALLBACK_MODES:
        errors.append(
            f"Invalid fallback_mode: {fallback_mode!r}. "
            f"Expected one of {_VALID_FALLBACK_MODES}"
        )

    int_cfg = config.get("integration", {})
    anchor_mode = int_cfg.get("anchor_mode")
    if anchor_mode is not None and anchor_mode not in _VALID_ANCHOR_MODES:
        errors.append(
            f"Invalid anchor_mode: {anchor_mode!r}. "
            f"Expected one of {_VALID_ANCHOR_MODES}"
        )

    strict_init = config.get("strict_predictor_init", False)
    ckpt = config.get("checkpoint_path")
    if strict_init and predictor_type in ("lstm", "gru"):
        if not ckpt:
            errors.append(
                f"strict_predictor_init=True requires checkpoint_path for {predictor_type}"
            )

    device = config.get("device")
    if device is not None and not _is_device_str(device):
        errors.append(f"Invalid device: {device!r}. Expected 'cpu' or 'cuda[:N]'")

    ckpt_strict = config.get("checkpoint_strict")
    if ckpt_strict is not None and not isinstance(ckpt_strict, bool):
        errors.append(f"checkpoint_strict must be bool, got {type(ckpt_strict).__name__}")

    # Unknown key check
    known_top = {
        "enabled", "predictor_type", "checkpoint_path", "strict_predictor_init",
        "device", "allow_device_fallback", "checkpoint_strict", "strict_checkpoint",
        "freeze_predictor_during_rl", "model", "history", "prediction",
        "integration", "normalization",
    }
    unknown = set(config.keys()) - known_top
    if unknown:
        msg = f"Unknown trajectory_prediction keys: {sorted(unknown)}"
        if on_unknown == "raise":
            errors.append(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)

    if errors:
        raise ValueError("trajectory_prediction config validation failed:\n" + "\n".join(errors))

    return errors
