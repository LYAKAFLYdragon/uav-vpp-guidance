"""Structured provenance helpers for reproducible experiments.

Tracks config overrides and runtime choices so that ablation results can be
traced back to the exact settings that produced them.
"""

from typing import Any, Dict, List, Optional


PROVENANCE_KEY = "provenance"
OVERRIDES_KEY = "config_overrides"


def record_config_override(
    config: Dict[str, Any],
    key: str,
    new_value: Any,
    old_value: Any = None,
    source: str = "script",
) -> Dict[str, Any]:
    """Record a config override in the provenance block.

    Scripts that programmatically mutate config should call this so the
    mutation is visible in saved manifests and step/reset telemetry.

    Args:
        config: Experiment config dict (mutated in place).
        key: Dot-notation or flat key that was overridden (e.g. "backend"
            or "guidance.mode_switch.enabled").
        new_value: Value after override.
        old_value: Value before override, if known.
        source: Human-readable source of the override (e.g. script name or
            CLI flag).

    Returns:
        The entry that was appended to config["provenance"]["config_overrides"].
    """
    if PROVENANCE_KEY not in config:
        config[PROVENANCE_KEY] = {}
    if OVERRIDES_KEY not in config[PROVENANCE_KEY]:
        config[PROVENANCE_KEY][OVERRIDES_KEY] = []

    entry = {
        "source": source,
        "key": key,
        "new_value": _serialize(new_value),
    }
    if old_value is not None:
        entry["old_value"] = _serialize(old_value)
    config[PROVENANCE_KEY][OVERRIDES_KEY].append(entry)
    return entry


def get_config_overrides(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return recorded config overrides, or an empty list."""
    return config.get(PROVENANCE_KEY, {}).get(OVERRIDES_KEY, [])


def record_config_override_if_changed(
    config: Dict[str, Any],
    key: str,
    new_value: Any,
    old_value: Any = None,
    source: str = "script",
) -> Optional[Dict[str, Any]]:
    """Record a config override only when the effective value changed."""
    if old_value == new_value:
        return None
    return record_config_override(
        config,
        key=key,
        new_value=new_value,
        old_value=old_value,
        source=source,
    )


def _serialize(value: Any) -> Any:
    """Make a value JSON-serialization friendly."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    return str(value)
