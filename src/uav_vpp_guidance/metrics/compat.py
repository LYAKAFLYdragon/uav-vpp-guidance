"""Metric-name compatibility aliases for legacy tracking outputs."""

METRIC_ALIAS = {
    "success_rate": "combat_success_rate",
    "time_to_kill": "combat_time_to_kill",
    "tracking_success_rate": "combat_success_rate",
    "mean_range_m": "_deprecated_tracking_mean_range_m",
    "mean_abs_ata_deg": "_deprecated_tracking_mean_abs_ata_deg",
    "envelope_fraction": "_deprecated_tracking_envelope_fraction",
}


def normalize_metric_name(name: str) -> str:
    """Return the canonical combat metric name when an alias exists."""
    return METRIC_ALIAS.get(name, name)
