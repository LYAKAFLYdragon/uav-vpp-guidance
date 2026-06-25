import importlib.util
from pathlib import Path

from uav_vpp_guidance.common.provenance import get_config_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_protection_gain_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "run_protection_gain_calibration", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_env_class_supports_sustained_turn():
    cls = MODULE.resolve_env_class(
        {"task": {"name": "sustained_turn", "env_class": "SustainedTurnEnv"}}
    )
    from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv

    assert cls is SustainedTurnEnv


def test_extract_primary_metric_uses_completed_orbits():
    metric_name, metric_value = MODULE.extract_primary_metric(
        "sustained_turn",
        {"completed_orbits": 2.75},
    )
    assert metric_name == "completed_orbits"
    assert metric_value == 2.75


def test_build_protection_config_records_altitude_hold_override():
    config = MODULE.build_protection_config(
        base_config={"backend": "simple", "env": {}, "low_level_controller": {}},
        task_config={"task": {"name": "sustained_turn", "env_class": "SustainedTurnEnv"}},
        max_bank_rad=1.3265,
        bank_protection_nz_increment=0.5,
        bank_protection_nz_increment_max=1.0,
        altitude_hold_gain=0.03,
    )
    overrides = get_config_overrides(config)
    keys = {entry["key"]: entry for entry in overrides}
    assert keys["low_level_controller.altitude_hold_gain"]["new_value"] == 0.03
