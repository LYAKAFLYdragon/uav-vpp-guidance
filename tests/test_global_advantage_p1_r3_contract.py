from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation import global_advantage_p1_r3_contract as r3


ROOT = Path(__file__).resolve().parent.parent
CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_global_advantage_v1_p1_r3_fresh_environment.yaml"
)
RUNNER_PATH = ROOT / "scripts" / "run_thesis_global_advantage_p1_r3_fresh_environment.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("test_r3_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeExec:
    def get_sim_time(self) -> float:
        return 0.0


class _FakeAircraft:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.jsbsim_exec = _FakeExec()

    def get_property_value(self, name: str) -> float:
        return self.offset + float(r3.FDM_PROPERTY_NAMES.index(name))

    def get_state(self) -> dict:
        return {
            "position_neu": np.array([self.offset, 2.0, 3.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "altitude_m": 5000.0,
        }


def _component(**fields: object) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def _fake_env() -> SimpleNamespace:
    env = SimpleNamespace()
    env._backend = "jsbsim"
    env.jsbsim_env = SimpleNamespace(
        _aircraft={"own": _FakeAircraft(0.0), "target": _FakeAircraft(10.0)}
    )
    for name in r3._ENV_RUNTIME_FIELDS:
        setattr(env, name, False if name.startswith("_") else 0)
    env.current_step = 0
    env._episode_count = 1
    env._sim_time_s = 0.0
    env._last_command_saturation = {"nz_saturated": 0.0}
    env._last_observation_schema = {"dim": 16}
    env._robustness_rng = np.random.default_rng(1)
    env._prediction_noise_rng = np.random.default_rng(2)
    env._domain_rand_rng = np.random.default_rng(3)
    env._merge_min_range_so_far_m = float("inf")
    env._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m = float(
        "nan"
    )
    env._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m = float(
        "nan"
    )
    env._post_merge_tactical_basis_recovery_profile_previous_altitude_m = float("nan")
    env.guidance = _component(prev_command=None)
    env._target_guidance = _component(prev_command=None)
    env._command_filter = _component(_filters={"nz_cmd": _component(prev=None)})
    env._target_command_filter = _component(_filters={"nz_cmd": _component(prev=None)})
    env._low_level_controller = _component(_prev_nz=1.0, _prev_roll_rate=0.0)
    env._target_low_level_controller = _component(_prev_nz=1.0, _prev_roll_rate=0.0)
    env.trajectory_predictor_adapter = _component(
        state_buffer=_component(_buffer=deque(maxlen=10), history_len=10)
    )
    env._prediction_error_tracker = _component(_errors=deque(maxlen=10))
    env.virtual_point_generator = _component(_last_virtual_point=None)
    env.reward_calculator = _component(_prev_command=None)
    env.termination_checker = _component(_success_counter=0)
    env.combat_hp = _component(own_hp=1.0, target_hp=1.0)
    env.opponent_policy = _component(
        _projection_count=0,
        delegate=_component(_last_action=np.zeros(3, dtype=np.float32)),
    )
    return env


def _observation() -> dict:
    names = [f"feature_{index}" for index in range(16)]
    return {
        "observation_vector": np.arange(16, dtype=np.float64),
        "observation_schema": {"feature_names": names, "dim": 16},
    }


def _summary(repeat: int, *, boundary: str = "boundary") -> dict:
    return {
        "opponent": "expert",
        "scenario_signature": "scenario_a",
        "repeat_index": repeat,
        "process_id": 100 + repeat,
        "reset_envelope_sha256": "reset",
        "first_reference_action_sha256": "reference",
        "first_opponent_action_sha256": "opponent",
        "trajectory_sha256": "trajectory",
        "boundary_envelope_sha256": boundary,
        "terminal_reason": "timeout",
        "telemetry_complete": True,
        "no_backend_or_prediction_fallback": True,
    }


def _envelope(
    env: SimpleNamespace,
    *,
    observation: dict | None = None,
    history: list[dict[str, float]] | None = None,
) -> dict:
    return r3.capture_runtime_envelope(
        env=env,
        observation=observation or _observation(),
        history=history or [{"range_m": 1000.0}],
        phase_tracker=_component(_first_pass=False),
        reference_metadata={"checkpoint_path": "frozen.pt", "obs_dim": 19},
    )


def test_r3_config_is_nonlearning_and_not_authorised_to_execute():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    plan = r3.build_p1_r3_plan(config)
    assert plan.source_id == r3.SOURCE_ID
    assert plan.execution_permitted is False
    assert plan.scenario_count == 30
    assert plan.opponents == r3.OPPONENTS
    assert plan.child_timeout_seconds == 900


def test_canonical_snapshot_rejects_nonfinite_values_and_is_quantised():
    first = r3.snapshot_hash({"value": 1.00000001}, 6)
    second = r3.snapshot_hash({"value": 1.00000002}, 6)
    assert first == second
    with pytest.raises(r3.GlobalAdvantageP1R3ContractError, match="non-finite"):
        r3.snapshot_hash({"value": float("nan")}, 6)
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(r3.GlobalAdvantageP1R3ContractError, match="cyclic runtime state"):
        r3.canonicalize(cycle)


def test_reset_envelope_records_nested_runtime_state_and_fdm_properties():
    env = _fake_env()
    first = _envelope(env)
    first_hash = r3.snapshot_hash(first, 6)
    assert set(first["fdm"]) == {"own", "target"}
    assert len(first["fdm"]["own"]["properties"]) == len(r3.FDM_PROPERTY_NAMES)
    assert first["components"]["opponent_policy"]["fields"]["delegate"]["fields"][
        "_last_action"
    ] == [0.0, 0.0, 0.0]
    env._low_level_controller._prev_nz = 2.0
    second = _envelope(env)
    assert r3.snapshot_hash(second, 6) != first_hash


def test_every_required_reset_input_changes_the_runtime_hash():
    baseline = r3.snapshot_hash(_envelope(_fake_env()), 6)

    changed_observation = _observation()
    changed_observation["observation_vector"][0] = 99.0
    assert r3.snapshot_hash(_envelope(_fake_env(), observation=changed_observation), 6) != baseline

    assert (
        r3.snapshot_hash(_envelope(_fake_env(), history=[{"range_m": 999.0}]), 6)
        != baseline
    )

    changed_predictor = _fake_env()
    changed_predictor.trajectory_predictor_adapter.state_buffer._buffer.append(
        np.ones(3, dtype=np.float32)
    )
    assert r3.snapshot_hash(_envelope(changed_predictor), 6) != baseline

    changed_controller = _fake_env()
    changed_controller._low_level_controller._prev_roll_rate = 0.4
    assert r3.snapshot_hash(_envelope(changed_controller), 6) != baseline

    changed_opponent = _fake_env()
    changed_opponent.opponent_policy.delegate._last_action[1] = -0.5
    assert r3.snapshot_hash(_envelope(changed_opponent), 6) != baseline

    changed_fdm = _fake_env()
    changed_fdm.jsbsim_env._aircraft["own"].offset = 0.25
    assert r3.snapshot_hash(_envelope(changed_fdm), 6) != baseline


def test_action_contract_and_repeat_gate_fail_closed_on_any_difference():
    assert r3.finite_action([0.1, -0.2, 0.3], name="action") == [0.1, -0.2, 0.3]
    with pytest.raises(r3.GlobalAdvantageP1R3ContractError, match="finite 3-D"):
        r3.finite_action([0.0, float("nan"), 0.0], name="action")
    passed = r3.compare_r3_repeats([_summary(0), _summary(1), _summary(2)])
    assert passed["passed"] is True
    failed = r3.compare_r3_repeats(
        [_summary(0), _summary(1, boundary="different"), _summary(2)]
    )
    assert failed["passed"] is False
    assert failed["comparison_rows"][0]["boundary_envelope_sha256_equal"] is False
    same_process = [_summary(0), _summary(1), _summary(2)]
    for item in same_process:
        item["process_id"] = 100
    failed_process = r3.compare_r3_repeats(same_process)
    assert failed_process["passed"] is False
    assert failed_process["comparison_rows"][0]["fresh_processes"] is False


def test_execution_authorization_requires_an_ancestor_and_exact_code_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner_module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authorization = config["global_advantage_p1_r3"]["authorization"]
    authorization["execution_permitted"] = True
    authorization["required_implementation_git_sha"] = "a" * 40
    code_path = ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "global_advantage_p1_r3_contract.py"
    authorization["authorized_code_files"] = [
        {
            "path": str(code_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        }
    ]
    config_path = tmp_path / "authorised_r3.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "_git_is_ancestor", lambda commit: commit == "a" * 40)
    monkeypatch.setattr(runner, "_git_changed_paths_since", lambda commit: set())
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(
        runner,
        "_git_value",
        lambda *args: "" if args == ("status", "--porcelain") else "test",
    )

    plan, *_ = runner._validate_sources(config_path)
    assert plan.execution_permitted is True

    monkeypatch.setattr(
        runner,
        "_git_changed_paths_since",
        lambda commit: {"src/uav_vpp_guidance/envs/tracking_env.py"},
    )
    with pytest.raises(r3.GlobalAdvantageP1R3ContractError, match="non-authorization changes"):
        runner._validate_sources(config_path)

    monkeypatch.setattr(runner, "_git_changed_paths_since", lambda commit: set())
    authorization["authorized_code_files"][0]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(r3.GlobalAdvantageP1R3ContractError, match="code-file SHA mismatch"):
        runner._validate_sources(config_path)
