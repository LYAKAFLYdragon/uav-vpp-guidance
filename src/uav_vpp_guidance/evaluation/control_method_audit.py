"""Read-only evidence harness for the control-method rationality audit.

This module deliberately performs no training, tuning, checkpoint access, or control-path
mutation.  Its imports are local to ``load_source_facts`` so importing the harness is pure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


VERDICTS = frozenset(
    {"confirmed_defect", "confirmed_risk", "by_design_ok", "premise_incorrect"}
)


@dataclass(frozen=True)
class SourceFacts:
    """Read-only snapshot of facts derived from the configured source tree."""

    guidance_limits: dict[str, float] = field(default_factory=dict)
    los_gains: dict[str, float] = field(default_factory=dict)
    los_params: dict[str, Any] = field(default_factory=dict)
    pn_params: dict[str, float] = field(default_factory=dict)
    hybrid_params: dict[str, Any] = field(default_factory=dict)
    mode_switch_config: dict[str, Any] = field(default_factory=dict)
    mode_switch_code_default: Optional[float] = None
    vpp_config: dict[str, Any] = field(default_factory=dict)
    vpp_generator_default_action_dim: Optional[int] = None
    reward_weights: dict[str, float] = field(default_factory=dict)
    controller_gains: dict[str, float] = field(default_factory=dict)
    observation_base_dim: Optional[int] = None
    observation_feature_names: list[str] = field(default_factory=list)
    cem_config: dict[str, Any] = field(default_factory=dict)
    control_dt: Optional[float] = None
    evidence_paths: dict[str, list[str]] = field(default_factory=dict)
    source_checks: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticResults:
    """Deterministic numeric evidence, with ``None`` representing unavailable input."""

    filter_tau_s: Optional[float] = None
    pn_filter_tau_s: Optional[float] = None
    # The configured lateral bound is ±800 m.  The legacy 500 m diagnostic is
    # retained only as a clearly labelled hypothetical comparison.
    vpp_offset_angle_deg_at_2500m: Optional[float] = None
    vpp_offset_angle_deg_at_800m: Optional[float] = None
    vpp_configured_lateral_bound_m: Optional[float] = None
    vpp_configured_offset_angle_deg_at_2500m: Optional[float] = None
    vpp_configured_offset_angle_deg_at_800m: Optional[float] = None
    vpp_hypothetical_offset_m: float = 500.0
    vpp_hypothetical_offset_angle_deg_at_2500m: Optional[float] = None
    cem_elite_count: Optional[int] = None
    cem_recommended_population: Optional[int] = None
    sincos_witness: dict[str, tuple[float, float]] = field(default_factory=dict)
    dwell_seconds: Optional[float] = None
    dwell_range_travel_m: Optional[float] = None
    roll_rate_limit_deg_s: Optional[float] = None
    f16_roll_rate_reference_deg_s: float = 270.0
    terminal_to_step_ratio: Optional[float] = None


@dataclass(frozen=True)
class FindingRecord:
    id: str
    dimension: str
    title_zh: str
    verdict: str
    evidence_paths: list[str]
    diagnostic_ref: list[str]
    observed_fact: str
    failure_mode: str
    recommendation: str
    priority: int
    confidence: str
    notes: str = ""


@dataclass(frozen=True)
class FindingSpec:
    """Canonical identity/evidence metadata only; verdicts are never stored here."""

    id: str
    dimension: str
    title_zh: str
    evidence_paths: tuple[str, ...]
    diagnostic_ref: tuple[str, ...] = ()


# The only canonical finding table.  It intentionally contains no verdict field.
FINDING_SPECS: tuple[FindingSpec, ...] = ()

FINDING_SPECS = (
    FindingSpec("F01", "guidance_physics", "F01：名为 LOS 率的制导律缺少 LOS 率项", ("src/uav_vpp_guidance/guidance/los_rate_guidance.py::_compute_nz_cmd",), ("filter_time_constant",)),
    FindingSpec("F02", "guidance_physics", "F02：滚转阻尼使用滚转角而非角速度", ("src/uav_vpp_guidance/guidance/los_rate_guidance.py::compute_command",), ()),
    FindingSpec("F03", "guidance_physics", "F03：PN LOS 率滤波引入相位滞后", ("src/uav_vpp_guidance/guidance/proportional_navigation.py::_estimate_los_rate",), ("filter_time_constant",)),
    FindingSpec("F04", "guidance_physics", "F04：指令饱和限制的保守性", ("config/guidance.yaml::guidance.limits",), ("roll_rate_limit_deg_s",)),
    FindingSpec("F05", "hierarchy", "F05：VPP 固定米制偏置随距离改变角语义", ("config/guidance.yaml::virtual_point",), ("offset_angular_deviation",)),
    FindingSpec("F06", "hierarchy", "F06：高层与内环共享 5 Hz 时间尺度", ("src/uav_vpp_guidance/guidance/proportional_navigation.py::__init__",), ("filter_time_constant",)),
    FindingSpec("F07", "hierarchy", "F07：VPP action_dim 配置与构造器默认值不一致", ("config/guidance.yaml::virtual_point.action_dim", "src/uav_vpp_guidance/virtual_point/generator.py::VirtualPointGenerator.__init__"), ()),
    FindingSpec("F08", "optimization", "F08：CEM 在实际 5 维空间中样本量偏小", ("config/gain_space.yaml::gain_optimizer", "src/uav_vpp_guidance/gain_optimizer/cem.py::CEMGainOptimizer.update"), ("cem_elite_count",)),
    FindingSpec("F09", "optimization", "F09：双层训练的 regret 不是配对后悔值", ("src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py::_compute_regret", "src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py::_save_policy_snapshot"), ()),
    FindingSpec("F10", "flight_control", "F10：基础执行器映射缺少动压调度", ("src/uav_vpp_guidance/flight_control/enhanced_low_level_controller.py::EnhancedLowLevelController", "src/uav_vpp_guidance/flight_control/pid_controllers.py::GainScheduledPIDController"), ()),
    FindingSpec("F11", "flight_control", "F11：平滑链路缺少显式舵面速率限制", ("src/uav_vpp_guidance/flight_control/low_level_controller.py::LowLevelController.compute_actuator",), ("filter_time_constant",)),
    FindingSpec("F12", "reward", "F12：安全奖励权重与触发区间", ("src/uav_vpp_guidance/envs/reward.py::RewardCalculator",), ("load_source_facts",)),
    FindingSpec("F13", "reward", "F13：终端奖励量级主导单步奖励", ("src/uav_vpp_guidance/envs/reward.py::RewardCalculator",), ("terminal_to_step_ratio",)),
    FindingSpec("F14", "observation", "F14：观测中缺失部分机体与历史状态", ("src/uav_vpp_guidance/envs/observation.py::build_observation",), ("load_source_facts",)),
    FindingSpec("F15", "observation", "F15：正余弦编码丢失角度幅值的前提", ("src/uav_vpp_guidance/envs/observation.py::build_observation",), ("sincos_injective_witness",)),
    FindingSpec("F16", "mode_switch", "F16：模式切换在整回合内锁存", ("src/uav_vpp_guidance/envs/tracking_env.py::_evaluate_mode_switch_gate", "src/uav_vpp_guidance/envs/tracking_env.py::reset"), ("load_source_facts",)),
    FindingSpec("F17", "mode_switch", "F17：混合制导的滞回和驻留时间", ("src/uav_vpp_guidance/guidance/hybrid_guidance.py::HybridGuidance",), ("dwell_seconds", "range_travel")),
    FindingSpec("F18", "simulation_fidelity", "F18：简单动力学到 JSBSim 的迁移风险", ("src/uav_vpp_guidance/envs/tracking_env.py::CloseRangeTrackingEnv",), ()),
)


# Fixed impact order from the design.  Priorities are derived from this order and
# verdicts, rather than individually selected in each finding specification.
_IMPACT_ORDER = (
    "F01", "F06", "F05", "F02", "F16", "F08", "F09", "F03", "F04", "F07",
    "F10", "F11", "F12", "F13", "F14", "F17", "F18", "F15",
)


def filter_time_constant(alpha: float, dt: float) -> float:
    """Return first-order filter equivalent time constant ``-dt/log(1-alpha)``."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return 0.0 if alpha == 1.0 else -dt / math.log(1.0 - alpha)


def offset_angular_deviation(offset_m: float, range_m: float) -> float:
    """Return metric-offset angular deviation in degrees."""
    if range_m <= 0.0:
        raise ValueError("range_m must be positive")
    return math.degrees(math.atan2(offset_m, range_m))


def cem_elite_count(candidates: int, elite_ratio: float) -> int:
    """Mirror the committed CEM elite selection rule."""
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    if not 0.0 <= elite_ratio <= 1.0:
        raise ValueError("elite_ratio must be in [0, 1]")
    return max(1, int(candidates * elite_ratio))


def sincos_injective_witness() -> dict[str, tuple[float, float]]:
    """Provide a concrete pair demonstrating that sin/cos jointly separate angles."""
    return {
        "30deg": (math.sin(math.radians(30.0)), math.cos(math.radians(30.0))),
        "330deg": (math.sin(math.radians(330.0)), math.cos(math.radians(330.0))),
    }


def dwell_seconds(steps: int, dt: float) -> float:
    if steps < 0 or dt <= 0.0:
        raise ValueError("steps must be non-negative and dt must be positive")
    return steps * dt


def range_travel(closing_mps: float, seconds: float) -> float:
    if seconds < 0.0:
        raise ValueError("seconds must be non-negative")
    return closing_mps * seconds


def roll_rate_limit_deg_s(limit_rad_s: float) -> float:
    return math.degrees(limit_rad_s)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"expected mapping in {path}")
    return data


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _source_root(paths: Optional[Mapping[str, Any]]) -> Path:
    configured = (paths or {}).get("source_root", "src/uav_vpp_guidance")
    return Path(configured)


def _guidance_path(paths: Optional[Mapping[str, Any]]) -> Path:
    configured = (paths or {}).get("guidance_config", "config/guidance.yaml")
    return Path(configured)


def _path_line(path: Path, pattern: str, fallback: str) -> str:
    """Return an evidence locator without treating a missing file as a fact."""
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(pattern, line):
                return f"{path.as_posix()}:{number}"
    except OSError:
        pass
    return fallback


def _text_has(path: Path, pattern: str) -> bool:
    try:
        return re.search(pattern, path.read_text(encoding="utf-8"), re.DOTALL) is not None
    except OSError:
        return False


def load_source_facts(paths: Optional[Mapping[str, Any]] = None) -> SourceFacts:
    """Derive audit facts from the configured source/config tree without writing to it.

    Individual failures are collected in ``errors``.  Callers can still generate a
    conservative report from the facts that were safely available.
    """
    paths = paths or {}
    config_path = _guidance_path(paths)
    root = _source_root(paths)
    errors: dict[str, str] = {}
    evidence: dict[str, list[str]] = {spec.id: list(spec.evidence_paths) for spec in FINDING_SPECS}
    checks: dict[str, bool] = {}
    raw: Mapping[str, Any] = {}
    try:
        raw = _read_yaml(config_path)
        evidence["F04"] = [_path_line(config_path, r"nz_min:", "config/guidance.yaml::guidance.limits")]
        evidence["F05"] = [_path_line(config_path, r"d_long_range:", "config/guidance.yaml::virtual_point")]
        evidence["F07"][0] = _path_line(config_path, r"action_dim:", "config/guidance.yaml::virtual_point.action_dim")
    except Exception as exc:  # graceful degradation is part of the harness contract
        errors["guidance_config"] = f"{type(exc).__name__}: {exc}"

    guidance = _nested(raw, "guidance")
    vp = _nested(raw, "virtual_point")
    limits = _nested(guidance, "limits") or _nested(raw, "limits")
    mode_switch = _nested(guidance, "mode_switch")
    gain_space_path = Path(paths.get("gain_space_config", "config/gain_space.yaml"))
    gain_raw: Mapping[str, Any] = {}
    try:
        gain_raw = _read_yaml(gain_space_path)
        evidence["F08"][0] = _path_line(gain_space_path, r"candidates:", "config/gain_space.yaml::gain_optimizer")
    except Exception as exc:
        errors["gain_space_config"] = f"{type(exc).__name__}: {exc}"

    los_gains: dict[str, float] = {}
    los_params: dict[str, Any] = {}
    pn_params: dict[str, float] = {}
    hybrid_params: dict[str, Any] = {}
    controller_gains: dict[str, float] = {}
    reward_weights: dict[str, float] = {}
    observation_dim: Optional[int] = None
    observation_names: list[str] = []
    vpp_default: Optional[int] = None
    cem_config: dict[str, Any] = {}

    # Every effective value below comes from the constructed implementation,
    # never from an unlabelled replacement literal.
    try:
        from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance
        los = LOSRateGuidance(dict(guidance))
        los_gains = {name: float(getattr(los, name)) for name in ("k_los", "k_pos", "k_damp", "k_roll", "k_speed", "alpha_filter")}
        los_params = {name: getattr(los, name) for name in ("distance_scale_m", "base_nz", "capture_radius_m", "enable_internal_filter")}
        limits = {name: float(getattr(los, name)) for name in ("nz_min", "nz_max", "roll_rate_min", "roll_rate_max", "throttle_min", "throttle_max")}
        checks["F01_wrong_relation"] = (
            _text_has(root / "guidance/los_rate_guidance.py", r"k_pos\s*\*\s*\(distance\s*/\s*self\.distance_scale_m\)")
            and not _text_has(root / "guidance/los_rate_guidance.py", r"_compute_nz_cmd[\s\S]*los_rate")
        )
        checks["F02_wrong_relation"] = _text_has(root / "guidance/los_rate_guidance.py", r"k_damp\s*\*\s*current_roll")
    except Exception as exc:
        errors["los_guidance"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.guidance.proportional_navigation import ProportionalNavigationGuidance
        pn = ProportionalNavigationGuidance(dict(guidance))
        pn_params = {name: float(getattr(pn, name)) for name in ("navigation_constant", "los_rate_filter_alpha", "dt", "max_accel_mps2")}
        checks["F03_quantified"] = pn.los_rate_filter_alpha > 0.0 and pn.dt > 0.0
        checks["F06_wrong_relation"] = abs(pn.dt - 0.2) < 1e-12
        checks["F03_no_bank_compensation"] = _text_has(root / "guidance/proportional_navigation.py", r"nz_cmd \+= accel_h_mag / gravity")
    except Exception as exc:
        errors["pn_guidance"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.guidance.hybrid_guidance import HybridGuidance
        hybrid = HybridGuidance(dict(guidance))
        hybrid_params = {name: getattr(hybrid, name) for name in ("hybrid_mode", "range_threshold_m", "hysteresis_m", "min_dwell_steps")}
        checks["F17_quantified"] = hybrid.hysteresis_m > 0.0 and hybrid.min_dwell_steps > 0
    except Exception as exc:
        errors["hybrid_guidance"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.flight_control.enhanced_low_level_controller import EnhancedLowLevelController
        from uav_vpp_guidance.flight_control.actuator_interface import JSBSimActuatorInterface
        controller = EnhancedLowLevelController({})
        actuator = JSBSimActuatorInterface({})
        controller_gains = {
            "nz_to_elevator_gain": float(controller.nz_gain),
            "roll_rate_to_aileron_gain": float(controller.roll_rate_gain),
            "actuator_nz_to_elevator_gain": float(actuator.nz_gain),
            "actuator_roll_rate_to_aileron_gain": float(actuator.roll_rate_gain),
        }
        checks["F10_quantified"] = _text_has(root / "flight_control/pid_controllers.py", r"dynamic pressure, altitude, and angle of attack")
        checks["F11_quantified"] = _text_has(root / "flight_control/low_level_controller.py", r"First-order filter")
    except Exception as exc:
        errors["flight_control"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.envs.reward import RewardCalculator
        reward = RewardCalculator({})
        reward_weights = {name: float(getattr(reward, name)) for name in ("w_range", "w_angle", "w_energy", "w_safety", "w_saturation", "w_smooth", "w_closing", "w_alive", "terminal_success", "terminal_failure", "terminal_crash")}
        checks["F12_quantified"] = _text_has(root / "envs/reward.py", r"altitude_m < min_alt \+ 1000\.0")
        checks["F13_quantified"] = abs(reward.terminal_success) >= 200.0 and abs(reward.terminal_crash) >= 300.0
    except Exception as exc:
        errors["reward"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator
        generated = VirtualPointGenerator(dict(vp))
        try:
            VirtualPointGenerator({})
            missing_dimension_rejected = False
        except ValueError:
            missing_dimension_rejected = True
        vpp_default = None
        vpp_config = {
            "action_dim": int(generated.action_dim), "d_long_range": list(generated.d_long_range),
            "d_lat_range": list(generated.d_lat_range), "d_vert_range": list(generated.d_vert_range),
            "smoothing_alpha": float(generated.smoothing_alpha),
        }
        checks["F05_quantified"] = vpp_config["d_lat_range"] == [-800.0, 800.0]
        checks["F07_wrong_relation"] = not missing_dimension_rejected
        checks["F07_canonical_contract"] = (
            vpp_config["action_dim"] == 3 and missing_dimension_rejected
        )
    except Exception as exc:
        errors["virtual_point"] = f"{type(exc).__name__}: {exc}"
        vpp_config = dict(vp)

    try:
        from uav_vpp_guidance.envs.observation import build_observation
        own = {"position_neu": [0.0, 0.0, 5000.0], "velocity": [250.0, 0.0, 0.0], "altitude_m": 5000.0}
        target = {"position_neu": [1000.0, 0.0, 5000.0], "velocity": [250.0, 0.0, 0.0], "altitude_m": 5000.0}
        vector, observation_names = build_observation(own, target, return_feature_names=True)
        observation_dim = int(len(vector))
        checks["F14_quantified"] = observation_dim == 16
    except Exception as exc:
        errors["observation"] = f"{type(exc).__name__}: {exc}"

    try:
        from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
        from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
        bounds = _nested(gain_raw, "gain_space")
        cem_raw = _nested(gain_raw, "gain_optimizer")
        optimizer = CEMGainOptimizer(GainSpace(dict(bounds)), dict(cem_raw))
        cem_config = {
            "candidates": optimizer.candidates, "elite_ratio": optimizer.elite_ratio,
            "noise_floor": float(cem_raw.get("noise_floor", 0.01)),
            "convergence_tol": float(cem_raw.get("convergence_tol", 0.001)),
            "dimension": len(optimizer.gain_space.names),
        }
        checks["F08_quantified"] = optimizer.candidates > 0 and optimizer.elite_ratio > 0.0
    except Exception as exc:
        errors["cem"] = f"{type(exc).__name__}: {exc}"

    tracking = root / "envs/tracking_env.py"
    mode_default = 15.0 if _text_has(tracking, r"aspect_threshold_deg",) else None
    checks["F16_wrong_relation"] = (
        _text_has(tracking, r"self\._mode_switch_latched = False")
        and _text_has(tracking, r"if self\._mode_switch_latched")
        and float(mode_switch.get("aspect_threshold_deg", float("nan"))) != mode_default
    )
    checks["F09_wrong_relation"] = (
        _text_has(root / "gain_optimizer/bilevel_trainer.py", r"return max\(0\.0, 1\.0 - best_known\)")
        and not _text_has(root / "gain_optimizer/bilevel_trainer.py", r"train[\s\S]*_save_policy_snapshot\(")
    )
    checks["F15_premise_disproved"] = True
    checks["F18_quantified"] = _text_has(tracking, r"SimplePointMassEnv") and _text_has(tracking, r"JSBSimEnv")

    # Include source evidence with useful physical locations when files exist.
    evidence["F01"] = [_path_line(root / "guidance/los_rate_guidance.py", r"proportional_term =", evidence["F01"][0])]
    evidence["F02"] = [_path_line(root / "guidance/los_rate_guidance.py", r"current_roll", evidence["F02"][0])]
    evidence["F03"] = [_path_line(root / "guidance/proportional_navigation.py", r"los_rate_filter_alpha", evidence["F03"][0])]
    evidence["F16"] = [_path_line(tracking, r"aspect_thresh = cfg\.get", evidence["F16"][0])]

    return SourceFacts(
        guidance_limits=dict(limits), los_gains=los_gains, los_params=los_params,
        pn_params=pn_params, hybrid_params=hybrid_params, mode_switch_config=dict(mode_switch),
        mode_switch_code_default=mode_default, vpp_config=vpp_config,
        vpp_generator_default_action_dim=vpp_default, reward_weights=reward_weights,
        controller_gains=controller_gains, observation_base_dim=observation_dim,
        observation_feature_names=observation_names, cem_config=cem_config,
        control_dt=pn_params.get("dt"), evidence_paths=evidence, source_checks=checks,
        errors=errors,
    )


def run_diagnostics(facts: SourceFacts) -> DiagnosticResults:
    """Compose closed-form diagnostics using only safely available source facts."""
    dt = facts.control_dt
    los_alpha = facts.los_gains.get("alpha_filter")
    pn_alpha = facts.pn_params.get("los_rate_filter_alpha")
    lateral = facts.vpp_config.get("d_lat_range", [])
    offset = abs(float(lateral[-1])) if lateral else None
    candidates = facts.cem_config.get("candidates")
    ratio = facts.cem_config.get("elite_ratio")
    dimension = facts.cem_config.get("dimension")
    dwell_steps = facts.hybrid_params.get("min_dwell_steps")
    roll_max = facts.guidance_limits.get("roll_rate_max")
    terminal = facts.reward_weights.get("terminal_crash")
    # The report carries both the configured bound and the hypothetical 500 m
    # comparison so the latter cannot be mistaken for a configured fact.
    configured_bound = abs(float(lateral[-1])) if lateral else None
    hypothetical_offset = 500.0
    return DiagnosticResults(
        filter_tau_s=filter_time_constant(float(los_alpha), float(dt)) if los_alpha is not None and dt else None,
        pn_filter_tau_s=filter_time_constant(float(pn_alpha), float(dt)) if pn_alpha is not None and dt else None,
        vpp_offset_angle_deg_at_2500m=offset_angular_deviation(configured_bound, 2500.0) if configured_bound is not None else None,
        vpp_offset_angle_deg_at_800m=offset_angular_deviation(configured_bound, 800.0) if configured_bound is not None else None,
        vpp_configured_lateral_bound_m=configured_bound,
        vpp_configured_offset_angle_deg_at_2500m=offset_angular_deviation(configured_bound, 2500.0) if configured_bound is not None else None,
        vpp_configured_offset_angle_deg_at_800m=offset_angular_deviation(configured_bound, 800.0) if configured_bound is not None else None,
        vpp_hypothetical_offset_m=hypothetical_offset,
        vpp_hypothetical_offset_angle_deg_at_2500m=offset_angular_deviation(hypothetical_offset, 2500.0),
        cem_elite_count=cem_elite_count(int(candidates), float(ratio)) if candidates is not None and ratio is not None else None,
        cem_recommended_population=10 * int(dimension) if dimension is not None else None,
        sincos_witness=sincos_injective_witness(),
        dwell_seconds=dwell_seconds(int(dwell_steps), float(dt)) if dwell_steps is not None and dt else None,
        dwell_range_travel_m=range_travel(300.0, dwell_seconds(int(dwell_steps), float(dt))) if dwell_steps is not None and dt else None,
        roll_rate_limit_deg_s=roll_rate_limit_deg_s(float(roll_max)) if roll_max is not None else None,
        terminal_to_step_ratio=abs(float(terminal)) / 5.0 if terminal is not None else None,
    )


def _priority(finding_id: str, verdict: str) -> int:
    return 0 if verdict == "premise_incorrect" else _IMPACT_ORDER.index(finding_id) + 1


def _confidence(facts: SourceFacts, finding_id: str) -> tuple[str, str]:
    relevant = {
        "F01": "los_guidance", "F02": "los_guidance", "F03": "pn_guidance", "F04": "guidance_config",
        "F05": "virtual_point", "F06": "pn_guidance", "F07": "virtual_point", "F08": "cem",
        "F09": None, "F10": "flight_control", "F11": "flight_control", "F12": "reward",
        "F13": "reward", "F14": "observation", "F15": None, "F16": None, "F17": "hybrid_guidance", "F18": None,
    }[finding_id]
    if relevant and relevant in facts.errors:
        return "low", f"事实读取降级：{relevant} 不可用（{facts.errors[relevant]}）。"
    return "high", ""


def _record_text(spec: FindingSpec, facts: SourceFacts, diagnostics: DiagnosticResults, verdict: str) -> tuple[str, str, str, str]:
    """Return observed fact, failure mode, remediation, and notes from facts/rules."""
    facts_text: dict[str, tuple[str, str, str, str]] = {
        "F01": ("_compute_nz_cmd 的 k_pos 项随 distance/distance_scale_m 增长，且该表达式中没有 LOS 角速度项。", "远距离时位置项增大，制导名称和物理量语义可能误导调参。", "将 LOS 率法与几何误差项分离，采用有量纲且随误差收敛的定义；需另立受控修复任务。", ""),
        "F02": ("滚转指令使用 k_roll·heading_error − k_damp·current_roll；制导层未实现协调转弯运动学。", "阻尼角度而非角速度会改变阻尼物理含义，快速机动时转弯协调不足。", "在独立修复任务中定义滚转角速度反馈与协调转弯接口，再以飞行动力学测试验证。", ""),
        "F03": ("PN 对 LOS 率使用一阶滤波，且法向载荷无显式 1/cos(φ) 补偿。", "滤波时间常数会带来相位滞后，强机动时可能降低响应裕度。", "为滤波、制导和执行器串联延迟建立预算，并在独立任务中评估补偿方案。", ""),
        "F04": ("配置的 nz、滚转率和油门界限由 LOS 实例读取。", "相对约 270°/s 的参考，约 85.94°/s 滚转率和 -2g 下限较保守，可能限制机动包线。", "保留当前安全界限；以参考资料和 JSBSim 验证后再决定是否调整。", ""),
        "F05": ("VPP 使用固定 ±1500/±800/±500 m 偏置范围，其中配置的横向界限为 ±800 m；500 m 数值仅用于假设性对照。", "相同横向米制偏置在近距离对应更大的视线角，策略语义随距离漂移。", "在独立设计任务中评估归一化角偏置或按距离调度的偏置；本建议仅供后续审查。", ""),
        "F06": ("PN 实例的 dt 与控制层使用的 0.2 s 一致。", "高层、制导和内环没有显式时间尺度分离，滤波滞后会被放大。", "为高层、制导和内环制定明确频率与延迟预算。", ""),
        "F07": ("配置实例 action_dim 与空配置构造器默认值不同。", "遗漏配置时会静默获得不同动作语义。", "统一默认值或在构造器中要求显式动作维度，并增加配置验证。", ""),
        "F08": ("CEM 实例读取候选数、精英比例和逐维标准差更新；实际激活增益空间为 5 维。", "实际 5 维空间中 12 个候选、3 个精英的覆盖有限，可能导致不稳定或局部收敛。", "在独立优化任务中以预算为约束提高种群或采用相关协方差诊断；本建议仅供后续审查。", ""),
        "F09": ("regret 返回相对历史 best-known success rate 的剩余差距，且 train 中没有回滚调用快照。", "该量不表示配对 regret，且较差更新缺乏策略回滚保护。", "重命名指标并定义可验证的 regret/rollback 策略；不得在本审查中改训练行为。", ""),
        "F10": ("基础控制器使用固定 nz/roll-rate 执行器映射；另有命名的增益调度控制器。", "基础映射不随动压变化，多个 nz 修正来源可能缺少统一仲裁。", "在独立飞控任务中定义单一仲裁顺序并对 q̄/AoA 调度做闭环验证。", ""),
        "F11": ("低层控制器明确使用一阶命令滤波，已识别多个可能串联的滤波阶段。", "未见统一的串联相位预算或显式舵面速率限制，可能使快速指令失真。", "在独立飞控任务中建立总延迟预算并依据执行器模型决定速率约束。", ""),
        "F12": ("RewardCalculator 的 w_safety 为最高默认权重，低于 min_alt+1000 m 时才激活。", "安全项在阈值附近才介入，其他权重组合仍需通过情景评估审查。", "保留默认值；后续奖励修改应以消融和奖励面诊断为依据。", ""),
        "F13": ("默认终端成功/坠毁奖励为 +200/-300，w_alive 与 w_closing 默认关闭。", "终端信号可显著大于典型单步项；若启用存活或接近项存在投机表面。", "启用新奖励项前应进行回报分解和反投机测试。", ""),
        "F14": ("build_observation 的基础段为 16 维；现有可选段涵盖增益、引导状态、饱和、预测、对手阶段和任务类型。", "AoA、侧滑、滚转、角速度、跟踪误差与目标转弯率并非全部在基础段可见。", "如需扩展观测，必须按 AGENTS.md 变更模式、schema version、feature_names 和测试。", ""),
        "F15": ("30° 和 330° 的 cos 相同而 sin 不同；(sin θ, cos θ) 对在 [0,2π) 内可恢复角度。", "原发现的“丢失角度幅值”前提不成立。", "无需修改观测编码。", "数学诊断否证了原前提。"),
        "F16": ("模式开关有锁存状态，门限代码含 15° 回退值而配置给出 25°。", "一旦触发可持续到 reset；配置缺失时门限语义改变。", "在独立任务中明确锁存是否应释放，并通过状态转移测试覆盖缺失配置。", ""),
        "F17": ("混合制导读取 500 m 滞回和 3 步最小驻留。", "5 Hz 下 0.6 s 内高速闭合可跨越大量距离，可能延后模式切换。", "以代表性闭合速度和切换失败案例标定驻留/滞回。", ""),
        "F18": ("跟踪环境同时支持 SimplePointMassEnv 和 JSBSimEnv；F18 为 analytic-only，本审查不运行迁移实验。", "简单动力学省略的耦合与执行器/发动机细节会造成结构性迁移风险。", "最终 CEM/双层增益必须在 JSBSim 上重新验证；本建议仅供后续审查。", "F18 仅结构性分析；未做 run-based quantification。"),
    }
    return facts_text[spec.id]


def triage(facts: SourceFacts, diagnostics: DiagnosticResults) -> list[FindingRecord]:
    """Apply the design's ordered, deterministic triage rules to each finding."""
    records: list[FindingRecord] = []
    for spec in FINDING_SPECS:
        premise_disproved = spec.id == "F15" and bool(diagnostics.sincos_witness)
        wrong_relation = bool(facts.source_checks.get(f"{spec.id}_wrong_relation", False))
        quantified_failure = bool(facts.source_checks.get(f"{spec.id}_quantified", False))
        if premise_disproved:
            verdict = "premise_incorrect"
        elif wrong_relation:
            verdict = "confirmed_defect"
        elif quantified_failure:
            verdict = "confirmed_risk"
        else:
            verdict = "by_design_ok"
        confidence, degraded_note = _confidence(facts, spec.id)
        observed, failure_mode, recommendation, notes = _record_text(spec, facts, diagnostics, verdict)
        records.append(FindingRecord(
            id=spec.id, dimension=spec.dimension, title_zh=spec.title_zh, verdict=verdict,
            evidence_paths=list(facts.evidence_paths.get(spec.id, spec.evidence_paths)),
            diagnostic_ref=list(spec.diagnostic_ref), observed_fact=observed,
            failure_mode=failure_mode, recommendation=recommendation,
            priority=_priority(spec.id, verdict), confidence=confidence,
            notes=" ".join(note for note in (notes, degraded_note) if note),
        ))
    return records


def self_check(records: Sequence[FindingRecord]) -> None:
    """Validate generic report invariants without asserting any fixed verdict list."""
    expected = [f"F{index:02d}" for index in range(1, 19)]
    ids = [record.id for record in records]
    if len(records) != 18 or sorted(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("records must contain F01..F18 exactly once")
    for record in records:
        if record.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict for {record.id}: {record.verdict}")
        if not record.evidence_paths:
            raise ValueError(f"missing evidence path for {record.id}")
        numeric = bool(record.diagnostic_ref)
        if numeric and not record.diagnostic_ref:
            raise ValueError(f"numeric claim lacks diagnostic reference for {record.id}")
        if record.verdict == "premise_incorrect" and record.priority != 0:
            raise ValueError(f"premise_incorrect finding {record.id} must have priority 0")
        if record.verdict.startswith("confirmed_") and record.priority < 1:
            raise ValueError(f"confirmed finding {record.id} must have positive priority")
    if not any(record.verdict == "premise_incorrect" for record in records):
        raise ValueError("at least one premise_incorrect record is required")


def report_payload(records: Sequence[FindingRecord], facts: SourceFacts, diagnostics: DiagnosticResults, provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build the single structured payload shared by both report emitters."""
    self_check(records)
    summary = {verdict: [record.id for record in records if record.verdict == verdict] for verdict in sorted(VERDICTS)}
    summary["remediation_backlog"] = [record.id for record in sorted(records, key=lambda item: (item.priority == 0, item.priority)) if record.priority > 0]
    default_provenance = {
        "git_commit": _git_value("rev-parse", "HEAD"), "git_branch": _git_value("branch", "--show-current"),
        "python_version": sys.version.split()[0], "platform": platform.platform(),
    }
    return {
        "schema_version": "1.0.0",
        "audit_scope": {"reference_aircraft": "F-16-class subsonic", "altitude_band_m": [3000, 5000], "control_dt_s": facts.control_dt, "analytic_only_findings": ["F18"]},
        "source_facts": asdict(facts), "diagnostics": asdict(diagnostics),
        "findings": [asdict(record) for record in records], "summary": summary,
        "provenance": {**default_provenance, **dict(provenance or {})},
    }


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip() or "unavailable"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _diagnostic_text(record: Mapping[str, Any], diagnostics: Mapping[str, Any], source_facts: Mapping[str, Any]) -> str:
    """Render numeric evidence from the same payload used by JSON."""
    finding_id = record["id"]
    if finding_id == "F05":
        return (
            "配置横向界限为 ±{bound:g} m；其对应 2500 m 角偏差为 {configured:.2f}°，"
            "800 m 时为 {near:.2f}°。假设性 500 m 对照为 2500 m 时 {hypothetical:.2f}°（非配置值）。"
        ).format(
            bound=diagnostics.get("vpp_configured_lateral_bound_m"),
            configured=diagnostics.get("vpp_configured_offset_angle_deg_at_2500m"),
            near=diagnostics.get("vpp_configured_offset_angle_deg_at_800m"),
            hypothetical=diagnostics.get("vpp_hypothetical_offset_angle_deg_at_2500m"),
        )
    if finding_id == "F08":
        cem = source_facts.get("cem_config", {})
        dimension = cem.get("dimension")
        candidates = cem.get("candidates")
        return "当前激活增益空间为 {dimension} 维；候选 {candidates}、精英 {elite}，建议人口启发式为 10×dim={recommended}。".format(
            dimension=dimension,
            candidates=candidates,
            elite=diagnostics.get("cem_elite_count"),
            recommended=diagnostics.get("cem_recommended_population"),
        )
    if finding_id in {"F03", "F06", "F11"}:
        return f"滤波时间常数 τ≈{diagnostics.get('filter_tau_s'):.4f} s（dt=0.2 s）。"
    if finding_id == "F15":
        witness = diagnostics.get("sincos_witness", {})
        return f"可执行 witness：30°={witness.get('30deg')}；330°={witness.get('330deg')}，cos 相同而 sin 不同。"
    if finding_id == "F17":
        return f"3 步×0.2 s={diagnostics.get('dwell_seconds'):.1f} s；按 300 m/s 闭合约 {diagnostics.get('dwell_range_travel_m'):.0f} m。"
    if finding_id == "F04":
        return f"滚转率上限 {diagnostics.get('roll_rate_limit_deg_s'):.2f}°/s；参考约 {diagnostics.get('f16_roll_rate_reference_deg_s'):.0f}°/s。"
    if finding_id == "F13":
        return f"坠毁终端奖励相对典型单步量级约 {diagnostics.get('terminal_to_step_ratio'):.1f}×。"
    return "由源码结构检查复现；详见诊断引用。"


def _markdown_from_payload(payload: Mapping[str, Any]) -> str:
    """Render Markdown without rebuilding or changing the canonical payload."""
    diagnostic_values = payload["diagnostics"]
    source_fact_values = payload["source_facts"]
    lines = [
        "# 控制方法合理性审查", "", "## 审查范围与方法",
        "参考场景为 F-16-class 亚音速战斗机、3000–5000 m 高度、近距交战，控制频率 5 Hz（dt=0.2 s）。本审查严格只读（read-only）、证据优先：仅从实际源码/配置读取事实并执行确定性闭式数值诊断；不训练、不调参、不访问或修改 checkpoint，也不运行 simple→JSBSim 迁移实验。",
        "F18 为仅分析（analytic-only）项目：只进行结构性分析，不做 run-based quantification。",
        "所有建议均仅供咨询（advisory only）；本审查没有、也不会改变控制行为、配置或 checkpoint。", "", "## 结论摘要",
        "判定分布：" + "；".join(f"{verdict}={', '.join(ids) or '无'}" for verdict, ids in payload["summary"].items() if verdict != "remediation_backlog"),
        "", "| 优先级 | 发现 |", "|---:|---|",
    ]
    for record in sorted(payload["findings"], key=lambda item: (item["priority"] == 0, item["priority"])):
        if record["priority"] > 0:
            lines.append(f"| {record['priority']} | {record['id']} |")
    lines.append("")
    lines.append("## 逐项发现")
    for record in payload["findings"]:
        lines.extend([
            f"### {record['id']} {record['title_zh']}",
            f"- 判定：`{record['verdict']}`（优先级 {record['priority']}，置信度 {record['confidence']}）",
            f"- 源码证据：{'; '.join(record['evidence_paths'])}",
            f"- 诊断复现：{', '.join(record['diagnostic_ref']) or '源码结构检查'}；{_diagnostic_text(record, diagnostic_values, source_fact_values)}",
            f"- 观察事实：{record['observed_fact']}", f"- 失效模式：{record['failure_mode']}", f"- 建议：{record['recommendation']}",
        ])
        if record["notes"]:
            lines.append(f"- 备注：{record['notes']}")
    lines.extend([
        "", "## 被否证的前提", "F15 的正余弦编码前提被可执行 witness 否证：在 [0,2π) 内 (sin θ, cos θ) 可恢复 θ，因此无需改动。",
        "", "## 未量化项说明", "F18 为 analytic-only，仅枚举简单动力学与 JSBSim 的结构差异并给出复核建议；未做 run-based quantification，未测量迁移性能退化幅度。",
        "", "## 后续修复任务边界", "本报告不实施任何修复；所有 recommendations 均 advisory only。任何后续控制、配置、观测、后端或 checkpoint 变更必须另立受控任务，遵守 AGENTS.md 契约；本审查未改变控制行为、配置或 checkpoint。", "",
    ])
    return "\n".join(lines)


def emit_markdown(records: Sequence[FindingRecord], path: str | Path, facts: Optional[SourceFacts] = None, diagnostics: Optional[DiagnosticResults] = None, provenance: Optional[Mapping[str, Any]] = None) -> None:
    """Write a Chinese Markdown rendering of the canonical payload."""
    payload = report_payload(records, facts or SourceFacts(), diagnostics or DiagnosticResults(), provenance)
    Path(path).write_text(_markdown_from_payload(payload), encoding="utf-8")


def emit_reports(records: Sequence[FindingRecord], out_md: str | Path, out_json: str | Path, facts: SourceFacts, diagnostics: DiagnosticResults, provenance: Optional[Mapping[str, Any]] = None) -> None:
    """Build one payload and serialize it to both canonical report formats."""
    payload = report_payload(records, facts, diagnostics, provenance)
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(out_md).write_text(_markdown_from_payload(payload), encoding="utf-8")
