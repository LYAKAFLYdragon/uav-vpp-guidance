"""Read-only alignment audit between phase-reachability v2 and the P4 sampler."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state


PHASES = ("pre_merge", "post_merge", "re_entry")
PRIMARY_ATTRIBUTIONS = (
    "sampler_mismatch",
    "66d_observability_gap",
    "physical_behavior_coverage_gap",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


class P4PhaseTracker:
    """Pure copy of the P4 phase semantics used during geometry pretraining."""

    def __init__(self, merge_range_m: float = 1000.0, reentry_rate_mps: float = -25.0) -> None:
        self.merge_range_m = float(merge_range_m)
        self.reentry_rate_mps = float(reentry_rate_mps)
        self._seen_merge = False
        self._post_merge_steps = 0

    def update(self, range_m: float, range_rate_mps: float) -> str:
        if not self._seen_merge and float(range_m) <= self.merge_range_m:
            self._seen_merge = True
        if not self._seen_merge:
            return "pre_merge"
        self._post_merge_steps += 1
        if self._post_merge_steps >= 5 and float(range_rate_mps) <= self.reentry_rate_mps:
            return "re_entry"
        return "post_merge"


@dataclass(frozen=True)
class AuditDecision:
    primary_attribution: str
    pilot_preregistration_eligible: bool
    reason: str


def _frame_dynamic_state(frame: Mapping[str, Any]) -> str:
    """Map recorder ATA/AA back to the taxonomy's documented LOS convention."""

    # The raw recorder stores ATA as target-to-own LOS and AA as own-to-target
    # LOS. This ordering is checked against each scenario's frozen initial label.
    return classify_dynamic_state(frame.get("aa_deg"), frame.get("ata_deg"))


def _exact_66d_fields_available(record: Mapping[str, Any]) -> dict[str, bool]:
    frames = [item for item in record.get("trajectory", []) if isinstance(item, Mapping)]
    has_vector = bool(frames) and all("observation_vector" in frame for frame in frames)
    has_history = bool(frames) and all("explicit_history" in frame for frame in frames)
    has_embedding = bool(frames) and all("temporal_embedding" in frame for frame in frames)
    has_profile = bool(frames) and all("intent_targets" in frame and "intent_weights" in frame for frame in frames)
    has_action = bool(frames) and all("normalized_vpp_action" in frame for frame in frames)
    return {
        "base16_vector_saved": has_vector,
        "explicit_history_saved": has_history,
        "p3_embedding_saved": has_embedding,
        "profile_conditioning_saved": has_profile,
        "normalized_vpp_action_saved": has_action,
        "exact_66d_to_3d_pair_constructable": all((has_vector, has_history, has_embedding, has_profile, has_action)),
    }


def audit_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise dynamic state, phase, prediction, VPP, and PID telemetry."""

    frames = [item for item in record.get("trajectory", []) if isinstance(item, Mapping)]
    if not frames:
        raise ValueError("v2 raw episode lacks trajectory")
    tracker = P4PhaseTracker()
    dynamic_phase = Counter()
    p4_phase = Counter()
    first_p4_post_merge_step: int | None = None
    first_v2_pass_step: int | None = None
    prediction_valid: list[float] = []
    prediction_fallback: list[float] = []
    vpp_forward: list[float] = []
    vpp_lateral: list[float] = []
    nz_cmd: list[float] = []
    roll_cmd: list[float] = []
    throttle_cmd: list[float] = []
    saturation: list[float] = []
    initial_dynamic = _frame_dynamic_state(frames[0])
    for frame in frames:
        range_m = _finite(frame.get("range_m"))
        range_rate = _finite(frame.get("range_rate_mps"))
        if range_m is None or range_rate is None:
            raise ValueError("v2 raw trajectory contains non-finite phase telemetry")
        phase = tracker.update(range_m, range_rate)
        state = _frame_dynamic_state(frame)
        if state == "unknown":
            raise ValueError("v2 raw trajectory contains unknown dynamic taxonomy")
        dynamic_phase[f"{state}:{phase}"] += 1
        p4_phase[phase] += 1
        if phase != "pre_merge" and first_p4_post_merge_step is None:
            first_p4_post_merge_step = int(frame.get("step", 0))
        if bool(frame.get("first_pass_complete", False)) and first_v2_pass_step is None:
            first_v2_pass_step = int(frame.get("step", 0))
        prediction_valid.append(float(bool(frame.get("prediction_valid", False))))
        prediction_fallback.append(float(bool(frame.get("prediction_fallback", False))))
        for destination, key in (
            (vpp_forward, "vp_forward_bias_m"),
            (vpp_lateral, "vp_lateral_bias_m"),
            (nz_cmd, "nz_cmd"),
            (roll_cmd, "roll_rate_cmd"),
            (throttle_cmd, "throttle_cmd"),
            (saturation, "saturation_flag"),
        ):
            value = _finite(frame.get(key))
            if value is not None:
                destination.append(value)
    metadata = record.get("scenario_metadata", {})
    expected_initial = metadata.get("taxonomy_geometry_state")
    if expected_initial and initial_dynamic != expected_initial:
        raise ValueError(
            f"ATA/AA convention mismatch: raw={initial_dynamic}, expected={expected_initial}"
        )
    exact = _exact_66d_fields_available(record)
    return {
        "opponent": record.get("opponent_stage"),
        "scenario": record.get("scenario"),
        "seed": record.get("seed"),
        "trajectory_steps": len(frames),
        "initial_dynamic_state": initial_dynamic,
        "v2_first_pass_step": first_v2_pass_step,
        "p4_first_post_merge_step": first_p4_post_merge_step,
        "p4_pre_merge_steps": int(p4_phase["pre_merge"]),
        "p4_post_merge_steps": int(p4_phase["post_merge"]),
        "p4_re_entry_steps": int(p4_phase["re_entry"]),
        "dynamic_phase_steps": dict(sorted(dynamic_phase.items())),
        "dynamic_disadvantage_post_merge_reentry_steps": int(
            dynamic_phase["disadvantage:post_merge"] + dynamic_phase["disadvantage:re_entry"]
        ),
        "prediction_valid_fraction": _mean(prediction_valid),
        "prediction_fallback_fraction": _mean(prediction_fallback),
        "mean_vp_forward_bias_m": _mean(vpp_forward),
        "mean_vp_lateral_bias_m": _mean(vpp_lateral),
        "max_abs_nz_cmd": max((abs(value) for value in nz_cmd), default=None),
        "mean_abs_roll_rate_cmd": _mean(abs(value) for value in roll_cmd),
        "mean_throttle_cmd": _mean(throttle_cmd),
        "saturation_fraction": _mean(saturation),
        **exact,
    }


def decide_primary_attribution(
    *,
    all_v2_episodes_reach_first_pass: bool,
    p4_v2_all_cells_supported: bool,
    exact_66d_pair_count: int,
    episode_count: int,
) -> AuditDecision:
    """Choose exactly one root attribution using a predeclared priority order."""

    if all_v2_episodes_reach_first_pass and not p4_v2_all_cells_supported:
        return AuditDecision(
            primary_attribution="sampler_mismatch",
            pilot_preregistration_eligible=False,
            reason=(
                "Physical first-pass is observed in every v2 episode, while the separately "
                "frozen P4-v2 physical sampler fails its dynamic-state x phase support gate."
            ),
        )
    if exact_66d_pair_count < episode_count:
        return AuditDecision(
            primary_attribution="66d_observability_gap",
            pilot_preregistration_eligible=False,
            reason="The raw evidence cannot reconstruct exact 66-D profile-conditioned observations and 3-D actions.",
        )
    return AuditDecision(
        primary_attribution="physical_behavior_coverage_gap",
        pilot_preregistration_eligible=False,
        reason="The selected physical envelope did not reach first-pass continuously under every opponent.",
    )


def aggregate_episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no episode rows")
    by_opponent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    dynamic_summary: Counter[tuple[str, str]] = Counter()
    for row in rows:
        opponent = str(row["opponent"])
        by_opponent[opponent].append(row)
        for key, count in row["dynamic_phase_steps"].items():
            dynamic_summary[(opponent, key)] += int(count)
    opponent_summary = {}
    telemetry_summary = []
    for opponent, items in by_opponent.items():
        opponent_summary[opponent] = {
            "episodes": len(items),
            "first_pass_episodes": sum(item["v2_first_pass_step"] is not None for item in items),
            "p4_post_merge_episodes": sum(item["p4_first_post_merge_step"] is not None for item in items),
            "dynamic_disadvantage_post_merge_reentry_steps": sum(
                int(item["dynamic_disadvantage_post_merge_reentry_steps"]) for item in items
            ),
            "exact_66d_to_3d_pair_episodes": sum(
                bool(item["exact_66d_to_3d_pair_constructable"]) for item in items
            ),
        }
        telemetry_summary.append(
            {
                "opponent": opponent,
                "mean_prediction_valid_fraction": _mean(
                    item["prediction_valid_fraction"] for item in items if item["prediction_valid_fraction"] is not None
                ),
                "mean_prediction_fallback_fraction": _mean(
                    item["prediction_fallback_fraction"] for item in items if item["prediction_fallback_fraction"] is not None
                ),
                "mean_vp_forward_bias_m": _mean(
                    item["mean_vp_forward_bias_m"] for item in items if item["mean_vp_forward_bias_m"] is not None
                ),
                "mean_vp_lateral_bias_m": _mean(
                    item["mean_vp_lateral_bias_m"] for item in items if item["mean_vp_lateral_bias_m"] is not None
                ),
                "max_abs_nz_cmd": max(
                    (float(item["max_abs_nz_cmd"]) for item in items if item["max_abs_nz_cmd"] is not None),
                    default=None,
                ),
                "mean_abs_roll_rate_cmd": _mean(
                    item["mean_abs_roll_rate_cmd"] for item in items if item["mean_abs_roll_rate_cmd"] is not None
                ),
                "mean_throttle_cmd": _mean(
                    item["mean_throttle_cmd"] for item in items if item["mean_throttle_cmd"] is not None
                ),
                "mean_saturation_fraction": _mean(
                    item["saturation_fraction"] for item in items if item["saturation_fraction"] is not None
                ),
            }
        )
    return {
        "opponent_summary": opponent_summary,
        "dynamic_phase_steps": [
            {"opponent": opponent, "dynamic_state_phase": state_phase, "steps": count}
            for (opponent, state_phase), count in sorted(dynamic_summary.items())
        ],
        "telemetry_summary": telemetry_summary,
    }
