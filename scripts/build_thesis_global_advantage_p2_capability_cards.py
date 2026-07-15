"""Build non-ranking, interaction-conditioned P2 capability cards from R5.

The input is frozen P1 R5 telemetry generated with the common run-in head-on
specialist.  This script is read-only: it never constructs JSBSim, loads a new
policy, or evaluates heldout240.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
R5_ROOT = Path(
    "E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/"
    "global_advantage_v1_p1_r5_json_telemetry"
)
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-CAPABILITY-CARD-V1"
R5_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P1-R5-JSON-TELEMETRY-REPRO-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
GRAVITY_MPS2 = 9.80665


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def specific_energy_j_per_kg(state: Mapping[str, Any]) -> float:
    """Return target specific mechanical energy from recorded SI telemetry."""

    altitude = _finite(state["altitude_m"], "altitude_m")
    speed = _finite(state["speed_mps"], "speed_mps")
    return GRAVITY_MPS2 * altitude + 0.5 * speed * speed


def _ratio(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _episode_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    header = payload["header"]
    summary = payload["summary"]
    steps = list(payload["steps"])
    if not steps:
        raise ValueError("episode has no telemetry steps")
    if header["source_id"] != R5_SOURCE_ID or summary["source_id"] != R5_SOURCE_ID:
        raise ValueError("episode source ID drifted")
    if int(header["repeat_index"]) != 0 or header["order_variant"] != "forward":
        raise ValueError("capability card must use the prespecified repeat_0/forward view")
    if not summary.get("telemetry_complete") or not summary.get("no_backend_or_prediction_fallback"):
        raise ValueError("episode telemetry/fallback contract failed")
    if any(step.get("effective_skill") != "run_in_head_on" for step in steps):
        raise ValueError("reference policy is not the common frozen run-in specialist")

    target_states = [step["target_state"] for step in steps]
    target_speeds = [_finite(state["speed_mps"], "target speed") for state in target_states]
    target_altitudes = [_finite(state["altitude_m"], "target altitude") for state in target_states]
    target_energy = [specific_energy_j_per_kg(state) for state in target_states]
    first_pass_steps = [int(step["step"]) for step in steps if step.get("first_pass_complete")]
    terminal = str(summary["terminal_reason"])
    return {
        "opponent": str(header["opponent"]),
        "scenario_signature": str(header["scenario_signature"]),
        "scenario_name": str(header["scenario_name"]),
        "step_count": int(summary["step_count"]),
        "target_speed_mean_mps": fmean(target_speeds),
        "target_altitude_mean_m": fmean(target_altitudes),
        "target_specific_energy_mean_j_per_kg": fmean(target_energy),
        "ego_attack_zone_fraction": _ratio([bool(step["ego_in_attack_zone"]) for step in steps]),
        "target_attack_zone_fraction": _ratio([bool(step["target_in_attack_zone"]) for step in steps]),
        "phase_fraction": {
            phase: _ratio([step["phase"] == phase for step in steps])
            for phase in ("pre_merge", "post_merge", "re_entry")
        },
        "first_pass_reached": bool(first_pass_steps),
        "first_pass_step": first_pass_steps[0] if first_pass_steps else None,
        "terminal_reason": terminal,
        "ego_crash_or_oob": terminal.startswith("ego_crash_or_out_of_bounds"),
        "target_crash_or_oob": terminal.startswith("target_crash_or_out_of_bounds"),
    }


def summarize_cards(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cards: dict[str, Any] = {}
    for opponent in OPPONENTS:
        items = [item for item in records if item["opponent"] == opponent]
        if len(items) != 30:
            raise ValueError(f"{opponent} must have exactly 30 repeat_0 episodes")
        cards[opponent] = {
            "episode_count": len(items),
            "reference_policy": "frozen_run_in_head_on_specialist",
            "target_speed_mean_mps": fmean(item["target_speed_mean_mps"] for item in items),
            "target_altitude_mean_m": fmean(item["target_altitude_mean_m"] for item in items),
            "target_specific_energy_mean_j_per_kg": fmean(
                item["target_specific_energy_mean_j_per_kg"] for item in items
            ),
            "ego_attack_zone_fraction": fmean(item["ego_attack_zone_fraction"] for item in items),
            "target_attack_zone_fraction": fmean(item["target_attack_zone_fraction"] for item in items),
            "phase_fraction": {
                phase: fmean(item["phase_fraction"][phase] for item in items)
                for phase in ("pre_merge", "post_merge", "re_entry")
            },
            "first_pass_episode_fraction": _ratio([item["first_pass_reached"] for item in items]),
            "first_pass_step_mean": (
                fmean(item["first_pass_step"] for item in items if item["first_pass_step"])
                if any(item["first_pass_step"] for item in items)
                else None
            ),
            "terminal_reason_counts": dict(sorted(Counter(item["terminal_reason"] for item in items).items())),
            "ego_crash_or_oob_fraction": _ratio([item["ego_crash_or_oob"] for item in items]),
            "target_crash_or_oob_fraction": _ratio([item["target_crash_or_oob"] for item in items]),
        }
    return cards


def build_cards(root: Path) -> dict[str, Any]:
    gate_path = root / "p1_r3_gate_summary.json"
    manifest_path = root / "run_manifest.json"
    gate = _load_json(gate_path)
    manifest = _load_json(manifest_path)
    if gate.get("source_id") != R5_SOURCE_ID or gate.get("passed") is not True:
        raise ValueError("R5 reproducibility gate is not a passed source")
    if manifest.get("source_id") != R5_SOURCE_ID or len(manifest.get("episodes", [])) != 270:
        raise ValueError("R5 manifest is incomplete")
    selected = [
        item
        for item in manifest["episodes"]
        if int(item["repeat_index"]) == 0 and item["order_variant"] == "forward"
    ]
    if len(selected) != 90:
        raise ValueError("R5 selected view must contain 90 episodes")
    records = [_episode_metrics(_load_json(Path(item["artifact_path"]))) for item in selected]
    if len({(item["opponent"], item["scenario_signature"]) for item in records}) != 90:
        raise ValueError("selected capability records are not unique")
    return {
        "source_id": SOURCE_ID,
        "status": "read_only_interaction_conditioned_capability_documentation",
        "claim_boundary": (
            "No Elo, total-strength ranking, or universal opponent ordering. Values describe "
            "pressure observed under the common frozen run-in reference on P1 dev30 only."
        ),
        "evidence": {
            "p1_source_id": R5_SOURCE_ID,
            "p1_gate_sha256": _sha256(gate_path),
            "p1_run_manifest_sha256": _sha256(manifest_path),
            "selected_repeat": {"repeat_index": 0, "order_variant": "forward"},
            "selection_rule": "P1 proved all three repeats equivalent before this fixed view was selected.",
        },
        "cards": summarize_cards(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-root", type=Path, default=R5_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "thesis_global_advantage_v1_p2_opponent_capability_cards.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cards = build_cards(args.r5_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cards, sort_keys=False), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "opponents": list(cards["cards"]), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
