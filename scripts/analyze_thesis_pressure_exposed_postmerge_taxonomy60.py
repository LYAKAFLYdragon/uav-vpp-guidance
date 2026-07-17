#!/usr/bin/env python3
"""Read-only audit for pressure-exposed post-merge windows in Taxonomy60.

This script never starts JSBSim, mutates an artifact, or changes a policy.  It
checks whether the frozen fixed-head-on trajectories already contain the
minimum pressure cohort required before a new collector may be justified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (  # noqa: E402
    classify_relative_geometry,
)
from uav_vpp_guidance.evaluation.pressure_exposed_postmerge_contract import (  # noqa: E402
    REQUIRED_PHASES,
    find_first_pressure_window,
)


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHOD = "fixed_head_on"
GATES = {
    "required_window_steps": 40,
    "target_attack_score_min": 0.1,
    "min_pressure_steps_per_window": 10,
}


class AuditError(RuntimeError):
    """Raised when the frozen formal source cannot be read safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _phase(frame: Mapping[str, Any]) -> str:
    if bool(frame.get("pre_merge", False)):
        return "pre_merge"
    rate = _finite(frame.get("range_rate_mps"))
    return "re_entry" if rate is not None and rate < 0.0 else "post_merge"


def _normalized_step(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    aa_deg = _finite(frame.get("aa_deg"))
    ata_deg = _finite(frame.get("ata_deg"))
    if aa_deg is None or ata_deg is None:
        return None
    phase = _phase(frame)
    return {
        "step": int(frame.get("step", 0)),
        "phase": phase,
        "dynamic_state": classify_relative_geometry(aa_deg, ata_deg),
        "target_attack_score": _finite(frame.get("target_attack_score")),
        "target_in_attack_zone": bool(frame.get("target_in_attack_zone", False)),
        "prediction_valid": bool(frame.get("prediction_valid", False)),
        "prediction_fallback": bool(frame.get("prediction_fallback", False)),
        "range_m": _finite(frame.get("range_m")),
        "range_rate_mps": _finite(frame.get("range_rate_mps")),
        "aa_deg": aa_deg,
        "ata_deg": ata_deg,
    }


def _manifest_metadata(path: Path) -> dict[str, Mapping[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("source_id") != SOURCE_ID:
        raise AuditError("Taxonomy60 manifest source ID mismatch")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 60:
        raise AuditError("Taxonomy60 manifest must contain exactly 60 scenarios")
    return {str(item["name"]): item["metadata"] for item in scenarios}


def audit(root: Path, manifest: Path) -> dict[str, Any]:
    verification = json.loads((root / "formal_verification.json").read_text(encoding="utf-8"))
    if verification.get("source_id") != SOURCE_ID or verification.get("verdict") != "pass":
        raise AuditError("formal verifier is not a passing Taxonomy60 receipt")
    metadata = _manifest_metadata(manifest)
    rows: list[dict[str, Any]] = []
    disadvantage_scenarios = {
        scenario: scenario_meta
        for scenario, scenario_meta in metadata.items()
        if scenario_meta.get("initial_class") == "disadvantage"
    }
    if len(disadvantage_scenarios) != 12:
        raise AuditError("Taxonomy60 must contain exactly 12 initial-disadvantage scenarios")
    for opponent in OPPONENTS:
        raw_dir = root / f"{SOURCE_ID}-{opponent}" / "raw"
        for scenario, scenario_meta in sorted(disadvantage_scenarios.items()):
            seed = int(scenario_meta["scenario_seed"])
            matches = sorted(raw_dir.glob(f"*/{METHOD}/seed_{seed}/episode_*.json"))
            if len(matches) != 1:
                raise AuditError(
                    f"expected one fixed-head-on raw record for {opponent}/{scenario}, got {len(matches)}"
                )
            raw_path = matches[0]
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if str(raw.get("controller") or raw.get("method") or "") != METHOD:
                raise AuditError(f"fixed-head-on controller mismatch: {raw_path}")
            if str(raw.get("scenario") or "") != scenario:
                raise AuditError(f"scenario identity mismatch: {raw_path}")
            if raw.get("backend") != "jsbsim" or raw.get("strict_backend") is not True:
                raise AuditError(f"strict JSBSim contract failed: {raw_path}")
            trajectory = raw.get("trajectory")
            if not isinstance(trajectory, list):
                raise AuditError(f"raw trajectory missing: {raw_path}")
            steps = [item for frame in trajectory if isinstance(frame, Mapping) if (item := _normalized_step(frame))]
            window = find_first_pressure_window(steps, GATES)
            postmerge = [step for step in steps if step["phase"] in REQUIRED_PHASES]
            pressure = [
                step for step in postmerge
                if step["target_attack_score"] is not None and step["target_attack_score"] >= GATES["target_attack_score_min"]
            ]
            rows.append(
                {
                    "opponent": opponent,
                    "scenario_id": scenario,
                    "scenario_signature": scenario_meta.get("mirror_pair_id"),
                    "mirror_sign": scenario_meta.get("mirror_sign"),
                    "raw_path": str(raw_path),
                    "raw_sha256": _sha256(raw_path),
                    "trajectory_steps": len(steps),
                    "postmerge_steps": len(postmerge),
                    "postmerge_pressure_steps": len(pressure),
                    "first_postmerge_pressure_step": pressure[0]["step"] if pressure else None,
                    "qualifying_window": window is not None,
                    "qualifying_window_start_step": window[0]["step"] if window else None,
                    "qualifying_window_pressure_steps": sum(
                        float(step["target_attack_score"]) >= GATES["target_attack_score_min"] for step in window
                    ) if window else 0,
                }
            )
    if len(rows) != 36:
        raise AuditError(f"expected 36 fixed-head-on disadvantage records, got {len(rows)}")
    per_opponent: dict[str, dict[str, Any]] = {}
    for opponent in OPPONENTS:
        group = [row for row in rows if row["opponent"] == opponent]
        qualifying = [row for row in group if row["qualifying_window"]]
        per_opponent[opponent] = {
            "records": len(group),
            "records_with_any_postmerge_pressure": sum(row["postmerge_pressure_steps"] > 0 for row in group),
            "qualifying_windows": len(qualifying),
            "signatures_with_any_postmerge_pressure": sorted({str(row["scenario_signature"]) for row in group if row["postmerge_pressure_steps"] > 0}),
            "mirror_signs_with_any_postmerge_pressure": sorted({str(row["mirror_sign"]) for row in group if row["postmerge_pressure_steps"] > 0}),
        }
    return {
        "audit_kind": "read_only_taxonomy60_pressure_postmerge_coverage",
        "source_id": SOURCE_ID,
        "method": METHOD,
        "gates": GATES,
        "records": rows,
        "per_opponent": per_opponent,
        "conclusion": "existing_taxonomy60_sufficient" if all(
            details["qualifying_windows"] >= 4 for details in per_opponent.values()
        ) else "existing_taxonomy60_insufficient_new_collector_required",
        "interpretation_boundary": "This audit measures only availability of a pressure-exposed causal cohort. It does not assess controller quality or opponent global strength.",
    }


def _write_outputs(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = list(result["records"])
    with (output_dir / "episode_pressure_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "pressure_coverage_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Taxonomy60 Pressure-Exposed Post-Merge Coverage Audit",
        "",
        f"- Frozen source: `{result['source_id']}`",
        f"- Controller: `{result['method']}`",
        f"- Conclusion: `{result['conclusion']}`",
        "- Scope: read-only inspection of fixed-head-on, initial-disadvantage raw trajectories; no JSBSim execution, training, tuning, or selection was performed.",
        "",
        "| Opponent | Records | Any post-merge pressure | 40-step qualifying windows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for opponent, details in result["per_opponent"].items():
        lines.append(
            f"| {opponent} | {details['records']} | {details['records_with_any_postmerge_pressure']} | {details['qualifying_windows']} |"
        )
    lines.extend(
        [
            "",
            "A nonzero attack-zone score before or outside a continuous post-merge interval is not a valid defensive-pressure endpoint. If this audit is insufficient, a new collector must be evaluated separately rather than reinterpreting these records as a controller failure.",
        ]
    )
    (output_dir / "pressure_coverage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/experiment/manifests/thesis_dualskill_taxonomy60_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.formal_root.resolve(), args.manifest.resolve())
        _write_outputs(args.output_dir.resolve(), result)
    except AuditError as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
