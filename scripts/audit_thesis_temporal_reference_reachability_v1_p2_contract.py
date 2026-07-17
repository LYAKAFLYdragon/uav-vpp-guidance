#!/usr/bin/env python3
"""Read-only audit of whether P2's historical handoffs expose tactical_basis_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_local_action_authority_map_v2 as historical
import run_thesis_temporal_reference_reachability_v1_p2 as p2


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit(config_path: Path, output: Path) -> dict[str, Any]:
    config = p2._design(config_path)
    _reference, cohort = p2._validated_inputs(config)
    rows: list[dict[str, Any]] = []
    for entry in cohort["entries"]:
        source_config, _scenario, registry = historical.base._source(entry)
        episode = historical.base._episode(entry, source_config, registry)
        try:
            generator = episode.env.virtual_point_generator
            rows.append(
                {
                    "cohort_key": str(entry["cohort_key"]),
                    "opponent": str(entry["opponent"]),
                    "historical_source_id": str(entry["historical_source_id"]),
                    "backend": str(getattr(episode.env, "_backend", None)),
                    "active_action_semantics": str(getattr(generator, "action_semantics", None)),
                    "configured_action_semantics": str(
                        getattr(generator, "configured_action_semantics", None)
                    ),
                }
            )
        finally:
            episode.close()
    tactical = [row for row in rows if row["active_action_semantics"] == "tactical_basis_v1"]
    result = {
        "source_id": p2.SOURCE_ID,
        "stage": p2.P2_STAGE,
        "audit_kind": "read_only_historical_handoff_action_semantics",
        "records": len(rows),
        "strict_jsbsim_records": sum(row["backend"] == "jsbsim" for row in rows),
        "tactical_basis_v1_records": len(tactical),
        "rows": rows,
        "verdict": (
            "p2_input_contract_incompatible"
            if not tactical
            else "p2_input_contract_partially_compatible"
        ),
        "p3_permitted": False,
        "reason": (
            "The P1 compiler requires tactical_basis_v1 vectors, but the replay-qualified "
            "historical handoffs expose cartesian_offset. No coordinate reinterpretation is allowed."
        ),
    }
    _write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.output), ensure_ascii=False, indent=2))
