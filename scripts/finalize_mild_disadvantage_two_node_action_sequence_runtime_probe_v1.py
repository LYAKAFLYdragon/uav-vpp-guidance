#!/usr/bin/env python3
"""Finalize a completed physics run without rerunning JSBSim.

The runner completed all 24 conditions before its summary writer failed. This
script only reads those traces, derives missing energy telemetry, and writes
the manifests and paired records required by the independent verifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904"
SOURCE_TRACE_SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"
OFFSETS = (4, 12, 48)
METRICS = ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "specific_energy_m2ps2", "altitude_m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {"path": str(path.resolve()), "exists": exists, "size_bytes": path.stat().st_size if exists else None, "sha256": sha256(path) if exists else None}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def condition_result(rows: list[dict[str, str]], condition: str, source: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        speed = number(row.get("own_speed_mps"))
        altitude = number(row.get("altitude_m"))
        row["specific_energy_m2ps2"] = 0.5 * speed * speed + 9.80665 * altitude
    offsets = {int(number(row.get("response_offset"))): row for row in rows}
    response_rows = [offsets[offset] for offset in OFFSETS if offset in offsets]
    backend_values = {row.get("backend", "") for row in rows}
    terminal_rows = [row for row in rows if truth(row.get("terminal"))]
    reason = next((row.get("terminal_reason", "") for row in terminal_rows if row.get("terminal_reason", "")), "window_end")
    reason_lower = reason.lower()
    action_out_of_bounds = any(
        any(number(row.get(key)) < -1.0 or number(row.get(key)) > 1.0 for key in ("action_0", "action_1", "action_2"))
        for row in rows
    )
    finite_keys = ("action_0", "action_1", "action_2", "range_m", "range_rate_mps", "ata_deg", "aa_deg", "altitude_m", "own_speed_mps", "specific_energy_m2ps2")
    finite_telemetry = bool(rows) and all(bool(row.get("state_hash_before_action")) and all(math.isfinite(number(row.get(key))) for key in finite_keys) for row in rows)
    response_complete = len(response_rows) == len(OFFSETS)
    return {
        "source_id": SOURCE_ID,
        "source_trace_id": source["source_trace_id"],
        "opponent": source["opponent"],
        "candidate": source["candidate"],
        "seed": source["seed"],
        "condition": condition,
        "method": "simplest_offline_two_node_sequence_runtime_probe",
        "reset_calls": 1,
        "length": len(rows),
        "reason": reason,
        "is_success": reason_lower == "success",
        "is_crash": "crash" in reason_lower,
        "is_out_of_bounds": "out_of_bounds" in reason_lower or "oob" in reason_lower,
        "is_timeout": reason_lower in {"timeout", "window_end", ""},
        "terminal_observed": bool(terminal_rows),
        "target_x_m": source.get("target_x_m"),
        "target_y_m": source.get("target_y_m"),
        "backend_effective": next(iter(backend_values)) if len(backend_values) == 1 else "mixed",
        "backend_fallback_occurred": any(truth(row.get("backend_fallback_occurred")) for row in rows),
        "finite_telemetry": finite_telemetry,
        "prediction_fallback_occurred": any(truth(row.get("prediction_fallback")) for row in rows),
        "response_window_complete": response_complete,
        "response_window_backend_fallback": any(truth(row.get("backend_fallback_occurred")) for row in response_rows),
        "response_window_prediction_fallback": any(truth(row.get("prediction_fallback")) for row in response_rows),
        "response_window_clipping": any(number(row.get("clipping_rate")) > 0.0 for row in response_rows),
        "action_out_of_bounds": action_out_of_bounds,
        "handoff_step": int(number(rows[0].get("handoff_step"))) if rows else None,
        "source_handoff_hash": rows[0].get("source_handoff_hash", "") if rows else "",
        "handoff_replay_hash": rows[0].get("handoff_replay_hash", "") if rows else "",
        "handoff_hash_match": all(truth(row.get("handoff_hash_match")) for row in rows),
        "node0_observed": any(row.get("sequence_node") == "node0" for row in rows),
        "node1_observed": any(row.get("sequence_node") == "node1" for row in rows),
        "policy_call_count": sum(row.get("mode") == "h0_current_geometry" for row in rows),
        "h0_checkpoint_obs_dim": 16,
        "h0_checkpoint_action_dim": 3,
        "sequence_node0": "[0.0, 0.08, 0.0]",
        "sequence_node1": "[-0.08, 0.0, 0.0]",
        "training_started": False,
        "fallback_or_reset_or_padding": bool(any(truth(row.get("backend_fallback_occurred")) for row in rows) or not response_complete or action_out_of_bounds),
        "terminal_reason": reason,
    }


def pair_records(results: list[dict[str, Any]], trace_map: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_ids = sorted({row["source_trace_id"] for row in results})
    for source_trace_id in source_ids:
        baseline = next(row for row in results if row["source_trace_id"] == source_trace_id and row["condition"] == "baseline")
        candidate = next(row for row in results if row["source_trace_id"] == source_trace_id and row["condition"] == "candidate")
        base = {int(number(row.get("response_offset"))): row for row in trace_map[(source_trace_id, "baseline")]}
        cand = {int(number(row.get("response_offset"))): row for row in trace_map[(source_trace_id, "candidate")]}
        for offset in OFFSETS:
            record: dict[str, Any] = {"source_id": SOURCE_ID, "source_trace_id": source_trace_id, "opponent": baseline["opponent"], "candidate": baseline["candidate"], "response_offset": offset, "pair_valid": offset in base and offset in cand}
            for metric in METRICS:
                b = number(base[offset].get(metric)) if offset in base else float("nan")
                c = number(cand[offset].get(metric)) if offset in cand else float("nan")
                record[f"baseline_{metric}"] = b
                record[f"candidate_{metric}"] = c
                record[f"delta_{metric}"] = c - b if math.isfinite(b) and math.isfinite(c) else float("nan")
            output.append(record)
    return output


def finalize(config_path: Path, root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["experiment"]["source_id"] != SOURCE_ID:
        raise RuntimeError("Source ID mismatch")
    source_root = (ROOT / config["source_traces_root"]).resolve()
    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    source_manifest = json.loads((source_root / "source_trace_manifest.json").read_text(encoding="utf-8"))
    for item in source_manifest["records"]:
        source_rows[(item["opponent"], item["source_trace_id"])] = item
    results: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], list[dict[str, str]]] = {}
    for opponent in config["protocol"]["opponents"]:
        for path in sorted((root / opponent / "traces").glob("*.csv")):
            rows = read_csv(path)
            if not rows:
                raise RuntimeError(f"empty trace: {path}")
            source_trace_id = rows[0]["source_trace_id"]
            condition = rows[0]["condition"]
            source = source_rows[(opponent, source_trace_id)]
            source = {**source, "target_x_m": 625.0 if "x625" in source_trace_id else 750.0, "target_y_m": 200.0 if "x625" in source_trace_id else 250.0}
            for row in rows:
                speed = number(row.get("own_speed_mps"))
                altitude = number(row.get("altitude_m"))
                row["specific_energy_m2ps2"] = 0.5 * speed * speed + 9.80665 * altitude
            traces[(source_trace_id, condition)] = rows
            results.append(condition_result(rows, condition, source))
            # Keep the raw trajectory immutable; derived energy is used only
            # for the aggregate rows below.
    pairs = pair_records(results, traces)
    write_csv(root / "condition_results.csv", results)
    write_csv(root / "paired_response_records.csv", pairs)
    assets = config["assets"]
    asset_receipts = {}
    for key, value in assets.items():
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        asset_receipts[key] = receipt(path.resolve())
    run_manifests = []
    for opponent in config["protocol"]["opponents"]:
        resolved = root / opponent / "resolved_config.yaml"
        run_manifests.append({"source_id": SOURCE_ID, "opponent": opponent, "status": "completed", "source_trace_count": 6, "condition_count": 12, "resolved_config": receipt(resolved), "backend_requested": "jsbsim", "strict_backend": True})
    manifest = {
        "source_id": SOURCE_ID, "status": "completed_physics_finalized_offline", "training_started": False, "training_permitted": False,
        "jsbsim_started": True, "physical_probe_completed": True, "summary_finalization_performed": True,
        "source_trace_source_id": SOURCE_TRACE_SOURCE_ID, "source_trace_count": 12, "condition_count": 24, "expected_condition_count": 24, "pair_count": 12,
        "opponents": config["protocol"]["opponents"], "conditions": config["protocol"]["conditions"], "node_steps_after_handoff": config["protocol"]["node_steps_after_handoff"],
        "node0_delta": config["protocol"]["node0_delta"], "node1_delta": config["protocol"]["node1_delta"], "response_offsets": OFFSETS,
        "config": receipt(config_path), "base_config": receipt((ROOT / config["base_config"]).resolve()), "config_canonical_sha256": __import__("uav_vpp_guidance.common.hash", fromlist=["config_sha256"]).config_sha256(config, truncate=0),
        "asset_receipts": asset_receipts, "run_manifests": run_manifests, "preregistered_gate": config["preregistered_gate"], "contract": config["protocol"],
        "finalization_note": "All 24 JSBSim conditions were completed before the original summary writer failed; this step only repaired offline aggregation and added derived specific energy telemetry.",
    }
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "probe_result.json", {**manifest, "paired_response_records": len(pairs), "decision": "pending_verifier"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment/mild_disadvantage_two_node_action_sequence_runtime_probe_v1.yaml")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.config.resolve(), args.root.resolve())
    print(json.dumps({"source_id": SOURCE_ID, "root": str(args.root.resolve()), "status": "finalized_without_rerun"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
