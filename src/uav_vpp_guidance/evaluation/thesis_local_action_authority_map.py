"""Pure unpooled local-action-authority gate for the five-state thesis lane."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
AXES = ("longitudinal", "lateral", "vertical")
METRICS = ("aa_deg", "ata_deg", "range_m", "range_rate_mps", "specific_energy_height_m", "altitude_m")
SCALES = {"aa_deg": 180.0, "ata_deg": 180.0, "range_m": 3000.0, "range_rate_mps": 200.0, "specific_energy_height_m": 1000.0, "altitude_m": 1000.0}
METHODS = ("reference", "longitudinal_plus", "longitudinal_minus", "lateral_plus", "lateral_minus", "vertical_plus", "vertical_minus")


def summarize(records: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    by_key = {(str(row.get("cohort_key")), str(row.get("method"))): row for row in records}
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    candidates: list[dict[str, str]] = []
    for axis in AXES:
        matrix[axis] = {}
        for metric in METRICS:
            per_opponent: dict[str, Any] = {}
            for opponent in OPPONENTS:
                values: list[float] = []
                safe = True
                for key, baseline in by_key.items():
                    cohort_key, method = key
                    if method != "reference" or baseline.get("opponent") != opponent:
                        continue
                    plus = by_key.get((cohort_key, f"{axis}_plus"))
                    minus = by_key.get((cohort_key, f"{axis}_minus"))
                    valid = all(row and row.get("replay_matched") and row.get("window_complete") and not row.get("pulse_clipping") for row in (baseline, plus, minus))
                    if not valid:
                        continue
                    if (not baseline.get("ego_failure")) and (plus.get("ego_failure") or minus.get("ego_failure")):
                        safe = False
                        continue
                    plus_value = plus.get("final_metrics", {}).get(metric)
                    minus_value = minus.get("final_metrics", {}).get(metric)
                    if isinstance(plus_value, (int, float)) and isinstance(minus_value, (int, float)):
                        values.append((float(plus_value) - float(minus_value)) / (2.0 * SCALES[metric]))
                magnitude = abs(float(median(values))) if values else None
                sign = 0 if not values else (1 if median(values) > 0 else -1 if median(values) < 0 else 0)
                sign_fraction = 0.0 if not values or sign == 0 else sum((value > 0) == (sign > 0) for value in values) / len(values)
                passed = bool(len(values) >= int(gates["min_pairs_per_opponent"]) and magnitude is not None and magnitude >= float(gates["min_abs_median_response"]) and sign_fraction >= float(gates["min_sign_fraction"]) and safe)
                per_opponent[opponent] = {"pairs": len(values), "median_response": None if not values else float(median(values)), "abs_median_response": magnitude, "sign": sign, "sign_fraction": sign_fraction, "safe": safe, "passes": passed}
            matrix[axis][metric] = per_opponent
            passed_cells = [cell for cell in per_opponent.values() if cell["passes"]]
            if len(passed_cells) >= int(gates["opponents_required"]) and len({cell["sign"] for cell in passed_cells}) == 1:
                candidates.append({"axis": axis, "metric": metric, "sign": str(passed_cells[0]["sign"])})
    return {"records": len(records), "expected_records": int(gates["expected_records"]), "matrix": matrix, "local_authority_candidates": candidates, "passes": len(records) == int(gates["expected_records"]) and bool(candidates)}
