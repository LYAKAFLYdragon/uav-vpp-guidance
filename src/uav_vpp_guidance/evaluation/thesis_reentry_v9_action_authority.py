"""Pure contracts for the V9 re-entry VPP action-authority probe.

V9 is deliberately non-learning.  It compares a frozen head-on action with
small, predeclared additive perturbations after a raw-SI re-entry handoff.
The helpers here make the perturbation, pairing, and gate rules auditable by
both the runner and an independent verifier.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


OPPONENTS: Tuple[str, ...] = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "frozen_fixed_head_on"


class ActionAuthorityContractError(ValueError):
    """Raised when a V9 action or persisted record violates its contract."""


def finite_action(value: Any) -> np.ndarray:
    """Return one bounded normalized VPP action."""
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-6):
        raise ActionAuthorityContractError("action must be a finite normalized 3-D VPP vector")
    return action


def apply_perturbation(reference_action: Any, delta: Any) -> Dict[str, Any]:
    """Clip an additive perturbation and expose the physically applied delta."""
    reference = finite_action(reference_action)
    requested_delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    if requested_delta.shape != (3,) or not np.isfinite(requested_delta).all():
        raise ActionAuthorityContractError("perturbation must be a finite 3-D vector")
    requested = reference + requested_delta
    applied = np.clip(requested, -1.0, 1.0).astype(np.float32)
    actual_delta = applied - reference
    return {
        "reference_action": reference.tolist(),
        "requested_delta": requested_delta.tolist(),
        "applied_action": applied.tolist(),
        "applied_delta": actual_delta.tolist(),
        "applied_delta_norm": float(np.linalg.norm(actual_delta)),
        "clipped": bool(not np.allclose(requested, applied, rtol=0.0, atol=1e-7)),
    }


def is_ego_failure(reason: Any) -> bool:
    """Keep target crash/OOB as legal wins rather than a safety failure."""
    normalized = str(reason or "").lower()
    if normalized.startswith("target_"):
        return False
    return "crash" in normalized or "out_of_bounds" in normalized


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _qualifying(records: Iterable[Mapping[str, Any]], *, min_window_steps: int) -> list[Mapping[str, Any]]:
    return [
        record for record in records
        if bool(record.get("handoff_reached"))
        and int(record.get("window_steps", 0)) >= min_window_steps
        and bool(record.get("contract_passed"))
    ]


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    method_deltas: Mapping[str, Sequence[float]],
    min_window_steps: int,
    min_pairs: int,
    min_applied_delta_norm: float,
    practical_delta: float,
    safety_delta_max: float,
) -> Dict[str, Any]:
    """Summarize one complete V9 matrix without pooling opponents."""
    by_method: Dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        method = str(record.get("method"))
        by_method.setdefault(method, []).append(record)
    if REFERENCE_METHOD not in by_method:
        raise ActionAuthorityContractError("reference method is missing")

    reference_by_key = {
        (str(row.get("opponent")), str(row.get("pair_key"))): row
        for row in by_method[REFERENCE_METHOD]
    }
    matrix: Dict[str, Any] = {}
    authority_candidates: list[str] = []
    for method, method_records in by_method.items():
        method_summary: Dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in method_records if str(row.get("opponent")) == opponent]
            qualifying = _qualifying(group, min_window_steps=min_window_steps)
            pairs = []
            handoff_hashes_match = True
            for row in qualifying:
                reference = reference_by_key.get((opponent, str(row.get("pair_key"))))
                if reference is None or reference not in _qualifying([reference], min_window_steps=min_window_steps):
                    continue
                handoff_hashes_match = handoff_hashes_match and row.get("handoff_state_hash") == reference.get("handoff_state_hash")
                pairs.append(float(row["intent_loss_auc20"]) - float(reference["intent_loss_auc20"]))
            mirrors = {str(row.get("mirror_sign")) for row in qualifying}
            heights = {str(row.get("height_condition")) for row in qualifying}
            ego_rate = sum(bool(row.get("ego_failure")) for row in group) / max(len(group), 1)
            reference_group = [row for row in by_method[REFERENCE_METHOD] if str(row.get("opponent")) == opponent]
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_group) / max(len(reference_group), 1)
            applied_norm = _mean([float(row.get("mean_applied_delta_norm", 0.0)) for row in qualifying])
            delta = _mean(pairs)
            method_summary[opponent] = {
                "records": len(group),
                "qualifying_episodes": len(qualifying),
                "paired_count": len(pairs),
                "mean_paired_intent_loss_delta_vs_reference": delta,
                "mean_applied_delta_norm": applied_norm,
                "mirrors": len(mirrors),
                "height_conditions": len(heights),
                "handoff_hashes_match": handoff_hashes_match,
                "ego_crash_oob_delta_vs_reference": ego_rate - reference_ego_rate,
                "safety_pass": ego_rate - reference_ego_rate <= safety_delta_max,
            }
        matrix[method] = method_summary
        if method != REFERENCE_METHOD:
            passed_opponents = [
                opponent for opponent, cell in method_summary.items()
                if cell["records"] == 12
                and cell["paired_count"] >= min_pairs
                and cell["mirrors"] == 2
                and cell["height_conditions"] >= 2
                and cell["handoff_hashes_match"]
                and cell["safety_pass"]
                and cell["mean_applied_delta_norm"] is not None
                and cell["mean_applied_delta_norm"] >= min_applied_delta_norm
                and cell["mean_paired_intent_loss_delta_vs_reference"] is not None
                and cell["mean_paired_intent_loss_delta_vs_reference"] <= practical_delta
            ]
            if len(passed_opponents) >= 2:
                authority_candidates.append(method)
    expected_records = len(method_deltas) * len(OPPONENTS) * 12
    return {
        "records": len(records),
        "expected_records": expected_records,
        "matrix": matrix,
        "authority_candidates": authority_candidates,
        "passes": len(records) == expected_records and bool(authority_candidates),
    }
