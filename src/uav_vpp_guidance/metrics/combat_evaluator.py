"""Combat metric aggregation utilities.

``survival_rate`` means the ego aircraft was not physically destroyed,
shot down, crashed, or out-of-bounds.  It can therefore count an HP-timeout
loss as survived: the aircraft lost the engagement but remained flyable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


LOSS_REASONS_FOR_SURVIVAL = {
    "ego_killed",
    "ego_crash_or_out_of_bounds",
    "crash",
    "out_of_bounds",
}


def compute_combat_metrics(
    episodes: Iterable[Dict[str, Any]],
    missing_time_to_kill: str = "nan",
) -> Dict[str, Any]:
    """Compute win/survival/HP/time-to-kill metrics from episode records."""
    rows = list(episodes)
    total = len(rows)
    wins_for_rate = 0
    losses_for_rate = 0
    survived = 0
    hp_advantages: List[float] = []
    ttk_values: List[float] = []

    for row in rows:
        outcome = _outcome(row)
        reason = str(row.get("combat_reason") or row.get("termination_reason") or "")
        exclude_from_rate = _exclude_from_win_rate(row, reason)
        if outcome == "win" and not exclude_from_rate:
            wins_for_rate += 1
        elif outcome == "loss" and not exclude_from_rate:
            losses_for_rate += 1

        failed_survival = outcome == "loss" and reason in LOSS_REASONS_FOR_SURVIVAL
        if not failed_survival:
            survived += 1

        if outcome == "win":
            ego_hp = _float(row.get("ego_hp", row.get("ego_hp_remaining")))
            target_hp = _float(row.get("target_hp", row.get("target_hp_remaining")))
            if np.isfinite(ego_hp) and np.isfinite(target_hp):
                hp_advantages.append(float(ego_hp - target_hp))

        ttk = _float(row.get("combat_time_to_kill", row.get("time_to_kill")))
        if not np.isfinite(ttk):
            if missing_time_to_kill == "max":
                ttk = _float(row.get("total_time_s", row.get("episode_max_time_s")))
            else:
                ttk = float("nan")
        ttk_values.append(ttk)

    denom = wins_for_rate + losses_for_rate
    return {
        "episodes": total,
        "wins": wins_for_rate,
        "losses": losses_for_rate,
        "win_rate": wins_for_rate / denom if denom else float("nan"),
        "combat_success_rate": wins_for_rate / denom if denom else float("nan"),
        "survival_rate": survived / total if total else float("nan"),
        "hp_advantage": _finite_mean(hp_advantages),
        "mean_time_to_kill": _finite_mean(ttk_values),
        "time_to_kill": ttk_values,
    }


def _outcome(row: Dict[str, Any]) -> str:
    outcome = row.get("combat_outcome")
    if outcome in {"win", "loss", "draw"}:
        return str(outcome)
    if row.get("win") is True or row.get("combat_success") is True:
        return "win"
    if row.get("loss") is True:
        return "loss"
    if row.get("draw") is True:
        return "draw"
    if row.get("success") is True:
        return "win"
    return "draw"


def _exclude_from_win_rate(row: Dict[str, Any], reason: str) -> bool:
    """Exclude unresolved timeout/draw rows, but keep HP-decided timeout wins/losses."""
    outcome = _outcome(row)
    if outcome == "draw":
        return True
    if reason in {"timeout_hp_advantage", "timeout_hp_disadvantage"}:
        return False
    if reason in {"timeout", "timeout_draw"}:
        return True
    return bool(row.get("is_timeout") is True and not reason)


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")
