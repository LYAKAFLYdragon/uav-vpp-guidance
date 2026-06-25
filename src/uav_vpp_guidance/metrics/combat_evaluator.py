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
    damage_dealt_values: List[float] = []
    damage_taken_values: List[float] = []
    damage_exchange_count = 0
    effective_engagement_count = 0
    damaging_win_count = 0

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

        damage_dealt, damage_taken = _extract_damage(row)
        if np.isfinite(damage_dealt):
            damage_dealt_values.append(float(damage_dealt))
        if np.isfinite(damage_taken):
            damage_taken_values.append(float(damage_taken))
        exchanged = (
            (np.isfinite(damage_dealt) and damage_dealt > 1e-6)
            or (np.isfinite(damage_taken) and damage_taken > 1e-6)
        )
        effective_engagement = np.isfinite(damage_dealt) and damage_dealt > 1e-6
        if exchanged:
            damage_exchange_count += 1
        if effective_engagement:
            effective_engagement_count += 1
        if outcome == "win" and effective_engagement:
            damaging_win_count += 1

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
        "damage_exchange_rate": damage_exchange_count / total if total else float("nan"),
        "effective_engagement_rate": (
            effective_engagement_count / total if total else float("nan")
        ),
        "damaging_win_rate": damaging_win_count / total if total else float("nan"),
        "mean_damage_dealt": _finite_mean(damage_dealt_values),
        "mean_damage_taken": _finite_mean(damage_taken_values),
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


def _extract_damage(row: Dict[str, Any]) -> tuple[float, float]:
    initial_hp = _float(row.get("combat_initial_hp", row.get("initial_hp", 100.0)))
    if not np.isfinite(initial_hp):
        initial_hp = 100.0

    ego_hp = _float(row.get("ego_hp", row.get("ego_hp_remaining")))
    target_hp = _float(row.get("target_hp", row.get("target_hp_remaining")))

    damage_dealt = (
        max(0.0, initial_hp - target_hp) if np.isfinite(target_hp) else float("nan")
    )
    damage_taken = (
        max(0.0, initial_hp - ego_hp) if np.isfinite(ego_hp) else float("nan")
    )
    return damage_dealt, damage_taken


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")
