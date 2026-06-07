"""
McNemar pairing validator.

Ensures McNemar exact tests are computed on strictly paired episodes,
aligned by canonical pairing keys that exclude the comparison dimension.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from uav_vpp_guidance.evaluation.statistical_comparison import mcnemar_exact_pvalue


def build_pairing_key(ep: dict, exclude: Optional[List[str]] = None) -> Tuple:
    """Canonical pairing key for McNemar alignment.

    Args:
        ep: Episode dict.
        exclude: Fields to exclude from the key (e.g., ["method"] when comparing methods).
    """
    exclude = set(exclude or [])
    parts = []
    if "scenario" not in exclude:
        parts.append(ep.get("scenario", ""))
    if "method" not in exclude:
        parts.append(ep.get("method", ""))
    if "guidance_mode" not in exclude:
        parts.append(ep.get("effective_guidance_mode", ep.get("guidance_mode", "")))
    if "training_seed" not in exclude:
        parts.append(ep.get("training_seed", -1))
    if "evaluation_seed" not in exclude:
        parts.append(ep.get("evaluation_seed", ep.get("eval_seed", -1)))
    if "episode_index" not in exclude:
        parts.append(ep.get("episode_index", -1))
    return tuple(parts)


def validate_mcnemar_pairing(episodes: List[dict]) -> Tuple[bool, List[str]]:
    """
    Validate that all episodes can be uniquely paired by key.

    Returns:
        (ok, issues)
    """
    issues = []
    key_counts = {}
    for ep in episodes:
        key = build_pairing_key(ep)
        key_counts[key] = key_counts.get(key, 0) + 1

    duplicates = [k for k, v in key_counts.items() if v > 1]
    if duplicates:
        issues.append(f"Duplicate pairing keys found: {len(duplicates)} keys")

    missing_keys = []
    for ep in episodes:
        key = build_pairing_key(ep)
        if any(v == -1 or v == "" for v in key):
            missing_keys.append(key)
    if missing_keys:
        issues.append(f"Episodes with missing pairing fields: {len(missing_keys)}")

    return len(issues) == 0, issues


def mcnemar_paired_exact_by_key(
    episodes_a: List[dict],
    episodes_b: List[dict],
    exclude_from_key: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
) -> Dict[Tuple, dict]:
    """
    Compute McNemar exact p-values on strictly paired episodes.

    Episodes are aligned by a pairing key that EXCLUDES the comparison dimension
    (e.g., exclude ["method"] when comparing methods, or ["guidance_mode"] when
    comparing guidance laws). Any episode present in only one group is excluded.

    Args:
        episodes_a: List of episode dicts for group A.
        episodes_b: List of episode dicts for group B.
        exclude_from_key: Fields to exclude from the pairing key.
        group_by: Additional fields to group results by (e.g., ["scenario"]).

    Returns:
        dict mapping group_key -> {"n_pairs", "a_success_b_failure",
        "a_failure_b_success", "mcnemar_exact_p", "a_success_rate",
        "b_success_rate", "missing_in_a", "missing_in_b"}
    """
    if exclude_from_key is None:
        exclude_from_key = []
    if group_by is None:
        group_by = []

    def index_eps(eps):
        d = {}
        for ep in eps:
            pair_key = build_pairing_key(ep, exclude=exclude_from_key)
            group_key = tuple(ep.get(k, "") for k in group_by)
            full_key = (group_key, pair_key)
            d[full_key] = ep
        return d

    idx_a = index_eps(episodes_a)
    idx_b = index_eps(episodes_b)

    all_keys = set(idx_a.keys()) | set(idx_b.keys())

    group_results = {}
    for full_key in all_keys:
        group_key, pair_key = full_key
        if group_key not in group_results:
            group_results[group_key] = {
                "n_pairs": 0,
                "a_success_b_failure": 0,
                "a_failure_b_success": 0,
                "a_success_count": 0,
                "b_success_count": 0,
                "missing_in_a": 0,
                "missing_in_b": 0,
            }

        ep_a = idx_a.get(full_key)
        ep_b = idx_b.get(full_key)

        if ep_a is None:
            group_results[group_key]["missing_in_a"] += 1
            continue
        if ep_b is None:
            group_results[group_key]["missing_in_b"] += 1
            continue

        succ_a = bool(ep_a.get("is_success", False))
        succ_b = bool(ep_b.get("is_success", False))
        group_results[group_key]["n_pairs"] += 1
        group_results[group_key]["a_success_count"] += int(succ_a)
        group_results[group_key]["b_success_count"] += int(succ_b)
        if succ_a and not succ_b:
            group_results[group_key]["a_success_b_failure"] += 1
        if not succ_a and succ_b:
            group_results[group_key]["a_failure_b_success"] += 1

    results = {}
    for group_key, stats in group_results.items():
        b_disc = stats["a_success_b_failure"]
        c_disc = stats["a_failure_b_success"]
        n = stats["n_pairs"]
        if n > 0:
            p_val = mcnemar_exact_pvalue(b_disc, c_disc)
            a_sr = stats["a_success_count"] / n
            b_sr = stats["b_success_count"] / n
        else:
            p_val = np.nan
            a_sr = np.nan
            b_sr = np.nan
        results[group_key] = {
            "group_key": group_key,
            "n_pairs": n,
            "a_success_b_failure": b_disc,
            "a_failure_b_success": c_disc,
            "mcnemar_exact_p": float(p_val),
            "a_success_rate": float(a_sr),
            "b_success_rate": float(b_sr),
            "missing_in_a": stats["missing_in_a"],
            "missing_in_b": stats["missing_in_b"],
        }

    return results


def mcnemar_from_dataframe(
    df: pd.DataFrame,
    group_cols: List[str],
    method_col: str,
    success_col: str = "is_success",
    pair_key_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute paired McNemar for all method pairs within each group.

    The DataFrame must contain canonical pairing keys:
    scenario, method, effective_guidance_mode, training_seed, evaluation_seed, episode_index

    Args:
        df: DataFrame with one row per episode.
        group_cols: Columns defining comparison groups (e.g., ["scenario", "effective_guidance_mode"]).
        method_col: Column containing method names.
        success_col: Column containing success boolean.
        pair_key_cols: Columns to use for pairing (default: all except method_col).

    Returns:
        DataFrame with one row per (group, method_a, method_b) comparison.
    """
    required = {"scenario", method_col, "effective_guidance_mode", "training_seed", "evaluation_seed", "episode_index", success_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if pair_key_cols is None:
        pair_key_cols = [c for c in ["scenario", "effective_guidance_mode", "training_seed", "evaluation_seed", "episode_index"] if c != method_col]

    rows = []
    for group_keys, group_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        methods = sorted(group_df[method_col].unique())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a_name, b_name = methods[i], methods[j]
                a_df = group_df[group_df[method_col] == a_name]
                b_df = group_df[group_df[method_col] == b_name]

                a_idx = {
                    tuple(r[c] for c in pair_key_cols): r[success_col]
                    for _, r in a_df.iterrows()
                }
                b_idx = {
                    tuple(r[c] for c in pair_key_cols): r[success_col]
                    for _, r in b_df.iterrows()
                }

                common_keys = set(a_idx.keys()) & set(b_idx.keys())
                missing_a = len(b_idx) - len(common_keys)
                missing_b = len(a_idx) - len(common_keys)

                a_succ = [a_idx[k] for k in common_keys]
                b_succ = [b_idx[k] for k in common_keys]

                b_disc = sum(1 for a, b in zip(a_succ, b_succ) if a and not b)
                c_disc = sum(1 for a, b in zip(a_succ, b_succ) if not a and b)
                n = len(common_keys)

                if n > 0:
                    p_val = mcnemar_exact_pvalue(b_disc, c_disc)
                    a_sr = sum(a_succ) / n
                    b_sr = sum(b_succ) / n
                else:
                    p_val = np.nan
                    a_sr = np.nan
                    b_sr = np.nan

                rows.append({
                    **{group_cols[k]: group_keys[k] for k in range(len(group_cols))},
                    "method_a": a_name,
                    "method_b": b_name,
                    "n_pairs": n,
                    "missing_in_a": missing_a,
                    "missing_in_b": missing_b,
                    "a_success_b_failure": b_disc,
                    "a_failure_b_success": c_disc,
                    "mcnemar_exact_p": float(p_val),
                    "a_success_rate": float(a_sr),
                    "b_success_rate": float(b_sr),
                })

    return pd.DataFrame(rows)
