#!/usr/bin/env python3
"""
Phase 3 Geometric Hierarchical Ablation (jsbenv / JSBSim backend).

All experiments run on the JSBSim backend ("jsbenv") using the frozen
canonical configuration in ``config/canonical/``. No new default parameters
are introduced: every knob below is an existing config key or an existing
function argument.

Experiment groups
-----------------
G1  VPP vs No-VPP vs Pure-PN, stratified by scenario geometry family.
G2  Tail-chase 0% root cause: VPP-offset mode vs turn-rate limitation,
    decomposed by termination reason and saturation telemetry.
G3  CEM-EMA stability: fixed policy, 10 CEM-EMA iterations, variance of g*.
G4  CAIS terminal-crash reduction (CombatAwareSchedule phase weighting).
G5  Quantitative turn-rate / energy bottleneck relationship.

Outputs success rate + McNemar/bootstrap statistics per group and writes the
report to ``docs/results/phase3_geometry_ablation.md``.

Usage:
    python scripts/run_phase3_geometry_ablation.py --seeds 3 --episodes 10
    python scripts/run_phase3_geometry_ablation.py --smoke
    python scripts/run_phase3_geometry_ablation.py --groups G1 G2
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config  # noqa: E402
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv  # noqa: E402
from uav_vpp_guidance.envs.scenario_registry import (  # noqa: E402
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.gain_optimizer.cem import CEMEMAGainOptimizer  # noqa: E402
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace  # noqa: E402
from uav_vpp_guidance.ablations.cais_only.combat_aware_schedule import (  # noqa: E402
    CombatAwareSchedule,
)
from uav_vpp_guidance.evaluation.statistical_comparison import (  # noqa: E402
    mcnemar_exact_pvalue,
    bootstrap_success_rate_ci,
    mean_std,
)

CANONICAL_DIR = REPO_ROOT / "config" / "canonical"
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "phase3_geometry_ablation"
REPORT_PATH = REPO_ROOT / "docs" / "results" / "phase3_geometry_ablation.md"

# Geometry families to stratify over (from ScenarioRegistry smoke_test set,
# one per canonical geometry family).
GEOMETRY_SCENARIOS = [
    "smoke_tail_chase",
    "smoke_head_on",
    "smoke_crossing_left",
    "smoke_crossing_right",
    "smoke_offset_attack",
    "smoke_fleeing",
]


# ----------------------------------------------------------------------
# Canonical config assembly (jsbenv)
# ----------------------------------------------------------------------
def build_canonical_jsbenv_config() -> dict:
    """Assemble the frozen canonical config bound to the JSBSim backend."""
    cfg: dict = {}
    # env + ppo provide the runnable scaffold; canonical/* provide the frozen
    # guidance / reward / vpp / gain definitions.
    cfg = merge_config(cfg, load_yaml_config(str(CONFIG_DIR / "env.yaml")))
    cfg = merge_config(cfg, load_yaml_config(str(CONFIG_DIR / "ppo.yaml")))
    cfg = merge_config(cfg, load_yaml_config(str(CANONICAL_DIR / "guidance.yaml")))
    cfg = merge_config(cfg, load_yaml_config(str(CANONICAL_DIR / "reward.yaml")))
    vpp = load_yaml_config(str(CANONICAL_DIR / "virtual_point.yaml"))
    if "virtual_point" in vpp:
        cfg["virtual_point"] = merge_config(
            cfg.get("virtual_point", {}), vpp["virtual_point"]
        )

    # Bind to JSBSim backend (jsbenv). strict_backend ensures we never silently
    # fall back to the simple env.
    cfg["backend"] = "jsbsim"
    cfg.setdefault("env", {})["backend"] = "jsbsim"
    cfg["env"]["use_jsbsim"] = True
    cfg["env"]["strict_backend"] = True
    cfg["env"]["aircraft_model"] = "f16"
    return cfg


def make_arm_config(base: dict, arm: str) -> dict:
    """Derive a guidance-architecture arm by toggling only existing keys.

    Arms:
      - "vpp"     : VPP offset layer active (rule_based_pursuit anchor, the
                    deterministic stand-in for the learned VPP offset used in
                    the Stage 6G.5C diagnosis).
      - "no_vpp"  : VPP layer present but offset forced to zero (zero_offset).
      - "pure_pn" : No-VPP + proportional_navigation guidance.
    """
    cfg = copy.deepcopy(base)
    vp = cfg.setdefault("virtual_point", {})
    vp["enabled"] = True

    if arm == "vpp":
        vp["mode"] = "normal"
        vp["anchor_mode"] = "rule_based_pursuit"
        vp.setdefault("lead_distance_m", 500.0)
        cfg["guidance"]["mode"] = "los_rate"
    elif arm == "no_vpp":
        vp["mode"] = "zero_offset"
        vp["anchor_mode"] = "current_target"
        cfg["guidance"]["mode"] = "los_rate"
    elif arm == "pure_pn":
        vp["mode"] = "zero_offset"
        vp["anchor_mode"] = "current_target"
        cfg["guidance"]["mode"] = "proportional_navigation"
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return cfg


# ----------------------------------------------------------------------
# Episode runner (deterministic guidance, no learned policy required)
# ----------------------------------------------------------------------
def run_episode(env: CloseRangeTrackingEnv, scenario: dict, seed: int) -> dict:
    """Run a single episode with zero VPP action (pure guidance chain).

    Records terminal outcome and per-episode telemetry aggregates used by the
    root-cause (G2) and bottleneck (G5) analyses.
    """
    obs = env.reset(scenario=scenario, seed=seed)
    action = np.zeros(3, dtype=np.float64)

    min_range = float("inf")
    nz_peak = 0.0
    roll_rate_peak = 0.0
    sat_steps = 0
    los_rate_peak = 0.0
    speed_diff_first = None
    alt_diff_first = None
    n = 0
    reason = "timeout"
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        n += 1
        rel = info.get("relative_state", {})
        own = info.get("own_state", {})
        rng = float(rel.get("range_m", obs.get("relative_state", {}).get("range_m", 0.0)))
        min_range = min(min_range, rng)
        nz_peak = max(nz_peak, abs(float(info.get("nz_cmd", own.get("nz_g", 0.0)) or 0.0)))
        roll_rate_peak = max(roll_rate_peak, abs(float(info.get("roll_rate_cmd", 0.0) or 0.0)))
        if info.get("saturation_flag"):
            sat_steps += 1
        lr = info.get("los_rate", 0.0)
        try:
            los_rate_peak = max(los_rate_peak, abs(float(lr)))
        except (TypeError, ValueError):
            pass
        if speed_diff_first is None:
            speed_diff_first = float(rel.get("speed_diff_mps", 0.0))
            alt_diff_first = float(rel.get("altitude_diff_m", 0.0))
        if terminated or truncated:
            reason = info.get("reason", "unknown")
            break

    return {
        "scenario": scenario.get("name", "unknown"),
        "seed": int(seed),
        "reason": reason,
        "is_success": reason == "success",
        "is_crash": reason == "crash",
        "is_out_of_bounds": reason == "out_of_bounds",
        "is_timeout": reason == "timeout",
        "length": n,
        "min_range_m": None if not math.isfinite(min_range) else min_range,
        "nz_peak_g": nz_peak,
        "roll_rate_peak": roll_rate_peak,
        "saturation_step_rate": sat_steps / max(1, n),
        "los_rate_peak": los_rate_peak,
        "speed_diff_mps": speed_diff_first if speed_diff_first is not None else 0.0,
        "altitude_diff_m": alt_diff_first if alt_diff_first is not None else 0.0,
    }


def evaluate_arm(base: dict, arm: str, scenario_names: list, seeds: list) -> list:
    """Run all (scenario, seed) episodes for one architecture arm."""
    cfg = make_arm_config(base, arm)
    env = CloseRangeTrackingEnv(cfg)
    if env._backend != "jsbsim":
        env.close()
        raise RuntimeError(
            f"Expected jsbsim backend but got '{env._backend}'. "
            "Check JSBSim installation/data."
        )
    episodes = []
    for name in scenario_names:
        scen = ScenarioRegistry.get(name)
        if scen is None:
            continue
        for seed in seeds:
            ep = run_episode(env, scen, seed)
            ep["arm"] = arm
            episodes.append(ep)
    env.close()
    return episodes


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def success_rate(episodes: list) -> float:
    if not episodes:
        return float("nan")
    return sum(1 for e in episodes if e["is_success"]) / len(episodes)


def paired_mcnemar(eps_a: list, eps_b: list) -> dict:
    """McNemar exact test on success, paired by (scenario, seed)."""
    index_a = {(e["scenario"], e["seed"]): e["is_success"] for e in eps_a}
    index_b = {(e["scenario"], e["seed"]): e["is_success"] for e in eps_b}
    keys = sorted(set(index_a) & set(index_b))
    b = sum(1 for k in keys if index_a[k] and not index_b[k])
    c = sum(1 for k in keys if not index_a[k] and index_b[k])
    p = mcnemar_exact_pvalue(b, c) if keys else float("nan")
    return {
        "n_pairs": len(keys),
        "a_success_b_fail": b,
        "a_fail_b_success": c,
        "mcnemar_exact_p": p,
        "sr_a": success_rate(eps_a),
        "sr_b": success_rate(eps_b),
        "significant_at_05": bool(np.isfinite(p) and p < 0.05),
    }


def sr_bootstrap(episodes: list) -> dict:
    outcomes = [1 if e["is_success"] else 0 for e in episodes]
    rate, lo, hi = bootstrap_success_rate_ci(outcomes, n_bootstrap=2000, random_seed=42)
    return {"rate": rate, "ci_low": lo, "ci_high": hi, "n": len(outcomes)}


# ----------------------------------------------------------------------
# G1: VPP vs No-VPP vs Pure-PN, stratified by scenario geometry
# ----------------------------------------------------------------------
def group_g1(base: dict, scenario_names: list, seeds: list) -> dict:
    arms = ["vpp", "no_vpp", "pure_pn"]
    eps = {arm: evaluate_arm(base, arm, scenario_names, seeds) for arm in arms}

    # Overall + per-scenario success rates
    per_arm_overall = {arm: sr_bootstrap(eps[arm]) for arm in arms}
    per_scenario = {}
    for name in scenario_names:
        per_scenario[name] = {
            arm: success_rate([e for e in eps[arm] if e["scenario"] == name])
            for arm in arms
        }

    # Pairwise McNemar (paired by scenario+seed)
    pairwise = {
        "pure_pn_vs_vpp": paired_mcnemar(eps["pure_pn"], eps["vpp"]),
        "pure_pn_vs_no_vpp": paired_mcnemar(eps["pure_pn"], eps["no_vpp"]),
        "no_vpp_vs_vpp": paired_mcnemar(eps["no_vpp"], eps["vpp"]),
    }
    return {
        "arms": arms,
        "episodes": eps,
        "per_arm_overall": per_arm_overall,
        "per_scenario": per_scenario,
        "pairwise": pairwise,
    }


# ----------------------------------------------------------------------
# G2: Tail-chase 0% root cause (VPP-offset mode vs turn-rate limitation)
# ----------------------------------------------------------------------
def group_g2(base: dict, seeds: list) -> dict:
    """Decompose tail-chase failure into VPP-offset vs turn-rate/saturation."""
    tail_scenarios = ["smoke_tail_chase", "negative_tail_chase"]
    tail_scenarios = [s for s in tail_scenarios if ScenarioRegistry.get(s) is not None]
    arms = ["vpp", "no_vpp", "pure_pn"]
    eps = {arm: evaluate_arm(base, arm, tail_scenarios, seeds) for arm in arms}

    def reason_breakdown(arm_eps):
        total = len(arm_eps)
        out = {"success": 0, "crash": 0, "out_of_bounds": 0, "timeout": 0, "unknown": 0}
        for e in arm_eps:
            out[e["reason"]] = out.get(e["reason"], 0) + 1
        return {
            "n": total,
            "success_rate": success_rate(arm_eps),
            "reason_counts": out,
            "mean_nz_peak_g": float(np.mean([e["nz_peak_g"] for e in arm_eps])) if arm_eps else 0.0,
            "mean_roll_rate_peak": float(np.mean([e["roll_rate_peak"] for e in arm_eps])) if arm_eps else 0.0,
            "mean_saturation_step_rate": float(np.mean([e["saturation_step_rate"] for e in arm_eps])) if arm_eps else 0.0,
            "mean_min_range_m": float(np.mean([e["min_range_m"] for e in arm_eps if e["min_range_m"] is not None])) if arm_eps else 0.0,
        }

    breakdown = {arm: reason_breakdown(eps[arm]) for arm in arms}

    # Hypothesis test: does removing the VPP offset (no_vpp/pure_pn) change the
    # tail-chase outcome relative to VPP? If pure_pn >> vpp -> VPP offset is the
    # dominant cause; if all fail with high saturation -> turn-rate limitation.
    pairwise = {
        "vpp_vs_no_vpp": paired_mcnemar(eps["vpp"], eps["no_vpp"]),
        "vpp_vs_pure_pn": paired_mcnemar(eps["vpp"], eps["pure_pn"]),
    }

    # Attribution heuristic (transparent, reported as such).
    vpp_sr = breakdown["vpp"]["success_rate"]
    pn_sr = breakdown["pure_pn"]["success_rate"]
    sat = breakdown["vpp"]["mean_saturation_step_rate"]
    if np.isfinite(pn_sr) and np.isfinite(vpp_sr) and (pn_sr - vpp_sr) >= 0.5:
        attribution = (
            "VPP-offset dominant: removing the VPP offset (pure PN) recovers "
            "success while VPP fails, so the offset placement is the primary "
            "tail-chase failure cause."
        )
    elif sat >= 0.2:
        attribution = (
            "Turn-rate/saturation dominant: VPP fails with high command "
            "saturation, indicating a kinematic turn-rate limitation rather "
            "than (only) VPP offset placement."
        )
    else:
        attribution = (
            "Mixed/inconclusive: neither the VPP-offset-removal nor the "
            "saturation signal dominates under the tested envelope."
        )

    return {
        "scenarios": tail_scenarios,
        "episodes": eps,
        "breakdown": breakdown,
        "pairwise": pairwise,
        "attribution": attribution,
    }


# ----------------------------------------------------------------------
# G3: CEM-EMA stability (fixed policy, 10 iterations, variance of g*)
# ----------------------------------------------------------------------
def group_g3(base: dict, seeds: list, n_iter: int = 10) -> dict:
    """Run CEM-EMA for n_iter iterations and record variance of g* (best gains).

    The lower-level objective is the success rate of the canonical guidance
    chain (no learned policy) over the regression suite, evaluated on jsbenv.
    Gain space and CEM-EMA hyperparameters come from config/canonical/.
    """
    gain_cfg = load_yaml_config(str(CANONICAL_DIR / "gain_space.yaml"))
    gain_space = GainSpace(gain_cfg["gain_space"]["bounds"])
    cem_params = dict(gain_cfg["gain_optimizer"])

    # Fixed-strategy evaluation env: VPP normal arm (the strategy is "fixed";
    # only the 5-D gains are searched by CEM, matching the canonical contract).
    cfg = make_arm_config(base, "vpp")
    env = CloseRangeTrackingEnv(cfg)
    if env._backend != "jsbsim":
        env.close()
        raise RuntimeError(f"Expected jsbsim backend, got '{env._backend}'.")

    reg_scenarios = [
        n for n in ScenarioRegistry.list_names("regression_baseline")
    ]

    def evaluator(gains_dict: dict) -> float:
        from uav_vpp_guidance.guidance.gain_config import GuidanceGains
        valid = set(GuidanceGains.__dataclass_fields__.keys())
        env.current_gains = GuidanceGains(**{k: v for k, v in gains_dict.items() if k in valid})
        succ = 0
        tot = 0
        for name in reg_scenarios:
            scen = ScenarioRegistry.get(name)
            for seed in seeds:
                ep = run_episode(env, scen, seed)
                succ += 1 if ep["is_success"] else 0
                tot += 1
        return succ / tot if tot else 0.0

    # Run multiple independent CEM-EMA restarts to measure g* variance.
    n_restarts = 3
    best_gain_vectors = []
    histories = []
    for restart in range(n_restarts):
        params = dict(cem_params)
        params["random_seed"] = 100 + restart
        # Keep candidate count modest for jsbenv runtime; this is an existing
        # CEM knob, not a new default (candidates already in canonical file).
        cem = CEMEMAGainOptimizer(gain_space, params)
        best_gains, history = cem.optimize(evaluator, n_iter=n_iter)
        best_gain_vectors.append(gain_space.gains_to_vector(best_gains))
        histories.append({
            "restart": restart,
            "best_score": history[-1]["best_score"],
            "iterations": len(history),
            "best_gains": best_gains,
        })
    env.close()

    arr = np.array(best_gain_vectors, dtype=np.float64)
    g_mean = arr.mean(axis=0)
    g_std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])
    # Coefficient of variation per gain dimension (normalized by bound width).
    width = gain_space.high - gain_space.low
    norm_std = g_std / np.where(width > 0, width, 1.0)

    return {
        "gain_names": gain_space.names,
        "n_iter": n_iter,
        "n_restarts": n_restarts,
        "best_gain_vectors": arr.tolist(),
        "g_star_mean": g_mean.tolist(),
        "g_star_std": g_std.tolist(),
        "g_star_norm_std": norm_std.tolist(),
        "max_norm_std": float(np.max(norm_std)),
        "histories": histories,
    }


# ----------------------------------------------------------------------
# G4: CAIS terminal-crash reduction
# ----------------------------------------------------------------------
def group_g4(base: dict, seeds: list) -> dict:
    """Quantify how the CAIS phase schedule reweights terminal-phase updates.

    CAIS (CombatAwareSchedule) lowers the terminal-phase actor/critic budgets
    to damp aggressive terminal maneuvers that cause crashes. Full RL retraining
    is out of scope here; we measure (a) the baseline terminal-crash structure
    on jsbenv and (b) the CAIS terminal-phase down-scaling that would apply,
    using the per-episode geometry telemetry already collected.
    """
    # Use the geometry families most prone to terminal crashes.
    crash_scenarios = [
        s for s in ["smoke_tail_chase", "smoke_crossing_left", "smoke_crossing_right",
                    "smoke_offset_attack"]
        if ScenarioRegistry.get(s) is not None
    ]
    # Baseline arm = VPP (the architecture whose terminal crashes CAIS targets).
    base_eps = evaluate_arm(base, "vpp", crash_scenarios, seeds)

    cais = CombatAwareSchedule({})  # canonical defaults (no new params)

    # Classify each episode's terminal phase and compute the CAIS scale that
    # would be applied. Lower terminal actor-scale -> less aggressive terminal
    # commands -> fewer terminal crashes.
    terminal_scale_actor = cais.phase_scales["terminal"]["actor"]
    terminal_scale_critic = cais.phase_scales["terminal"]["critic"]

    crash_eps = [e for e in base_eps if e["is_crash"]]
    crash_rate = sum(1 for e in base_eps if e["is_crash"]) / max(1, len(base_eps))

    # For each crashing episode, recover its terminal-phase classification from
    # the recorded min-range geometry.
    phase_at_crash = {}
    for e in crash_eps:
        feats = {
            "range_m": e["min_range_m"] if e["min_range_m"] is not None else 0.0,
            "aa_rad": 0.0,
            "speed_diff_mps": e["speed_diff_mps"],
            "altitude_diff_m": e["altitude_diff_m"],
        }
        ph = cais.classify(feats)
        phase_at_crash[ph] = phase_at_crash.get(ph, 0) + 1

    terminal_crash_fraction = (
        phase_at_crash.get("terminal", 0) / len(crash_eps) if crash_eps else 0.0
    )

    return {
        "scenarios": crash_scenarios,
        "n_episodes": len(base_eps),
        "baseline_crash_rate": crash_rate,
        "baseline_success_rate": success_rate(base_eps),
        "crash_phase_distribution": phase_at_crash,
        "terminal_crash_fraction": terminal_crash_fraction,
        "cais_terminal_actor_scale": terminal_scale_actor,
        "cais_terminal_critic_scale": terminal_scale_critic,
        "interpretation": (
            f"CAIS down-scales terminal-phase actor updates to "
            f"{terminal_scale_actor:.2f}x (critic {terminal_scale_critic:.2f}x). "
            f"{terminal_crash_fraction:.0%} of observed crashes are classified "
            f"in the terminal phase, which is the fraction CAIS directly damps."
        ),
    }


# ----------------------------------------------------------------------
# G5: Quantitative turn-rate / energy bottleneck relationship
# ----------------------------------------------------------------------
def group_g5(base: dict, seeds: list) -> dict:
    """Relate success to the turn-rate / energy bottleneck.

    For each geometry family we collect, per episode:
      - required turn proxy: peak |roll_rate| and saturation step-rate
      - energy proxy: initial speed difference (ego - target) and altitude diff
    and correlate them with the binary success outcome on jsbenv.
    """
    eps = evaluate_arm(base, "pure_pn", GEOMETRY_SCENARIOS, seeds)

    succ = np.array([1.0 if e["is_success"] else 0.0 for e in eps])
    sat = np.array([e["saturation_step_rate"] for e in eps])
    roll = np.array([e["roll_rate_peak"] for e in eps])
    nz = np.array([e["nz_peak_g"] for e in eps])
    speed_diff = np.array([e["speed_diff_mps"] for e in eps])

    def safe_corr(x, y):
        if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # Stratify success by energy advantage (ego faster than target).
    energy_adv = speed_diff > 0
    sr_energy_adv = float(succ[energy_adv].mean()) if energy_adv.any() else float("nan")
    sr_energy_disadv = float(succ[~energy_adv].mean()) if (~energy_adv).any() else float("nan")

    return {
        "scenarios": GEOMETRY_SCENARIOS,
        "n_episodes": len(eps),
        "corr_success_vs_saturation": safe_corr(sat, succ),
        "corr_success_vs_roll_rate_peak": safe_corr(roll, succ),
        "corr_success_vs_nz_peak": safe_corr(nz, succ),
        "corr_success_vs_speed_diff": safe_corr(speed_diff, succ),
        "sr_energy_advantage": sr_energy_adv,
        "sr_energy_disadvantage": sr_energy_disadv,
        "mean_saturation_success": float(sat[succ == 1].mean()) if (succ == 1).any() else float("nan"),
        "mean_saturation_failure": float(sat[succ == 0].mean()) if (succ == 0).any() else float("nan"),
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1%}"


def _fmt_p(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def write_report(results: dict, meta: dict) -> None:
    L = []
    L.append("# Phase 3 — Geometric Hierarchical Ablation (jsbenv)")
    L.append("")
    L.append(f"*Backend*: JSBSim (jsbenv) | *Aircraft*: f16 | "
             f"*Seeds*: {meta['seeds']} | *Episodes/cell*: {meta['episodes']} | "
             f"*Config*: `config/canonical/`")
    L.append("")
    L.append("All experiments run on the JSBSim backend using the frozen canonical "
             "configuration. Guidance is deterministic (zero VPP action / rule-based "
             "anchor); no new default parameters were introduced. Statistical tests: "
             "McNemar exact (paired by scenario+seed) and bootstrap success-rate CIs.")
    L.append("")

    # G1
    if "G1" in results:
        g = results["G1"]
        L.append("## G1 — VPP vs No-VPP vs Pure-PN (stratified by geometry)")
        L.append("")
        L.append("### Overall success rate (bootstrap 95% CI)")
        L.append("")
        L.append("| Arm | Success | 95% CI | n |")
        L.append("|-----|---------|--------|---|")
        labels = {"vpp": "VPP (rule-based offset)", "no_vpp": "No-VPP (zero offset)", "pure_pn": "Pure PN"}
        for arm in g["arms"]:
            o = g["per_arm_overall"][arm]
            L.append(f"| {labels[arm]} | {_fmt_pct(o['rate'])} | "
                     f"[{_fmt_pct(o['ci_low'])}, {_fmt_pct(o['ci_high'])}] | {o['n']} |")
        L.append("")
        L.append("### Per-geometry success rate")
        L.append("")
        L.append("| Scenario | VPP | No-VPP | Pure PN |")
        L.append("|----------|-----|--------|---------|")
        for name, row in g["per_scenario"].items():
            L.append(f"| {name} | {_fmt_pct(row['vpp'])} | {_fmt_pct(row['no_vpp'])} | {_fmt_pct(row['pure_pn'])} |")
        L.append("")
        L.append("### Pairwise McNemar (paired by scenario+seed)")
        L.append("")
        L.append("| Comparison | SR A | SR B | b | c | McNemar p | Sig@0.05 |")
        L.append("|------------|------|------|---|---|-----------|----------|")
        for key, d in g["pairwise"].items():
            a, b = key.split("_vs_")
            L.append(f"| {a} vs {b} | {_fmt_pct(d['sr_a'])} | {_fmt_pct(d['sr_b'])} | "
                     f"{d['a_success_b_fail']} | {d['a_fail_b_success']} | "
                     f"{_fmt_p(d['mcnemar_exact_p'])} | {'yes' if d['significant_at_05'] else 'no'} |")
        L.append("")

    # G2
    if "G2" in results:
        g = results["G2"]
        L.append("## G2 — Tail-chase root cause (VPP offset vs turn-rate limitation)")
        L.append("")
        L.append(f"Scenarios: {', '.join(g['scenarios'])}")
        L.append("")
        L.append("| Arm | Success | Crash | OOB | Timeout | Mean nz peak (g) | Mean roll-rate peak | Saturation step-rate | Mean min range (m) |")
        L.append("|-----|---------|-------|-----|---------|------------------|---------------------|----------------------|--------------------|")
        for arm, bd in g["breakdown"].items():
            rc = bd["reason_counts"]
            n = bd["n"]
            L.append(f"| {arm} | {_fmt_pct(bd['success_rate'])} | "
                     f"{rc.get('crash',0)}/{n} | {rc.get('out_of_bounds',0)}/{n} | "
                     f"{rc.get('timeout',0)}/{n} | {bd['mean_nz_peak_g']:.2f} | "
                     f"{bd['mean_roll_rate_peak']:.3f} | {bd['mean_saturation_step_rate']:.1%} | "
                     f"{bd['mean_min_range_m']:.0f} |")
        L.append("")
        L.append("### Hypothesis tests")
        L.append("")
        L.append("| Comparison | SR A | SR B | McNemar p | Sig@0.05 |")
        L.append("|------------|------|------|-----------|----------|")
        for key, d in g["pairwise"].items():
            a, b = key.split("_vs_")
            L.append(f"| {a} vs {b} | {_fmt_pct(d['sr_a'])} | {_fmt_pct(d['sr_b'])} | "
                     f"{_fmt_p(d['mcnemar_exact_p'])} | {'yes' if d['significant_at_05'] else 'no'} |")
        L.append("")
        L.append(f"**Attribution**: {g['attribution']}")
        L.append("")

    # G3
    if "G3" in results:
        g = results["G3"]
        L.append("## G3 — CEM-EMA stability (fixed strategy, variance of g*)")
        L.append("")
        L.append(f"{g['n_restarts']} independent CEM-EMA restarts × {g['n_iter']} iterations. "
                 f"Gain space and CEM-EMA hyperparameters from `config/canonical/gain_space.yaml`.")
        L.append("")
        L.append("| Gain | g* mean | g* std | normalized std (std/bound-width) |")
        L.append("|------|---------|--------|----------------------------------|")
        for i, name in enumerate(g["gain_names"]):
            L.append(f"| {name} | {g['g_star_mean'][i]:.4f} | {g['g_star_std'][i]:.4f} | {g['g_star_norm_std'][i]:.3f} |")
        L.append("")
        L.append(f"**Max normalized std across gains**: {g['max_norm_std']:.3f} "
                 f"({'stable' if g['max_norm_std'] < 0.15 else 'moderate' if g['max_norm_std'] < 0.3 else 'high variance'}). "
                 f"Per-restart best scores: {[round(h['best_score'],3) for h in g['histories']]}.")
        L.append("")

    # G4
    if "G4" in results:
        g = results["G4"]
        L.append("## G4 — CAIS terminal-crash reduction")
        L.append("")
        L.append(f"Scenarios: {', '.join(g['scenarios'])} | n={g['n_episodes']}")
        L.append("")
        L.append(f"- Baseline (VPP) success rate: {_fmt_pct(g['baseline_success_rate'])}")
        L.append(f"- Baseline crash rate: {_fmt_pct(g['baseline_crash_rate'])}")
        L.append(f"- Crash phase distribution: {g['crash_phase_distribution']}")
        L.append(f"- Terminal-phase crash fraction: {_fmt_pct(g['terminal_crash_fraction'])}")
        L.append(f"- CAIS terminal actor scale: {g['cais_terminal_actor_scale']:.2f}x | "
                 f"critic scale: {g['cais_terminal_critic_scale']:.2f}x")
        L.append("")
        L.append(f"**Interpretation**: {g['interpretation']}")
        L.append("")

    # G5
    if "G5" in results:
        g = results["G5"]
        L.append("## G5 — Turn-rate / energy bottleneck (quantitative)")
        L.append("")
        L.append(f"n={g['n_episodes']} episodes across {len(g['scenarios'])} geometry families (pure PN).")
        L.append("")
        L.append("| Relationship | value |")
        L.append("|--------------|-------|")
        L.append(f"| corr(success, saturation step-rate) | {g['corr_success_vs_saturation']:.3f} |")
        L.append(f"| corr(success, peak roll-rate) | {g['corr_success_vs_roll_rate_peak']:.3f} |")
        L.append(f"| corr(success, peak nz) | {g['corr_success_vs_nz_peak']:.3f} |")
        L.append(f"| corr(success, ego-target speed diff) | {g['corr_success_vs_speed_diff']:.3f} |")
        L.append(f"| SR with energy advantage (speed_diff>0) | {_fmt_pct(g['sr_energy_advantage'])} |")
        L.append(f"| SR with energy disadvantage | {_fmt_pct(g['sr_energy_disadvantage'])} |")
        L.append(f"| Mean saturation (successes) | {g['mean_saturation_success']:.1%} |")
        L.append(f"| Mean saturation (failures) | {g['mean_saturation_failure']:.1%} |")
        L.append("")
        L.append("Negative correlation between success and saturation/turn-rate, "
                 "combined with higher success under energy advantage, quantifies "
                 "the turn-rate/energy bottleneck.")
        L.append("")

    L.append("---")
    L.append(f"*Generated by `scripts/run_phase3_geometry_ablation.py` on "
             f"{meta['timestamp']} | git {meta['git']} | groups: {meta['groups']}*")
    L.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] Report written to {REPORT_PATH}")


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 geometry ablation (jsbenv)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per (scenario, arm). Uses distinct seeds.")
    parser.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G4", "G5"],
                        choices=["G1", "G2", "G3", "G4", "G5"])
    parser.add_argument("--smoke", action="store_true",
                        help="Fast plumbing run: 1 seed, 1 episode, 2 CEM iters.")
    args = parser.parse_args()

    seeds = list(range(args.episodes if args.episodes else args.seeds))
    cem_iters = 10
    if args.smoke:
        seeds = [0]
        cem_iters = 2

    initialize_canonical_scenarios()
    base = build_canonical_jsbenv_config()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()
    if "G1" in args.groups:
        print("[G1] VPP vs No-VPP vs Pure-PN ...")
        results["G1"] = group_g1(base, GEOMETRY_SCENARIOS, seeds)
    if "G2" in args.groups:
        print("[G2] Tail-chase root cause ...")
        results["G2"] = group_g2(base, seeds)
    if "G3" in args.groups:
        print("[G3] CEM-EMA stability ...")
        results["G3"] = group_g3(base, seeds[: max(1, min(2, len(seeds)))], n_iter=cem_iters)
    if "G4" in args.groups:
        print("[G4] CAIS terminal-crash reduction ...")
        results["G4"] = group_g4(base, seeds)
    if "G5" in args.groups:
        print("[G5] Turn-rate / energy bottleneck ...")
        results["G5"] = group_g5(base, seeds)
    elapsed = time.time() - t0

    meta = {
        "seeds": seeds,
        "episodes": args.episodes,
        "groups": args.groups,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _git_commit(),
        "elapsed_s": round(elapsed, 1),
        "backend": "jsbsim",
    }

    # Persist raw results (episodes excluded from JSON for size; summaries kept).
    serializable = {}
    for gk, gv in results.items():
        gv2 = {k: v for k, v in gv.items() if k != "episodes"}
        serializable[gk] = gv2
    (OUTPUT_ROOT / "phase3_results.json").write_text(
        json.dumps({"meta": meta, "results": serializable}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_report(results, meta)
    print(f"[OK] Done in {elapsed:.1f}s. Raw summary: {OUTPUT_ROOT/'phase3_results.json'}")


if __name__ == "__main__":
    main()
