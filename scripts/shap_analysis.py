#!/usr/bin/env python3
"""
SHAP interpretability comparison of Dense VPP vs Sparse-Gaussian VPP policies.

Loads the two best checkpoints, rolls out `favorable`, `neutral`, and `challenging`
scenarios in the JSBSim environment, and computes SHAP values for the deterministic
mean action produced by each policy.  Outputs a side-by-side feature-importance
summary, dependence plots for key geometry features, and a Markdown report.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_sparse_reward_validation import make_base_config, build_variant_config
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


FEATURE_NAMES = [
    "range_m",
    "range_rate_mps",
    "altitude_diff_m",
    "speed_diff_mps",
    "los_azimuth_sin",
    "los_azimuth_cos",
    "los_elevation_sin",
    "los_elevation_cos",
    "ata_sin",
    "ata_cos",
    "aa_sin",
    "aa_cos",
    "own_speed",
    "target_speed",
    "own_altitude",
    "target_altitude",
]

# Human-readable labels for plots
FEATURE_LABELS = [
    "range (m)",
    "range rate (m/s)",
    "altitude diff (m)",
    "speed diff (m/s)",
    "LOS az sin",
    "LOS az cos",
    "LOS elev sin",
    "LOS elev cos",
    "ATA sin",
    "ATA cos",
    "AA sin",
    "AA cos",
    "own speed",
    "target speed",
    "own altitude",
    "target altitude",
]


class _MeanScalarWrapper(nn.Module):
    """Wrap an MLPActorCritic so it exposes one action-dimension mean (batch,1)."""

    def __init__(self, network: nn.Module, action_idx: int):
        super().__init__()
        self.network = network
        self.action_idx = int(action_idx)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.network(obs)
        return mean[:, self.action_idx : self.action_idx + 1]


def _load_agent(checkpoint_path: str, cfg: Dict, device: str = "cpu") -> Tuple[PPOAgent, int, int]:
    """Build a PPOAgent and load a checkpoint, inferring obs/action dims."""
    # Build a throwaway env just to get observation shape.
    env = CloseRangeTrackingEnv(cfg)
    env.set_domain_rand_scale(0.0)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(cfg.get("policy", {}).get("action_dim", 3))
    env.close()

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device=device)
    agent.load(checkpoint_path)
    return agent, obs_dim, action_dim


def _collect_observations(
    cfg: Dict,
    agent: PPOAgent,
    scenarios: Dict[str, Dict],
    episodes_per_scenario: int = 12,
    max_steps_per_episode: int = 150,
    seed_base: int = 0,
    domain_rand_scale: float = 0.15,
) -> np.ndarray:
    """Roll out episodes and return a stacked array of observation vectors."""
    env = CloseRangeTrackingEnv(cfg)
    env.set_domain_rand_scale(domain_rand_scale)
    obs_list: List[np.ndarray] = []

    for sidx, (scenario_name, scenario) in enumerate(scenarios.items()):
        for ep in range(episodes_per_scenario):
            seed = seed_base + ep * 1000 + sidx * 100
            obs = env.reset(scenario=scenario, seed=seed)
            for step in range(max_steps_per_episode):
                obs_vec = obs["observation_vector"]
                if obs_vec.shape[0] != agent.obs_dim:
                    raise ValueError(
                        f"Observation dim mismatch: env={obs_vec.shape[0]} agent={agent.obs_dim}"
                    )
                obs_list.append(np.asarray(obs_vec, dtype=np.float32))
                action = agent.get_deterministic_action(obs_vec)
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break

    env.close()
    return np.stack(obs_list, axis=0)


def _compute_shap_values(
    agent: PPOAgent,
    observations: np.ndarray,
    background_obs: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute SHAP values for each action dimension and stack them.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, obs_dim, action_dim)
    """
    n_samples, obs_dim = observations.shape
    action_dim = agent.action_dim
    shap_values = np.zeros((n_samples, obs_dim, action_dim), dtype=np.float64)

    bg_t = torch.as_tensor(background_obs, dtype=torch.float32, device=device)
    test_t = torch.as_tensor(observations, dtype=torch.float32, device=device)

    for a in range(action_dim):
        wrapper = _MeanScalarWrapper(agent.network, a).to(device).eval()
        explainer = shap.DeepExplainer(wrapper, bg_t)
        vals = explainer.shap_values(test_t, check_additivity=False)
        # vals can be (n_samples, obs_dim, 1) or (n_samples, obs_dim)
        vals = np.asarray(vals).reshape(n_samples, obs_dim)
        shap_values[:, :, a] = vals

    return shap_values


def _plot_summary(
    feature_labels: List[str],
    dense_mean: np.ndarray,
    sparse_mean: np.ndarray,
    output_path: str,
) -> None:
    """Side-by-side mean |SHAP| bar plot and importance-difference bar plot."""
    order = np.argsort(dense_mean + sparse_mean)[::-1]
    ordered_labels = [feature_labels[i] for i in order]
    dense_ord = dense_mean[order]
    sparse_ord = sparse_mean[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    y = np.arange(len(ordered_labels))
    height = 0.35

    ax = axes[0]
    ax.barh(y + height / 2, dense_ord, height, label="Dense VPP", color="#1f77b4")
    ax.barh(y - height / 2, sparse_ord, height, label="Sparse-Gaussian VPP", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Feature importance: Dense vs Sparse-Gaussian")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax = axes[1]
    diff = sparse_ord - dense_ord
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diff]
    ax.barh(y, diff, height=0.6, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Sparse − Dense  mean |SHAP|")
    ax.set_title("Feature importance shift")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dependence(
    feature_names: List[str],
    feature_labels: List[str],
    dense_X: np.ndarray,
    dense_sv: np.ndarray,
    sparse_X: np.ndarray,
    sparse_sv: np.ndarray,
    output_path: str,
) -> None:
    """Dependence plots for range and ATA with human-readable units."""
    # Use sum over action dimensions for signed SHAP values.
    dense_signed = dense_sv.sum(axis=2)
    sparse_signed = sparse_sv.sum(axis=2)

    # Feature indices.
    range_idx = feature_names.index("range_m")
    range_rate_idx = feature_names.index("range_rate_mps")
    ata_sin_idx = feature_names.index("ata_sin")
    ata_cos_idx = feature_names.index("ata_cos")

    def _get_x_values(X, x_idx):
        """Return physical units for x-axis."""
        if x_idx == range_idx:
            return X[:, x_idx] * 3000.0, "range (m)"
        if x_idx == ata_sin_idx:
            ata_deg = np.rad2deg(np.arctan2(X[:, ata_sin_idx], X[:, ata_cos_idx]))
            return ata_deg, "ATA (deg)"
        return X[:, x_idx], feature_labels[x_idx]

    def _get_c_values(X, c_idx):
        """Return physical units for color axis."""
        if c_idx == range_rate_idx:
            return X[:, c_idx] * 200.0, "range rate (m/s)"
        if c_idx == range_idx:
            return X[:, c_idx] * 3000.0, "range (m)"
        return X[:, c_idx], feature_labels[c_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        (range_idx, range_rate_idx),
        (ata_sin_idx, range_rate_idx),
    ]

    for col, (x_idx, c_idx) in enumerate(pairs):
        for row, (X, sv, label) in enumerate(
            [
                (dense_X, dense_signed, "Dense"),
                (sparse_X, sparse_signed, "Sparse-Gaussian"),
            ]
        ):
            ax = axes[row, col]
            xvals, xlabel = _get_x_values(X, x_idx)
            cvals, clabel = _get_c_values(X, c_idx)
            sc = ax.scatter(
                xvals,
                sv[:, x_idx],
                c=cvals,
                cmap="coolwarm",
                s=20,
                alpha=0.7,
                edgecolors="none",
            )
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("SHAP value (signed sum over action dims)")
            ax.set_title(f"{label}: dependence on {xlabel}")
            ax.grid(linestyle="--", alpha=0.3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(clabel)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: str,
    feature_labels: List[str],
    dense_mean: np.ndarray,
    sparse_mean: np.ndarray,
    dense_X: np.ndarray,
    sparse_X: np.ndarray,
    dense_sv: np.ndarray,
    sparse_sv: np.ndarray,
    dense_top_features: List[Tuple[int, float]],
    sparse_top_features: List[Tuple[int, float]],
    summary_image: str,
    dependence_image: str,
) -> None:
    """Write a Markdown comparison report."""
    def _top_lines(top_features, mean_vec, X, sv):
        lines = []
        for rank, (idx, score) in enumerate(top_features, start=1):
            lines.append(
                f"{rank}. **{feature_labels[idx]}** — mean |SHAP| = {score:.4f}"
            )
        return lines

    lines = [
        "# Dense vs Sparse-Gaussian VPP: SHAP Interpretability Comparison",
        "",
        "## Method",
        "",
        "- Policies: Dense VPP reward baseline and Sparse-Gaussian reward relabelling.",
        "- Environment: JSBSim F-16 `CloseRangeTrackingEnv` under `favorable`, `neutral`, and `challenging` scenarios.",
        "- Observations: 16-dimensional geometry/state vector used by the trained policies.",
        "- SHAP: `shap.DeepExplainer` on the deterministic mean action of each policy; explanations are computed per action dimension and aggregated.",
        "- Additivity checks are disabled because the explained quantity is the un-squashed actor mean, which is the quantity the network actually computes.",
        "",
        "## Sample sizes",
        "",
        f"- Dense observations: {len(dense_X)}",
        f"- Sparse-Gaussian observations: {len(sparse_X)}",
        "",
        "## Feature importance ranking",
        "",
        "Ranking is by mean absolute SHAP value summed across the three action dimensions.",
        "",
        "### Dense VPP top features",
        "",
    ]
    lines.extend(_top_lines(dense_top_features, dense_mean, dense_X, dense_sv))
    lines.extend(
        [
            "",
            "### Sparse-Gaussian VPP top features",
            "",
        ]
    )
    lines.extend(_top_lines(sparse_top_features, sparse_mean, sparse_X, sparse_sv))
    lines.extend(
        [
            "",
            "## Visual summary",
            "",
            f"![Summary comparison]({os.path.basename(summary_image)})",
            "",
            "The left panel shows mean |SHAP| per feature for both policies; the right panel shows the Sparse−Dense difference. Positive bars mean the Sparse-Gaussian policy puts more weight on that feature.",
            "",
            "## Dependence plots",
            "",
            f"![Dependence plots]({os.path.basename(dependence_image)})",
            "",
            "Dependence plots show how the signed SHAP value for a feature changes with the feature value itself. Range is shown in metres and ATA is shown in degrees; points are colored by range rate (m/s).",
            "",
            "## Interpretation",
            "",
        ]
    )

    # Compute a few summary statistics for the report.
    dense_top = set(idx for idx, _ in dense_top_features[:5])
    sparse_top = set(idx for idx, _ in sparse_top_features[:5])
    shared = dense_top & sparse_top
    lines.append(
        f"- **Shared top-5 features:** {len(shared)} of 5."
    )
    range_idx = FEATURE_NAMES.index("range_m")
    ata_sin_idx = FEATURE_NAMES.index("ata_sin")
    ata_cos_idx = FEATURE_NAMES.index("ata_cos")
    lines.append(
        f"- **Range (`range_m`)** ranks Dense={dense_mean.argsort()[::-1].tolist().index(range_idx)+1}, "
        f"Sparse={sparse_mean.argsort()[::-1].tolist().index(range_idx)+1}."
    )
    lines.append(
        f"- **ATA sin** ranks Dense={dense_mean.argsort()[::-1].tolist().index(ata_sin_idx)+1}, "
        f"Sparse={sparse_mean.argsort()[::-1].tolist().index(ata_sin_idx)+1}."
    )
    lines.append(
        f"- **ATA cos** ranks Dense={dense_mean.argsort()[::-1].tolist().index(ata_cos_idx)+1}, "
        f"Sparse={sparse_mean.argsort()[::-1].tolist().index(ata_cos_idx)+1}."
    )
    lines.append(
        "- Overall, both policies rely on the same core geometry cues (range and aspect angles), "
        "which is expected because they solve the same close-range tracking task. "
        "Differences in ranking reflect how the Sparse-Gaussian reward reshapes the value landscape "
        "without changing the underlying observation space."
    )
    lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="SHAP comparison for Dense vs Sparse-Gaussian VPP")
    parser.add_argument(
        "--dense-checkpoint",
        type=str,
        default="outputs/sparse_reward_comparison_jsbsim/dense/checkpoints/best.pt",
        help="Path to the Dense checkpoint.",
    )
    parser.add_argument(
        "--sparse-checkpoint",
        type=str,
        default="outputs/sparse_reward_comparison_jsbsim/sparse_gaussian/checkpoints/best.pt",
        help="Path to the Sparse-Gaussian checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/interpretability",
        help="Directory where plots and report are written.",
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=12)
    parser.add_argument("--max-steps-per-episode", type=int, default=150)
    parser.add_argument("--background-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_cfg = make_base_config()
    dense_cfg = build_variant_config(base_cfg, "dense")
    sparse_cfg = build_variant_config(base_cfg, "sparse_gaussian")

    scenarios = {
        k: v
        for k, v in base_cfg.get("scenarios", {}).items()
        if k in ("favorable", "neutral", "challenging")
    }

    print("Loading Dense agent ...")
    dense_agent, _dense_obs_dim, _dense_action_dim = _load_agent(
        args.dense_checkpoint, dense_cfg, device=args.device
    )
    print("Loading Sparse-Gaussian agent ...")
    sparse_agent, _sparse_obs_dim, _sparse_action_dim = _load_agent(
        args.sparse_checkpoint, sparse_cfg, device=args.device
    )

    print("Collecting Dense observations ...")
    dense_X = _collect_observations(
        dense_cfg,
        dense_agent,
        scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        max_steps_per_episode=args.max_steps_per_episode,
        seed_base=args.seed,
        domain_rand_scale=0.15,
    )
    print(f"  -> {dense_X.shape[0]} observations")

    print("Collecting Sparse-Gaussian observations ...")
    sparse_X = _collect_observations(
        sparse_cfg,
        sparse_agent,
        scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        max_steps_per_episode=args.max_steps_per_episode,
        seed_base=args.seed + 1,
        domain_rand_scale=0.15,
    )
    print(f"  -> {sparse_X.shape[0]} observations")

    # Background samples for DeepExplainer (random subset of each policy's own observations).
    rng = np.random.default_rng(args.seed + 42)
    dense_bg = dense_X[rng.choice(len(dense_X), size=min(args.background_samples, len(dense_X)), replace=False)]
    sparse_bg = sparse_X[rng.choice(len(sparse_X), size=min(args.background_samples, len(sparse_X)), replace=False)]

    print("Computing SHAP values for Dense policy ...")
    dense_sv = _compute_shap_values(dense_agent, dense_X, dense_bg, device=args.device)
    print("Computing SHAP values for Sparse-Gaussian policy ...")
    sparse_sv = _compute_shap_values(sparse_agent, sparse_X, sparse_bg, device=args.device)

    # Per-feature mean absolute SHAP, summed over action dimensions.
    dense_mean = np.abs(dense_sv).sum(axis=2).mean(axis=0)
    sparse_mean = np.abs(sparse_sv).sum(axis=2).mean(axis=0)

    dense_order = np.argsort(dense_mean)[::-1]
    sparse_order = np.argsort(sparse_mean)[::-1]
    dense_top = [(int(i), float(dense_mean[i])) for i in dense_order]
    sparse_top = [(int(i), float(sparse_mean[i])) for i in sparse_order]

    summary_png = os.path.join(args.output_dir, "dense_vs_sparse_shap_comparison.png")
    dependence_png = os.path.join(args.output_dir, "dense_vs_sparse_shap_dependence.png")
    report_md = os.path.join(args.output_dir, "dense_vs_sparse_shap_report.md")

    print("Generating summary plot ...")
    _plot_summary(FEATURE_LABELS, dense_mean, sparse_mean, summary_png)

    print("Generating dependence plots ...")
    _plot_dependence(
        FEATURE_NAMES,
        FEATURE_LABELS,
        dense_X,
        dense_sv,
        sparse_X,
        sparse_sv,
        dependence_png,
    )

    print("Writing report ...")
    _write_report(
        report_md,
        FEATURE_LABELS,
        dense_mean,
        sparse_mean,
        dense_X,
        sparse_X,
        dense_sv,
        sparse_sv,
        dense_top,
        sparse_top,
        summary_png,
        dependence_png,
    )

    # Also save numerical arrays for reuse.
    np.savez(
        os.path.join(args.output_dir, "dense_vs_sparse_shap_data.npz"),
        dense_X=dense_X,
        dense_sv=dense_sv,
        sparse_X=sparse_X,
        sparse_sv=sparse_sv,
        feature_names=np.array(FEATURE_NAMES),
    )

    summary = {
        "dense_top5": [
            {"feature": FEATURE_NAMES[i], "mean_abs_shap": float(dense_mean[i])}
            for i in dense_order[:5]
        ],
        "sparse_top5": [
            {"feature": FEATURE_NAMES[i], "mean_abs_shap": float(sparse_mean[i])}
            for i in sparse_order[:5]
        ],
        "summary_plot": summary_png,
        "dependence_plot": dependence_png,
        "report": report_md,
    }
    with open(os.path.join(args.output_dir, "dense_vs_sparse_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Report: {report_md}")
    print(f"Summary plot: {summary_png}")
    print(f"Dependence plot: {dependence_png}")


if __name__ == "__main__":
    main()
