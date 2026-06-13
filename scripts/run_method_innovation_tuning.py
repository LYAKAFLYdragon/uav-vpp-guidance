#!/usr/bin/env python3
"""Quick tuning experiment for CR-PPO and Intentional PPO hyperparameters."""
import subprocess, sys
from pathlib import Path

# Run 3 seeds x 20k steps with tuned hyperparameters
ALGOS = {
    "baseline_tuned": ("ppo", {}),
    "cr_ppo_high": ("cr_ppo", {"ppo": {"complexity_coef": 1.0e-2, "cr_n_bins": 8}}),
    "intentional_high": ("intentional_ppo", {
        "ppo": {
            "use_intentional_critic": True,
            "use_intentional_actor": True,
            "use_combat_aware_eta": True,
            "eta_critic": 1.0,
            "eta_actor": 0.1,
            "iu_eps": 1.0e-8,
            "beta_adv": 0.999,
        },
        "combat_aware": {
            "eta_actor": 0.1,
            "eta_critic": 1.0,
            "range_thresholds_m": [3000.0, 6000.0],
            "terminal_range_m": 1200.0,
            "aspect_threshold_deg": 30.0,
        },
    }),
}

for algo, (algorithm, overrides) in ALGOS.items():
    for seed in range(3):
        out = Path(f"outputs/method_innovation_tuning/{algo}/seed{seed}")
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.yaml"
        # Reuse runner config builder by importing
        from run_method_innovation_comparison import build_config
        cfg = build_config("config/method_innovation_comparison.yaml", overrides, total_timesteps=20000, seed=seed)
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        cmd = [sys.executable, "scripts/train_curriculum_ppo.py", "--config", str(cfg_path),
               "--seed", str(seed), "--output-dir", str(out), "--device", "cpu", "--backend", "simple", "--algorithm", algorithm]
        print(f"[RUN] {algo} seed={seed}")
        rc = subprocess.run(cmd, cwd=Path(__file__).parent.parent).returncode
        print(f"[DONE] {algo} seed={seed} exit={rc}")
