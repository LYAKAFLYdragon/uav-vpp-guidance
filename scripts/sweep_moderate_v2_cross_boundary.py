#!/usr/bin/env python3
"""Crossing boundary sweep for moderate v2 scene."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = _PROJECT_ROOT / "config" / "experiment"


def base_config(name: str, geo: dict) -> str:
    return f"""# Auto-generated moderate crossing boundary sweep config
includes:
  - maneuver_target_vpp_pilot_disadvantage_focused_v5.yaml

experiment:
  name: {name}

ppo:
  total_timesteps: 5000

evaluation:
  eval_interval: 5000
  num_episodes: 10

curriculum:
  stage_gate_sr: 0.50
  stages:
    - progress_end: 1.0
      scenario_names:
        - disadvantage_moderate
      weights:
        disadvantage_moderate: 1.0

scenarios:
  disadvantage_moderate:
    name: disadvantage_moderate
    description: "{geo['description']}"
    own_init:
      position_m: [0.0, 0.0, 5000.0]
      velocity_mps: {geo['own_v']:.1f}
      heading_deg: 0.0
    target_init:
      position_m: [{geo['tgt_x']:.1f}, {geo['tgt_y']:.1f}, 5000.0]
      velocity_mps: {geo['tgt_v']:.1f}
      heading_deg: {geo['tgt_h']:.1f}
"""


def geometry_metrics(geo: dict):
    dx, dy = geo["tgt_x"], geo["tgt_y"]
    rng = math.hypot(dx, dy)
    los = math.degrees(math.atan2(dy, dx))
    ata = abs((los - 0 + 180) % 360 - 180)
    aa = abs((0 - geo["tgt_h"] + 180) % 360 - 180)
    ev = [geo["own_v"], 0.0]
    tv = [
        geo["tgt_v"] * math.cos(math.radians(geo["tgt_h"])),
        geo["tgt_v"] * math.sin(math.radians(geo["tgt_h"])),
    ]
    rel = [ev[0] - tv[0], ev[1] - tv[1]]
    u = [dx / rng, dy / rng]
    closure = rel[0] * u[0] + rel[1] * u[1]
    return rng, ata, aa, closure


def train_and_parse(config_path: Path, seed: int = 0) -> float | None:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "train_curriculum_ppo.py"),
        "--config", str(config_path),
        "--backend", "jsbsim",
        "--device", "cpu",
        "--seed", str(seed),
        "--total-timesteps", "5000",
    ]
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
    out = result.stdout + result.stderr
    m = re.search(r"Success:\s+([0-9.]+)%", out)
    if not m:
        print(f"  [warn] could not parse success for {config_path.name}")
        print(out[-500:])
        return None
    return float(m.group(1))


def main():
    candidates = [
        (185.0, 225.0, 305.0, 500.0, 1050.0),  # 100%
        (184.0, 225.0, 304.0, 490.0, 1060.0),
        (183.0, 225.0, 303.0, 485.0, 1065.0),
        (182.0, 225.0, 302.0, 480.0, 1070.0),
        (181.0, 225.0, 301.0, 475.0, 1075.0),
        (180.0, 225.0, 300.0, 470.0, 1080.0),  # 0%
    ]

    results = []
    for i, (o_v, t_v, t_h, x, y) in enumerate(candidates):
        geo = {
            "own_v": o_v, "tgt_v": t_v, "tgt_h": t_h,
            "tgt_x": x, "tgt_y": y,
            "description": f"Cross boundary candidate {i}: ego {o_v:.0f}/tgt {t_v:.0f}/h{t_h:.0f}",
        }
        rng, ata, aa, closure = geometry_metrics(geo)
        cfg_name = f"_tmp_moderate_cross_bnd_{i:02d}"
        cfg_path = CONFIG_DIR / f"{cfg_name}.yaml"
        cfg_path.write_text(base_config(cfg_name, geo), encoding="utf-8")

        print(f"[{i}] ego={o_v:.0f} tgt={t_v:.0f} h={t_h:.0f} pos=[{x:.0f},{y:.0f}] "
              f"rng={rng:.0f} ATA={ata:.1f} AA={aa:.0f} closure={closure:.0f}")
        sr = train_and_parse(cfg_path)
        print(f"    -> train SR = {sr}%")
        results.append({
            "i": i, "own_v": o_v, "tgt_v": t_v, "tgt_h": t_h,
            "tgt_x": x, "tgt_y": y, "rng": rng, "ata": ata, "aa": aa,
            "closure": closure, "train_sr": sr,
        })
        cfg_path.unlink()

    out_path = _PROJECT_ROOT / "outputs" / "disadvantage_v2" / "moderate_cross_boundary_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved cross-boundary sweep results to {out_path}")


if __name__ == "__main__":
    main()
