#!/usr/bin/env python
"""Train and register independent H0 and end-to-end policy assets.

The runner never writes to historical experiment directories. It records the
exact command, config hash, Git state, checkpoint receipt, and checkpoint
contract for every lane. Training is intentionally on the simple backend;
strict JSBSim is reserved for the later heldout evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.hash import file_sha256
from uav_vpp_guidance.evaluation.lstm_retraining_v2 import inspect_policy_checkpoint


SOURCE_ID = "THESIS-PREDICTED-GEOMETRY-H0-CURRENT-AND-E2E-ASSETS-V1-20260903"
DEFAULT_ROOT = REPO_ROOT / "outputs/prediction_highlevel_h0_asset_retraining_v1"
H0_CONFIG = REPO_ROOT / "config/experiment/train_h0_current_geometry_asset_v1.yaml"
E2E_CONFIG = REPO_ROOT / "config/experiment/train_end_to_end_opponent_asset_v1.yaml"
H0_SEEDS = (9201, 9202, 9203)
E2E_SEED = 9204


def receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def write_resolved_config(path: Path, config: dict[str, Any], seed: int) -> None:
    config = json.loads(json.dumps(config))
    config.setdefault("experiment", {})["source_id"] = SOURCE_ID
    config["experiment"]["seed"] = int(seed)
    config.setdefault("ppo", {})["seed"] = int(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_lane(
    *,
    role: str,
    seed: int,
    config_path: Path,
    output_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    resolved_config = output_dir / "resolved_config.yaml"
    write_resolved_config(resolved_config, config, seed)
    module = (
        "uav_vpp_guidance.training.train_no_prediction_vpp_ppo"
        if role == "h0_current_geometry"
        else "uav_vpp_guidance.training.train_end_to_end_ppo"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(resolved_config),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--backend",
        "simple",
    ]
    if smoke:
        command.append("--smoke")
    log_path = output_dir / "training.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.write("START_UTC: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        result = subprocess.run(command, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
    command_result = {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.resolve()),
    }
    manifest: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "role": role,
        "stage": "smoke" if smoke else "full_training",
        "seed": seed,
        "config": receipt(resolved_config),
        "config_source": receipt(config_path),
        "git_info": get_git_info(str(REPO_ROOT)),
        "training_backend": "simple",
        "formal_evaluation_backend": "jsbsim",
        "command_result": command_result,
        "training_requested_timesteps": 512 if smoke else 200000,
        "training_started": True,
    }
    if result.returncode != 0:
        manifest["status"] = "failed"
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"{role} seed {seed} failed; see {log_path}")

    checkpoint = output_dir / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        manifest["status"] = "failed_missing_checkpoint"
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"missing checkpoint after {role} seed {seed}: {checkpoint}")
    try:
        contract = inspect_policy_checkpoint(checkpoint, min_timesteps=100000 if not smoke else 1)
    except Exception as exc:
        manifest["status"] = "failed_checkpoint_contract"
        manifest["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output_dir / "run_manifest.json", manifest)
        raise
    manifest.update({"status": "completed", "checkpoint": contract, "checkpoint_receipt": receipt(checkpoint)})
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--h0-seeds", type=int, nargs="+", default=list(H0_SEEDS))
    parser.add_argument("--e2e-seed", type=int, default=E2E_SEED)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.rglob("*")):
        raise SystemExit(f"refusing to overwrite non-empty source output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    smoke = args.phase == "smoke"
    source_manifest: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "stage": "asset_training",
        "phase": args.phase,
        "git_info": get_git_info(str(REPO_ROOT)),
        "h0_seeds": args.h0_seeds,
        "e2e_seed": args.e2e_seed,
        "runs": [],
        "historical_assets_reused": False,
    }
    for seed in args.h0_seeds:
        source_manifest["runs"].append(run_lane(
            role="h0_current_geometry",
            seed=seed,
            config_path=H0_CONFIG,
            output_dir=root / "h0_current_geometry" / f"seed_{seed}",
            smoke=smoke,
        ))
    source_manifest["runs"].append(run_lane(
        role="end_to_end_opponent",
        seed=args.e2e_seed,
        config_path=E2E_CONFIG,
        output_dir=root / "end_to_end_opponent" / f"seed_{args.e2e_seed}",
        smoke=smoke,
    ))
    write_json(root / "source_run_manifest.json", source_manifest)
    if not smoke:
        assets = []
        for run in source_manifest["runs"]:
            assets.append({
                "role": run["role"],
                "seed": run["seed"],
                "source_id": SOURCE_ID,
                "contract": run["checkpoint"],
                "training_manifest": str((Path(run["checkpoint_receipt"]["path"]).parent.parent / "run_manifest.json").resolve()),
            })
        write_json(root / "asset_manifest.json", {
            "asset_manifest_id": SOURCE_ID + "-ASSETS",
            "source_id": SOURCE_ID,
            "historical_assets_reused": False,
            "assets": assets,
            "git_info": get_git_info(str(REPO_ROOT)),
        })
    print(json.dumps(source_manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
