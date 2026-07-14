"""Run a no-result strict-JSBSim compatibility preflight for Heldout40.

The script uses two disposable compatibility fixtures, not any Heldout40
scenario.  It performs exactly one simulation step per opponent/task to prove
that frozen policies, the 16-D feature-name adapters, and strict JSBSim can
coexist before the preregistered evaluation is launched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_jsbsim_hrl_comparison as comparison  # noqa: E402


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
FIXED_METHODS = ("head_on_vpp_specialist_no_routing", "crossing_vpp_specialist_no_routing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(task: str) -> dict[str, Any]:
    if task == "head_on":
        target_position = [2400.0, 120.0, 5000.0]
        target_heading = 180.0
    elif task == "crossing_feasible":
        target_position = [2200.0, 1400.0, 5000.0]
        target_heading = 250.0
    else:
        raise ValueError(f"Unsupported preflight task: {task}")
    return {
        "name": f"heldout40_compatibility_fixture_{task}",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 235.0, "heading_deg": 0.0},
        "target_init": {"position_m": target_position, "velocity_mps": 240.0, "heading_deg": target_heading},
        "metadata": {"fixture_only": True, "task_registry_key": task, "scenario_seed": 99117},
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = comparison._load_config_with_includes(path)
    config = comparison._materialize_scenario_manifest(config, path)
    if config.get("heldout40_protocol", {}).get("source_id") != SOURCE_ID:
        raise ValueError("Not a Heldout40 comparison config")
    return config


def _asset_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for asset in config.get("frozen_assets", {}).get("artifacts", []):
        path = Path(asset["path"])
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "id": asset["id"],
                "path": str(path),
                "exists": path.is_file(),
                "expected_sha256": asset.get("sha256"),
                "actual_sha256": actual,
                "passed": path.is_file() and actual == asset.get("sha256"),
            }
        )
    return checks


def run_preflight(*, config_path: Path, jsbsim_root: str, device: str) -> dict[str, Any]:
    config = _load_config(config_path)
    asset_checks = _asset_checks(config)
    failures = [item["id"] for item in asset_checks if not item["passed"]]
    checks: list[dict[str, Any]] = []
    if failures:
        return {"source_id": SOURCE_ID, "passed": False, "asset_checks": asset_checks, "failures": failures, "checks": checks}

    for opponent in OPPONENTS:
        for task in ("head_on", "crossing_feasible"):
            cfg = comparison.build_eval_config(
                config,
                "canonical_ppo_high_level_policy",
                task,
                backend="jsbsim",
                opponent_stage=opponent,
                jsbsim_root=jsbsim_root,
            )
            env = comparison._build_env(cfg)
            try:
                obs = env.reset(seed=99117, scenario=_fixture(task))
                checkpoint = Path(config["methods"]["canonical_ppo_high_level_policy"]["checkpoint"])
                agent = comparison._build_agent(
                    config["methods"]["canonical_ppo_high_level_policy"], cfg, checkpoint, obs, device
                )
                if hasattr(agent, "set_env"):
                    agent.set_env(env)
                if hasattr(agent, "set_task_name"):
                    agent.set_task_name(task)
                action = agent.get_deterministic_action(obs["observation_vector"])
                _, _, _, _, info = env.step(action)
                diagnostics = env.opponent_policy.get_diagnostics() if env.opponent_policy else {}
                backend = str(info.get("backend") or obs.get("provenance", {}).get("backend"))
                adapter_required = opponent != "expert"
                adapter_passed = (not adapter_required) or (
                    diagnostics.get("opponent_adapter") == "feature_name_base16_projection"
                    and int(diagnostics.get("opponent_adapter_projection_count", 0)) >= 1
                )
                passed = backend == "jsbsim" and not bool(info.get("backend_fallback_occurred", False)) and adapter_passed
                checks.append(
                    {
                        "opponent": opponent,
                        "task": task,
                        "backend": backend,
                        "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
                        "ego_observation_dim": int(obs["observation_vector"].shape[0]),
                        "ego_feature_names": list(obs.get("observation_schema", {}).get("feature_names", [])),
                        "opponent_adapter": diagnostics.get("opponent_adapter"),
                        "opponent_adapter_projection_count": diagnostics.get("opponent_adapter_projection_count"),
                        "passed": passed,
                    }
                )
                if not passed:
                    failures.append(f"{opponent}/{task}")
            except Exception as exc:
                checks.append({"opponent": opponent, "task": task, "passed": False, "error": str(exc)})
                failures.append(f"{opponent}/{task}: {exc}")
            finally:
                env.close()

    # Shape-check both frozen low-level specialists on the task they receive.
    for method, task in zip(FIXED_METHODS, ("head_on", "crossing_feasible")):
        cfg = comparison.build_eval_config(
            config, method, task, backend="jsbsim", opponent_stage="expert", jsbsim_root=jsbsim_root
        )
        env = comparison._build_env(cfg)
        try:
            obs = env.reset(seed=99118, scenario=_fixture(task))
            checkpoint = Path(config["methods"][method]["checkpoint"])
            audit = comparison._build_observation_audit(
                method, task, config["methods"][method], cfg, checkpoint, obs
            )
            passed = bool(audit["obs_dim_matches_checkpoint"] and audit["action_dim_matches_checkpoint"])
            checks.append({"method": method, "task": task, "specialist_shape_audit": audit, "passed": passed})
            if not passed:
                failures.append(f"{method}/{task}: shape")
        except Exception as exc:
            checks.append({"method": method, "task": task, "passed": False, "error": str(exc)})
            failures.append(f"{method}/{task}: {exc}")
        finally:
            env.close()

    return {
        "source_id": SOURCE_ID,
        "passed": not failures,
        "asset_checks": asset_checks,
        "checks": checks,
        "failures": failures,
        "training_or_tuning_performed": False,
        "heldout_scenario_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jsbsim-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_preflight(
        config_path=args.config,
        jsbsim_root=str(args.jsbsim_root),
        device=str(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failures": result["failures"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
