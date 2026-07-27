"""Static F18 simple-to-strict-JSBSim transfer preflight; never runs policies."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def preflight(config: dict, root: Path) -> dict[str, Any]:
    defaults = config["run_defaults"]
    methods = defaults["main_methods"]
    checks = []
    for method in methods:
        entry = config["methods"][method]
        for role, raw in (("policy_checkpoint", entry.get("checkpoint")), ("config_path", entry.get("config_path"))):
            path = root / raw if raw else None
            checks.append({"method": method, "role": role, "path": str(path) if path else None, "exists": bool(path and path.exists())})
    return {"matrix": {"tasks": defaults["tasks"], "seeds": defaults["formal_small_seeds"], "episodes_per_seed": defaults["n_episodes"], "methods": methods, "decision_frequency_hz": config["env"]["decision_freq"], "action_contract": "method checkpoint action metadata and evaluation action space must match", "metrics_required": ["success", "safety/termination category", "return", "actuator/energy/command traces", "backend/fallback provenance"]}, "backend_pairs": [{"requested": backend, "strict_backend": backend == "jsbsim", "accept_as_jsbsim_evidence": backend == "jsbsim", "fallback_allowed": False} for backend in ("simple", "jsbsim")], "input_checks": checks, "missing_required_inputs": [item for item in checks if not item["exists"]], "acceptance_rules": {"strict_jsbsim": "final backend must be jsbsim and fallback_occurred must be false", "simple": "final backend must be simple", "pairing": "same task, seed, method, checkpoint/gain hash, control frequency, action contract, and metrics"}}
