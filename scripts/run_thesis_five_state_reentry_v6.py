#!/usr/bin/env python3
"""Run the preregistered V6 frozen-controller re-entry reachability matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_vpp_guidance.evaluation.frozen_specialist_projection import action_for_frozen_specialist
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import FiveStateRawSIV5Episode, build_record_only_env
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine

SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-REACHABILITY-CONTROL-MATRIX-V6"
CONTROLLERS = ("frozen_fixed_head_on", "frozen_fixed_crossing")
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def _load(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict): raise ValueError("expected YAML mapping")
    return value

def _hash(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _path(value):
    path = Path(value); return path if path.is_absolute() else ROOT / path
def _write(path: Path, value): engine.write_json(path, value)

def _check(config):
    if config.get("source_id") != SOURCE_ID or config.get("authorization", {}).get("execution_permitted") is not True: raise ValueError("V6 is not authorized")
    if any(config["authorization"].get(k) is not False for k in config["authorization"] if k != "execution_permitted"): raise ValueError("V6 permits execution only")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(): raise ValueError("V6 requires clean worktree")
    for key, name in (("manifest", "manifest"), ("runtime_registry", "runtime registry"), ("runtime_template", "runtime template"), ("frozen_p3_encoder", "P3 encoder")):
        path = _path(config["inputs"][key]); expected = config["inputs"][key + "_sha256"]
        if not path.is_file() or _hash(path) != expected: raise ValueError(name + " hash mismatch")
    registry = _load(_path(config["inputs"]["runtime_registry"]))
    for name in CONTROLLERS:
        entry = registry["specialists"][name]
        if _hash(Path(entry["checkpoint"])) != entry["checkpoint_sha256"]: raise ValueError(name + " checkpoint hash mismatch")
    manifest = _load(_path(config["inputs"]["manifest"]))
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12: raise ValueError("V6 manifest drift")
    output = Path(config["outputs"]["root"])
    if not output.is_absolute() or output.exists() or shutil.disk_usage(output.parent).free / 1024**3 < float(config["observability"]["min_free_disk_gb"]): raise ValueError("V6 output/disk gate failed")
    return registry, manifest

def _episode(config, registry, scenario, opponent, controller):
    env = build_record_only_env(config, registry, opponent)
    capture = {}; original = env.reset
    def reset(*args, **kwargs):
        value = original(*args, **kwargs); capture["state"] = env.jsbsim_env.get_state(); return value
    env.reset = reset
    try:
        policy = engine._specialist(registry, controller)
        encoder = engine.FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
        episode = FiveStateRawSIV5Episode(env, policy, encoder, config, controller_method=controller, action_adapter=action_for_frozen_specialist)
        try: result, ledger = episode.run(scenario, scenario["metadata"]["scenario_seed"])
        finally: episode.close()
    finally:
        try: env.close()
        except Exception: pass
    steps = ledger["steps"]
    reentry = [s for s in steps if s["raw_si_first_pass_latched"] and s["phase"] == "re_entry" and s["observation_66d"] is not None]
    if any(s["controller_projection"].get("implicit_truncation_or_padding") for s in steps): raise ValueError("V6 implicit specialist adaptation")
    return {"source_id": SOURCE_ID, "opponent": opponent, "controller": controller, "pair_key": scenario["metadata"]["pair_key"], "metadata": scenario["metadata"], "result": {**result, "reentry_66d_steps": len(reentry), "reentry_reached": bool(reentry), "scenario_reset_captured": bool(capture.get("state")), "projection_contract_passed": True}}, ledger

def _gate(records):
    matrix = {}
    for controller in CONTROLLERS:
        matrix[controller] = {}
        for opponent in OPPONENTS:
            rows = [r for r in records if r["controller"] == controller and r["opponent"] == opponent]
            qualifying = [r for r in rows if r["result"]["reentry_reached"]]
            matrix[controller][opponent] = {"records": len(rows), "reentry_episodes": len(qualifying), "reentry_66d_steps": sum(r["result"]["reentry_66d_steps"] for r in rows), "mirrors": len({r["metadata"]["mirror_sign"] for r in qualifying}), "heights": len({r["metadata"]["height_condition"] for r in qualifying})}
            cell = matrix[controller][opponent]
            cell["passes"] = cell["records"] == 12 and cell["reentry_episodes"] >= 2 and cell["reentry_66d_steps"] >= 20 and cell["mirrors"] == 2 and cell["heights"] >= 2
    return {"source_id": SOURCE_ID, "pooling_prohibited": True, "matrix": matrix, "any_controller_cross_opponent_reentry_feasible": any(all(matrix[c][o]["passes"] for o in OPPONENTS) for c in CONTROLLERS), "training_unlocked": False}

def run(config_path):
    config = _load(config_path); registry, manifest = _check(config); output = Path(config["outputs"]["root"]); output.mkdir(parents=True)
    records = []
    try:
        _write(output / "resolved_config.json", config); shutil.copy2(config_path, output / "frozen_config.yaml"); shutil.copy2(_path(config["inputs"]["manifest"]), output / "frozen_manifest.yaml")
        for controller in CONTROLLERS:
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    record, ledger = _episode(config, registry, scenario, opponent, controller); records.append(record)
                    _write(output / "episodes" / controller / opponent / (scenario["name"] + ".json"), {"record": record, "ledger": ledger})
        gate = _gate(records); _write(output / "reentry_reachability_gate.json", gate); _write(output / "run_manifest.json", {"source_id": SOURCE_ID, "record_count": len(records), "git_sha": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(), "gate": gate, "paper_safe": False}); return gate
    except Exception as error:
        _write(output / "failure.json", {"source_id": SOURCE_ID, "completed_record_count": len(records), "error": repr(error)}); raise

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true"); args=parser.parse_args()
    if not args.execute: print(json.dumps({"mode":"preflight", "source_id":SOURCE_ID},indent=2)); _check(_load(args.config)); return 0
    print(json.dumps(run(args.config), indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
