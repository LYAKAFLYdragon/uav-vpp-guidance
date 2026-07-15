from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "scripts" / "audit_thesis_reentry_terminal_reward_pilot_v1_completion.py"


def _load():
    spec = importlib.util.spec_from_file_location("terminal_reward_completion_audit", AUDIT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(signature: str, loss: float, *, ego_failure: bool = False) -> dict[str, object]:
    return {
        "scenario_signature": signature,
        "qualifying": True,
        "intent_loss_auc20": loss,
        "ego_failure": ego_failure,
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completion_audit_recomputes_selection_and_attributes_post_completion_stdout_failure(tmp_path: Path):
    module = _load()
    root = tmp_path / "result"
    baseline = {}
    candidate_records = {}
    for index, opponent in enumerate(module.OPPONENTS):
        signature = f"{opponent}-scenario"
        baseline[opponent] = {"frozen_fixed_head_on": [_record(signature, 1.0 + index * 0.1)]}
        candidate_records[opponent] = [_record(signature, 0.99 + index * 0.1)]
    _write(root / "dev_baseline.json", baseline)
    _write(root / "dev_candidate" / "step_10000.json", {"records": candidate_records, "summary": {
        "step": 10000,
        "selection_score_worst_opponent_mean_delta": -0.01,
        "safety_pass": True,
        "per_opponent": {opponent: {"paired_count": 1, "mean_paired_intent_loss_delta_vs_head_on": -0.01, "ego_crash_oob_delta_vs_head_on": 0.0, "safety_pass": True} for opponent in module.OPPONENTS},
    }})
    selected_records = {opponent: [_record(f"{opponent}-scenario", 0.98 + index * 0.1)] for index, opponent in enumerate(module.OPPONENTS)}
    _write(root / "dev_candidate" / "step_20000.json", {"records": selected_records, "summary": {
        "step": 20000,
        "selection_score_worst_opponent_mean_delta": -0.02,
        "safety_pass": True,
        "per_opponent": {opponent: {"paired_count": 1, "mean_paired_intent_loss_delta_vs_head_on": -0.02, "ego_crash_oob_delta_vs_head_on": 0.0, "safety_pass": True} for opponent in module.OPPONENTS},
    }})
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_20000.pt").write_bytes(b"selected")
    (checkpoint_dir / "best.pt").write_bytes(b"selected")
    _write(root / "training_summary.json", {"source_id": module.SOURCE_ID, "training_steps": 20000, "attempts": 12, "stopped": False, "stop_reason": None, "heldout_executed": False})
    _write(root / "run_manifest.json", {"source_id": module.SOURCE_ID, "git_sha": "synthetic"})
    _write(root / "failure.json", {"source_id": module.SOURCE_ID, "error": "OSError(22, 'Invalid argument')"})
    for offset, name in enumerate(("training_summary.json", "run_manifest.json", "failure.json")):
        os.utime(root / name, ns=(1_000_000_000 + offset, 1_000_000_000 + offset))

    report = module.audit_completion(root)

    assert report["selected_checkpoint"]["step"] == 20000
    assert report["selected_checkpoint"]["best_matches_selected_sha256"] is True
    assert report["dev_recomputation"][1]["stored_summary_matches_recomputed"] is True
    assert report["decision"]["dev_safety_pass"] is True
    assert report["decision"]["practical_improvement_opponents"] == []
    assert report["classification"] == "infrastructure_failure_attribution_inconclusive"
