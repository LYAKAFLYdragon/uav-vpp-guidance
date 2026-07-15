from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_thesis_global_advantage_p2_b5_reachability_atlas.py"


def _module():
    spec = importlib.util.spec_from_file_location("b5_atlas", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(module, opponent: str, index: int, eligible: bool) -> dict:
    steps = []
    if eligible:
        steps = [
            {
                "first_pass_complete_before_step": True,
                "structural_contract_valid": True,
                "prediction_valid": True,
                "prediction_fallback": False,
                "dynamic_state": "neutral",
                "phase": "post_merge",
                "target_pair_allowed": True,
            }
            for _ in range(10)
        ]
    return {
        "source_id": module.SOURCE_ID,
        "terminal_reason": "horizon",
        "ledger": {
            "header": {
                "source_id": module.SOURCE_ID,
                "opponent": opponent,
                "scenario": f"{opponent}_{index}",
                "scenario_signature": f"package_{index // 2}|height_{index % 3}",
                "mirror_sign": "negative" if index % 2 == 0 else "positive",
            },
            "summary": {"first_pass_observed": eligible, "continuity_valid": eligible},
            "steps": steps,
        },
    }


def test_atlas_keeps_an_opponent_level_zero_from_becoming_a_pooled_candidate(tmp_path: Path):
    module = _module()
    records = []
    for opponent in module.OPPONENTS:
        for index in range(12):
            eligible = opponent != "independent_ppo_vpp" and index < 2
            episode = _episode(module, opponent, index, eligible)
            path = tmp_path / "episodes" / opponent / f"{index}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(episode), encoding="utf-8")
            records.append(
                {
                    "opponent": opponent,
                    "path": str(path.relative_to(tmp_path)).replace("\\", "/"),
                    "sha256": _sha(path),
                }
            )
    (tmp_path / "phase_feasible_sampler_gate.json").write_text(
        json.dumps({"source_id": module.SOURCE_ID, "verdict": "phase_feasible_data_contract_not_established"}),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"source_id": module.SOURCE_ID, "records": records}), encoding="utf-8"
    )

    result = module.analyze(tmp_path)
    neutral = next(
        candidate
        for candidate in result["candidates"]
        if candidate["state"] == "neutral" and candidate["phase"] == "post_merge"
    )

    assert result["cross_opponent_candidate_exists"] is False
    assert neutral["per_opponent"]["expert"]["eligible_steps"] == 20
    assert neutral["per_opponent"]["end_to_end"]["eligible_steps"] == 20
    assert neutral["per_opponent"]["independent_ppo_vpp"]["eligible_steps"] == 0
    assert neutral["meets_b5_coverage_gate"] is False
