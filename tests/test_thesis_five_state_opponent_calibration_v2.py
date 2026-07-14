from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
ATLAS_TEST = ROOT / "tests" / "test_thesis_five_state_heldout40_capability_atlas.py"
SCRIPT = ROOT / "scripts" / "analyze_thesis_five_state_opponent_calibration_v2.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_heldout40_v1.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_v2_uses_one_reference_and_never_pools_opponents(tmp_path):
    fixture = _load(ATLAS_TEST, "calibration_fixture")
    module = _load(SCRIPT, "opponent_calibration_v2")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in module.OPPONENTS:
        path = tmp_path / opponent
        fixture._write_run(path, opponent, manifest["scenarios"])
        run_dirs[opponent] = path
    report = module.analyze(manifest_path=MANIFEST, run_dirs=run_dirs, output_dir=tmp_path / "out")
    assert report["mode"] == "read_only"
    assert report["reference_controller"] == "legacy_static_oracle_task_gate"
    assert report["episode_count"] == 120
    for name in report["artifacts"]:
        assert (tmp_path / "out" / name).is_file()
    cards = __import__("json").loads((tmp_path / "out" / "opponent_calibration_v2_cards.json").read_text(encoding="utf-8"))
    assert set(cards) == set(module.OPPONENTS)
    assert all(card["all_states"]["episodes"] == 40 for card in cards.values())
    assert "not an Elo" in (tmp_path / "out" / "opponent_calibration_v2_report.md").read_text(encoding="utf-8")
