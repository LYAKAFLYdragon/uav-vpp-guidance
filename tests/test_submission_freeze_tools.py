"""Contracts for CAN-20260705 packaging and worktree-lane tooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str):
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_manifest_uses_only_current_v5_sources():
    manifest = yaml.safe_load(
        (REPO_ROOT / "drones" / "submission_manifest.yaml").read_text(encoding="utf-8")
    )
    latex_sources = {entry["source"] for entry in manifest["latex_documents"]}
    assert "drones/dfar_ast_final_v5.tex" in latex_sources
    assert "drones/supplementary_material_v4.tex" in latex_sources
    assert "drones/supplement_index.tex" in latex_sources
    assert all("dfartv2" not in source for source in latex_sources)
    assert manifest["canonical_evidence_id"] == "CAN-20260705"


def test_worktree_classifier_has_explicit_lane_for_operational_paths():
    module = _load_script_module("classify_worktree_inventory.py")
    assert module.classify_path("drones/build_b1b9/main.pdf") == "build"
    assert module.classify_path("config/experiment/noncanonical_example.yaml") == "noncanonical"
    assert module.classify_path("src/uav_vpp_guidance/agents/commander_double_dqn_agent.py") == "historical"
    assert module.classify_path("loop-run-log.md") == "canonical"


def test_can_bundle_tool_pins_expected_commit_and_runs():
    module = _load_script_module("can20260705_artifact_bundle.py")
    assert module.CANONICAL_COMMIT == "9a9f9bf6d68560afa81560f5260b78f318dcd9b1"
    assert len(module.RUN_IDS) == 2
    assert "formal_expert_20260705" in module.RUN_IDS[0]
    assert "formal_end_to_end_20260705" in module.RUN_IDS[1]
