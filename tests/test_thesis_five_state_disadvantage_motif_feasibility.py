from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
ATLAS_TEST_PATH = ROOT / "tests" / "test_thesis_five_state_heldout40_capability_atlas.py"
MOTIF_PATH = ROOT / "scripts" / "analyze_thesis_five_state_disadvantage_motif_feasibility.py"
MANIFEST_PATH = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_heldout40_v1.yaml"


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_disadvantage_motif_audit_is_read_only_and_uses_atlas_hashes(tmp_path):
    atlas_test = _load(ATLAS_TEST_PATH, "heldout40_atlas_fixture")
    atlas = atlas_test._atlas()
    motif = _load(MOTIF_PATH, "heldout40_disadvantage_motif")
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    input_hashes = {}
    for opponent in atlas.OPPONENTS:
        run = tmp_path / opponent
        atlas_test._write_run(run, opponent, manifest["scenarios"])
        run_dirs[opponent] = run
        input_hashes[opponent] = _hash(run / "aggregate" / "episode_records.json")
    analysis_r2 = tmp_path / "analysis_r2"
    atlas_test._write_analysis_r2(analysis_r2)
    atlas_dir = tmp_path / "atlas"
    atlas.analyze(
        run_dirs=run_dirs,
        manifest_path=MANIFEST_PATH,
        analysis_dir=analysis_r2,
        output_dir=atlas_dir,
    )

    output_dir = tmp_path / "motif"
    report = motif.analyze(
        manifest_path=MANIFEST_PATH,
        atlas_dir=atlas_dir,
        run_dirs=run_dirs,
        output_dir=output_dir,
    )

    assert report["mode"] == "read_only"
    assert report["selected_scenario_count"] == 6
    assert report["selected_episode_count"] == 24
    assert report["audited_episode_count"] == 24
    assert report["phase_sentinel_episode_count"] == 0
    assert report["phase_row_count"] == 72
    assert report["full_disadvantage_all_method_loss_pairs"] == 24
    assert report["representative_all_method_loss_pairs"] == 6
    assert report["full_phase_coverage"]["method_episode_rows"] == 96
    for name in report["artifacts"]:
        assert (output_dir / name).is_file(), name
    assert "All four methods also share the downstream guidance/PID execution chain" in (
        output_dir / "heldout40_atlas_decision_memo_zh.md"
    ).read_text(encoding="utf-8")
    for opponent, run in run_dirs.items():
        assert _hash(run / "aggregate" / "episode_records.json") == input_hashes[opponent]
        assert report["input_artifacts"]["formal_episode_records"][opponent]["sha256"] == input_hashes[opponent]


def test_disadvantage_motif_rejects_drifted_atlas_hash(tmp_path):
    atlas_test = _load(ATLAS_TEST_PATH, "heldout40_atlas_fixture_drift")
    atlas = atlas_test._atlas()
    motif = _load(MOTIF_PATH, "heldout40_disadvantage_motif_drift")
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in atlas.OPPONENTS:
        run = tmp_path / opponent
        atlas_test._write_run(run, opponent, manifest["scenarios"])
        run_dirs[opponent] = run
    analysis_r2 = tmp_path / "analysis_r2"
    atlas_test._write_analysis_r2(analysis_r2)
    atlas_dir = tmp_path / "atlas"
    atlas.analyze(
        run_dirs=run_dirs,
        manifest_path=MANIFEST_PATH,
        analysis_dir=analysis_r2,
        output_dir=atlas_dir,
    )
    record_path = run_dirs["expert"] / "aggregate" / "episode_records.json"
    record_path.write_text(record_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="hash no longer matches"):
        motif.analyze(
            manifest_path=MANIFEST_PATH,
            atlas_dir=atlas_dir,
            run_dirs=run_dirs,
            output_dir=tmp_path / "motif",
        )


def test_nonrepresentative_post_merge_case_is_appended_as_coverage_sentinel(tmp_path):
    atlas_test = _load(ATLAS_TEST_PATH, "heldout40_atlas_fixture_sentinel")
    atlas = atlas_test._atlas()
    motif = _load(MOTIF_PATH, "heldout40_disadvantage_motif_sentinel")
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in atlas.OPPONENTS:
        run = tmp_path / opponent
        atlas_test._write_run(run, opponent, manifest["scenarios"])
        run_dirs[opponent] = run

    disadvantage_names = sorted(
        scenario["name"]
        for scenario in manifest["scenarios"]
        if scenario["metadata"]["initial_class"] == "disadvantage"
    )
    sentinel_name = disadvantage_names[len(disadvantage_names) // 2]
    aggregate_path = run_dirs["end_to_end"] / "aggregate" / "episode_records.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    for record in aggregate["episodes"]:
        if record["scenario"] == sentinel_name:
            record["trajectory"][1]["range_rate_mps"] = 80.0
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    analysis_r2 = tmp_path / "analysis_r2"
    atlas_test._write_analysis_r2(analysis_r2)
    atlas_dir = tmp_path / "atlas"
    atlas.analyze(
        run_dirs=run_dirs,
        manifest_path=MANIFEST_PATH,
        analysis_dir=analysis_r2,
        output_dir=atlas_dir,
    )
    report = motif.analyze(
        manifest_path=MANIFEST_PATH,
        atlas_dir=atlas_dir,
        run_dirs=run_dirs,
        output_dir=tmp_path / "motif",
    )

    assert report["selected_scenario_count"] == 6
    assert report["phase_sentinel_scenario_count"] == 1
    assert report["phase_sentinel_episode_count"] == 4
    assert report["audited_episode_count"] == 28
