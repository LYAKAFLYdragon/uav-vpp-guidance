"""Tests for the portable artifact pipeline core."""

import json
from pathlib import Path

import pytest

from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.hash import config_sha256, file_info, file_sha256
from uav_vpp_guidance.common.manifest import RunManifest
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult


class TestGitInfo:
    def test_returns_expected_keys(self):
        info = get_git_info()
        assert set(info.keys()) == {
            "commit",
            "short_commit",
            "branch",
            "dirty",
            "description",
        }
        if info["commit"] is not None:
            assert len(info["commit"]) == 40
            assert info["short_commit"] == info["commit"][:12]


class TestHashUtilities:
    def test_config_sha256_stable_across_order(self, tmp_path):
        a = {"x": 1, "y": {"z": 2}}
        b = {"y": {"z": 2}, "x": 1}
        assert config_sha256(a) == config_sha256(b)

    def test_file_sha256_matches_file_info(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        assert file_sha256(path) == file_info(path)["sha256"]


class TestArtifactContract:
    def test_validate_detects_missing_files(self, tmp_path):
        contract = ArtifactContract(required_files=["a.txt"], required_directories=["dir"])
        result = contract.validate(tmp_path)
        assert result["valid"] is False
        assert "a.txt" in result["missing"]
        assert "dir" in result["missing"]

    def test_validate_passes_when_present(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "dir").mkdir()
        contract = ArtifactContract(required_files=["a.txt"], required_directories=["dir"])
        result = contract.validate(tmp_path)
        assert result["valid"] is True


class TestRunManifest:
    def test_save_and_load_roundtrip(self, tmp_path):
        manifest = RunManifest(stage_name="test", output_dir=str(tmp_path))
        manifest.mark_completed()
        manifest.save(tmp_path)
        loaded = RunManifest.load(tmp_path / "run_manifest.json")
        assert loaded.stage_name == "test"
        assert loaded.status == "completed"

    def test_config_hash_computed(self):
        manifest = RunManifest(stage_name="test", resolved_config={"a": 1})
        manifest.compute_config_hash()
        assert manifest.config_hash is not None
        assert len(manifest.config_hash) == 16

    def test_invalid_for_paper_toggles_flag(self):
        manifest = RunManifest(stage_name="test")
        manifest.mark_completed()
        assert manifest.paper_safe is True
        manifest.add_invalid_for_paper_reason("test reason")
        assert manifest.paper_safe is False


class DummyStage(PipelineStage):
    stage_name = "dummy"

    def __init__(self, config, output_dir, fail=False):
        super().__init__(config, output_dir)
        self.fail = fail
        self.artifact_contract = ArtifactContract(required_files=["result.txt"])

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.snapshot_config()
        if self.fail:
            return self.finalize(success=False)
        (self.output_dir / "result.txt").write_text("ok")
        self.manifest.record_output_file("result.txt", self.output_dir / "result.txt")
        return self.finalize(success=True)


class TestPipelineStage:
    def test_successful_stage_writes_manifest(self, tmp_path):
        stage = DummyStage({}, str(tmp_path / "out"))
        result = stage.run()
        assert result.success is True
        assert (tmp_path / "out" / "run_manifest.json").exists()
        assert (tmp_path / "out" / "artifact_contract.json").exists()

    def test_failed_stage_reports_missing_artifacts(self, tmp_path):
        stage = DummyStage({}, str(tmp_path / "out"), fail=True)
        result = stage.run()
        assert result.success is False
        assert "result.txt" in result.errors


class TestArtifactBundle:
    def test_create_and_verify_roundtrip(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "result.txt").write_text("ok")

        manifest = RunManifest(stage_name="test", output_dir=str(source))
        manifest.record_output_file("result.txt", source / "result.txt")
        manifest.mark_completed()
        manifest.save(source)
        ArtifactContract(required_files=["result.txt"]).save(source)

        bundle_dir = tmp_path / "bundle"
        bundle = ArtifactBundle(source_dir=source, bundle_dir=bundle_dir)
        info = bundle.create()
        assert (bundle_dir / "bundle_info.json").exists()
        assert (bundle_dir / "result.txt").exists()

        verification = bundle.verify()
        assert verification["valid"] is True

    def test_verify_detects_tampering(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "result.txt").write_text("ok")

        manifest = RunManifest(stage_name="test", output_dir=str(source))
        manifest.record_output_file("result.txt", source / "result.txt")
        manifest.mark_completed()
        manifest.save(source)
        ArtifactContract(required_files=["result.txt"]).save(source)

        bundle_dir = tmp_path / "bundle"
        bundle = ArtifactBundle(source_dir=source, bundle_dir=bundle_dir)
        bundle.create()
        (bundle_dir / "result.txt").write_text("tampered")

        verification = bundle.verify()
        assert verification["valid"] is False
        assert len(verification["mismatches"]) == 1
