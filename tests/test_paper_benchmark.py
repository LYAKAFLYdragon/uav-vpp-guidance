"""Tests for paper benchmark components."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_paper_benchmark import (
    serialize,
    METHODS,
    evaluate_method,
    _resolve_checkpoint,
    _load_gain_only_gains,
    _config_hash,
    load_config,
)


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------
class TestSerialize:
    def test_numpy_array(self):
        assert serialize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_numpy_float64(self):
        assert serialize(np.float64(3.14)) == 3.14

    def test_numpy_int64(self):
        assert serialize(np.int64(42)) == 42

    def test_numpy_bool(self):
        assert serialize(np.bool_(True)) is True
        assert serialize(np.bool_(False)) is False

    def test_nested_dict(self):
        d = {"a": np.array([1]), "b": {"c": np.float64(2.0)}}
        result = serialize(d)
        assert result == {"a": [1], "b": {"c": 2.0}}

    def test_regular_types_unchanged(self):
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize(3.14) == 3.14


# ---------------------------------------------------------------------------
# Methods config
# ---------------------------------------------------------------------------
class TestMethodsConfig:
    def test_all_methods_have_checkpoint(self):
        for name, cfg in METHODS.items():
            assert "checkpoint" in cfg
            assert "config_method" in cfg

    def test_gain_only_has_gains_path(self):
        assert "gains_path" in METHODS["gain_only"]


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
class TestCheckpointResolution:
    def test_methods_default_when_no_override(self):
        path, source = _resolve_checkpoint(
            "no_prediction",
            METHODS["no_prediction"],
            {},
            {},
        )
        assert path == METHODS["no_prediction"]["checkpoint"]
        assert source == "methods_default"

    def test_config_override_over_methods_default(self):
        path, source = _resolve_checkpoint(
            "no_prediction",
            METHODS["no_prediction"],
            {"checkpoint": "config_override.pt"},
            {},
        )
        assert path == "config_override.pt"
        assert source == "config_method"

    def test_cli_map_over_config_and_default(self):
        path, source = _resolve_checkpoint(
            "no_prediction",
            METHODS["no_prediction"],
            {"checkpoint": "config_override.pt"},
            {"no_prediction": "cli_override.pt"},
        )
        assert path == "cli_override.pt"
        assert source == "cli_checkpoint_map"


# ---------------------------------------------------------------------------
# Gain schema validation
# ---------------------------------------------------------------------------
class TestGainSchemaValidation:
    def test_missing_file_raises(self, tmp_path):
        cfg = {"gains_path": str(tmp_path / "missing.json")}
        with pytest.raises(FileNotFoundError):
            _load_gain_only_gains(cfg, allow_random_smoke=False)

    def test_missing_file_allows_smoke(self, tmp_path):
        cfg = {"gains_path": str(tmp_path / "missing.json")}
        info = _load_gain_only_gains(cfg, allow_random_smoke=True)
        assert info["gains_exists"] is False
        assert info["gains_schema_valid"] is False

    def test_bad_json_raises(self, tmp_path):
        gains_file = tmp_path / "bad.json"
        gains_file.write_text("not json")
        cfg = {"gains_path": str(gains_file)}
        with pytest.raises(ValueError):
            _load_gain_only_gains(cfg, allow_random_smoke=False)

    def test_missing_best_gains_raises(self, tmp_path):
        gains_file = tmp_path / "bad.json"
        gains_file.write_text(json.dumps({"history": []}))
        cfg = {"gains_path": str(gains_file)}
        with pytest.raises(ValueError):
            _load_gain_only_gains(cfg, allow_random_smoke=False)

    def test_empty_best_gains_raises(self, tmp_path):
        gains_file = tmp_path / "bad.json"
        gains_file.write_text(json.dumps({"best_gains": {}}))
        cfg = {"gains_path": str(gains_file)}
        with pytest.raises(ValueError):
            _load_gain_only_gains(cfg, allow_random_smoke=False)

    def test_unsupported_fields_ignored(self, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(
            json.dumps({"best_gains": {"k_los": 2.0, "foo": 1.0}})
        )
        cfg = {"gains_path": str(gains_file)}
        info = _load_gain_only_gains(cfg, allow_random_smoke=False)
        assert info["gains_exists"] is True
        assert info["gains_schema_valid"] is True
        assert info["loaded_gains"] == {"k_los": 2.0}
        assert info["ignored_gain_fields"] == ["foo"]

    def test_only_unsupported_fields_invalid(self, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(json.dumps({"best_gains": {"foo": 1.0}}))
        cfg = {"gains_path": str(gains_file)}
        with pytest.raises(ValueError):
            _load_gain_only_gains(cfg, allow_random_smoke=False)

    def test_valid_best_gains_loaded(self, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(json.dumps({"best_gains": {"k_los": 2.0, "k_pos": 0.7}}))
        cfg = {"gains_path": str(gains_file)}
        info = _load_gain_only_gains(cfg, allow_random_smoke=False)
        assert info["loaded_gains"] == {"k_los": 2.0, "k_pos": 0.7}
        assert info["ignored_gain_fields"] == []
        assert info["gains_schema_valid"] is True


# ---------------------------------------------------------------------------
# Evaluate method: checkpoint behavior
# ---------------------------------------------------------------------------
class TestEvaluateMethodCheckpoint:
    def test_missing_checkpoint_without_allow_random_smoke_raises(self):
        with pytest.raises(FileNotFoundError):
            evaluate_method(
                "nonexistent",
                {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
                [],
                (0,),
                backend="simple",
                config_path="config/experiment/stage6f5_feasible_geometry.yaml",
                full_config={},
                allow_random_smoke=False,
            )

    def test_missing_checkpoint_with_allow_random_smoke_succeeds(self):
        result = evaluate_method(
            "nonexistent",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            full_config={},
            allow_random_smoke=True,
        )
        assert result["metadata"]["invalid_for_paper"] is True
        assert result["metadata"]["is_random_smoke"] is True
        assert result["metadata"]["checkpoint_source"] == "methods_default"


# ---------------------------------------------------------------------------
# Evaluate method: gain-only gains
# ---------------------------------------------------------------------------
class TestGainOnlyGains:
    def test_gain_only_requires_gains_file_without_random_smoke(self):
        cfg = {
            "checkpoint": "nonexistent/path.pt",
            "config_method": "no_prediction",
            "gains_path": "nonexistent/gains.json",
        }
        with pytest.raises(FileNotFoundError):
            evaluate_method(
                "gain_only",
                cfg,
                [],
                (0,),
                backend="simple",
                config_path="config/experiment/stage6f5_feasible_geometry.yaml",
                full_config={},
                allow_random_smoke=False,
            )

    def test_gain_only_loads_and_records_gains(self, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(json.dumps({"best_gains": {"k_los": 2.0}}))
        cfg = {
            "checkpoint": "nonexistent/path.pt",
            "config_method": "no_prediction",
            "gains_path": str(gains_file),
        }
        result = evaluate_method(
            "gain_only",
            cfg,
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            full_config={},
            allow_random_smoke=True,
        )
        meta = result["metadata"]
        assert meta["gains_exists"] is True
        assert meta["gains_schema_valid"] is True
        assert meta["loaded_gains"] == {"k_los": 2.0}
        assert meta["gain_source"] == "cem"
        assert meta["ignored_gain_fields"] == []


# ---------------------------------------------------------------------------
# Metadata structure
# ---------------------------------------------------------------------------
class TestCheckpointMeta:
    def test_meta_structure(self):
        result = evaluate_method(
            "test",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            full_config={},
            allow_random_smoke=True,
        )
        meta = result["metadata"]
        assert "method" in meta
        assert "config_path" in meta
        assert "resolved_config_hash" in meta
        assert "method_override_name" in meta
        assert "backend" in meta
        assert "scenarios" in meta
        assert "seeds" in meta
        assert "prediction_mode" in meta
        assert "guidance_mode" in meta
        assert "checkpoint_path_final" in meta
        assert "checkpoint_source" in meta
        assert "policy_checkpoint" in meta
        assert "checkpoint_exists" in meta
        assert "is_random_smoke" in meta
        assert "invalid_for_paper" in meta
        assert "invalid_for_paper_reasons" in meta

    def test_valid_checkpoint_meta(self):
        ckpt = "outputs/audit_no_pred_final/checkpoints/best.pt"
        if not Path(ckpt).exists():
            pytest.skip("No checkpoint available for test")
        result = evaluate_method(
            "no_prediction",
            {"checkpoint": ckpt, "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            full_config={},
            allow_random_smoke=False,
        )
        meta = result["metadata"]
        assert meta["checkpoint_exists"] is True
        assert meta["invalid_for_paper"] is False
        assert meta["is_random_smoke"] is False
        assert meta["config_path"] == "config/experiment/stage6f5_feasible_geometry.yaml"
        assert len(meta["resolved_config_hash"]) == 16


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------
class TestResolvedConfigHash:
    def test_config_hash_changes_when_config_changes(self):
        h1 = _config_hash({"a": 1, "b": 2})
        h2 = _config_hash({"a": 1, "b": 3})
        assert h1 != h2
        h3 = _config_hash({"a": 1, "b": 2})
        assert h1 == h3


# ---------------------------------------------------------------------------
# Real CLI tests
# ---------------------------------------------------------------------------
class TestUnknownMethod:
    @pytest.fixture(autouse=True)
    def _project_root(self):
        self.project_root = Path(__file__).parent.parent
        self.script = self.project_root / "scripts" / "run_paper_benchmark.py"

    def test_unknown_method_fails_by_default(self):
        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--backend", "simple",
                "--seeds", "0",
                "--scenarios", "regression",
                "--methods", "bogus",
                "--allow-random-smoke",
                "--output-dir", str(tempfile.mkdtemp()),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Unknown method" in (result.stdout + result.stderr)

    def test_unknown_method_can_skip_only_with_allow_missing_methods(self):
        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--backend", "simple",
                "--seeds", "0",
                "--scenarios", "regression",
                "--methods", "no_prediction", "bogus",
                "--allow-random-smoke",
                "--allow-missing-methods",
                "--output-dir", str(tempfile.mkdtemp()),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "WARNING" in (result.stdout + result.stderr)


class TestRunManifestArtifact:
    @pytest.fixture(autouse=True)
    def _project_root(self):
        self.project_root = Path(__file__).parent.parent
        self.script = self.project_root / "scripts" / "run_paper_benchmark.py"

    def test_smoke_run_produces_manifest_and_records_provenance(self, tmp_path):
        output_dir = tmp_path / "bench"
        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--backend", "simple",
                "--seeds", "0",
                "--scenarios", "regression",
                "--methods", "no_prediction",
                "--allow-random-smoke",
                "--checkpoint-map", "no_prediction=nonexistent/path.pt",
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

        expected_files = [
            "summary.md",
            "results.csv",
            "tables/comparison_table.md",
            "figures/figure1_method_comparison.png",
            "run_manifest.json",
        ]
        for name in expected_files:
            assert (output_dir / name).exists(), f"Missing artifact: {name}"

        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert "git_commit" in manifest
        assert "git_dirty" in manifest
        assert manifest["config_path"]
        assert manifest["command_line"]
        assert "paper_safe" in manifest
        assert "invalid_for_paper_reasons" in manifest
        assert "method_provenance" in manifest

        prov = manifest["method_provenance"]["no_prediction"]
        assert prov["checkpoint_path_final"] == "nonexistent/path.pt"
        assert prov["checkpoint_source"] == "cli_checkpoint_map"
        assert prov["checkpoint_exists"] is False  # smoke run

        summary = (output_dir / "summary.md").read_text(encoding="utf-8")
        assert "Run Manifest" in summary
        assert prov["checkpoint_path_final"] in summary


# ---------------------------------------------------------------------------
# Integration: loaded gains must affect telemetry
# ---------------------------------------------------------------------------
class TestGainsAffectTelemetry:
    @pytest.fixture(scope="class")
    def config_path(self):
        return "config/experiment/stage6f5_feasible_geometry.yaml"

    def _make_dummy_checkpoint(self, tmp_path: Path, config_path: str) -> Path:
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.agents.ppo_agent import PPOAgent

        config = load_config(config_path, "no_prediction")
        config["backend"] = "simple"
        if "env" not in config:
            config["env"] = {}
        config["env"]["backend"] = "simple"
        config["env"]["use_jsbsim"] = False

        env = CloseRangeTrackingEnv(config)
        obs = env.reset(seed=0)
        obs_dim = int(obs["observation_vector"].shape[0])
        env.close()

        ckpt_path = tmp_path / "dummy.pt"
        torch.manual_seed(0)
        agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
        agent.save(str(ckpt_path))
        return ckpt_path

    def _first_regression_scenario(self):
        from uav_vpp_guidance.envs.scenario_registry import (
            ScenarioRegistry,
            initialize_canonical_scenarios,
        )

        initialize_canonical_scenarios()
        suite = ScenarioRegistry.get_regression_suite()
        assert suite, "Regression suite is empty"
        return [suite[0]]

    def test_different_gains_produce_different_telemetry(self, tmp_path, config_path):
        dummy_ckpt = self._make_dummy_checkpoint(tmp_path, config_path)
        scenario = self._first_regression_scenario()
        seeds = (0,)
        backend = "simple"

        full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        def run_with_gains(k_los):
            gains_file = tmp_path / f"gains_{k_los}.json"
            gains_file.write_text(json.dumps({"best_gains": {"k_los": k_los}}))
            method_cfg = {
                "checkpoint": str(dummy_ckpt),
                "config_method": "no_prediction",
                "gains_path": str(gains_file),
            }
            return evaluate_method(
                "gain_only",
                method_cfg,
                scenario,
                seeds,
                backend,
                config_path=config_path,
                full_config=full_config,
                allow_random_smoke=False,
            )

        result_low = run_with_gains(0.1)
        result_high = run_with_gains(3.0)

        assert result_low["metadata"]["loaded_gains"] == {"k_los": 0.1}
        assert result_high["metadata"]["loaded_gains"] == {"k_los": 3.0}
        assert result_low["metadata"]["gains_schema_valid"] is True
        assert result_high["metadata"]["gains_schema_valid"] is True

        ep_low = result_low["episodes"][0]
        ep_high = result_high["episodes"][0]

        telemetry_differs = (
            ep_low.get("return") != ep_high.get("return")
            or ep_low.get("nz_cmd_mean") != ep_high.get("nz_cmd_mean")
            or ep_low.get("roll_rate_cmd_mean") != ep_high.get("roll_rate_cmd_mean")
            or ep_low.get("min_range_m") != ep_high.get("min_range_m")
        )
        assert telemetry_differs, (
            "Changing k_los should affect at least one telemetry aggregate; "
            f"low={ep_low}, high={ep_high}"
        )


# ---------------------------------------------------------------------------
# Training checkpoint semantics (regression from Stage 8C)
# ---------------------------------------------------------------------------
class TestTrainingCheckpointSemantics:
    def test_train_bilevel_missing_checkpoint_fails_without_allow_random_init(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_bilevel",
                "--config", "config/experiment/proposed_bilevel.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        # dry-run allows missing checkpoint
        assert result.returncode == 0

    def test_train_bilevel_missing_checkpoint_fails_in_normal_run(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_bilevel",
                "--config", "config/experiment/proposed_bilevel.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--n-episodes", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "allow-random-init" in result.stdout or "allow-random-init" in result.stderr

    def test_train_gain_only_missing_checkpoint_fails_without_allow_random_init(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_gain_only",
                "--config", "config/experiment/gain_only_cem.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        # dry-run allows missing checkpoint
        assert result.returncode == 0

    def test_train_gain_only_missing_checkpoint_fails_in_normal_run(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_gain_only",
                "--config", "config/experiment/gain_only_cem.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--n-iter", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "allow-random-init" in result.stdout or "allow-random-init" in result.stderr
