"""Targeted tests for OracleTaskGatePolicy and comparison runner oracle path."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
ORACLE_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_oracle_task_gate_mvp.yaml"


def _run_dry_run(tmp_path: Path, *extra_args: str) -> Path:
    run_id = "test_oracle_gate"
    output_root = tmp_path / "outputs"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--backend",
        "simple",
        "--dry-run",
        "--allow-missing-checkpoints",
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return output_root / run_id


class TestOracleTaskGatePolicyUnit:
    """Unit tests for OracleTaskGatePolicy in isolation (mocked)."""

    @pytest.fixture
    def mock_specialist(self):
        """Return a mock specialist that can be injected into OracleTaskGatePolicy."""
        specialist = MagicMock()
        specialist.obs_dim = 16
        specialist.get_deterministic_action = MagicMock(
            return_value=np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        return specialist

    @pytest.fixture
    def gate_with_mock(self, mock_specialist):
        """Create an OracleTaskGatePolicy with a mocked specialist."""
        from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy

        with patch(
            "uav_vpp_guidance.evaluation.oracle_task_gate_policy.load_experiment_config",
            return_value={},
        ), patch(
            "torch.load",
            return_value={"obs_dim": 16, "action_dim": 3},
        ), patch(
            "uav_vpp_guidance.evaluation.oracle_task_gate_policy.PPOAgent",
            return_value=mock_specialist,
        ):
            agent = OracleTaskGatePolicy(
                specialists_config={
                    "head_on": {
                        "checkpoint": "dummy.pt",
                        "config_path": "dummy.yaml",
                    }
                },
                device="cpu",
            )
            yield agent

    def test_task_routing_sets_current_task_name(self, gate_with_mock):
        """Verify that set_task_name correctly sets the current task."""
        assert gate_with_mock.current_task_name is None
        gate_with_mock.set_task_name("head_on")
        assert gate_with_mock.current_task_name == "head_on"

    def test_get_deterministic_action_without_set_task_name_raises(self, gate_with_mock):
        """Verify that calling get_deterministic_action before set_task_name raises."""
        with pytest.raises(RuntimeError, match="current_task_name is not set"):
            gate_with_mock.get_deterministic_action(np.zeros(16, dtype=np.float32))

    def test_observation_truncation_when_current_dim_larger_than_target(self, gate_with_mock):
        """Verify that obs is truncated when current_dim > target_dim."""
        gate_with_mock.set_task_name("head_on")
        obs = np.arange(19, dtype=np.float32)
        result = gate_with_mock.get_deterministic_action(obs)
        # The mock specialist returns [0.1, 0.2, 0.3] regardless, but we can verify
        # that the mock was called with the truncated array.
        specialist = gate_with_mock._specialists["head_on"]
        called_obs = specialist.get_deterministic_action.call_args[0][0]
        assert called_obs.shape == (16,)
        np.testing.assert_array_equal(called_obs, obs[:16])

    def test_observation_padding_when_current_dim_smaller_than_target(self, gate_with_mock):
        """Verify that obs is padded with zeros when current_dim < target_dim."""
        gate_with_mock.set_task_name("head_on")
        obs = np.arange(10, dtype=np.float32)
        result = gate_with_mock.get_deterministic_action(obs)
        specialist = gate_with_mock._specialists["head_on"]
        called_obs = specialist.get_deterministic_action.call_args[0][0]
        assert called_obs.shape == (16,)
        expected = np.concatenate([obs, np.zeros(6, dtype=np.float32)])
        np.testing.assert_array_equal(called_obs, expected)

    def test_fallback_to_first_specialist_when_task_not_found(self, gate_with_mock):
        """Verify fallback to first available specialist when task is unknown."""
        gate_with_mock.set_task_name("unknown_task")
        obs = np.arange(16, dtype=np.float32)
        result = gate_with_mock.get_deterministic_action(obs)
        specialist = gate_with_mock._specialists["head_on"]
        assert specialist.get_deterministic_action.called

    def test_no_specialists_loaded_raises(self):
        """Verify that an empty specialist config raises on action."""
        from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy

        agent = OracleTaskGatePolicy(specialists_config={}, device="cpu")
        agent.set_task_name("head_on")
        with pytest.raises(RuntimeError, match="No specialists loaded"):
            agent.get_deterministic_action(np.zeros(16, dtype=np.float32))

    def test_load_is_noop(self, gate_with_mock):
        """Verify that load() is a no-op."""
        # Should not raise
        gate_with_mock.load("some_path")

    def test_specialist_config_uses_include_resolved_loader(self, mock_specialist):
        """Oracle gate should resolve includes like FrozenSpecialistPolicy does."""
        from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy

        resolved_config = {"virtual_point": {"action_semantics": "tactical_basis_v1"}}
        with patch(
            "uav_vpp_guidance.evaluation.oracle_task_gate_policy.load_experiment_config",
            return_value=resolved_config,
        ) as load_cfg, patch(
            "torch.load",
            return_value={"obs_dim": 16, "action_dim": 3},
        ), patch(
            "uav_vpp_guidance.evaluation.oracle_task_gate_policy.PPOAgent",
            return_value=mock_specialist,
        ):
            OracleTaskGatePolicy(
                specialists_config={
                    "head_on": {
                        "checkpoint": "dummy.pt",
                        "config_path": "dummy.yaml",
                    }
                },
                device="cpu",
            )

        load_cfg.assert_called_once_with("dummy.yaml")

    def test_specialist_config_matches_resolved_real_head_on_training_config(self, mock_specialist):
        """Real included specialist YAML should expose the full merged config surface."""
        from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy
        from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config

        head_on_cfg_path = (
            REPO_ROOT
            / "config"
            / "experiment"
            / (
                "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_"
                "tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted.yaml"
            )
        )
        expected = load_experiment_config(str(head_on_cfg_path))
        captured_config = {}

        def _capture_agent(*, obs_dim, action_dim, config, device):
            captured_config["config"] = config
            return mock_specialist

        with patch(
            "torch.load",
            return_value={"obs_dim": 16, "action_dim": 3},
        ), patch(
            "uav_vpp_guidance.evaluation.oracle_task_gate_policy.PPOAgent",
            side_effect=_capture_agent,
        ):
            OracleTaskGatePolicy(
                specialists_config={
                    "head_on": {
                        "checkpoint": "dummy.pt",
                        "config_path": str(head_on_cfg_path),
                    }
                },
                device="cpu",
            )

        actual = captured_config["config"]
        assert actual == expected
        assert actual["virtual_point"]["action_semantics"] == "tactical_basis_v1"
        assert actual["trajectory_prediction"]["enabled"] is True
        assert actual["observation"]["include_task_type"] is True
        assert actual["attack_zone"]["enabled"] is True


class TestOracleTaskGateRunnerPath:
    """Integration tests for the oracle_task_gate path through the comparison runner."""

    def test_dry_run_oracle_task_gate_generates_manifest_and_contract(self, tmp_path):
        """Verify that oracle_task_gate can run through the full dry-run pipeline."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
        )

        manifest_path = run_dir / "run_manifest.json"
        contract_path = run_dir / "artifact_contract.json"
        assert manifest_path.exists()
        assert contract_path.exists()
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "dry_run_checks.json").exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["extra"]["methods"] == ["oracle_task_gate"]
        assert manifest["extra"]["tasks"] == ["head_on"]

    def test_dry_run_oracle_task_gate_config_snapshot_has_virtual_point(self, tmp_path):
        """Verify that the oracle config snapshot includes the full virtual_point block."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
        )

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        snapshot_path = Path(
            manifest["extra"]["config_snapshots"]["oracle_task_gate/head_on"]["path"]
        )
        cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

        # Verify virtual_point is present and not default
        assert cfg["virtual_point"]["action_semantics"] == "tactical_basis_v1"
        assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
        assert cfg["virtual_point"]["close_range_anchor_mode_by_task"]["head_on"] == "current_target"
        assert cfg["virtual_point"]["offset_frame_by_task"]["head_on"] == "target_velocity"

        # Verify trajectory_prediction is present
        assert cfg["trajectory_prediction"]["enabled"] is True
        assert cfg["trajectory_prediction"]["predictor_type"] == "lstm"
        assert cfg["trajectory_prediction"]["checkpoint_path"] == "outputs/trajectory_prediction/best_model.pt"

    def test_dry_run_oracle_task_gate_attack_zone_override_recorded(self, tmp_path):
        """Verify that attack-zone AoA override is recorded in config overrides."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
            "--attack-zone-close-range-max-aoa-deg",
            "60",
        )

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["extra"]["attack_zone_overrides"] == {"close_range_max_aoa_deg": 60}

        snapshot_path = Path(
            manifest["extra"]["config_snapshots"]["oracle_task_gate/head_on"]["path"]
        )
        cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60
        assert any(
            item["key"] == "attack_zone.close_range_max_aoa_deg"
            and item["new_value"] == 60
            for item in cfg["provenance"]["config_overrides"]
        )

    def test_dry_run_oracle_task_gate_no_checkpoint_check_error(self, tmp_path):
        """Verify that oracle_task_gate skips checkpoint validation (no checkpoint needed)."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
        )

        checks_path = run_dir / "dry_run_checks.json"
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        checkpoint_checks = checks.get("checkpoint_checks", [])
        # oracle_task_gate should not appear in checkpoint checks
        oracle_checks = [c for c in checkpoint_checks if c.get("method") == "oracle_task_gate"]
        assert len(oracle_checks) == 0

    def test_dry_run_oracle_task_gate_specialists_config_in_manifest(self, tmp_path):
        """Verify that the oracle specialists config is preserved in the resolved config."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
        )

        resolved_config = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        oracle_def = resolved_config["methods"]["oracle_task_gate"]
        assert oracle_def["agent_type"] == "oracle_task_gate"
        assert "specialists" in oracle_def
        assert "head_on" in oracle_def["specialists"]
        assert "crossing_feasible" in oracle_def["specialists"]
        assert oracle_def["specialists"]["head_on"]["checkpoint"].endswith("best.pt")

    def test_dry_run_oracle_task_gate_multiple_tasks(self, tmp_path):
        """Verify that oracle_task_gate can run with multiple tasks in dry-run."""
        run_dir = _run_dry_run(
            tmp_path,
            "--config",
            str(ORACLE_CONFIG),
            "--methods",
            "oracle_task_gate",
            "--tasks",
            "head_on",
            "crossing_feasible",
        )

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["extra"]["tasks"] == ["head_on", "crossing_feasible"]
        # Should have config snapshots for both tasks
        snapshots = manifest["extra"]["config_snapshots"]
        assert "oracle_task_gate/head_on" in snapshots
        assert "oracle_task_gate/crossing_feasible" in snapshots
