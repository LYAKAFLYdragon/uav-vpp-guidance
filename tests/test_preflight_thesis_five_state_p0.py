from __future__ import annotations

import hashlib
import importlib.util
from collections import namedtuple
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "preflight_thesis_five_state_p0.py"
_SPEC = importlib.util.spec_from_file_location("thesis_p0_preflight", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODULE)


def _write_contracts(tmp_path: Path, *, eligible: bool, sha256: str | None = None):
    asset_path = tmp_path / "asset.bin"
    asset_path.write_bytes(b"frozen-asset")
    actual_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    manifest = {
        "lane": {"family": "thesis_five_state_shared_intent_v1"},
        "assets": [
            {
                "id": "fixture",
                "frozen_path": str(asset_path),
                "sha256": sha256 or actual_hash,
                "expected_obs_dim": None,
                "expected_action_dim": None,
            }
        ],
        "third_ppo_vpp_opponent": {
            "eligible": eligible,
            "status": "blocked_no_independent_ppo_vpp_checkpoint",
        },
    }
    retention = {
        "output_root": str(tmp_path / "new-output"),
        "minimum_free_gb_before_run": 120,
        "checkpoint_retention": {"top_k": 3},
        "raw_telemetry": {
            "full_allowed_splits": ["dev30", "heldout60"],
            "prohibited_splits": ["train"],
        },
    }
    manifest_path = tmp_path / "assets.yaml"
    retention_path = tmp_path / "retention.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    retention_path.write_text(yaml.safe_dump(retention), encoding="utf-8")
    return manifest_path, retention_path


def _ample_disk_usage(_path: str):
    Usage = namedtuple("Usage", "total used free")
    return Usage(500 * 1024**3, 100 * 1024**3, 300 * 1024**3)


def test_preflight_keeps_asset_integrity_separate_from_training_readiness(tmp_path):
    manifest, retention = _write_contracts(tmp_path, eligible=False)

    report = MODULE.build_preflight_report(
        manifest,
        retention,
        disk_usage=_ample_disk_usage,
    )

    assert report["asset_integrity_pass"] is True
    assert report["third_opponent_ready"] is False
    assert report["training_ready"] is False
    assert report["blockers"] == ["blocked_no_independent_ppo_vpp_checkpoint"]


def test_preflight_accepts_an_eligible_third_opponent_when_assets_match(tmp_path):
    manifest, retention = _write_contracts(tmp_path, eligible=True)

    report = MODULE.build_preflight_report(
        manifest,
        retention,
        disk_usage=_ample_disk_usage,
    )

    assert report["asset_integrity_pass"] is True
    assert report["third_opponent_ready"] is True
    assert report["training_ready"] is True


def test_preflight_rejects_a_hash_mismatch(tmp_path):
    manifest, retention = _write_contracts(tmp_path, eligible=True, sha256="0" * 64)

    report = MODULE.build_preflight_report(
        manifest,
        retention,
        disk_usage=_ample_disk_usage,
    )

    assert report["asset_integrity_pass"] is False
    assert "fixture:sha256_mismatch" in report["errors"]
