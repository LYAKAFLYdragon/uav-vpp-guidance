from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
CONFIG = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
DEV = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_dev30.yaml"
HELDOUT = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_heldout60.yaml"


def _load_script():
    path = SCRIPTS / "build_thesis_five_state_train_distribution.py"
    spec = importlib.util.spec_from_file_location("five_state_distribution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_training_distribution_is_balanced_and_blocks_evaluation_leakage():
    config = _load(CONFIG)
    contract = config["sampling_contract"]

    assert set(contract["initial_states"]) == {
        "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"
    }
    assert {item["weight"] for item in contract["initial_states"].values()} == {1.0}
    assert {item["weight"] for item in contract["height_conditions"].values()} == {1.0}
    assert contract["opponent_sampling"] == "balanced_round_robin"
    assert config["leakage_controls"]["normalization_fit_split"] == "train_only"
    assert config["leakage_controls"]["train_raw_telemetry_retention"] == "prohibited"
    assert config["integrity"]["dev30_training_support_overlap_count"] == 0
    assert config["integrity"]["heldout60_training_support_overlap_count"] == 0


def test_distribution_integrity_rebuilds_without_mutating_frozen_config():
    builder = _load_script()
    config = _load(CONFIG)
    rebuilt = builder.build_distribution_record(copy.deepcopy(config), _load(DEV), _load(HELDOUT))

    assert rebuilt == config
    payload = copy.deepcopy(config)
    expected_hash = payload["integrity"].pop("payload_sha256")
    assert builder._payload_sha256(payload) == expected_hash
