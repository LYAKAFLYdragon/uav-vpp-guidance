from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_opponent_registry.yaml"
ASSET_PATH = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_asset_manifest.yaml"
CARDS_PATH = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_opponent_capability_cards.yaml"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def test_registry_is_balanced_and_binds_all_three_frozen_opponents():
    registry = _load(REGISTRY_PATH)
    assets = {item["id"]: item for item in _load(ASSET_PATH)["assets"]}

    assert registry["training_sampling"]["strategy"] == "balanced_round_robin"
    assert set(registry["opponents"]) == {
        "expert_rule_based", "end_to_end_neural", "independent_ppo_vpp"
    }
    assert set(registry["training_sampling"]["weights"].values()) == {1.0}
    assert registry["evaluation_reporting"]["aggregate_opponents"] == "prohibited"
    assert registry["action_cadence_contract"]["environment_high_level_dt_s"] == 0.2

    e2e = registry["opponents"]["end_to_end_neural"]
    independent = registry["opponents"]["independent_ppo_vpp"]
    assert e2e["checkpoint_sha256"] == assets["canonical_end_to_end_neural_opponent"]["sha256"]
    assert independent["checkpoint_sha256"] == assets["independent_ppo_vpp_opponent_v1"]["sha256"]
    assert _sha256(Path(e2e["checkpoint"])) == e2e["checkpoint_sha256"]
    assert _sha256(Path(independent["checkpoint"])) == independent["checkpoint_sha256"]
    assert independent["observation_schema"]["role_reversed"] is True


def test_three_registry_classes_load_and_vpp_adapter_obeys_contract():
    registry = _load(REGISTRY_PATH)["opponents"]
    expert = _class(registry["expert_rule_based"]["class"])(config={})
    end_to_end = _class(registry["end_to_end_neural"]["class"])(
        checkpoint_path=registry["end_to_end_neural"]["checkpoint"], device="cpu"
    )
    independent = _class(registry["independent_ppo_vpp"]["class"])(
        checkpoint_path=registry["independent_ppo_vpp"]["checkpoint"], device="cpu"
    )

    assert expert.action_mode == "vpp"
    assert end_to_end.action_mode == "direct_command"
    action = independent.act(
        {"observation_schema": {"role_reversed": True}, "observation_vector": np.zeros(16)}
    )
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert independent.get_diagnostics()["opponent_type"] == "independent_ppo_vpp"


def test_capability_cards_keep_identity_and_strength_boundaries_explicit():
    cards = _load(CARDS_PATH)

    assert cards["claim_boundary"] == "capability documentation only; no Elo or total-strength ranking"
    assert set(cards["cards"]) == {
        "expert_rule_based", "end_to_end_neural", "independent_ppo_vpp"
    }
    assert "not a general opponent-strength" in cards["cards"]["independent_ppo_vpp"]["known_boundary"]
