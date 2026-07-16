from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_dualskill_taxonomy60_formal.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("taxonomy60_formal_preflight", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_preflight_accepts_the_frozen_formal_contract():
    module = _load_module()
    manifest = module._load_yaml(module.MANIFEST_PATH)
    config = module.comparison._load_config_with_includes(module.FORMAL_CONFIG)
    config = module.comparison._materialize_scenario_manifest(config, module.FORMAL_CONFIG)
    decision = module._load_yaml(module.DECISION_CONTRACT_PATH)

    assert module.validate_static_contract(
        manifest=manifest, formal_config=config, decision_contract=decision
    ) == []


def test_static_preflight_rejects_an_unfrozen_method_matrix():
    module = _load_module()
    manifest = module._load_yaml(module.MANIFEST_PATH)
    config = module.comparison._load_config_with_includes(module.FORMAL_CONFIG)
    config = module.comparison._materialize_scenario_manifest(config, module.FORMAL_CONFIG)
    decision = module._load_yaml(module.DECISION_CONTRACT_PATH)
    changed = copy.deepcopy(config)
    changed["thesis_dualskill_taxonomy60_formal"]["methods"] = ["canonical_ppo"]

    errors = module.validate_static_contract(
        manifest=manifest, formal_config=changed, decision_contract=decision
    )
    assert "formal config methods mismatch" in errors


def test_static_preflight_rejects_an_incomplete_no_go_route_mapping():
    module = _load_module()
    manifest = module._load_yaml(module.MANIFEST_PATH)
    config = module.comparison._load_config_with_includes(module.FORMAL_CONFIG)
    config = module.comparison._materialize_scenario_manifest(config, module.FORMAL_CONFIG)
    decision = module._load_yaml(module.DECISION_CONTRACT_PATH)
    changed = copy.deepcopy(decision)
    changed["classification_routes"].pop("routing_sensitive_candidate")

    errors = module.validate_static_contract(
        manifest=manifest, formal_config=config, decision_contract=changed
    )
    assert "decision contract has unmapped classifications" in errors
