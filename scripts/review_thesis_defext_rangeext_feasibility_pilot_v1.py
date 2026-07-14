#!/usr/bin/env python3
"""Independent, non-executing review of the defensive-extension pilot preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_thesis_defext_rangeext_feasibility_pilot_v1_manifests import (  # noqa: E402
    V2_SIGNATURES,
    package_signature,
    payload_sha256,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping: {path}")
    return payload


def _repo_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def review(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "detail": detail})

    check("source_id", config.get("source_id") == "THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1", str(config.get("source_id")))
    authorization = config.get("authorization", {})
    locked = (
        "training_permitted",
        "pilot_execution_permitted",
        "baseline_evaluation_permitted",
        "heldout_evaluation_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
    )
    check("execution_locked", all(authorization.get(key) is False for key in locked), str({key: authorization.get(key) for key in locked}))

    freeze = config.get("implementation_freeze", {})
    implementation_sha = str(freeze.get("git_sha", ""))
    check("implementation_commit_exists", _commit_exists(implementation_sha), implementation_sha)
    freeze_path = _repo_path(str(freeze.get("manifest", "")))
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    file_results = []
    for entry in freeze_manifest.get("files", []):
        path = _repo_path(entry["path"])
        actual = _sha256(path) if path.is_file() else None
        file_results.append(actual == entry.get("sha256"))
    check("implementation_source_hashes", bool(file_results) and all(file_results), f"{sum(file_results)}/{len(file_results)} matched")
    dirty_evidence = freeze_manifest.get("dirty_run_evidence", {})
    dirty_gate_path = Path(str(dirty_evidence.get("gate_path", "")))
    dirty_gate_hash_valid = (
        dirty_gate_path.is_file()
        and _sha256(dirty_gate_path) == dirty_evidence.get("gate_sha256_at_freeze")
    )
    check(
        "dirty_run_not_promoted",
        dirty_evidence.get("promotion_to_paper_safe") is False
        and dirty_evidence.get("status") == "research_only_not_paper_safe"
        and dirty_gate_hash_valid,
        str(dirty_evidence),
    )

    p3 = config.get("fixed_contract", {}).get("encoder", {})
    p3_path = Path(str(p3.get("checkpoint", "")))
    p3_sha = _sha256(p3_path) if p3_path.is_file() else None
    check("p3_frozen_hash", p3.get("trainable") is False and p3.get("fallback") == "prohibited" and p3_sha == p3.get("sha256"), str(p3_sha))

    fixed = config.get("fixed_contract", {})
    check("fixed_skill_profile", fixed.get("skill") == "defensive_extension" and fixed.get("profile") == "range_extension", f"{fixed.get('skill')}/{fixed.get('profile')}")
    check("routing_disabled", fixed.get("routing_enabled") is False and fixed.get("high_level_policy_present") is False, str(fixed))

    methods = config.get("methods", {})
    expected_methods = {"candidate_fixed_defensive_extension", "frozen_fixed_head_on", "frozen_fixed_crossing"}
    check("minimal_method_matrix", set(methods) == expected_methods, str(sorted(methods)))
    check("candidate_untrained", methods.get("candidate_fixed_defensive_extension", {}).get("checkpoint") is None, str(methods.get("candidate_fixed_defensive_extension", {})))
    asset_results = []
    for method in ("frozen_fixed_head_on", "frozen_fixed_crossing"):
        entry = methods[method]
        path = Path(entry["checkpoint"])
        asset_results.append(path.is_file() and _sha256(path) == entry.get("checkpoint_sha256"))
    check("baseline_checkpoint_hashes", all(asset_results), f"{sum(asset_results)}/2 matched")

    sources = config.get("sources", {})
    train = _load_yaml(_repo_path(sources["train_distribution"]))
    dev = _load_yaml(_repo_path(sources["dev_manifest"]))
    heldout = _load_yaml(_repo_path(sources["heldout_manifest"]))
    check("manifest_payload_hashes", payload_sha256(dev) == sources.get("dev_manifest_payload_sha256") and payload_sha256(heldout) == sources.get("heldout_manifest_payload_sha256"), f"dev={payload_sha256(dev)} heldout={payload_sha256(heldout)}")
    check("manifest_counts", len(dev.get("scenarios", [])) == 12 and len(heldout.get("scenarios", [])) == 24, f"dev={len(dev.get('scenarios', []))} heldout={len(heldout.get('scenarios', []))}")

    dev_packages = dev.get("generation_contract", {}).get("packages", [])
    heldout_packages = heldout.get("generation_contract", {}).get("packages", [])
    dev_signatures = {package_signature(item) for item in dev_packages}
    heldout_signatures = {package_signature(item) for item in heldout_packages}
    check("v2_package_disjointness", not ((dev_signatures | heldout_signatures) & V2_SIGNATURES), str(sorted(dev_signatures | heldout_signatures)))
    check("dev_heldout_disjointness", not (dev_signatures & heldout_signatures), str(dev_signatures & heldout_signatures))
    train_range = train.get("sampling_contract", {}).get("initial_range_m", [])
    fixed_ranges = {signature[0] for signature in dev_signatures | heldout_signatures | V2_SIGNATURES}
    check("train_fixed_package_disjointness", len(train_range) == 2 and all(not float(train_range[0]) <= value <= float(train_range[1]) for value in fixed_ranges), f"train_range={train_range} fixed_ranges={sorted(fixed_ranges)}")
    all_scenarios = list(dev.get("scenarios", [])) + list(heldout.get("scenarios", []))
    scenario_contract = all(
        item.get("metadata", {}).get("taxonomy_geometry_state") == "disadvantage"
        and item.get("metadata", {}).get("continuous_run_in") is True
        and item.get("metadata", {}).get("routing_enabled") is False
        and item.get("metadata", {}).get("fixed_skill") == "defensive_extension"
        and item.get("metadata", {}).get("fixed_profile") == "range_extension"
        and float(item.get("metadata", {}).get("initial_range_m", 0.0)) > 1000.0
        for item in all_scenarios
    )
    check("scenario_contract", scenario_contract, f"{len(all_scenarios)} scenarios")

    opponents = config.get("opponents", {})
    check("opponents_separate", opponents.get("order") == ["expert", "end_to_end", "independent_ppo_vpp"] and opponents.get("report_separately") is True and opponents.get("pooled_gate") == "prohibited", str(opponents))
    training = config.get("proposed_training", {})
    check("fixed_training_budget", training.get("total_timesteps") == 50000 and training.get("budget_extension") == "prohibited" and training.get("heldout_use_for_selection") == "prohibited", str(training))

    gates = config.get("gates", {})
    minimum = gates.get("per_opponent_contract_minimum", {})
    claim_ready = gates.get("per_opponent_claim_ready_coverage", {})
    thresholds_ok = (
        minimum.get("qualifying_paired_episodes") == 2
        and minimum.get("valid_target_steps") == 20
        and minimum.get("distinct_scenario_signatures") == 2
        and minimum.get("distinct_mirror_signs") == 2
        and claim_ready.get("qualifying_paired_episodes") == 8
        and claim_ready.get("valid_target_steps") == 160
        and gates.get("safety_noninferiority_vs_head_on", {}).get("ego_crash_oob_rate_delta_max") == 0.05
        and gates.get("geometry_noninferiority_vs_head_on", {}).get("paired_intent_loss_auc20_delta_max") == 0.02
        and gates.get("practical_improvement_vs_head_on", {}).get("paired_intent_loss_auc20_delta_max") == -0.05
        and gates.get("practical_improvement_vs_best_existing_specialist", {}).get("paired_intent_loss_auc20_delta_max") == -0.02
    )
    check("preregistered_thresholds", thresholds_ok, str(gates))
    stop = config.get("safety_stop_rule", {})
    check("fail_closed_stop_rule", len(stop.get("immediate_abort_on", [])) >= 6 and stop.get("after_abort", {}).get("freeze_as_negative_evidence") is True and stop.get("after_abort", {}).get("add_training_steps") == "prohibited" and stop.get("after_abort", {}).get("rerun_same_source_id") == "prohibited", str(stop))
    check("decision_tree_complete", set(config.get("decision_logic", {})) == {"new_skill_gap_supported", "existing_library_routing_or_composition_gap", "defensive_extension_hypothesis_not_supported", "safety_no_go"}, str(config.get("decision_logic", {}).keys()))

    output_root = Path(config.get("outputs", {}).get("root", ""))
    check("fresh_output_id", output_root.name == "defensive_extension_range_extension_feasibility_v1" and not output_root.exists() and config.get("outputs", {}).get("creation_permitted_by_this_config") is False, str(output_root))
    passed = all(item["passed"] for item in checks)
    return {
        "source_id": config.get("source_id"),
        "review_mode": "independent_design_review_no_training_no_evaluation",
        "passed": passed,
        "pilot_execution_authorized": False,
        "training_authorized": False,
        "checks": checks,
        "decision": "preregistration_ready_for_separate_authorization" if passed else "preregistration_not_ready",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Defensive-Extension Feasibility Pilot V1 独立预注册复核",
        "",
        f"**复核结论：** `{'PASS' if result['passed'] else 'FAIL'}`",
        "",
        "本复核只检查设计、资产、分割、门槛和 stop rule；未启动训练、JSBSim evaluation 或 output root。通过也不构成执行授权。",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in result["checks"]:
        detail = str(item["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| `{item['id']}` | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    lines.extend(
        [
            "",
            "## 授权边界",
            "",
            "- `training_authorized=false`。",
            "- `pilot_execution_authorized=false`。",
            "- 下一步必须由用户单独授权，且执行前重新验证 clean worktree、fresh output root、资产 SHA 与 config source ID。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=ROOT / "reports" / "thesis_defext_rangeext_feasibility_pilot_v1_independent_review_20260714.json")
    parser.add_argument("--md-output", type=Path, default=ROOT / "reports" / "thesis_defext_rangeext_feasibility_pilot_v1_independent_review_20260714_zh.md")
    args = parser.parse_args()
    result = review(args.config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.md_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "decision": result["decision"], "json": str(args.json_output), "markdown": str(args.md_output)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
