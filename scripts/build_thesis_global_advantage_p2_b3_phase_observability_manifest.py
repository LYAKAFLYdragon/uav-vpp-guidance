"""Build the non-overlapping diagnostic manifest for P2-B3 observability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_thesis_five_state_manifests as v1
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


MANIFEST_DIR = ROOT / "config" / "experiment" / "manifests"
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-MANIFEST30-V1"
PACKAGE = {
    "id": "p2_b3_diagnostic_far_fast_e",
    "initial_range_m": 3800.0,
    "own_speed_mps": 270.0,
    "target_speed_mps": 250.0,
}
SEED_BASE = 13001
DISJOINT_MANIFESTS = (
    MANIFEST_DIR / "thesis_global_advantage_v1_p2_physical_preflight60.yaml",
    MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml",
    MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml",
    MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_signature(scenario: Mapping[str, Any]) -> str:
    return _payload_sha256(scenario)


def _scenario(
    *, initial_class: str, height_condition: str, mirror_sign: str, index: int
) -> dict[str, Any]:
    scenario = v1._build_scenario(
        split="p2_b3_phase_observability30",
        package=PACKAGE,
        initial_class=initial_class,
        height_condition=height_condition,
        mirror_sign=mirror_sign,
        scenario_index=index,
    )
    seed = SEED_BASE + index
    metadata = scenario["metadata"]
    cell_id = f"{PACKAGE['id']}__{initial_class}__{height_condition}__{mirror_sign}"
    scenario["name"] = f"global_advantage_p2_b3_{cell_id}_s{seed}"
    metadata.update(
        {
            "manifest_family": "thesis_global_advantage_v1",
            "split": "p2_b3_phase_observability30",
            "scenario_seed": seed,
            "diagnostic_seed": seed,
            "geometry_cell_id": cell_id,
            "p2_b3_phase_observability_only": True,
            "physical_geometry_signature": scenario_signature(scenario),
        }
    )
    return scenario


def _validate(manifest: Mapping[str, Any], disjoint_sources: Mapping[str, Mapping[str, Any]]) -> None:
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 30 or len({item["name"] for item in scenarios}) != 30:
        raise ValueError("P2-B3 requires exactly 30 named scenarios")
    if {item["metadata"]["initial_class"] for item in scenarios} != set(
        v1.INITIAL_CLASS_ANGLES_DEG
    ):
        raise ValueError("P2-B3 must retain all five initial states")
    if {item["metadata"]["height_condition"] for item in scenarios} != set(v1.HEIGHT_OFFSETS_M):
        raise ValueError("P2-B3 must retain all three height conditions")
    if {item["metadata"]["mirror_sign"] for item in scenarios} != {"negative", "positive"}:
        raise ValueError("P2-B3 must retain mirror pairs")
    if not all(item["metadata"]["phase_at_reset"] == "pre_merge" for item in scenarios):
        raise ValueError("P2-B3 reset phase contract drifted")
    instance_signatures = {_instance_signature(item) for item in scenarios}
    physical_signatures = {scenario_signature(item) for item in scenarios}
    if len(instance_signatures) != 30 or len(physical_signatures) != 30:
        raise ValueError("P2-B3 scenarios must be physically and seed-wise unique")
    for label, source in disjoint_sources.items():
        source_scenarios = list(source.get("scenarios") or [])
        source_instances = {_instance_signature(item) for item in source_scenarios}
        source_physical = {scenario_signature(item) for item in source_scenarios}
        if instance_signatures.intersection(source_instances) or physical_signatures.intersection(
            source_physical
        ):
            raise ValueError(f"P2-B3 overlaps frozen source: {label}")


def build_manifest(disjoint_sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    index = 0
    for initial_class in v1.INITIAL_CLASS_ANGLES_DEG:
        for height_condition in v1.HEIGHT_OFFSETS_M:
            for mirror_sign in ("negative", "positive"):
                scenarios.append(
                    _scenario(
                        initial_class=initial_class,
                        height_condition=height_condition,
                        mirror_sign=mirror_sign,
                        index=index,
                    )
                )
                index += 1
    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "split": "p2_b3_phase_observability30",
        "protocol": {
            "training_permitted": False,
            "tuning_permitted": False,
            "policy_evaluation_permitted": False,
            "formal_heldout_claims_permitted": False,
            "allowed_use": "strict_jsbsim_phase_observability_preflight_only",
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
        },
        "generation_contract": {
            "initial_classes": list(v1.INITIAL_CLASS_ANGLES_DEG),
            "height_offsets_target_minus_own_m": v1.HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "package": dict(PACKAGE),
            "diagnostic_seed_base": SEED_BASE,
            "scenario_count": 30,
        },
        "disjointness": {
            label: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _file_sha256(path),
                "instance_intersection_count": 0,
                "physical_intersection_count": 0,
            }
            for label, path in zip(
                ("p2_b2", "heldout240", "dev30", "historical_heldout60"), DISJOINT_MANIFESTS
            )
        },
        "scenarios": scenarios,
    }
    _validate(manifest, disjoint_sources)
    manifest["integrity"] = {"payload_sha256": _payload_sha256(manifest)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_DIR / "thesis_global_advantage_v1_p2_b3_phase_observability30.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources = {
        label: _load_yaml(path)
        for label, path in zip(
            ("p2_b2", "heldout240", "dev30", "historical_heldout60"), DISJOINT_MANIFESTS
        )
    }
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {"source_id": SOURCE_ID, "scenario_count": len(manifest["scenarios"]), **manifest["integrity"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
