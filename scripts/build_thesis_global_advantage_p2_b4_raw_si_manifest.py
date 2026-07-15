"""Build a new, non-overlapping 30-scenario manifest for the P2-B4 SI check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_thesis_five_state_manifests as v1
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


MANIFEST_DIR = ROOT / "config" / "experiment" / "manifests"
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-MANIFEST30-V1"
PACKAGE = {
    "id": "p2_b4_diagnostic_far_fast_f",
    "initial_range_m": 4000.0,
    "own_speed_mps": 275.0,
    "target_speed_mps": 255.0,
}
SEED_BASE = 14001
DISJOINT_MANIFESTS = (
    MANIFEST_DIR / "thesis_global_advantage_v1_p2_physical_preflight60.yaml",
    MANIFEST_DIR / "thesis_global_advantage_v1_p2_b3_phase_observability30.yaml",
    MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml",
    MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml",
    MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario(initial_class: str, height_condition: str, mirror_sign: str, index: int) -> dict[str, Any]:
    scenario = v1._build_scenario(
        split="p2_b4_raw_si_phase30",
        package=PACKAGE,
        initial_class=initial_class,
        height_condition=height_condition,
        mirror_sign=mirror_sign,
        scenario_index=index,
    )
    seed = SEED_BASE + index
    metadata = scenario["metadata"]
    cell = f"{PACKAGE['id']}__{initial_class}__{height_condition}__{mirror_sign}"
    scenario["name"] = f"global_advantage_p2_b4_{cell}_s{seed}"
    metadata.update(
        {
            "manifest_family": "thesis_global_advantage_v1",
            "split": "p2_b4_raw_si_phase30",
            "scenario_seed": seed,
            "diagnostic_seed": seed,
            "geometry_cell_id": cell,
            "p2_b4_raw_si_only": True,
            "physical_geometry_signature": scenario_signature(scenario),
        }
    )
    return scenario


def _validate(manifest: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> None:
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 30 or len({item["name"] for item in scenarios}) != 30:
        raise ValueError("P2-B4 requires 30 unique scenarios")
    if {item["metadata"]["initial_class"] for item in scenarios} != set(v1.INITIAL_CLASS_ANGLES_DEG):
        raise ValueError("P2-B4 must cover all five states")
    if {item["metadata"]["height_condition"] for item in scenarios} != set(v1.HEIGHT_OFFSETS_M):
        raise ValueError("P2-B4 must cover all height conditions")
    if {item["metadata"]["mirror_sign"] for item in scenarios} != {"negative", "positive"}:
        raise ValueError("P2-B4 must cover mirror pairs")
    if not all(item["metadata"]["phase_at_reset"] == "pre_merge" for item in scenarios):
        raise ValueError("P2-B4 must declare pre-merge reset")
    signatures = {scenario_signature(item) for item in scenarios}
    if len(signatures) != 30:
        raise ValueError("P2-B4 physical signatures are not unique")
    for label, source in sources.items():
        source_signatures = {scenario_signature(item) for item in source.get("scenarios") or []}
        if signatures.intersection(source_signatures):
            raise ValueError(f"P2-B4 overlaps frozen source: {label}")


def build_manifest(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios = []
    index = 0
    for initial_class in v1.INITIAL_CLASS_ANGLES_DEG:
        for height_condition in v1.HEIGHT_OFFSETS_M:
            for mirror_sign in ("negative", "positive"):
                scenarios.append(_scenario(initial_class, height_condition, mirror_sign, index))
                index += 1
    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "split": "p2_b4_raw_si_phase30",
        "protocol": {
            "training_permitted": False,
            "tuning_permitted": False,
            "policy_evaluation_permitted": False,
            "formal_heldout_claims_permitted": False,
            "allowed_use": "strict_jsbsim_raw_si_phase_preflight_only",
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
                "physical_intersection_count": 0,
            }
            for label, path in zip(
                ("p2_b2", "p2_b3", "heldout240", "dev30", "historical_heldout60"),
                DISJOINT_MANIFESTS,
            )
        },
        "scenarios": scenarios,
    }
    _validate(manifest, sources)
    manifest["integrity"] = {"payload_sha256": _hash(manifest)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_DIR / "thesis_global_advantage_v1_p2_b4_raw_si_phase30.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    labels = ("p2_b2", "p2_b3", "heldout240", "dev30", "historical_heldout60")
    sources = {label: _load_yaml(path) for label, path in zip(labels, DISJOINT_MANIFESTS)}
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "scenario_count": 30, **manifest["integrity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
