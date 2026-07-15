"""Derive a non-evaluation 60-cell physical preflight from heldout240."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "config" / "experiment" / "manifests"
HELDOUT240 = MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml"
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1"
PREFLIGHT_SEED_BASE = 12001


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_manifest(heldout240: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = list(heldout240["scenarios"])
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        groups.setdefault(str(scenario["metadata"]["geometry_cell_id"]), []).append(scenario)
    if len(groups) != 60 or any(len(items) != 4 for items in groups.values()):
        raise ValueError("heldout240 must provide exactly 60 four-seed geometry cells")

    preflight: list[dict[str, Any]] = []
    for index, (cell_id, candidates) in enumerate(sorted(groups.items())):
        source = copy.deepcopy(min(candidates, key=lambda item: int(item["metadata"]["evaluation_seed"])))
        metadata = source["metadata"]
        heldout_seed = int(metadata["evaluation_seed"])
        preflight_seed = PREFLIGHT_SEED_BASE + index
        source["name"] = f"global_advantage_p2_physical_{cell_id}_s{preflight_seed}"
        metadata.update(
            {
                "split": "p2_physical_preflight60",
                "scenario_seed": preflight_seed,
                "preflight_seed": preflight_seed,
                "source_heldout_evaluation_seed": heldout_seed,
                "preflight_only": True,
            }
        )
        preflight.append(source)

    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "split": "p2_physical_preflight60",
        "protocol": {
            "training_permitted": False,
            "tuning_permitted": False,
            "policy_evaluation_permitted": False,
            "allowed_use": "strict_jsbsim_physical_reachability_preflight_only",
            "formal_heldout_claims_permitted": False,
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
        },
        "source_heldout240": {
            "source_id": heldout240["source_id"],
            "payload_sha256": heldout240["integrity"]["payload_sha256"],
            "geometry_cells_reused": 60,
            "evaluation_seeds_reused": False,
        },
        "scenarios": preflight,
    }
    if len({item["name"] for item in preflight}) != 60:
        raise ValueError("preflight names must be unique")
    if len({item["metadata"]["geometry_cell_id"] for item in preflight}) != 60:
        raise ValueError("preflight must retain every geometry cell")
    if any(
        item["metadata"]["preflight_seed"] == item["metadata"]["source_heldout_evaluation_seed"]
        for item in preflight
    ):
        raise ValueError("preflight must not reuse heldout evaluation seeds")
    manifest["integrity"] = {"payload_sha256": _sha256(manifest)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MANIFEST_DIR / "thesis_global_advantage_v1_p2_physical_preflight60.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = build_manifest(_load(HELDOUT240))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "count": len(manifest["scenarios"]), "sha256": manifest["integrity"]["payload_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
