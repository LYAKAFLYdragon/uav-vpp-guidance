"""Build non-ranking capability cards for the three frozen v1 opponents."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = (
    REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_opponent_registry.yaml"
)
DEFAULT_ASSETS = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_asset_manifest.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_opponent_capability_cards.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def build_cards(registry: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    asset_by_id = {item["id"]: item for item in assets.get("assets", [])}
    end_to_end = asset_by_id["canonical_end_to_end_neural_opponent"]
    independent = asset_by_id["independent_ppo_vpp_opponent_v1"]
    return {
        "schema_version": 1,
        "source_id": "THESIS-FIVE-STATE-SHARED-INTENT-V1-OPPONENT-CAPABILITY-CARDS",
        "lane": "noncanonical_thesis_extension",
        "claim_boundary": "capability documentation only; no Elo or total-strength ranking",
        "shared_runtime": registry["action_cadence_contract"],
        "cards": {
            "expert_rule_based": {
                "identity": "ExpertOpponent / ExpertVPPPolicy",
                "control_mechanism": "rule-based VPP through guidance and PID",
                "checkpoint": None,
                "reference_evidence": "rule mechanism is auditable; no training-budget claim",
                "known_boundary": "not a learned or Elo-calibrated opponent",
            },
            "end_to_end_neural": {
                "identity": "frozen 16-D to 3-D direct-command neural opponent",
                "checkpoint": end_to_end["frozen_path"],
                "checkpoint_sha256": end_to_end["sha256"],
                "action_mode": "direct_command",
                "known_boundary": "historical frozen checkpoint; strength is reported per opponent rather than ranked",
            },
            "independent_ppo_vpp": {
                "identity": "independent frozen 16-D to 3-D PPO/VPP opponent",
                "checkpoint": independent["frozen_path"],
                "checkpoint_sha256": independent["sha256"],
                "action_mode": "vpp",
                "reference_evidence": assets["third_ppo_vpp_opponent"]["capability_audit"],
                "known_boundary": assets["third_ppo_vpp_opponent"]["claim_boundary"],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cards = build_cards(_load_yaml(args.registry), _load_yaml(args.asset_manifest))
    cards["integrity"] = {
        "registry_sha256": _sha256_file(args.registry),
        "asset_manifest_sha256": _sha256_file(args.asset_manifest),
        "builder_code_sha256": _sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cards, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
