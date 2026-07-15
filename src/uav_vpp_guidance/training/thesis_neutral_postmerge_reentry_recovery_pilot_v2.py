"""V2 execution wrapper with manifest-defined pairing keys.

V1 is retained unchanged as frozen implementation-failure evidence.  V2
reuses its strictly tested simulation engine only through a scoped adapter
that replaces the ambiguous metadata lookup with the V2 manifest contract.
"""

from __future__ import annotations

from contextlib import contextmanager
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_pairing_v2 import (
    pair_key,
)
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as v1


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2"


class NeutralPostMergePilotV2Error(v1.NeutralPostMergePilotError):
    """Raised when a V2 execution input violates its explicit pairing contract."""


class V2Episode(v1.NeutralPostMergeEpisode):
    """Only changes ledger identity; control mechanics stay in the V1 engine."""

    def ledger(
        self,
        *,
        opponent: str,
        method: str,
        scenario: Mapping[str, Any],
        handoff: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = pair_key(scenario)
        steps = self.runin_steps + self.post_handoff_steps
        return {
            "source_id": SOURCE_ID,
            "header": {
                "opponent": opponent,
                "method": method,
                "scenario": scenario["name"],
                "pair_key": key,
                "scenario_signature": key,
                "scenario_seed": scenario["metadata"]["scenario_seed"],
                "mirror_sign": scenario["metadata"]["mirror_sign"],
            },
            "handoff": dict(handoff),
            "steps": steps,
            "summary": {
                "step_count": len(steps),
                "handoff_reached": bool(handoff.get("handoff_reached")),
                "no_backend_fallback": all(not bool(item.get("backend_fallback_occurred")) for item in steps),
                "full_post_handoff_telemetry": all(
                    item.get("observation_66d") is not None and len(item["observation_66d"]) == 66
                    for item in self.post_handoff_steps
                ),
            },
        }


def _episode_result(
    opponent: str,
    method: str,
    scenario: Mapping[str, Any],
    handoff: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    key = pair_key(scenario)
    valid = [item for item in metrics if item.get("valid_target_step")][:20]
    final = metrics[-1] if metrics else {}
    # ``scenario_signature`` remains only as an engine-compatibility alias;
    # every persisted V2 result also exposes the explicit pair_key.
    return {
        "opponent": opponent,
        "method": method,
        "scenario": scenario["name"],
        "pair_key": key,
        "scenario_signature": key,
        "mirror_sign": scenario["metadata"]["mirror_sign"],
        "scenario_seed": scenario["metadata"]["scenario_seed"],
        "handoff_reached": bool(handoff.get("handoff_reached")),
        "handoff_step": handoff.get("handoff_step"),
        "handoff_state_sha256": handoff.get("handoff_state_sha256"),
        "valid_target_steps": sum(bool(item.get("valid_target_step")) for item in metrics),
        "qualifying": len(valid) == 20,
        "intent_loss_auc20": float(np.mean([item["after_intent_loss"] for item in valid])) if len(valid) == 20 else None,
        "range_opening_fraction_auc20": float(np.mean([float(item["range_rate_mps"] > 0.0) for item in valid])) if valid else None,
        "specific_energy_height_delta_m_at_20": float(valid[-1]["specific_energy_height_delta_m"]) if len(valid) == 20 else None,
        "target_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["target_in_attack_zone"]) for item in valid])) if valid else None,
        "ego_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["ego_in_attack_zone"]) for item in valid])) if valid else None,
        "terminal_reason": final.get("terminal_reason", handoff.get("terminal_reason", "no_handoff")),
        "ego_failure": bool(final.get("ego_failure", False)),
        "win": bool(final.get("win", False)),
        "loss": bool(final.get("loss", False)),
        "draw": bool(final.get("draw", False)),
        "ego_hp": final.get("ego_hp"),
        "target_hp": final.get("target_hp"),
        "telemetry_complete": bool(ledger["summary"]["full_post_handoff_telemetry"]),
    }


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_authorization(
    config: Mapping[str, Any], config_path: Path, base_path: Path
) -> tuple[Mapping[str, Any], Path]:
    if config.get("source_id") != SOURCE_ID:
        raise NeutralPostMergePilotV2Error("unexpected V2 source ID")
    auth = config.get("authorization") or {}
    for key in ("execution_permitted", "training_permitted", "baseline_evaluation_permitted", "heldout_evaluation_permitted"):
        if auth.get(key) is not True:
            raise NeutralPostMergePilotV2Error(f"authorization.{key} must be true")
    for key in ("high_level_ppo_training_permitted", "four_skill_training_permitted", "combat_finetune_permitted", "tuning_permitted", "vpp_change_permitted", "guidance_change_permitted", "pid_change_permitted", "snapshot_restore", "future_state_injection", "history_padding_permitted"):
        if auth.get(key) is not False:
            raise NeutralPostMergePilotV2Error(f"authorization.{key} must remain false")
    execution = config.get("execution") or {}
    required_sha = str(execution.get("required_implementation_git_sha", ""))
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False)
    if not required_sha or ancestor.returncode != 0 or _git_value("status", "--porcelain"):
        raise NeutralPostMergePilotV2Error("V2 execution requires a clean worktree and frozen implementation ancestor")
    allowed = {str(item) for item in execution.get("authorization_delta_paths", [])}
    changed = {line.replace("\\", "/") for line in _git_value("diff", "--name-only", f"{required_sha}..HEAD").splitlines() if line.strip()}
    if changed - allowed:
        raise NeutralPostMergePilotV2Error(f"unexpected changes after V2 implementation freeze: {sorted(changed - allowed)}")
    v1._require_hash(base_path, execution.get("base_config_sha256"), "V2 base config")
    v1._require_hash(v1._repo_path(str(execution["runtime_template"])), execution.get("runtime_template_sha256"), "V2 runtime template")
    registry_path = v1._repo_path(str(execution["runtime_registry"]))
    v1._require_hash(registry_path, execution.get("runtime_registry_sha256"), "V2 runtime registry")
    code_files = execution.get("authorized_code_files")
    if not isinstance(code_files, Sequence) or not code_files:
        raise NeutralPostMergePilotV2Error("V2 execution requires authorized code hashes")
    for entry in code_files:
        v1._require_hash(v1._repo_path(str(entry.get("path", ""))), entry.get("sha256"), "V2 authorized code")
    # The parent design must validate before any JSBSim environment is created.
    from scripts.preflight_thesis_neutral_postmerge_reentry_recovery_pilot_v2 import validate_design

    validate_design(base_path)
    registry = v1.load_yaml(registry_path)
    if tuple(registry.get("opponents", ())) != v1.OPPONENTS:
        raise NeutralPostMergePilotV2Error("V2 runtime registry opponent order drifted")
    for opponent in v1.OPPONENTS:
        entry = registry["opponents"][opponent]
        if opponent != "expert":
            v1._require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), f"V2 opponent {opponent}")
    for name in ("run_in_head_on", "frozen_fixed_head_on", "frozen_fixed_crossing"):
        entry = registry["specialists"][name]
        v1._require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), f"V2 specialist {name}")
    output = Path(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise NeutralPostMergePilotV2Error("V2 output root must be absolute, absent, and fresh")
    if shutil.disk_usage(output.parent).free / (1024**3) < float(config["contract"]["min_free_disk_gb"]):
        raise NeutralPostMergePilotV2Error("V2 output root failed disk gate")
    return registry, output


@contextmanager
def _v2_engine() -> Iterator[None]:
    saved = (v1.SOURCE_ID, v1.NeutralPostMergeEpisode, v1._episode_result, v1._validate_authorization)
    v1.SOURCE_ID = SOURCE_ID
    v1.NeutralPostMergeEpisode = V2Episode
    v1._episode_result = _episode_result
    v1._validate_authorization = _validate_authorization
    try:
        yield
    finally:
        v1.SOURCE_ID, v1.NeutralPostMergeEpisode, v1._episode_result, v1._validate_authorization = saved


def run(config_path: Path) -> dict[str, Any]:
    with _v2_engine():
        return v1.run(config_path)


def preflight(config_path: Path) -> dict[str, Any]:
    with _v2_engine():
        result = v1.preflight(config_path)
    result["pairing_key"] = "metadata.geometry_cell_id + metadata.scenario_seed"
    return result
