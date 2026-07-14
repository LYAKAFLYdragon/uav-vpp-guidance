#!/usr/bin/env python3
"""Run the authorised non-learning global-advantage P1 preflight."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_runin_contract import (  # noqa: E402
    GlobalAdvantageP1ContractError,
    GlobalAdvantageP1Plan,
    base_observation,
    classify_step,
    compare_repeats,
    stable_hash,
    summarize_episode,
    telemetry_record,
)
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as pilot  # noqa: E402
from uav_vpp_guidance.training.thesis_shared_skill_geometry import (  # noqa: E402
    FrozenP3Encoder,
    PhaseTracker,
)
from uav_vpp_guidance.evaluation.global_advantage_runin_contract import (  # noqa: E402
    build_p1_plan,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_global_advantage_v1_p1_runin_preflight.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GlobalAdvantageP1ContractError(f"YAML root must be a mapping: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _variant_order(
    scenarios: Sequence[Mapping[str, Any]], variant: str
) -> list[Mapping[str, Any]]:
    if variant == "forward":
        return list(scenarios)
    if variant == "reverse":
        return list(reversed(scenarios))
    if variant != "mirror_interleaved":
        raise GlobalAdvantageP1ContractError(f"unknown order variant: {variant}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        metadata = scenario.get("metadata", {})
        key = str(metadata.get("mirror_pair_id") or scenario.get("name"))
        groups.setdefault(key, []).append(scenario)
    ordered: list[Mapping[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        group = sorted(
            groups[key],
            key=lambda item: str(item.get("metadata", {}).get("mirror_sign", "")),
        )
        if index % 2:
            group.reverse()
        ordered.extend(group)
    return ordered


def _history_embedding(
    history: deque[np.ndarray], encoder: FrozenP3Encoder
) -> list[float] | None:
    if len(history) != history.maxlen:
        return None
    embedding = np.asarray(encoder.encode(np.stack(tuple(history), axis=0))).reshape(-1)
    if embedding.shape != (32,) or not np.isfinite(embedding).all():
        raise GlobalAdvantageP1ContractError("P3 temporal embedding is malformed")
    return [float(value) for value in embedding]


def _step_state(
    observation: Mapping[str, Any],
    history: deque[np.ndarray],
    phase_tracker: PhaseTracker,
    encoder: FrozenP3Encoder,
) -> tuple[dict[str, float], str, str, list[float] | None]:
    base = base_observation(observation)
    history.append(np.asarray([base[name] for name in base], dtype=np.float32))
    phase = phase_tracker.update(base["range_m"], base["range_rate_mps"])
    dynamic_state = classify_step(base, phase)
    return base, phase, dynamic_state, _history_embedding(history, encoder)


def _reset_record(
    observation: Mapping[str, Any],
    history: deque[np.ndarray],
    phase_tracker: PhaseTracker,
    encoder: FrozenP3Encoder,
    decimal_places: int,
) -> dict[str, Any]:
    base, phase, dynamic_state, embedding = _step_state(
        observation, history, phase_tracker, encoder
    )
    return telemetry_record(
        step=0,
        base=base,
        phase=phase,
        dynamic_state=dynamic_state,
        history=[
            {name: float(value) for name, value in zip(base, frame)}
            for frame in history
        ],
        temporal_embedding=embedding,
        action=None,
        info={"backend": "jsbsim"},
        decimal_places=decimal_places,
    )


def _history_payload(
    history: deque[np.ndarray], base_names: Sequence[str]
) -> list[dict[str, float]]:
    return [
        {name: float(value) for name, value in zip(base_names, frame)}
        for frame in history
    ]


def _run_episode(
    *,
    env: Any,
    reference: Any,
    encoder: FrozenP3Encoder,
    scenario: Mapping[str, Any],
    opponent: str,
    repeat_index: int,
    order_variant: str,
    plan: GlobalAdvantageP1Plan,
) -> dict[str, Any]:
    metadata = dict(scenario.get("metadata") or {})
    seed = int(metadata["scenario_seed"])
    observation = env.reset(scenario=dict(scenario), seed=seed)
    if getattr(env, "_backend", None) != "jsbsim":
        raise GlobalAdvantageP1ContractError("P1 backend is not JSBSim")
    if hasattr(env, "set_runtime_specialist_context"):
        env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="run_in_head_on_specialist",
            specialist_reason="global_advantage_p1_reference_only",
        )
    schema = dict(observation["observation_schema"])
    history: deque[np.ndarray] = deque(maxlen=plan.history_window_steps)
    phase_tracker = PhaseTracker()
    reset = _reset_record(
        observation,
        history,
        phase_tracker,
        encoder,
        plan.hash_decimal_places,
    )
    header = {
        "source_id": plan.source_id,
        "opponent": opponent,
        "scenario_name": str(scenario["name"]),
        "scenario_signature": stable_hash(scenario, plan.hash_decimal_places),
        "scenario_metadata": metadata,
        "scenario_seed": seed,
        "repeat_index": repeat_index,
        "order_variant": order_variant,
        "observation_schema_sha256": stable_hash(schema, plan.hash_decimal_places),
    }
    steps: list[dict[str, Any]] = []
    terminal_reason = "horizon"
    for step in range(1, plan.max_high_level_steps + 1):
        action = reference.get_deterministic_action(observation["observation_vector"])
        next_observation, _reward, terminated, truncated, info = env.step(action)
        base, phase, dynamic_state, embedding = _step_state(
            next_observation, history, phase_tracker, encoder
        )
        record = telemetry_record(
            step=step,
            base=base,
            phase=phase,
            dynamic_state=dynamic_state,
            history=_history_payload(history, tuple(base)),
            temporal_embedding=embedding,
            action=action,
            info=info,
            decimal_places=plan.hash_decimal_places,
        )
        steps.append(record)
        observation = next_observation
        if terminated or truncated:
            terminal_reason = str(info.get("reason") or "terminated")
            break
    summary = summarize_episode(
        header=header,
        reset=reset,
        steps=steps,
        terminal_reason=terminal_reason,
        decimal_places=plan.hash_decimal_places,
    )
    summary["step_hashes"] = [record["step_sha256"] for record in steps]
    return {"header": header, "reset": reset, "steps": steps, "summary": summary}


def _output_path(root: Path, episode: Mapping[str, Any]) -> Path:
    header = episode["header"]
    scenario = str(header["scenario_name"]).replace("/", "_").replace("\\", "_")
    return (
        root
        / "episodes"
        / str(header["opponent"])
        / f"repeat_{int(header['repeat_index'])}_{header['order_variant']}"
        / f"{scenario}.json"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n",
        encoding="utf-8",
    )


def _markdown(gate: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    lines = [
        "# Global-Advantage P1 Run-in Reproducibility Preflight",
        "",
        f"Source ID: `{gate['source_id']}`",
        "",
        f"Formal result: **`{'PASS' if gate['passed'] else 'NO-GO'}`**.",
        "",
        "| Opponent | Scenario cells | Failed cells |",
        "|---|---:|---:|",
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in gate["comparison_rows"]:
        grouped.setdefault(str(row["opponent"]), []).append(row)
    for opponent, rows in sorted(grouped.items()):
        lines.append(
            f"| {opponent} | {len(rows)} | {sum(not bool(row['passed']) for row in rows)} |"
        )
    lines.extend(
        [
            "",
            "The preflight compares three repeat/order variants per opponent-scenario cell. "
            "It requires equality of reset, full trajectory, boundary, terminal semantics, "
            "and complete telemetry without backend or prediction fallback.",
            "",
            f"Simulation SHA: `{provenance['git_sha']}`",
            f"Config SHA-256: `{provenance['config_sha256']}`",
            f"Scenario manifest SHA-256: `{provenance['scenario_manifest_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def validate_inputs(config_path: Path) -> tuple[GlobalAdvantageP1Plan, dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    config = _load_yaml(config_path)
    plan = build_p1_plan(config)
    protocol = config["global_advantage_p1"]
    runtime_config_path = _repo_path(str(protocol["sources"]["runtime_config"]))
    registry_path = _repo_path(str(protocol["sources"]["runtime_registry"]))
    manifest_path = _repo_path(str(protocol["sources"]["scenario_manifest"]))
    for path, field in (
        (runtime_config_path, "runtime_config"),
        (registry_path, "runtime_registry"),
        (manifest_path, "scenario_manifest"),
    ):
        if not path.is_file():
            raise GlobalAdvantageP1ContractError(f"missing P1 source: {field}")
    expected = protocol["sources"]
    for path, key in (
        (runtime_config_path, "runtime_config_sha256"),
        (registry_path, "runtime_registry_sha256"),
        (manifest_path, "scenario_manifest_sha256"),
    ):
        if _sha256(path) != str(expected[key]).lower():
            raise GlobalAdvantageP1ContractError(f"P1 source hash mismatch: {key}")
    runtime_config = _load_yaml(runtime_config_path)
    registry = _load_yaml(registry_path)
    manifest = _load_yaml(manifest_path)
    frozen_p3 = runtime_config["fixed_contract"]["encoder"]
    expected_p3 = protocol["sources"]
    p3_path = Path(str(frozen_p3["checkpoint"]))
    if (
        p3_path != Path(str(expected_p3["frozen_p3_checkpoint"]))
        or str(frozen_p3["sha256"]).lower()
        != str(expected_p3["frozen_p3_checkpoint_sha256"]).lower()
        or not p3_path.is_file()
        or _sha256(p3_path) != str(expected_p3["frozen_p3_checkpoint_sha256"]).lower()
    ):
        raise GlobalAdvantageP1ContractError("P1 frozen P3 checkpoint contract drifted")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != plan.scenario_count:
        raise GlobalAdvantageP1ContractError("P1 scenario count drifted")
    if manifest.get("source_id") != protocol["sources"]["scenario_manifest_source_id"]:
        raise GlobalAdvantageP1ContractError("P1 scenario manifest source ID drifted")
    return plan, runtime_config, registry, scenarios, Path(protocol["outputs"]["root"])


def run(config_path: Path) -> dict[str, Any]:
    plan, runtime_config, registry, scenarios, output_root = validate_inputs(config_path)
    if output_root.exists():
        raise GlobalAdvantageP1ContractError("P1 output root must be fresh and absent")
    free_gb = shutil.disk_usage(output_root.parent).free / (1024**3)
    if free_gb < plan.min_free_disk_gb:
        raise GlobalAdvantageP1ContractError(
            f"P1 disk gate failed: {free_gb:.1f} GB < {plan.min_free_disk_gb:.1f} GB"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    config = _load_yaml(config_path)
    fixed = runtime_config["fixed_contract"]["encoder"]
    encoder = FrozenP3Encoder(Path(fixed["checkpoint"]), str(fixed["sha256"]))
    episodes: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    try:
        for repeat_index, variant in enumerate(plan.order_variants):
            ordered_scenarios = _variant_order(scenarios, variant)
            for opponent in plan.opponents:
                env = pilot._build_env(runtime_config, registry, opponent)
                reference = pilot._specialist(registry, "run_in_head_on")
                try:
                    for scenario in ordered_scenarios:
                        episode = _run_episode(
                            env=env,
                            reference=reference,
                            encoder=encoder,
                            scenario=scenario,
                            opponent=opponent,
                            repeat_index=repeat_index,
                            order_variant=variant,
                            plan=plan,
                        )
                        path = _output_path(output_root, episode)
                        _write_json(path, episode)
                        summary = dict(episode["summary"])
                        summary["artifact_path"] = str(path)
                        summary["artifact_sha256"] = _sha256(path)
                        episodes.append(summary)
                        artifacts.append(
                            {
                                "path": str(path),
                                "sha256": summary["artifact_sha256"],
                                "size_bytes": path.stat().st_size,
                            }
                        )
                finally:
                    env.close()
        gate = compare_repeats(episodes)
        provenance = {
            "git_sha": _git_value("rev-parse", "HEAD"),
            "git_dirty": _git_value("status", "--porcelain") != "",
            "python": sys.version,
            "platform": platform.platform(),
            "config_sha256": _sha256(config_path),
            "scenario_manifest_sha256": _sha256(
                _repo_path(config["global_advantage_p1"]["sources"]["scenario_manifest"])
            ),
            "runtime_config_sha256": _sha256(
                _repo_path(config["global_advantage_p1"]["sources"]["runtime_config"])
            ),
            "runtime_registry_sha256": _sha256(
                _repo_path(config["global_advantage_p1"]["sources"]["runtime_registry"])
            ),
        }
        _write_json(output_root / "p1_gate_summary.json", gate)
        _write_json(
            output_root / "run_manifest.json",
            {
                "source_id": plan.source_id,
                "mode": "non_learning_runin_reproducibility_preflight",
                "training_permitted": False,
                "heldout_evaluation_permitted": False,
                "episodes": episodes,
                "artifacts": artifacts,
                "gate": gate,
                "provenance": provenance,
            },
        )
        (output_root / "p1_report.md").write_text(
            _markdown(gate, provenance), encoding="utf-8"
        )
        return {"output_root": str(output_root), "gate": gate, "provenance": provenance}
    except Exception:
        failure = {
            "source_id": plan.source_id,
            "status": "runner_exception_output_preserved_for_audit",
            "completed_episode_count": len(episodes),
        }
        _write_json(output_root / "runner_failure_manifest.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        plan, _runtime, _registry, scenarios, output = validate_inputs(args.config)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "source_id": plan.source_id,
                        "mode": "validated_not_executed",
                        "training_permitted": False,
                        "scenario_count": len(scenarios),
                        "repeat_count": len(plan.order_variants),
                        "episode_count": len(scenarios)
                        * len(plan.opponents)
                        * len(plan.order_variants),
                        "output_root": str(output),
                    },
                    indent=2,
                )
            )
            return 0
        result = run(args.config)
        print(json.dumps(result, indent=2))
        return 0 if result["gate"]["passed"] else 3
    except (OSError, ValueError, yaml.YAMLError, GlobalAdvantageP1ContractError) as error:
        print(f"P1 preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
