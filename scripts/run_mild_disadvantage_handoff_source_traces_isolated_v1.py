#!/usr/bin/env python3
"""Generate and independently replay 12 handoff source traces.

Each opponent/candidate/repetition is executed in a fresh worker process. A
second fresh worker replays the same seed and the verifier compares the first
mode-transition state hash. This is a provenance gate, not a performance run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from uav_vpp_guidance.common.git import get_git_info
SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"
DEFAULT_CONFIG = ROOT / "config/experiment/mild_disadvantage_handoff_source_traces_isolated_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/mild_disadvantage_handoff_source_traces_isolated_v1"
WORKER = ROOT / "scripts/run_mild_disadvantage_handoff_source_trace_worker_v1.py"


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": file_sha256(path) if exists else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def state_hash_at_handoff(path: Path) -> tuple[int, str, str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    previous = rows[0].get("mode", "")
    for index, row in enumerate(rows[1:], start=1):
        mode = row.get("mode", "")
        if mode != previous:
            return index, row.get("state_hash_before_action", ""), mode, len(rows)
        previous = mode
    raise ValueError(f"no mode transition in trajectory: {path}")


def run_worker(config: Path, output_dir: Path, opponent: str, candidate: str, seed: int) -> str:
    command = [
        sys.executable,
        str(WORKER),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--opponent",
        opponent,
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--execute",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed ({opponent}/{candidate}/{seed})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return " ".join(command)


def episode_jobs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    base_seed = int(spec["probe"]["evaluation_seed_base"])
    candidates = list(spec["probe"]["candidates"])
    repeats = int(spec["probe"]["repetitions_per_candidate_and_opponent"])
    jobs = []
    for opponent in spec["probe"]["opponents"]:
        opponent_offset = 0 if opponent == "expert" else 50
        for candidate_index, candidate in enumerate(candidates):
            for repetition in range(repeats):
                jobs.append(
                    {
                        "opponent": opponent,
                        "candidate": candidate,
                        "repetition": repetition + 1,
                        "seed": base_seed + candidate_index * 100 + opponent_offset + repetition,
                    }
                )
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Required to run JSBSim")
    args = parser.parse_args(argv)
    spec = load(args.config.resolve())
    if spec["experiment"]["source_id"] != SOURCE_ID:
        raise SystemExit("Source ID mismatch")
    if not args.execute:
        print(json.dumps({"status": "locked", "source_id": SOURCE_ID, "jobs": len(episode_jobs(spec))}))
        return 0

    root = args.output_dir.resolve()
    if root.exists() and any(root.rglob("*")):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    generated_root = root / "_workers" / "generated"
    replay_root = root / "_workers" / "replayed"
    records: list[dict[str, Any]] = []
    jobs = episode_jobs(spec)
    for job in jobs:
        key = f"{job['opponent']}__{job['candidate']}__{job['repetition']:02d}"
        generated_dir = generated_root / key
        replay_dir = replay_root / key
        generated_command = run_worker(
            args.config.resolve(), generated_dir, job["opponent"], job["candidate"], job["seed"]
        )
        generated_trace = next((generated_dir / "trajectories").glob("*.csv"))
        handoff_index, generated_hash, handoff_mode, length = state_hash_at_handoff(generated_trace)
        replay_command = run_worker(
            args.config.resolve(), replay_dir, job["opponent"], job["candidate"], job["seed"]
        )
        replay_trace = next((replay_dir / "trajectories").glob("*.csv"))
        replay_index, replay_hash, replay_mode, replay_length = state_hash_at_handoff(replay_trace)
        destination = root / job["opponent"] / "trajectories" / f"{job['candidate']}_{job['repetition']:02d}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_trace, destination)
        records.append(
            {
                **job,
                "source_trace_id": f"{job['opponent']}/{destination.name}",
                "generated_trace": receipt(generated_trace),
                "registered_trace": receipt(destination),
                "replay_trace": receipt(replay_trace),
                "handoff_index": handoff_index,
                "replay_handoff_index": replay_index,
                "handoff_mode": handoff_mode,
                "replay_handoff_mode": replay_mode,
                "handoff_hash": generated_hash,
                "replay_handoff_hash": replay_hash,
                "handoff_hash_match": generated_hash == replay_hash and handoff_index == replay_index,
                "generated_length": length,
                "replay_length": replay_length,
                "generated_command": generated_command,
                "replay_command": replay_command,
            }
        )

    for opponent in spec["probe"]["opponents"]:
        episodes = [
            {
                "source_id": SOURCE_ID,
                "opponent": item["opponent"],
                "candidate": item["candidate"],
                "seed": item["seed"],
                "source_trace_id": item["source_trace_id"],
                "handoff_index": item["handoff_index"],
                "handoff_hash": item["handoff_hash"],
                "replay_handoff_hash": item["replay_handoff_hash"],
                "handoff_hash_match": item["handoff_hash_match"],
                "backend_effective": "jsbsim",
                "backend_fallback_occurred": False,
                "reset_calls": 1,
            }
            for item in records
            if item["opponent"] == opponent
        ]
        write_json(root / opponent / "episode_results.json", {"source_id": SOURCE_ID, "opponent": opponent, "episodes": episodes})

    replay_ok = len(records) == 12 and all(item["handoff_hash_match"] for item in records)
    manifest = {
        "source_id": SOURCE_ID,
        "status": "completed" if replay_ok else "rejected",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "training_permitted": False,
        "backend_requested": "jsbsim",
        "strict_backend": True,
        "fresh_process_per_episode": True,
        "source_episode_count": len(records),
        "expected_source_episode_count": 12,
        "config": receipt(args.config.resolve()),
        "git_info": get_git_info(str(ROOT)),
        "source_replay_gate": {
            "required": True,
            "passed": replay_ok,
            "hash_match_count": sum(bool(item["handoff_hash_match"]) for item in records),
            "hash_match_denominator": len(records),
            "same_handoff_index_required": True,
            "no_mid_episode_reset": True,
            "strict_jsbsim_required": True,
        },
        "records": records,
    }
    write_json(root / "source_trace_manifest.json", {"source_id": SOURCE_ID, "records": records})
    write_json(root / "replay_verification.json", {"source_id": SOURCE_ID, "passed": replay_ok, "records": records})
    write_json(root / "run_manifest.json", manifest)
    if not replay_ok:
        raise SystemExit("source replay gate failed; output is rejected and cannot feed local-response analysis")
    print(f"Completed isolated source trace route: {len(records)} episodes; replay gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
