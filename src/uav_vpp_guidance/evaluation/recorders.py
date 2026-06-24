"""
Independent JSON recorders for the flight-control comparison benchmark.

Recording logic lives here so that the training/evaluation runner does not
need to inline JSON formatting or trajectory bookkeeping.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .flight_control_metrics import (
    compute_break_turn_metrics,
    compute_multi_waypoint_metrics,
    compute_sustained_turn_metrics,
)


def _tolist(arr):
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


class EpisodeRecorder:
    """
    Records one episode for the flight-control comparison benchmark.

    The produced JSON schema contains the per-step trajectory, task-specific
    statistics, and provenance required by ``export_table3.py`` and
    ``plot_flight_control_figures.py``.
    """

    def __init__(
        self,
        run_id: str,
        task: str,
        controller: str,
        seed: int,
        episode: int,
        config: dict,
        config_sha256: str,
        git_commit: str,
        backend: str = "jsbsim",
        strict_backend: bool = True,
        save_full: bool = True,
    ):
        self.run_id = run_id
        self.task = task
        self.controller = controller
        self.seed = seed
        self.episode = episode
        self.config = config
        self.config_sha256 = config_sha256
        self.git_commit = git_commit
        self.backend = backend
        self.strict_backend = strict_backend
        self.save_full = save_full

        self.trajectory: List[Dict[str, Any]] = []
        self._prev_active_idx = 0

    def record_step(
        self,
        step: int,
        time_s: float,
        own_state: dict,
        target_state: dict,
        info: dict,
        reward: float,
    ) -> None:
        """Append one step to the trajectory."""
        if not self.save_full:
            return

        current_active_idx = info.get("active_waypoint_index", self._prev_active_idx)
        switch_event = current_active_idx != self._prev_active_idx
        self._prev_active_idx = current_active_idx

        point = {
            "step": step,
            "time_s": float(time_s),
            "own_pos_m": _tolist(
                own_state.get("position_m", own_state.get("position_neu"))
            ),
            "target_pos_m": _tolist(
                target_state.get("position_m", target_state.get("position_neu"))
            ),
            "range_m": float(info.get("range_m", np.nan)),
            "heading_deg": float(np.degrees(own_state.get("yaw_rad", 0.0))),
            "speed_mps": float(own_state.get("speed_mps", 250.0)),
            "nz_g": float(own_state.get("nz_g", 1.0)),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "aggressiveness": info.get("aggressiveness"),
            "gain_scale": info.get("gain_scale"),
            "saturation_flag": bool(info.get("saturation_flag", False)),
            "active_waypoint_index": current_active_idx,
            "switch_event": bool(switch_event),
            "switch_events": copy.deepcopy(info.get("switch_events", [])),
            "waypoints": [copy.deepcopy(wp) for wp in info.get("waypoints", [])],
            "completed_waypoints": info.get("completed_waypoints", 0),
            "completed_orbits": info.get("completed_orbits", 0.0),
            "turn_radius_m": info.get("turn_radius_m", np.nan),
            "orbit_direction": info.get("orbit_direction", np.nan),
            "virtual_point_m": _tolist(
                info.get("virtual_point", {}).get("position_neu")
                if info.get("virtual_point") is not None
                else None
            ),
            "reward": float(reward),
        }
        self.trajectory.append(point)

    def _compute_statistics(self) -> Dict[str, Any]:
        if self.task == "multi_waypoint":
            return compute_multi_waypoint_metrics(self.trajectory)
        if self.task == "sustained_turn":
            return compute_sustained_turn_metrics(self.trajectory)
        if self.task == "break_turn":
            return compute_break_turn_metrics(self.trajectory)
        return {}

    def finalize(
        self,
        steps: int,
        total_time_s: float,
        total_reward: float,
        termination_reason: str,
        success: bool,
        final_position_m: Any,
        final_speed_mps: float,
        final_altitude_m: float,
    ) -> Dict[str, Any]:
        """Build the final episode JSON."""
        statistics = self._compute_statistics()

        ep: Dict[str, Any] = {
            "run_id": self.run_id,
            "task": self.task,
            "controller": self.controller,
            "seed": self.seed,
            "episode": self.episode,
            "backend": self.backend,
            "strict_backend": self.strict_backend,
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
            "success": success,
            "termination_reason": termination_reason,
            "steps": steps,
            "total_time_s": total_time_s,
            "total_reward": total_reward,
            "trajectory": self.trajectory if self.save_full else [],
            "statistics": statistics,
        }

        if self.task == "multi_waypoint":
            if self.trajectory:
                ep["completed_waypoints"] = int(
                    self.trajectory[-1].get("completed_waypoints", 0)
                )
                ep["switch_events"] = self.trajectory[-1].get("switch_events", [])
                ep["waypoints"] = self.trajectory[-1].get("waypoints", [])
            else:
                ep["completed_waypoints"] = 0
                ep["switch_events"] = []
                ep["waypoints"] = []
        elif self.task == "sustained_turn":
            ep["completed_orbits"] = float(statistics.get("completed_orbits", 0.0))

        return ep


class RunRecorder:
    """
    Records all episodes in a run and writes aggregate outputs.

    Keeps a list of episode JSONs and, at the end of a run, writes:
      - per-episode JSON files under ``raw/``
      - ``aggregate/episode_records.json`` (all episodes in one file)
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.aggregate_dir = self.run_dir / "aggregate"
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def write(self) -> None:
        """Write every episode JSON and the aggregate JSON."""
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)

        for rec in self.records:
            task = rec["task"]
            controller = rec["controller"]
            seed = rec["seed"]
            episode = rec["episode"]
            out_path = (
                self.raw_dir
                / task
                / controller
                / f"seed_{seed:02d}"
                / f"episode_{episode:03d}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)

        aggregate_path = self.aggregate_dir / "episode_records.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(self.run_dir),
                    "n_episodes": len(self.records),
                    "episodes": self.records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
