"""Pipeline stage wrapper for Stage 6G guidance-law limitation probe."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ...common.artifact_contract import ArtifactContract
from ...common.hash import file_info
from ..stage_runner import PipelineStage, StageResult


class Stage6GGuidanceLimitationProbe(PipelineStage):
    """Run the Stage 6G.1 probe and produce a portable artifact bundle."""

    stage_name = "stage6g_guidance_limitation_probe"
    stage_version = "1.0.0"

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        command_line: List[str] = None,
        config_path: str = None,
        smoke: bool = False,
        allow_incomplete: bool = False,
    ):
        super().__init__(config, output_dir, command_line=command_line, config_path=config_path)
        self.smoke = smoke
        self.allow_incomplete = allow_incomplete
        self.artifact_contract = ArtifactContract(
            required_files=[
                "run_manifest.json",
                "artifact_contract.json",
                "resolved_config.yaml",
                "raw_episodes.csv",
                "scenario_method_summary.csv",
                "pairwise_mcnemar.csv",
            ],
            required_directories=["cells"],
            optional_files=[
                "paper_safe_claims.md",
                "README_result_block.md",
                "run.log",
            ],
        )

    def _hash_inputs(self):
        """Record hashes for checkpoints and configs referenced by the stage."""
        methods = self.config.get("methods", {})
        for method_name, method_cfg in methods.items():
            ckpt = method_cfg.get("checkpoint")
            if ckpt:
                self.manifest.record_input_file(f"checkpoint_{method_name}", ckpt)
            gains = method_cfg.get("gains_file") or method_cfg.get("gains")
            if isinstance(gains, str):
                self.manifest.record_input_file(f"gains_{method_name}", gains)
        base_configs = self.config.get("scenario_configs", {})
        for scenario, cfg_path in base_configs.items():
            self.manifest.record_input_file(f"config_{scenario}", cfg_path)

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build subprocess command to existing Stage 6G runner
        cmd = [
            sys.executable,
            "scripts/run_stage6g_guidance_limitation_probe.py",
            "--output-dir",
            str(self.output_dir),
        ]
        if self.smoke:
            cmd.append("--smoke")
        if self.allow_incomplete:
            cmd.append("--allow-incomplete")
        if self.config.get("episodes_per_scenario"):
            cmd.extend(["--episodes-per-scenario", str(self.config["episodes_per_scenario"])])
        if self.config.get("eval_seeds"):
            cmd.extend(["--eval-seeds"] + [str(s) for s in self.config["eval_seeds"]])

        self.manifest.command_line = cmd
        self._hash_inputs()
        self.snapshot_config()

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0 and not self.allow_incomplete:
                self.manifest.mark_failed(result.stderr or "Stage 6G runner returned non-zero")
                self.manifest.save(self.output_dir)
                return StageResult(
                    success=False,
                    output_dir=str(self.output_dir),
                    manifest=self.manifest,
                    errors=[result.stderr or "Stage 6G runner returned non-zero"],
                )
        except Exception as exc:
            self.manifest.mark_failed(str(exc))
            self.manifest.save(self.output_dir)
            return StageResult(success=False, output_dir=str(self.output_dir), manifest=self.manifest, errors=[str(exc)])

        # Read the runner's manifest if present and merge key fields
        runner_manifest_path = self.output_dir / "run_manifest.json"
        if runner_manifest_path.exists():
            with open(runner_manifest_path, "r", encoding="utf-8") as f:
                runner_manifest = json.load(f)
            self.manifest.extra["runner_manifest"] = runner_manifest
            # Propagate paper-safe status and any invalid reasons
            if not runner_manifest.get("paper_safe", True):
                self.manifest.paper_safe = False
                for reason in runner_manifest.get("invalid_for_paper_reasons", []):
                    self.manifest.add_invalid_for_paper_reason(reason)

        # Register produced outputs
        for rel in self.artifact_contract.required_files + self.artifact_contract.optional_files:
            path = self.output_dir / rel
            if path.exists():
                self.manifest.record_output_file(rel, path)
        for rel in self.artifact_contract.required_directories:
            path = self.output_dir / rel
            if path.exists():
                self.manifest.artifacts_present[rel] = True

        # Move cells into a stable subdirectory if they are at top level
        cell_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "cells"]
        if cell_dirs:
            cells_dir = self.output_dir / "cells"
            cells_dir.mkdir(exist_ok=True)
            for cell_dir in cell_dirs:
                if cell_dir.name not in {"cells", "bundle"}:
                    shutil.move(str(cell_dir), str(cells_dir / cell_dir.name))

        return self.finalize(success=True, paper_safe=self.manifest.paper_safe)
