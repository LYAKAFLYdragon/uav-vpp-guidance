"""Pipeline stage wrapper for Stage 6G guidance-law limitation probe."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

    def _find_runner_output_dir(self, existing_run_dirs: Set[Path]) -> Optional[Path]:
        """Return the runner-created run directory, if one was produced."""
        run_dirs = [
            path
            for path in self.output_dir.glob("run_*")
            if path.is_dir() and path.resolve() not in existing_run_dirs
        ]
        if not run_dirs:
            run_dirs = [
                path
                for path in self.output_dir.glob("run_*")
                if path.is_dir() and (path / "run_manifest.json").exists()
            ]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda path: path.stat().st_mtime)

    def _copy_runner_artifacts(self, runner_output_dir: Path) -> None:
        """Copy timestamped runner outputs into the stable stage directory."""
        top_level_files = [
            "resolved_config.yaml",
            "raw_episodes.csv",
            "scenario_method_summary.csv",
            "pairwise_mcnemar.csv",
            "paper_safe_claims.md",
            "README_result_block.md",
            "run.log",
        ]
        for rel in top_level_files:
            src = runner_output_dir / rel
            if src.exists():
                shutil.copy2(src, self.output_dir / rel)

        runner_manifest = runner_output_dir / "run_manifest.json"
        if runner_manifest.exists():
            shutil.copy2(runner_manifest, self.output_dir / "runner_run_manifest.json")

        cells_dir = self.output_dir / "cells"
        cells_dir.mkdir(exist_ok=True)
        for child in runner_output_dir.iterdir():
            if not child.is_dir():
                continue
            dst = cells_dir / child.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_run_dirs = {
            path.resolve()
            for path in self.output_dir.glob("run_*")
            if path.is_dir()
        }

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

        runner_output_dir = self._find_runner_output_dir(existing_run_dirs)
        if runner_output_dir is not None:
            self._copy_runner_artifacts(runner_output_dir)
            self.manifest.extra["runner_output_dir"] = str(runner_output_dir)

        # Read the runner's manifest if present and merge key fields
        runner_manifest_path = (
            runner_output_dir / "run_manifest.json"
            if runner_output_dir is not None
            else self.output_dir / "run_manifest.json"
        )
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

        return self.finalize(success=True, paper_safe=self.manifest.paper_safe)
