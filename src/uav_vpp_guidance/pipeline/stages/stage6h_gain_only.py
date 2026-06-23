"""Pipeline stage wrapper for Stage 6H gain-only CEM optimization."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ...common.artifact_contract import ArtifactContract
from ...common.hash import file_info
from ..stage_runner import PipelineStage, StageResult


class Stage6HGainOnly(PipelineStage):
    """Run gain-only CEM optimization and produce a portable artifact bundle."""

    stage_name = "stage6h_gain_only_cem"
    stage_version = "1.0.0"

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        command_line: List[str] = None,
        config_path: str = None,
        checkpoint: str = None,
    ):
        super().__init__(config, output_dir, command_line=command_line, config_path=config_path)
        self.checkpoint = checkpoint or config.get("checkpoint")
        self.artifact_contract = ArtifactContract(
            required_files=[
                "run_manifest.json",
                "artifact_contract.json",
                "resolved_config.yaml",
                "cem_results.json",
            ],
            optional_files=[
                "summary.md",
                "run.log",
            ],
        )

    def _hash_inputs(self):
        if self.checkpoint:
            self.manifest.record_input_file("policy_checkpoint", self.checkpoint)
        if self.config_path:
            self.manifest.record_input_file("experiment_config", self.config_path)

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "uav_vpp_guidance.training.train_gain_only",
            "--config",
            self.config_path or self.config.get("config_path", ""),
            "--output-dir",
            str(self.output_dir),
        ]
        if self.checkpoint:
            cmd.extend(["--checkpoint", self.checkpoint])
        for key in ["n_iter", "candidates", "elite_ratio", "seeds", "scenarios"]:
            if self.config.get(key) is not None:
                val = self.config[key]
                flag = f"--{key.replace('_', '-')}"
                if isinstance(val, list):
                    cmd.extend([flag] + [str(v) for v in val])
                else:
                    cmd.extend([flag, str(val)])

        self.manifest.command_line = cmd
        self._hash_inputs()
        self.snapshot_config()

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                self.manifest.mark_failed(result.stderr or "Gain-only CEM returned non-zero")
                self.manifest.save(self.output_dir)
                return StageResult(success=False, output_dir=str(self.output_dir), manifest=self.manifest, errors=[result.stderr])
        except Exception as exc:
            self.manifest.mark_failed(str(exc))
            self.manifest.save(self.output_dir)
            return StageResult(success=False, output_dir=str(self.output_dir), manifest=self.manifest, errors=[str(exc)])

        cem_path = self.output_dir / "cem_results.json"
        if cem_path.exists():
            with open(cem_path, "r", encoding="utf-8") as f:
                cem_results = json.load(f)
            self.manifest.extra["cem_results"] = {
                "best_gains": cem_results.get("best_gains"),
                "best_score": cem_results.get("best_score"),
                "n_iterations": cem_results.get("n_iterations"),
                "invalid_for_paper": cem_results.get("invalid_for_paper", False),
            }
            if cem_results.get("invalid_for_paper"):
                self.manifest.add_invalid_for_paper_reason("Gain-only CEM used random policy initialization")

        for rel in self.artifact_contract.required_files + self.artifact_contract.optional_files:
            path = self.output_dir / rel
            if path.exists():
                self.manifest.record_output_file(rel, path)

        return self.finalize(success=True, paper_safe=self.manifest.paper_safe)
