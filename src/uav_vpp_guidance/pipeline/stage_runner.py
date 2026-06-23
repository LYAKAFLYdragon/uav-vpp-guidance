"""Base classes for portable pipeline stages."""

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.artifact_contract import ArtifactContract
from ..common.manifest import RunManifest
from ..common.provenance import get_config_overrides, record_config_override


@dataclass
class StageResult:
    """Result of running a pipeline stage."""

    success: bool
    output_dir: Optional[str] = None
    manifest: Optional[RunManifest] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class PipelineStage:
    """Abstract base for a reproducible pipeline stage.

    Subclasses implement `run()` and declare an `artifact_contract`. The base
    class handles manifest creation, config snapshotting, and artifact
    validation.
    """

    stage_name: str = "abstract_stage"
    stage_version: str = "1.0.0"
    artifact_contract: ArtifactContract = field(
        default_factory=lambda: ArtifactContract()
    )

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        command_line: Optional[List[str]] = None,
        config_path: Optional[str] = None,
    ):
        self.config = copy.deepcopy(config)
        self.output_dir = Path(output_dir)
        self.command_line = command_line or sys.argv
        self.config_path = config_path
        self.manifest = RunManifest(
            stage_name=self.stage_name,
            stage_version=self.stage_version,
            command_line=list(self.command_line),
            output_dir=str(self.output_dir),
            config_path=self.config_path,
            resolved_config=copy.deepcopy(self.config),
        )
        self.manifest.compute_config_hash()

    def record_override(self, key: str, new_value: Any, old_value: Any = None, source: Optional[str] = None):
        """Record a config override both in config and in the manifest."""
        src = source or f"{self.stage_name}:stage_runner"
        record_config_override(self.config, key, new_value, old_value=old_value, source=src)
        self.manifest.config_overrides = list(get_config_overrides(self.config))

    def snapshot_config(self, filename: str = "resolved_config.yaml"):
        """Write the resolved config (with provenance) to the output dir."""
        import yaml

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False, allow_unicode=True)
        self.manifest.record_output_file(filename, path)
        return path

    def run(self) -> StageResult:
        """Execute the stage. Subclasses must implement."""
        raise NotImplementedError

    def validate(self) -> Dict[str, Any]:
        """Validate produced artifacts against the contract."""
        return self.artifact_contract.validate(
            self.output_dir, manifest=self.manifest.to_dict()
        )

    def finalize(self, success: bool, paper_safe: bool = True) -> StageResult:
        """Write manifest, validate, and return a StageResult."""
        if success:
            self.manifest.mark_completed(paper_safe=paper_safe)
        else:
            self.manifest.mark_failed("Stage run failed")

        self.artifact_contract.save(self.output_dir)
        self.manifest.save(self.output_dir)
        validation = self.validate()
        if not validation["valid"]:
            self.manifest.add_invalid_for_paper_reason(
                f"Artifact contract violation: missing {validation['missing']}"
            )
            self.manifest.save(self.output_dir)

        return StageResult(
            success=success and validation["valid"],
            output_dir=str(self.output_dir),
            manifest=self.manifest,
            artifacts=dict(self.manifest.outputs),
            errors=validation["missing"] if not validation["valid"] else [],
        )
