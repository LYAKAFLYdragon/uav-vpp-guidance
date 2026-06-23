"""Structured run manifest for reproducible experiment artifacts."""

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .git import get_git_info
from .hash import config_sha256, file_info


SCHEMA_VERSION = "1.0.0"


@dataclass
class RunManifest:
    """Standard run manifest for a reproducible experiment stage.

    This dataclass captures everything needed to verify what was run, with what
    inputs, and what outputs were produced.
    """

    stage_name: str
    stage_version: str = "1.0.0"
    schema_version: str = SCHEMA_VERSION
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "pending"  # pending | running | completed | failed
    command_line: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None
    config_hash: Optional[str] = None
    resolved_config: Optional[Dict[str, Any]] = None
    git_info: Dict[str, Any] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=platform.platform)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts_present: Dict[str, bool] = field(default_factory=dict)
    config_overrides: List[Dict[str, Any]] = field(default_factory=list)
    paper_safe: bool = True
    invalid_for_paper_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc).isoformat()
        if not self.git_info:
            self.git_info = get_git_info()
        if self.working_directory is None:
            self.working_directory = str(Path.cwd())

    def mark_started(self):
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, paper_safe: bool = True):
        self.status = "completed"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.paper_safe = paper_safe and not self.invalid_for_paper_reasons

    def mark_failed(self, reason: str):
        self.status = "failed"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.notes.append(reason)

    def add_invalid_for_paper_reason(self, reason: str):
        self.invalid_for_paper_reasons.append(reason)
        self.paper_safe = False

    def record_input_file(self, name: str, path: Union[str, Path]):
        self.inputs[name] = file_info(path)

    def record_output_file(self, name: str, path: Union[str, Path]):
        self.outputs[name] = file_info(path)
        self.artifacts_present[name] = Path(path).exists()

    def compute_config_hash(self):
        if self.resolved_config is not None:
            self.config_hash = config_sha256(self.resolved_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "stage_version": self.stage_version,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "command_line": self.command_line,
            "working_directory": self.working_directory,
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "git_info": self.git_info,
            "python_version": self.python_version,
            "platform": self.platform,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "artifacts_present": self.artifacts_present,
            "config_overrides": self.config_overrides,
            "paper_safe": self.paper_safe,
            "invalid_for_paper_reasons": self.invalid_for_paper_reasons,
            "notes": self.notes,
            "extra": self.extra,
        }

    def save(self, output_dir: Union[str, Path], filename: str = "run_manifest.json"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        # Update self-presence before writing
        self.artifacts_present[filename] = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RunManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop fields not in __init__
        data.pop("schema_version", None)
        return cls(**data)
