"""Artifact contract: declare required files and validate a run directory."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ArtifactContract:
    """Declares the artifacts a stage promises to produce."""

    required_files: List[str] = field(default_factory=list)
    required_directories: List[str] = field(default_factory=list)
    optional_files: List[str] = field(default_factory=list)
    schema_version: str = "1.0.0"

    def validate(
        self, output_dir: Union[str, Path], manifest: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate that required artifacts exist.

        Returns a dict with valid: bool, missing: list, present: list, and
        manifest_consistent: bool.
        """
        root = Path(output_dir)
        present = []
        missing = []

        for rel in self.required_files:
            if (root / rel).exists():
                present.append(rel)
            else:
                missing.append(rel)

        for rel in self.required_directories:
            if (root / rel).is_dir():
                present.append(rel)
            else:
                missing.append(rel)

        manifest_consistent = True
        manifest_missing = []
        if manifest is not None:
            artifacts_present = manifest.get("artifacts_present", {})
            for rel in self.required_files + self.required_directories:
                if artifacts_present.get(rel) is False:
                    manifest_consistent = False
                    manifest_missing.append(rel)

        valid = len(missing) == 0 and manifest_consistent
        return {
            "valid": valid,
            "missing": missing,
            "present": present,
            "optional_present": [
                rel for rel in self.optional_files if (root / rel).exists()
            ],
            "manifest_consistent": manifest_consistent,
            "manifest_missing": manifest_missing,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "required_files": self.required_files,
            "required_directories": self.required_directories,
            "optional_files": self.optional_files,
        }

    def save(self, output_dir: Union[str, Path], filename: str = "artifact_contract.json"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
