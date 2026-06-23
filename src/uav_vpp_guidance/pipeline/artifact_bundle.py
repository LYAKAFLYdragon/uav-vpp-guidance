"""Portable artifact bundle: collect a stage's outputs into a verifiable unit."""

import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.artifact_contract import ArtifactContract
from ..common.hash import file_info
from ..common.manifest import RunManifest


@dataclass
class ArtifactBundle:
    """A portable bundle of run artifacts.

    Bundles collect the manifest, contract, config snapshot, and key outputs
    into a single directory that can be archived, transferred, and verified on
    another machine.
    """

    source_dir: Union[str, Path]
    bundle_dir: Union[str, Path]
    manifest: Optional[RunManifest] = None
    extra_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.source_dir = Path(self.source_dir)
        self.bundle_dir = Path(self.bundle_dir)

    def create(self) -> Dict[str, Any]:
        """Create the bundle directory with symlinks/copies of artifacts.

        Returns a dict describing the bundle.
        """
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.source_dir / "run_manifest.json"
        if manifest_path.exists():
            self.manifest = RunManifest.load(manifest_path)

        bundle_manifest = copy.deepcopy(self.manifest.to_dict()) if self.manifest else {}
        bundle_files = {}

        # Copy core contract files
        for filename in ["run_manifest.json", "artifact_contract.json", "resolved_config.yaml"]:
            src = self.source_dir / filename
            if src.exists():
                dst = self.bundle_dir / filename
                shutil.copy2(src, dst)
                bundle_files[filename] = file_info(dst)

        # Copy declared outputs
        if self.manifest:
            for name, info in self.manifest.outputs.items():
                src = Path(info["path"])
                if src.exists():
                    dst = self.bundle_dir / src.name
                    # Avoid name collisions by prefixing with output key
                    if dst.exists():
                        dst = self.bundle_dir / f"{name}_{src.name}"
                    shutil.copy2(src, dst)
                    bundle_files[name] = file_info(dst)

        # Copy extra requested files
        for rel in self.extra_files:
            src = self.source_dir / rel
            if src.exists():
                dst = self.bundle_dir / src.name
                shutil.copy2(src, dst)
                bundle_files[rel] = file_info(dst)

        bundle_info = {
            "bundle_dir": str(self.bundle_dir),
            "source_dir": str(self.source_dir),
            "files": bundle_files,
            "manifest": bundle_manifest,
        }
        with open(self.bundle_dir / "bundle_info.json", "w", encoding="utf-8") as f:
            json.dump(bundle_info, f, indent=2, default=str)
        return bundle_info

    def verify(self) -> Dict[str, Any]:
        """Verify bundle integrity against stored hashes.

        Returns validation dict.
        """
        info_path = self.bundle_dir / "bundle_info.json"
        if not info_path.exists():
            return {"valid": False, "error": "bundle_info.json missing"}

        with open(info_path, "r", encoding="utf-8") as f:
            bundle_info = json.load(f)

        mismatches = []
        missing = []
        for name, expected in bundle_info.get("files", {}).items():
            path = self.bundle_dir / Path(expected["path"]).name
            if not path.exists():
                missing.append(name)
                continue
            actual = file_info(path)
            if expected.get("sha256") and actual["sha256"] != expected["sha256"]:
                mismatches.append({"name": name, "expected": expected["sha256"], "actual": actual["sha256"]})

        # Validate artifact contract if present
        contract_path = self.bundle_dir / "artifact_contract.json"
        contract_valid = True
        contract_missing = []
        if contract_path.exists():
            contract = ArtifactContract(**json.load(open(contract_path, "r", encoding="utf-8")))
            validation = contract.validate(self.bundle_dir)
            contract_valid = validation["valid"]
            contract_missing = validation["missing"]

        valid = len(mismatches) == 0 and len(missing) == 0 and contract_valid
        return {
            "valid": valid,
            "missing": missing,
            "mismatches": mismatches,
            "contract_valid": contract_valid,
            "contract_missing": contract_missing,
        }
