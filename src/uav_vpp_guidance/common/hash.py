"""Canonical hashing utilities for configs and files."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def config_sha256(config: Dict[str, Any], truncate: int = 16) -> str:
    """Return a canonical SHA-256 hex digest of a JSON-serializable config.

    The config is serialized with sorted keys and no extra whitespace so the
    hash is stable across equivalent dict orderings.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if truncate:
        return digest[:truncate]
    return digest


def file_sha256(path: Union[str, Path], truncate: int = 0) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if truncate:
        return digest[:truncate]
    return digest


def file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Return a dict with path, size, sha256, and mtime for a file."""
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "sha256": file_sha256(p) if p.exists() else None,
        "mtime": p.stat().st_mtime if p.exists() else None,
    }
