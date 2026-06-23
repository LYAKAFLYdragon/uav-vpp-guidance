"""Unified git metadata collection for reproducibility manifests."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Return git metadata for the repository at repo_path.

    Returns:
        dict with keys:
          - commit: full SHA-1 of HEAD (or None if not a git repo)
          - short_commit: first 12 characters of HEAD
          - branch: current branch name
          - dirty: True if working tree has uncommitted changes
          - description: "branch@short_commit-dirty" style string
    """
    cwd = Path(repo_path) if repo_path else Path.cwd()
    info = {
        "commit": None,
        "short_commit": None,
        "branch": None,
        "dirty": None,
        "description": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        short_commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = (
            len(
                subprocess.check_output(
                    ["git", "status", "--short"],
                    cwd=cwd,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            > 0
        )
        info.update(
            {
                "commit": commit,
                "short_commit": short_commit,
                "branch": branch or None,
                "dirty": dirty,
                "description": f"{branch or 'HEAD'}@{short_commit}{'-dirty' if dirty else ''}",
            }
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"Could not collect git info: {exc}")
    return info
