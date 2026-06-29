#!/usr/bin/env python3
"""Generate and run retrospective combat-finetune checkpoint comparisons."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.evaluation.combat_checkpoint_retrospective import main


if __name__ == "__main__":
    raise SystemExit(main())
