#!/usr/bin/env python3
"""Run the V2-only replay-valid local-action-authority source."""
from __future__ import annotations
import run_thesis_local_action_authority_map_v1 as base
import uav_vpp_guidance.evaluation.thesis_local_action_authority_map as gate

SOURCE_ID = "THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V2"
base.SOURCE_ID = SOURCE_ID
gate.SOURCE_ID = SOURCE_ID

if __name__ == "__main__": raise SystemExit(base.main())
