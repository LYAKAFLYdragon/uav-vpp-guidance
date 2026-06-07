"""
Stage 10.1 JSBSim F-16 divergence root-cause diagnosis runner.

Usage:
    python scripts/run_stage10_1_jsbsim_diagnosis.py \
        --config config/experiment/no_prediction_vpp_jsbsim.yaml \
        --methods hold direct_pn low_gain_direct no_prediction gain_only \
        --scenarios smoke_head_on smoke_tail_chase \
        --seeds 0 1 2 \
        --output-dir outputs/stage10_jsbsim_diagnosis
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.evaluation.jsbsim_diagnosis import main  # noqa: E402

if __name__ == "__main__":
    main()
