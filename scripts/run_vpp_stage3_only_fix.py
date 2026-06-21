"""Run only Stage 3 of the VPP curriculum using the new conservative config.

Reuses an existing Stage 2 pursuer checkpoint so we can validate the Stage 3
fix without retraining Stage 1 and Stage 2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_curriculum_adversarial import stage2_or_3_train_target_and_pursuer


def main():
    output_dir = "outputs/adversarial_curriculum_pilot/vpp_stage3_only_fix"
    # Existing Stage 2 pursuer from the original full-gate25 run
    prev_pursuer_ckpt = (
        "outputs/adversarial_curriculum_pilot/vpp_full_gate25_s0/stage2/pursuer/checkpoints/best.pt"
    )

    if not os.path.exists(prev_pursuer_ckpt):
        raise FileNotFoundError(f"Stage 2 pursuer checkpoint not found: {prev_pursuer_ckpt}")

    stage3_ckpt = stage2_or_3_train_target_and_pursuer(
        output_dir=output_dir,
        stage=3,
        target_config_path="config/adversarial/train_target_pilot.yaml",
        pursuer_config_path="config/adversarial/train_pursuer_v2_pilot_aggressive_stage3.yaml",
        prev_pursuer_ckpt=prev_pursuer_ckpt,
        smoke=False,
        seed=0,
        target_difficulty="hard",
    )
    print(f"\nStage 3 only fix complete. Checkpoint: {stage3_ckpt}")


if __name__ == "__main__":
    main()
