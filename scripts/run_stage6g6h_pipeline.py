#!/usr/bin/env python3
"""Portable artifact pipeline for Stage 6G/6H result production.

This orchestrator runs Stage 6G guidance-law limitation probes and/or Stage 6H
gain/bilevel training stages, writes a unified manifest, and produces a
portable artifact bundle that can be verified on another machine.

Example:
    python scripts/run_stage6g6h_pipeline.py \
        --stage stage6g --smoke \
        --output-root outputs/pipeline/stage6g_smoke

    python scripts/run_stage6g6h_pipeline.py \
        --stage stage6h_gain_only \
        --config config/experiment/gain_only_cem.yaml \
        --checkpoint outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt \
        --output-root outputs/pipeline/stage6h_gain_only

    python scripts/run_stage6g6h_pipeline.py \
        --stage all \
        --config config/experiment/proposed_bilevel.yaml \
        --checkpoint outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt \
        --output-root outputs/pipeline/stage6g6h
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from uav_vpp_guidance.common.manifest import RunManifest
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stages import (
    Stage6GGuidanceLimitationProbe,
    Stage6HGainOnly,
    Stage6HBilevelTraining,
)


def load_config(config_path: str) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_stage(stage_name: str, config: dict, output_dir: Path, args) -> dict:
    common_kwargs = {
        "config": config,
        "output_dir": str(output_dir),
        "command_line": sys.argv,
        "config_path": args.config,
    }

    if stage_name == "stage6g":
        stage = Stage6GGuidanceLimitationProbe(
            **common_kwargs,
            smoke=args.smoke,
            allow_incomplete=args.allow_incomplete,
        )
    elif stage_name == "stage6h_gain_only":
        stage = Stage6HGainOnly(**common_kwargs, checkpoint=args.checkpoint)
    elif stage_name == "stage6h_bilevel":
        stage = Stage6HBilevelTraining(**common_kwargs, checkpoint=args.checkpoint)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    result = stage.run()
    return {
        "stage": stage_name,
        "success": result.success,
        "output_dir": result.output_dir,
        "paper_safe": result.manifest.paper_safe if result.manifest else False,
        "errors": result.errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 6G/6H portable artifact pipeline")
    parser.add_argument(
        "--stage",
        choices=["stage6g", "stage6h_gain_only", "stage6h_bilevel", "all"],
        required=True,
        help="Which stage(s) to run.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to frozen PPO checkpoint.")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory.")
    parser.add_argument("--smoke", action="store_true", help="Run Stage 6G in smoke mode.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow Stage 6G to complete with missing cells.")
    parser.add_argument("--skip-bundle", action="store_true", help="Do not create portable artifact bundle.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = {}
    if args.config:
        config = load_config(args.config)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    stages_to_run = []
    if args.stage == "all":
        stages_to_run = ["stage6g", "stage6h_gain_only", "stage6h_bilevel"]
    else:
        stages_to_run = [args.stage]

    pipeline_manifest = RunManifest(
        stage_name="stage6g6h_pipeline",
        stage_version="1.0.0",
        command_line=sys.argv,
        output_dir=str(output_root),
        config_path=args.config,
        resolved_config=copy.deepcopy(config),
    )
    pipeline_manifest.compute_config_hash()

    stage_results = []
    all_success = True
    for stage_name in stages_to_run:
        stage_output = output_root / stage_name
        result_summary = run_stage(stage_name, config, stage_output, args)
        stage_results.append(result_summary)
        if not result_summary["success"]:
            all_success = False
            pipeline_manifest.add_invalid_for_paper_reason(
                f"Stage {stage_name} failed: {result_summary['errors']}"
            )
        elif not result_summary["paper_safe"]:
            pipeline_manifest.add_invalid_for_paper_reason(
                f"Stage {stage_name} is not paper-safe"
            )

    pipeline_manifest.extra["stage_results"] = stage_results
    pipeline_manifest.mark_completed(paper_safe=all_success and pipeline_manifest.paper_safe)
    pipeline_manifest.save(output_root)

    # Create portable bundles for each stage
    bundle_infos = []
    if not args.skip_bundle:
        for stage_name in stages_to_run:
            stage_output = output_root / stage_name
            if stage_output.exists():
                bundle_dir = output_root / "bundles" / stage_name
                bundle = ArtifactBundle(source_dir=stage_output, bundle_dir=bundle_dir)
                bundle_infos.append({"stage": stage_name, "info": bundle.create()})
        if bundle_infos:
            with open(output_root / "bundle_index.json", "w", encoding="utf-8") as f:
                json.dump(bundle_infos, f, indent=2, default=str)

    print("=" * 60)
    print("Stage 6G/6H pipeline complete")
    print(f"Output root: {output_root}")
    print(f"All stages successful: {all_success}")
    print(f"Paper-safe: {pipeline_manifest.paper_safe}")
    for r in stage_results:
        print(f"  {r['stage']}: success={r['success']} paper_safe={r['paper_safe']}")
    if not args.skip_bundle:
        print(f"Bundles: {output_root / 'bundles'}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
