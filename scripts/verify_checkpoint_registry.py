#!/usr/bin/env python3
"""Verify checkpoint registry integrity.

Checks:
  1. Registry YAML is valid and follows schema
  2. All training entries have required fields
  3. All evaluation_methods reference valid training entries
  4. No duplicate checkpoint paths (within each stage)
  5. All checkpoint paths are parseable (seed templates valid)
  6. Optional: physical file existence (--check-existence)

Exit codes:
  0: all checks passed
  1: registry format error
  2: validation error (missing fields, broken references, duplicates)
  3: file existence failure (only with --check-existence)
"""

import argparse
import sys
from pathlib import Path

import yaml


REGISTRY_PATH = Path("config/checkpoint_registry.yaml")


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def load_registry(path: Path) -> dict:
    if not path.exists():
        _error(f"Registry not found: {path}")
        sys.exit(1)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        _error(f"Invalid YAML: {e}")
        sys.exit(1)
    if not isinstance(data, dict):
        _error("Registry root must be a mapping")
        sys.exit(1)
    return data


def check_training_entries(registry: dict) -> int:
    """Validate training section. Returns number of errors."""
    errors = 0
    training = registry.get("training", {})
    if not training:
        _error("Missing 'training' section")
        return 1

    for name, entry in training.items():
        if not isinstance(entry, dict):
            _error(f"Training entry '{name}' must be a mapping")
            errors += 1
            continue

        required = ["output_dir", "checkpoint", "seeds", "description"]
        for field in required:
            if field not in entry:
                _error(f"Training entry '{name}' missing required field '{field}'")
                errors += 1

        if "seeds" in entry and not isinstance(entry["seeds"], list):
            _error(f"Training entry '{name}': 'seeds' must be a list")
            errors += 1

        # Validate seed template
        if "checkpoint" in entry and "{seed}" in entry["checkpoint"] and "seeds" in entry:
            try:
                for s in entry["seeds"]:
                    entry["checkpoint"].format(seed=s)
            except (KeyError, ValueError) as e:
                _error(f"Training entry '{name}': invalid seed template: {e}")
                errors += 1

    _info(f"Training entries: {len(training)} checked")
    return errors


def check_evaluation_methods(registry: dict) -> int:
    """Validate evaluation_methods section. Returns number of errors."""
    errors = 0
    eval_methods = registry.get("evaluation_methods", {})
    training = registry.get("training", {})

    if not eval_methods:
        _warn("No 'evaluation_methods' section found")
        return 0

    for stage, methods in eval_methods.items():
        if not isinstance(methods, dict):
            _error(f"Stage '{stage}' must be a mapping of methods")
            errors += 1
            continue

        seen_paths = {}
        for method_name, method_cfg in methods.items():
            if not isinstance(method_cfg, dict):
                _error(f"Stage '{stage}'.'{method_name}' must be a mapping")
                errors += 1
                continue

            ckpt = method_cfg.get("checkpoint", "")
            if not ckpt:
                _error(f"Stage '{stage}'.'{method_name}' missing 'checkpoint'")
                errors += 1

            # Check duplicate paths within stage
            if ckpt:
                if ckpt in seen_paths:
                    _warn(
                        f"Stage '{stage}': duplicate checkpoint path '{ckpt}' "
                        f"between '{seen_paths[ckpt]}' and '{method_name}'"
                    )
                else:
                    seen_paths[ckpt] = method_name

            # Validate source_training reference
            src = method_cfg.get("source_training")
            if src and src not in training:
                _error(
                    f"Stage '{stage}'.'{method_name}': source_training "
                    f"'{src}' not found in training entries"
                )
                errors += 1

        _info(f"Stage '{stage}': {len(methods)} methods checked")

    return errors


def check_cem_bilevel(registry: dict) -> int:
    """Validate CEM/Bilevel section. Returns number of errors."""
    errors = 0
    cem = registry.get("cem_bilevel", {})
    for name, entry in cem.items():
        if not isinstance(entry, dict):
            _error(f"CEM entry '{name}' must be a mapping")
            errors += 1
            continue
        if "checkpoint" not in entry:
            _error(f"CEM entry '{name}' missing 'checkpoint'")
            errors += 1
        if "gains" not in entry:
            _warn(f"CEM entry '{name}' missing 'gains' field")
    _info(f"CEM/Bilevel entries: {len(cem)} checked")
    return errors


def check_predictors(registry: dict) -> int:
    """Validate predictors section. Returns number of errors."""
    errors = 0
    preds = registry.get("predictors", {})
    for name, entry in preds.items():
        if not isinstance(entry, dict):
            _error(f"Predictor entry '{name}' must be a mapping")
            errors += 1
            continue
        if "checkpoint" not in entry:
            _error(f"Predictor entry '{name}' missing 'checkpoint'")
            errors += 1
    _info(f"Predictor entries: {len(preds)} checked")
    return errors


def check_aliases(registry: dict) -> int:
    """Validate aliases section. Returns number of errors."""
    errors = 0
    aliases = registry.get("aliases", {})
    training = registry.get("training", {})
    for alias, target in aliases.items():
        if target not in training:
            _error(f"Alias '{alias}' → '{target}' not found in training entries")
            errors += 1
    _info(f"Aliases: {len(aliases)} checked")
    return errors


def check_file_existence(registry: dict, check_training_existence: bool = False) -> int:
    """Check whether referenced files actually exist on disk.

    By default only checks evaluation checkpoints (fast).
    With --check-training-existence also checks all training seeds.
    """
    errors = 0
    project_root = Path(__file__).parent.parent

    def _exists(relative_path: str) -> bool:
        return (project_root / relative_path).exists()

    # Check evaluation checkpoints
    eval_methods = registry.get("evaluation_methods", {})
    missing_eval = []
    for stage, methods in eval_methods.items():
        for method_name, method_cfg in methods.items():
            ckpt = method_cfg.get("checkpoint", "")
            if ckpt and not _exists(ckpt):
                missing_eval.append((stage, method_name, ckpt))
            gains = method_cfg.get("gains")
            if gains and not _exists(gains):
                missing_eval.append((stage, f"{method_name} gains", gains))

    if missing_eval:
        _warn(f"Missing evaluation checkpoints ({len(missing_eval)}):")
        for stage, method, path in missing_eval:
            print(f"  [{stage}][{method}] {path}", file=sys.stderr)
        errors += len(missing_eval)

    # Check CEM/Bilevel
    cem = registry.get("cem_bilevel", {})
    for name, entry in cem.items():
        ckpt = entry.get("checkpoint", "")
        if ckpt and not _exists(ckpt):
            _warn(f"Missing CEM checkpoint [{name}]: {ckpt}")
            errors += 1
        gains = entry.get("gains")
        if gains and not _exists(gains):
            _warn(f"Missing CEM gains [{name}]: {gains}")
            errors += 1

    # Check predictors
    preds = registry.get("predictors", {})
    for name, entry in preds.items():
        ckpt = entry.get("checkpoint", "")
        if ckpt and not _exists(ckpt):
            _warn(f"Missing predictor checkpoint [{name}]: {ckpt}")
            errors += 1

    if check_training_existence:
        training = registry.get("training", {})
        missing_train = []
        for name, entry in training.items():
            ckpt_template = entry.get("checkpoint", "")
            seeds = entry.get("seeds", [0])
            for s in seeds:
                resolved = ckpt_template.format(seed=s)
                if not _exists(resolved):
                    missing_train.append((name, s, resolved))
        if missing_train:
            _warn(f"Missing training checkpoints ({len(missing_train)}):")
            for name, seed, path in missing_train:
                print(f"  [{name} seed={seed}] {path}", file=sys.stderr)
            errors += len(missing_train)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify checkpoint registry")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Path to checkpoint_registry.yaml",
    )
    parser.add_argument(
        "--check-existence",
        action="store_true",
        help="Also verify files exist on disk (evaluation checkpoints only)",
    )
    parser.add_argument(
        "--check-training-existence",
        action="store_true",
        help="Also verify training checkpoints exist (all seeds)",
    )
    args = parser.parse_args()

    _info(f"Loading registry from: {args.registry}")
    registry = load_registry(args.registry)

    total_errors = 0
    total_errors += check_training_entries(registry)
    total_errors += check_evaluation_methods(registry)
    total_errors += check_cem_bilevel(registry)
    total_errors += check_predictors(registry)
    total_errors += check_aliases(registry)

    if args.check_existence or args.check_training_existence:
        total_errors += check_file_existence(
            registry, check_training_existence=args.check_training_existence
        )

    if total_errors == 0:
        print("\n" + "=" * 60)
        print("Registry verification PASSED")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print(f"Registry verification FAILED: {total_errors} error(s)")
        print("=" * 60)
        return 2


if __name__ == "__main__":
    sys.exit(main())
