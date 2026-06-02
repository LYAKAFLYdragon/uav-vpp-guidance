"""
Configuration loading and merging utilities.
"""

import os
import yaml


def load_yaml_config(path):
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(base, override):
    """
    Recursively merge override dict into base dict.

    Args:
        base (dict): Base configuration.
        override (dict): Override configuration.

    Returns:
        dict: Merged configuration.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_paths(config, root_dir="."):
    """
    Resolve relative paths in a configuration to absolute paths.

    Args:
        config (dict): Configuration dictionary.
        root_dir (str): Root directory for relative path resolution.

    Returns:
        dict: Configuration with resolved paths.
    """
    # TODO: Implement path resolution for config_file references.
    return config
