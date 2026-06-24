"""Tests for the structured provenance helper."""

import pytest

from uav_vpp_guidance.common.provenance import (
    get_config_overrides,
    record_config_override,
    record_config_override_if_changed,
)


class TestRecordConfigOverride:
    def test_creates_provenance_block(self):
        config = {}
        record_config_override(config, "backend", "simple")
        assert "provenance" in config
        assert "config_overrides" in config["provenance"]

    def test_records_source_key_new_value(self):
        config = {}
        entry = record_config_override(
            config,
            "backend",
            "jsbsim",
            old_value="simple",
            source="test_script",
        )
        assert entry["source"] == "test_script"
        assert entry["key"] == "backend"
        assert entry["new_value"] == "jsbsim"
        assert entry["old_value"] == "simple"

    def test_get_config_overrides_returns_list(self):
        config = {}
        record_config_override(config, "a", 1)
        record_config_override(config, "b", 2)
        overrides = get_config_overrides(config)
        assert len(overrides) == 2
        assert overrides[0]["key"] == "a"
        assert overrides[1]["key"] == "b"

    def test_get_config_overrides_empty_when_none_recorded(self):
        assert get_config_overrides({}) == []
        assert get_config_overrides({"provenance": {}}) == []

    def test_serializes_nested_values(self):
        config = {}
        record_config_override(config, "gains", {"k_los": [0.1, 3.0]})
        entry = get_config_overrides(config)[0]
        assert entry["new_value"] == {"k_los": [0.1, 3.0]}

    def test_serializes_non_primitive_to_string(self):
        config = {}
        record_config_override(config, "obj", object())
        entry = get_config_overrides(config)[0]
        assert isinstance(entry["new_value"], str)

    def test_if_changed_skips_equal_values(self):
        config = {}
        entry = record_config_override_if_changed(
            config,
            "backend",
            "simple",
            old_value="simple",
            source="test_script",
        )
        assert entry is None
        assert get_config_overrides(config) == []

    def test_if_changed_records_different_values(self):
        config = {}
        entry = record_config_override_if_changed(
            config,
            "backend",
            "jsbsim",
            old_value="simple",
            source="test_script",
        )
        assert entry["key"] == "backend"
        assert entry["old_value"] == "simple"
        assert entry["new_value"] == "jsbsim"
