import json
from pathlib import Path

from scripts.verify_h0_current_asset_retraining_v1 import (
    EXPECTED_H0_SEEDS,
    SOURCE_ID,
    verify,
)


def test_asset_verifier_rejects_missing_assets(tmp_path: Path):
    result = verify(tmp_path, tmp_path / "missing_predictor.pt")
    assert result["ok"] is False
    assert any("checkpoint" in error or "predictor" in error for error in result["errors"])


def test_asset_contract_constants_are_independent():
    assert SOURCE_ID.endswith("20260903")
    assert EXPECTED_H0_SEEDS == (9201, 9202, 9203)
