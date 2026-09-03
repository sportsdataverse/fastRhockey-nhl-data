"""``artifact_calibration`` must actually re-score the frame it is handed.

A calibration probe that returns the same numbers whatever frame it sees would hide
exactly the defect it exists to catch, so the test asserts the output MOVES when the
input frame changes (dropping ``MISSED_SHOT``), not merely that the call succeeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from nhl_data_build.xg_parity import artifact_calibration

FIX = Path(__file__).parent / "fixtures"
MODELS = Path(__file__).resolve().parents[1] / "models"
GID = 2024020001


def _pbp() -> pl.DataFrame:
    final = json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))
    return pl.DataFrame(final["all_plays"], infer_schema_length=None)


def _booster(variant: str):
    xgb = pytest.importorskip("xgboost")
    path = MODELS / f"xg_model_{variant}.json"
    if not path.is_file():
        pytest.skip(f"{path} not committed")
    b = xgb.Booster()
    b.load_model(str(path))
    return b


def test_artifact_calibration_scores_the_current_frame() -> None:
    out = artifact_calibration(_pbp(), _booster("5v5"), variant="5v5")
    assert out["n"] > 0 and out["sum_xg"] > 0
    assert out["ratio"] is not None and 0.0 <= out["ratio"] < 10.0
    assert out["per_season"], "per-season table empty — the drift statistic would be unavailable"
    assert out["missing_features"] == [], f"frame lacks booster features: {out['missing_features']}"


def test_artifact_calibration_moves_with_the_frame() -> None:
    pbp = _pbp()
    full = artifact_calibration(pbp, _booster("5v5"), variant="5v5")
    fewer = artifact_calibration(pbp.filter(pl.col("event_type") != "MISSED_SHOT"), _booster("5v5"), variant="5v5")
    assert fewer["n"] < full["n"], "dropping MISSED_SHOT changed no rows — fixture cannot detect drift"
    assert fewer["sum_xg"] < full["sum_xg"], "ΣxG did not move with the frame — probe is a no-op"
