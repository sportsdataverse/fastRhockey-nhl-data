"""Fast hermetic check of the xG training feature recipe (full train is smoke-tested live).

``prepare_training_frame`` must emit exactly the 36 (5v5) / 38 (st) features the canonical
boosters expect, so a retrained model is drop-in for the inference path. The full
grid-search/CV training (``train_xg_models``) is validated on a live season — see the
commit message — since it needs the multi-MB pbp corpus and is slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nhl_data_build.xg_train import prepare_training_frame

FIX = Path(__file__).parent / "fixtures"
GID = 2024020001


def _pbp() -> pl.DataFrame:
    final = json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))
    return pl.DataFrame(final["all_plays"], infer_schema_length=None)


def test_5v5_feature_recipe() -> None:
    f5 = prepare_training_frame(_pbp(), variant="5v5")
    feats = [c for c in f5.columns if c not in ("game_id", "goal")]
    assert len(feats) == 36, f"expected 36 5v5 features, got {len(feats)}"
    assert {"wrist_shot", "last_faceoff", "era_2025_on", "shot_distance", "empty_net"} <= set(feats)
    assert "goal" in f5.columns and f5.height > 0


def test_st_feature_recipe() -> None:
    fst = prepare_training_frame(_pbp(), variant="st")
    feats = [c for c in fst.columns if c not in ("game_id", "goal")]
    assert len(feats) == 38, f"expected 38 st features, got {len(feats)}"
    assert {"total_skaters_on", "event_team_advantage"} <= set(feats)
