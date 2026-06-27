"""Publish mapping + dry-run (no network) — every dataset maps to an nhl_* release tag."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nhl_data_build.config import DATASETS
from nhl_data_build.publish import _PUBLISH, publish_season


def test_publish_covers_all_datasets() -> None:
    keys = {key for _prefix, _tag, key in _PUBLISH}
    assert {k for k, _f, _p, _t in DATASETS} <= keys  # every registry dataset
    assert {"pbp_lite", "player_box"} <= keys  # + the derived ones
    assert all(tag.startswith("nhl_") for _p, tag, _k in _PUBLISH)


def test_publish_season_dry_run(tmp_path: Path) -> None:
    for prefix, key in (("skater_box", "skater_box"), ("play_by_play_lite", "pbp_lite")):
        d = tmp_path / key / "parquet"
        d.mkdir(parents=True)
        pl.DataFrame({"x": [1]}).write_parquet(d / f"{prefix}_2025.parquet")
    done = publish_season(tmp_path, 2025, dry_run=True)
    tags = {t for t, _name in done}
    assert "nhl_skater_boxscores" in tags
    assert "nhl_pbp_lite" in tags
