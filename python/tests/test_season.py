"""Season reader/writer: read final/{gid}.json from a dir -> datasets -> flattened parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nhl_data_build.season import build_season_from_dir, read_final_dir, write_datasets

FIX = Path(__file__).parent / "fixtures"
GID = 2024020001


def _stage_final_dir(tmp_path: Path) -> Path:
    """Stage the fixture as ``{gid}.json`` (the reader keys on the integer game id)."""
    fd = tmp_path / "final"
    fd.mkdir()
    (fd / f"{GID}.json").write_text((FIX / f"final_{GID}.json").read_text(encoding="utf-8"), encoding="utf-8")
    return fd


def test_reader_season_filter(tmp_path: Path) -> None:
    fd = _stage_final_dir(tmp_path)
    assert len(read_final_dir(fd, season_end_year=2025)) == 1  # 2024020001 -> 2024-25
    assert read_final_dir(fd, season_end_year=2024) == []  # wrong season filtered out


def test_build_and_write(tmp_path: Path) -> None:
    fd = _stage_final_dir(tmp_path)
    season = build_season_from_dir(fd, season_end_year=2025)
    assert season["skater_box"].height == 36 and "pbp" in season

    out = tmp_path / "out"
    written = write_datasets(season, out, 2025)
    assert written["skater_box"] == 36
    assert (out / "skater_box" / "parquet" / "skater_box_2025.parquet").exists()
    assert (out / "pbp_lite" / "parquet" / "play_by_play_lite_2025.parquet").exists()

    # nested datasets are flattened on write
    scoring = pl.read_parquet(out / "scoring" / "parquet" / "scoring_2025.parquet")
    assert "firstName.default" in scoring.columns
