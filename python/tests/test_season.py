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


def test_build_season_batch_invariant() -> None:
    # The streaming/batched compile must be invariant to batch_size: splitting games across
    # batch boundaries must not change the per-dataset output. A string-remapped copy is a
    # distinct second game so the boundary actually combines two games' rows.
    import json

    from nhl_data_build.build import build_season

    text = (FIX / f"final_{GID}.json").read_text(encoding="utf-8")
    g1 = json.loads(text)
    g2 = json.loads(text.replace(str(GID), str(GID + 1)))
    games = [g1, g2]
    one = build_season(games, batch_size=1)  # each game its own batch -> cross-batch concat
    big = build_season(games, batch_size=100)  # single batch, no boundary
    assert set(one) == set(big)
    for k in one:
        assert one[k].height == big[k].height, f"{k}: {one[k].height} vs {big[k].height}"
        assert sorted(one[k].columns) == sorted(big[k].columns), k
    # the boundary actually combined two games (more pbp rows than a single game alone)
    assert one["pbp"].height > build_season([g1])["pbp"].height


def test_rds_carries_the_make_fastRhockey_data_stamp(tmp_path: Path) -> None:
    """R's rbindlist_with_attrs reads these attrs off the file — NULL prints a blank header.

    Asserted against the raw RDS stream (names are stored as ASCII) rather than via
    readRDS, so the check needs no R toolchain in CI.
    """
    import gzip

    season = build_season_from_dir(_stage_final_dir(tmp_path))
    out = tmp_path / "out"
    write_datasets(season, out, 2025)

    blob = gzip.decompress((out / "pbp" / "rds" / "play_by_play_2025.rds").read_bytes())
    for token in (
        b"fastRhockey_data",  # the class chain load_nhl_*() expects
        b"tbl_df",
        b"data.table",
        b"data.frame",
        b"fastRhockey_timestamp",
        b"fastRhockey_type",
        b"NHL play-by-play data (full)",  # the key's registry description
    ):
        assert token in blob, f"{token!r} missing from the rds stamp"


def test_derived_datasets_get_their_own_type_label(tmp_path: Path) -> None:
    """pbp_lite/player_box have no tribble row — without an explicit label they'd
    stamp their bare key as the printed header."""
    import gzip

    season = build_season_from_dir(_stage_final_dir(tmp_path))
    out = tmp_path / "out"
    write_datasets(season, out, 2025)

    lite = out / "pbp_lite" / "rds" / "play_by_play_lite_2025.rds"
    assert b"NHL play-by-play data (lite)" in gzip.decompress(lite.read_bytes())
