"""Hermetic parity: Python reshaper vs R-produced season parquet (one-game slices).

Feeds the ``final.json`` the Python scraper produces and asserts the extracted datasets
reproduce R's ``nhl/{key}/parquet/*`` output (sliced to game 2024020001). Pilot covers the
standard-case datasets; the special cases (scoring/penalties/linescore/...) land next.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from nhl_data_build.build import build_season

FIX = Path(__file__).parent / "fixtures"
GID = 2024020001


def _season() -> dict[str, pl.DataFrame]:
    final = json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))
    return build_season([final])


def _eq(a: object, b: object) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-4
    return a == b


def _assert_parity(got: pl.DataFrame, oracle: pl.DataFrame, sort_key: list[str]) -> None:
    assert got.height == oracle.height, f"rows {got.height} vs {oracle.height}"
    assert set(got.columns) == set(oracle.columns), f"col mismatch {set(got.columns) ^ set(oracle.columns)}"
    g = got.select(oracle.columns).sort(sort_key).to_dicts()
    o = oracle.sort(sort_key).to_dicts()
    for gr, orow in zip(g, o):
        for c in oracle.columns:
            assert _eq(gr[c], orow[c]), f"col {c}: {gr[c]!r} vs {orow[c]!r}"


# Flat datasets (no nested struct columns) — parity directly off extract_all/build_season.
@pytest.mark.parametrize(
    "key,sort_key",
    [
        ("skater_box", ["player_id"]),
        ("goalie_box", ["player_id"]),
        ("team_box", ["team_id"]),
        ("game_info", ["game_id"]),
        ("game_rosters", ["player_id"]),
        ("officials", ["role", "name"]),
        ("scratches", ["id"]),
        ("linescore", ["game_id"]),
        ("shifts", ["game_seconds", "event_team"]),
    ],
)
def test_flat_dataset_parity(key: str, sort_key: list[str]) -> None:
    season = _season()
    oracle = pl.read_parquet(FIX / f"oracle_{key}_{GID}.parquet")
    _assert_parity(season[key], oracle, sort_key)


def test_pbp_full_and_lite_shape() -> None:
    from nhl_data_build.build import pbp_lite

    pbp = _season()["pbp"]
    assert pbp.height == 850, "pbp_full carries CHANGE rows (current all_plays design)"
    assert pbp.filter(pl.col("event_type") == "CHANGE").height == 501
    assert pbp_lite(pbp).height == 349


# NOTE on pbp event-column parity: the reshaper only passes all_plays through (+ game_date),
# so the per-event columns (coords / shot geometry / on-ice / strength / xG / descriptions)
# are validated upstream in fastRhockey-nhl-raw against the *current* final.json (SP-B coords
# 90/90, xG 90/90, on-ice 294/294). The local pbp oracle here predates CHANGE-in-all_plays and
# was scraped from an earlier API snapshot (revised coordinates), so a re-assertion would be
# both redundant and unreliable. The shape test above is the reshaper-side check.
