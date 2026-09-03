"""Contract tests for the game_rosters handedness join.

Offline: the index is written to a tmp parquet, so nothing here reaches the
network. What is pinned is the join contract that makes the enrichment safe to
run inside a season build -- the dtype boundary, the refusal to lose roster rows,
and the refusal to change the dataset's shape depending on whether the index
happened to be reachable.
"""

from __future__ import annotations

import polars as pl
import pytest
from nhl_data_build.player_bio import SCHEMA, attach_player_bio, coverage, load_player_bio

BIO = pl.DataFrame(
    {
        "player_id": ["8478402", "8471214"],
        "shoots_catches": ["L", "L"],
        "position_code_bio": ["C", "C"],
    }
)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64])
def test_join_matches_on_either_roster_id_width(dtype):
    """Roster ``player_id`` is Int32 in older seasons and Int64 in newer ones.

    Both must match: the boundary casts the INTEGER to Utf8, so neither width --
    nor a float, which would stringify as ``8478402.0`` and match nothing -- can
    silently produce an all-null column.
    """
    rosters = pl.DataFrame({"player_id": pl.Series([8478402, 8471214], dtype=dtype)})
    out = attach_player_bio(rosters, BIO)
    assert out["shoots_catches"].to_list() == ["L", "L"]
    assert out.schema["player_id"] == dtype  # the join must not rewrite the roster's own column


def test_unmatched_roster_row_survives_with_a_null():
    """A left join, not an inner one: an uncaptured player keeps its roster row.

    Dropping roster rows to satisfy the join would turn an enrichment into
    silent data loss in the published dataset.
    """
    rosters = pl.DataFrame({"player_id": [8478402, 9999999]})
    out = attach_player_bio(rosters, BIO)
    assert out.height == 2
    assert out["shoots_catches"].to_list() == ["L", None]
    assert coverage(out) == {"rows": 2, "matched": 1, "pct": 50.0}


def test_missing_index_still_yields_the_column():
    """An unreachable index nulls the column -- it does not fail the season build
    and does not change the dataset's column set."""
    rosters = pl.DataFrame({"player_id": [8478402]})
    out = attach_player_bio(rosters, pl.DataFrame(schema=SCHEMA))
    assert out["shoots_catches"].to_list() == [None]
    assert out.schema["shoots_catches"] == pl.Utf8


def test_unreachable_source_is_empty_not_an_error(tmp_path):
    assert load_player_bio(tmp_path / "nope.parquet").height == 0


def test_index_position_cannot_shadow_the_rosters_own(tmp_path):
    """The index ships ``position_code``; the roster frame already uses that name
    for the player's position IN THAT GAME. The loader renames it on read, so a
    join can never overwrite the roster's value."""
    p = tmp_path / "bio.parquet"
    pl.DataFrame({"player_id": ["1"], "shoots_catches": ["R"], "position_code": ["D"]}).write_parquet(p)
    bio = load_player_bio(p)
    assert "position_code" not in bio.columns
    assert bio["position_code_bio"].to_list() == ["D"]
