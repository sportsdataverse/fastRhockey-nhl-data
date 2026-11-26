"""``nhl_data_build.pregame`` on a real captured day.

``fixtures/pregame_2026-11-25.json`` is the raw file nhl-raw's stage 03 wrote from nine real ESPN
probes of the 2026-11-25 slate (14 games), every 15 minutes from 00:10:47Z to 02:11:06Z; four were
state changes. ``pregame_schedule_2026-11-25.parquet`` is that date's rows of the season schedule
and ``pregame_rosters.parquet`` every roster row (any season) whose folded name appears in the
captures, plus one 2026-27 row per team: exactly the rows the bridge can read. See
``fixtures/PREGAME.md``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from nhl_data_build import pregame

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads((FIX / "pregame_2026-11-25.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table(doc) -> pl.DataFrame:
    return pregame.compile_day(
        doc,
        pl.read_parquet(FIX / "pregame_schedule_2026-11-25.parquet"),
        pl.read_parquet(FIX / "pregame_rosters.parquet"),
    )


def _as_of(df: pl.DataFrame, t) -> pl.DataFrame:
    return df.filter((pl.col("valid_from") <= t) & (pl.col("valid_to").is_null() | (pl.lit(t) < pl.col("valid_to"))))


def test_schema_and_every_event_bridges(table, doc):
    assert dict(table.schema) == pregame.SCHEMA
    assert table["espn_event_id"].n_unique() == len(doc["events"]) == 14
    assert table["game_id"].null_count() == 0 and table["team_id"].null_count() == 0
    assert table.filter(pl.col("kind") == "probable_goalie")["player_id"].null_count() == 0
    # an ESPN id never stands in for an NHL id: NHL player ids are 8-digit 84xxxxx
    assert table["player_id"].drop_nulls().is_between(8_000_000, 9_999_999).all()


def test_intervals_never_overlap_and_end_on_a_capture(table, doc):
    """Gate 3: per (game, team, kind, player) intervals are disjoint, and each starts and ends on a capture."""
    stamps = {pregame._ts(c["captured_at"]) for c in doc["captures"]}
    assert set(table["valid_from"]) <= stamps
    assert set(table["valid_to"].drop_nulls()) <= stamps
    assert (
        (table.filter(pl.col("valid_to").is_not_null()).select(pl.col("valid_to") > pl.col("valid_from")))
        .to_series()
        .all()
    )
    for _, g in table.sort("valid_from").group_by("game_id", "team_id", "kind", "espn_athlete_id"):
        ends = g["valid_to"].to_list()
        starts = g["valid_from"].to_list()
        assert all(e is not None and e <= s for e, s in zip(ends, starts[1:]))


def test_coverage_numbers_on_the_real_day(table, doc):
    """The gate-2 report. Probing began at 00:10Z, after the first puck drop, so few games have a lead."""
    rep = pregame.coverage(table, [doc])
    assert rep == {
        "days": 1,
        "events": 14,
        "events_bridged": 14,
        "goalie_both_teams_at_lead": 0,  # -3 h was before the first probe for every game
        "rows": 128,
        "rows_no_game_id": 0,
        "rows_no_player_id": 8,
        "max_probe_gap_min": pytest.approx(15.1, abs=0.1),
    }
    # puck drop minus 1 h: the three games starting 02:30Z-03:00Z
    assert pregame.coverage(table, [doc], lead=timedelta(hours=1))["goalie_both_teams_at_lead"] == 3


def test_a_goalie_confirmed_after_the_probe_is_expected_before_it(table):
    """NSH Saros went Expected -> Confirmed at the 01:10:53Z probe: two intervals, end on start."""
    saros = table.filter(pl.col("player_name") == "Juuse Saros").sort("valid_from")
    assert saros["status"].to_list() == ["Expected", "Confirmed"]
    assert saros["valid_to"][0] == saros["valid_from"][1]
    assert saros["valid_to"][1] is None


def test_no_state_row_is_duplicated(table):
    key = ["espn_event_id", "team_id", "kind", "espn_athlete_id", "status", "valid_from"]
    assert table.select(key).is_duplicated().sum() == 0


def test_reading_as_of_each_capture_gives_back_that_capture(table, doc):
    """The table is lossless: the state known at a capture's time is exactly what it held."""
    playing = {t for e in doc["events"] for t in (e["home_id"], e["away_id"])}
    for cap in doc["captures"]:
        t = pregame._ts(cap["captured_at"])
        got = {
            (r["kind"], r["espn_athlete_id"], r["status"])
            for r in _as_of(table, t).select("kind", "espn_athlete_id", "status").iter_rows(named=True)
        }
        want = {("probable_goalie", p["athlete_id"], p["status"]) for p in cap["probables"]} | {
            ("injury", i["athlete_id"], i["status"]) for i in cap["injuries"] if i["team_id"] in playing
        }
        assert got == want, cap["captured_at"]


def test_nothing_is_known_before_the_first_capture(table, doc):
    first = min(pregame._ts(c["captured_at"]) for c in doc["captures"])
    assert _as_of(table, first - timedelta(seconds=1)).height == 0
    assert _as_of(table, first).height > 0


def test_an_ambiguous_name_stays_unmatched_rather_than_guessed(table):
    """Two different players share the name 'Elias Pettersson' (both VAN): no id is picked."""
    rosters = pl.read_parquet(FIX / "pregame_rosters.parquet")
    assert rosters.filter(pl.col("full_name") == "Elias Pettersson")["player_id"].n_unique() >= 2
    rows = table.filter(pl.col("player_name") == "Elias Pettersson")
    assert rows.height >= 1 and rows["player_id"].null_count() == rows.height
    assert rows["game_id"].null_count() == 0


def test_name_key_folds_accents_case_and_punctuation():
    assert pregame.name_key("Tim Stützle") == pregame.name_key("tim stutzle") == "timstutzle"
    assert pregame.name_key("Jake O'Brien") == "jakeobrien"
    assert pregame.name_key(None) is None


def test_main_writes_the_season_parquet(tmp_path, doc):
    raw = tmp_path / "raw" / "2027"
    raw.mkdir(parents=True)
    (raw / "2026-11-25.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "nhl"
    (out / "schedules" / "parquet").mkdir(parents=True)
    (out / "rosters" / "parquet").mkdir(parents=True)
    pl.read_parquet(FIX / "pregame_schedule_2026-11-25.parquet").write_parquet(
        out / "schedules" / "parquet" / "nhl_schedule_2027.parquet"
    )
    pl.read_parquet(FIX / "pregame_rosters.parquet").write_parquet(out / "rosters" / "parquet" / "rosters_2027.parquet")
    assert pregame.main(["-s", "2027", "--pregame-dir", str(tmp_path / "raw"), "--out-dir", str(out)]) == 0
    written = pl.read_parquet(out / "pregame_snapshots" / "parquet" / "nhl_pregame_snapshots_2027.parquet")
    assert dict(written.schema) == pregame.SCHEMA and written.height > 0
