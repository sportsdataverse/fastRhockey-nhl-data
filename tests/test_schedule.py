"""The compiled schedule must keep the R-written release shape (nhl_schedule_2026.parquet):
same 35 columns in R's order, int32 integers, plain strings."""

from __future__ import annotations

import io

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from nhl_data_build.schedule import add_completion_flags, fetch_schedule, write_schedule

S, I32, B = pa.string(), pa.int32(), pa.bool_()
RAW = [
    ("game_id", I32), ("season_full", S), ("game_type", S), ("game_date", S), ("game_time", S),
    ("home_team_abbr", S), ("away_team_abbr", S), ("home_team_name", S), ("away_team_name", S),
    ("home_score", I32), ("away_score", I32), ("game_state", S), ("venue", S), ("series_letter", S),
    ("playoff_round", I32), ("series_game_number", I32), ("season", I32), ("game_json", B), ("game_json_url", S),
]  # fmt: skip
FLAGS = [
    "PBP", "team_box", "player_box", "skater_box", "goalie_box", "game_info", "game_rosters", "scoring",
    "penalties", "scratches", "linescore", "three_stars", "shifts", "officials", "shots_by_period", "shootout",
]  # fmt: skip


def test_compiled_schedule_keeps_the_r_release_schema(tmp_path) -> None:
    raw = pa.table({n: pa.array([None, None], type=t) for n, t in RAW})
    raw = raw.set_column(0, "game_id", pa.array([2026020001, 2026020002], type=I32))
    raw = raw.set_column(3, "game_date", pa.array(["2026-10-07", "2026-10-08"]))
    buf = io.BytesIO()
    pq.write_table(raw, buf)

    sched = fetch_schedule(2027, opener=lambda url: buf.getvalue())
    season = {"pbp": pl.DataFrame({"game_id": [2026020001]})}
    write_schedule(add_completion_flags(sched, season), tmp_path, 2027)

    got = pq.read_table(tmp_path / "schedules" / "parquet" / "nhl_schedule_2027.parquet")
    assert [(f.name, f.type) for f in got.schema] == RAW + [(f, B) for f in FLAGS]
    pbp = dict(zip(got["game_id"].to_pylist(), got["PBP"].to_pylist()))
    assert pbp == {2026020001: True, 2026020002: False}
