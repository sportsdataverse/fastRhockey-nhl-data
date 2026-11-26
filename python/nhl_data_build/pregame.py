"""Compile the raw pregame captures into ``nhl_pregame_snapshots``: one row per state interval.

Input: ``fastRhockey-nhl-raw``'s ``nhl/pregame/{END_YEAR}/{YYYY-MM-DD}.json``, written by
``python/nhl_raw_03_pregame_snapshot.py`` every 30 minutes on game days. Each file holds the
kept captures of one ET game date (a capture is kept only when the probable goalies or the
injury list changed). Output: ``pregame_snapshots/parquet/nhl_pregame_snapshots_{END_YEAR}.parquet``.

A state is ``(game, team, kind, player, status)``: a probable goalie with ESPN's
``Expected`` / ``Confirmed`` status, or an injured player with ESPN's status (``Out``,
``Day-To-Day``, ...). ``valid_from`` is the first capture that showed it and ``valid_to``
the first capture that no longer did, null while it is still current. A model reads the
table as of its decision time ``t`` (``valid_from <= t < valid_to``), with
``sportsdataverse.espn_snapshots.pregame_state_as_of``; nothing here is known before the
capture that first showed it, so the table cannot leak a later lineup into an earlier read.

ESPN ids are bridged to NHL ids: the event by ET date + the two team abbreviations against
the season schedule, the team by abbreviation, and the player by a folded-name key against
the season roster (team first, then the whole league when the name is unique). Rows that
do not bridge stay in the table with a null NHL id and are counted.

The R twin, ``R/nhl_pregame_snapshots.R``, must write the same table (``tests/test_pregame_r_parity.py``).

    uv run python -m nhl_data_build.pregame -s 2027 \\
        --pregame-dir ../../fastRhockey-nhl-raw/nhl/pregame --out-dir ../nhl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

#: ESPN scoreboard abbreviations that differ from the NHL schedule's. Every other team
#: uses the same three letters in both (checked against the 2026-11-25 slate).
ESPN_TO_NHL_ABBR: dict[str, str] = {"NJ": "NJD", "TB": "TBL", "SJ": "SJS", "LA": "LAK"}

SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int32,
    "game_date": pl.Utf8,
    "game_id": pl.Int64,
    "espn_event_id": pl.Int64,
    "start_time_utc": pl.Datetime("us", "UTC"),
    "team_id": pl.Int64,
    "team_abbr": pl.Utf8,
    "kind": pl.Utf8,
    "player_id": pl.Int64,
    "espn_athlete_id": pl.Int64,
    "player_name": pl.Utf8,
    "status": pl.Utf8,
    "valid_from": pl.Datetime("us", "UTC"),
    "valid_to": pl.Datetime("us", "UTC"),
    "source": pl.Utf8,
}

_STATE = (
    "espn_event_id",
    "espn_team_id",
    "kind",
    "espn_athlete_id",
    "player_name",
    "status",
    "source",
)


def _ts(text: str | None) -> datetime | None:
    """ESPN / capture stamps (``2026-11-26T00:00Z``, ``...:00:04Z``) as aware UTC."""
    if not text:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ" if text.count(":") == 2 else "%Y-%m-%dT%H:%MZ"
    return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)


def name_key(name: str | None) -> str | None:
    """Fold a player name to a join key: accents stripped, lowercase, letters only."""
    if not name:
        return None
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", folded.lower()) or None


def intervals(doc: dict) -> list[dict]:
    """One day's kept captures -> ESPN-keyed state intervals (no NHL ids yet)."""
    captures = sorted(doc.get("captures") or [], key=lambda c: c["captured_at"])
    teams = {}
    for e in doc.get("events") or []:
        teams[e["home_id"]] = e["event_id"]
        teams[e["away_id"]] = e["event_id"]
    open_: dict[tuple, datetime] = {}
    out: list[dict] = []
    for cap in captures:
        at = _ts(cap["captured_at"])
        seen = set()
        for p in cap.get("probables") or []:
            seen.add(
                (
                    p["event_id"],
                    p["team_id"],
                    "probable_goalie",
                    p["athlete_id"],
                    p["athlete_display_name"],
                    p["status"],
                    "espn_scoreboard",
                )
            )
        for i in cap.get("injuries") or []:
            if i["team_id"] in teams:  # only teams playing on this date
                seen.add(
                    (
                        teams[i["team_id"]],
                        i["team_id"],
                        "injury",
                        i["athlete_id"],
                        i["athlete_display_name"],
                        i["status"],
                        "espn_injuries",
                    )
                )
        for state in [s for s in open_ if s not in seen]:
            out.append(dict(zip(_STATE, state)) | {"valid_from": open_.pop(state), "valid_to": at})
        for state in seen:
            open_.setdefault(state, at)
    out += [dict(zip(_STATE, s)) | {"valid_from": v, "valid_to": None} for s, v in open_.items()]
    return out


def compile_day(doc: dict, schedule: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """One raw day file -> rows in :data:`SCHEMA`, ESPN ids bridged to NHL ids."""
    rows = intervals(doc)
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    day = doc["date"]
    nhl = lambda a: ESPN_TO_NHL_ABBR.get(a, a)  # noqa: E731
    ev = pl.DataFrame(
        [
            {
                "espn_event_id": e["event_id"],
                "start_time_utc": _ts(e["event_date"]),
                "home": nhl(e["home_abbreviation"]),
                "away": nhl(e["away_abbreviation"]),
                "home_id": e["home_id"],
                "away_id": e["away_id"],
            }
            for e in doc["events"]
        ],
        schema={
            "espn_event_id": pl.Int64,
            "start_time_utc": pl.Datetime("us", "UTC"),
            "home": pl.Utf8,
            "away": pl.Utf8,
            "home_id": pl.Int64,
            "away_id": pl.Int64,
        },
    )
    games = schedule.filter(pl.col("game_date") == day).select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("home_team_abbr").alias("home"),
        pl.col("away_team_abbr").alias("away"),
    )
    ev = ev.join(games, on=["home", "away"], how="left")
    team_abbr = pl.concat(
        [
            ev.select(espn_team_id="home_id", team_abbr="home"),
            ev.select(espn_team_id="away_id", team_abbr="away"),
        ]
    ).unique()
    season = int(day[:4]) + (1 if int(day[5:7]) >= 9 else 0)
    current = rosters.filter(pl.col("season") == season)
    team_ids = current.group_by("team_abbr").agg(pl.col("team_id").first())
    keyed = rosters.select(
        "season",
        "team_abbr",
        "player_id",
        key=pl.col("full_name").map_elements(name_key, return_dtype=pl.Utf8),
    )
    # this season's team roster first; then the last three seasons' rosters when the name is
    # unique league-wide (an injured player who has not dressed yet is not on this season's).
    # The window keeps a rookie from inheriting a long-retired namesake's id.
    by_team = (
        keyed.filter(pl.col("season") == season)
        .select("team_abbr", "player_id", "key")
        .unique()
        .unique(["team_abbr", "key"], keep="none")
    )
    league = (
        keyed.filter(pl.col("season").is_between(season - 2, season))
        .select("player_id", "key")
        .unique()
        .unique("key", keep="none")
        .rename({"player_id": "league_pid"})
    )

    df = pl.DataFrame(
        rows,
        schema_overrides={
            "valid_from": pl.Datetime("us", "UTC"),
            "valid_to": pl.Datetime("us", "UTC"),
        },
    )
    df = df.with_columns(
        pl.col("espn_event_id", "espn_team_id", "espn_athlete_id").cast(pl.Int64),
        key=pl.col("player_name").map_elements(name_key, return_dtype=pl.Utf8),
    )
    # id dtype agreement before the bridge joins (a mismatch only shows as empty matches)
    for left, right, key in (
        (df, ev, "espn_event_id"),
        (df, team_abbr, "espn_team_id"),
        (team_ids, by_team, "team_abbr"),
    ):
        if left.schema[key] != right.schema[key]:
            raise ValueError(
                f"pregame bridge: {key} is {left.schema[key]} on one side, {right.schema[key]} on the other"
            )
    df = (
        df.join(
            ev.select("espn_event_id", "game_id", "start_time_utc"),
            on="espn_event_id",
            how="left",
            validate="m:1",
        )
        .join(team_abbr, on="espn_team_id", how="left", validate="m:1")
        .join(team_ids, on="team_abbr", how="left", validate="m:1")
        .join(by_team, on=["team_abbr", "key"], how="left", validate="m:1")
        .join(league, on="key", how="left", validate="m:1")
        .with_columns(
            player_id=pl.coalesce("player_id", "league_pid"),
            season=pl.lit(season, dtype=pl.Int32),
            game_date=pl.lit(day),
        )
    )
    return (
        df.select(list(SCHEMA))
        .cast(SCHEMA)
        .sort("game_id", "team_id", "kind", "player_name", "espn_athlete_id", "status", "valid_from")
    )


def compile_season(
    pregame_dir: str | Path, end_year: int, schedule: pl.DataFrame, rosters: pl.DataFrame
) -> tuple[pl.DataFrame, dict]:
    """Every day file of ``end_year`` -> ``(table, report)``; ``report`` counts what did not bridge."""
    files = sorted((Path(pregame_dir) / str(end_year)).glob("*.json"))
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    frames = [compile_day(d, schedule, rosters) for d in docs]
    df = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    return df, coverage(df, docs)


def coverage(df: pl.DataFrame, docs: list[dict], lead: timedelta = timedelta(hours=3)) -> dict:
    """The spec's gate-2 numbers: event bridge rate, goalie coverage at ``lead``, longest probe gap."""
    events = {(d["date"], e["event_id"]) for d in docs for e in d.get("events") or []}
    games = df.unique("espn_event_id")
    t = pl.col("start_time_utc") - lead
    known = df.filter(
        (pl.col("kind") == "probable_goalie")
        & (pl.col("valid_from") <= t)
        & (pl.col("valid_to").is_null() | (t < pl.col("valid_to")))
    )
    both = known.group_by("espn_event_id").agg(pl.col("team_id").n_unique().alias("n")).filter(pl.col("n") == 2)
    gaps = []
    for d in docs:
        stamps = sorted(_ts(s) for s in d.get("probed") or [])
        gaps += [(b - a).total_seconds() / 60 for a, b in zip(stamps, stamps[1:]) if b - a < timedelta(hours=12)]
    return {
        "days": len(docs),
        "events": len(events),
        "events_bridged": games.filter(pl.col("game_id").is_not_null()).height,
        "goalie_both_teams_at_lead": both.height,
        "rows": df.height,
        "rows_no_game_id": df["game_id"].null_count(),
        "rows_no_team_id": df["team_id"].null_count(),
        "rows_no_player_id": df["player_id"].null_count(),
        "max_probe_gap_min": max(gaps) if gaps else None,
    }


#: First day nhl-raw stage 03 captured pregame state.
CAPTURE_START = "2026-11-25"
#: Months the sdv-orch pregame probe and commit deployments run (their crons: ``10-12,1-6``).
PROBE_MONTHS = frozenset({10, 11, 12, 1, 2, 3, 4, 5, 6})
#: A played game date this recent with no capture file means capture is failing now. Older gaps
#: can't be recaptured, so they stop failing the build after this many days.
GAP_WINDOW_DAYS = 3


def missing_capture_dates(game_dates: pl.Series, captured: set[str], today: date) -> list[str]:
    """Played game dates (``YYYY-MM-DD``) from the last ``GAP_WINDOW_DAYS`` days, on or after
    ``CAPTURE_START`` and in ``PROBE_MONTHS``, that have no capture file. Today's slate isn't over,
    so it's never counted."""
    lo = max(CAPTURE_START, (today - timedelta(days=GAP_WINDOW_DAYS)).isoformat())
    days = set(game_dates.cast(pl.String).drop_nulls().unique().to_list())
    return sorted(d for d in days if lo <= d < today.isoformat() and int(d[5:7]) in PROBE_MONTHS and d not in captured)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-s", "--season", type=int, required=True, help="END year (2027 = 2026-27)")
    ap.add_argument("--pregame-dir", required=True)
    ap.add_argument("--out-dir", default="nhl")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if not Path(a.pregame_dir).is_dir():  # a wrong path must fail, not publish nothing
        raise FileNotFoundError(f"pregame dir not found: {a.pregame_dir}")
    sched_path = out / "schedules" / "parquet" / f"nhl_schedule_{a.season}.parquet"
    captured = {p.stem for p in (Path(a.pregame_dir) / str(a.season)).glob("*.json")}
    if sched_path.exists():
        # a recent slate with no capture file means the probes are failing: fail, don't publish stale
        gaps = missing_capture_dates(
            pl.read_parquet(sched_path, columns=["game_date"])["game_date"], captured, datetime.now(timezone.utc).date()
        )
        if gaps:
            raise RuntimeError(f"[pregame] {a.season}: no capture file for played game dates {gaps}")
    if not captured:
        print(f"[pregame] {a.season}: no captures", flush=True)  # seasons before capture began
        return 0
    schedule = pl.read_parquet(sched_path)
    # every season's roster: the league-wide name fallback reaches players who have not dressed this season
    rosters = pl.concat(
        [
            pl.read_parquet(f, columns=["season", "team_abbr", "team_id", "player_id", "full_name"])
            for f in sorted((out / "rosters" / "parquet").glob("rosters_*.parquet"))
        ],
        how="vertical_relaxed",
    )
    df, report = compile_season(a.pregame_dir, a.season, schedule, rosters)
    print(f"[pregame] {a.season}: {report}", flush=True)
    if df.height:
        dest = out / "pregame_snapshots" / "parquet" / f"nhl_pregame_snapshots_{a.season}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
