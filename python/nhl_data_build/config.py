"""Dataset registry — Python port of ``nhl_data_creation.R``'s ``DATASETS`` tribble.

Each entry maps a per-game ``final.json`` block to a season-level dataset + its
``sportsdataverse-data`` release tag. ``player_box`` (skater+goalie) and ``pbp_lite``
(pbp minus CHANGE) are derived, not direct blocks.
"""

from __future__ import annotations

# (key, json_field, file_prefix, release_tag, description)
#
# ``description`` is not decoration: it is stamped onto each released rds as the
# ``fastRhockey_type`` attribute, which is what ``print.fastRhockey_data`` shows as the
# header. Keep the strings identical to the R tribble's ~description column.
DATASETS: list[tuple[str, str, str, str, str]] = [
    ("pbp", "all_plays", "play_by_play", "nhl_pbp_full", "NHL play-by-play data (full)"),
    ("skater_box", "skater_stats", "skater_box", "nhl_skater_boxscores", "NHL skater boxscores"),
    ("goalie_box", "goalie_stats", "goalie_box", "nhl_goalie_boxscores", "NHL goalie boxscores"),
    ("team_box", "team_box_parsed", "team_box", "nhl_team_boxscores", "NHL team boxscores"),
    ("game_info", "game_info", "game_info", "nhl_game_info", "NHL game info"),
    ("game_rosters", "rosters", "game_rosters", "nhl_game_rosters", "NHL per-game rosters"),
    ("shifts", "shifts", "shifts", "nhl_shifts", "NHL shifts"),
    ("scoring", "scoring", "scoring", "nhl_scoring", "NHL scoring summary"),
    ("penalties", "penalties", "penalties", "nhl_penalties", "NHL penalty summary"),
    ("scratches", "scratches", "scratches", "nhl_scratches", "NHL scratches"),
    ("linescore", "linescore", "linescore", "nhl_linescore", "NHL linescore"),
    ("three_stars", "decisions", "three_stars", "nhl_three_stars", "NHL three stars / decisions"),
    ("officials", "officials", "officials", "nhl_officials", "NHL on-ice officials"),
    ("shots_by_period", "shots_by_period", "shots_by_period", "nhl_shots_by_period", "NHL shots by period"),
    ("shootout", "shootout", "shootout_summary", "nhl_shootout", "NHL shootout summary"),
]

# key -> fastRhockey_type stamped on the rds. The two derived datasets have no tribble
# row, so their labels are declared here.
TYPES: dict[str, str] = {key: desc for key, _f, _p, _t, desc in DATASETS} | {
    "pbp_lite": "NHL play-by-play data (lite)",
    "player_box": "NHL player boxscores",
}

# Blocks that are a direct list-of-rows in final.json -> attach game_id/season/game_date.
STANDARD_KEYS = {"pbp", "skater_box", "goalie_box", "team_box", "game_info", "game_rosters", "shifts"}
