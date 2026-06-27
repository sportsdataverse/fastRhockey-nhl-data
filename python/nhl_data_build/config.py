"""Dataset registry — Python port of ``nhl_data_creation.R``'s ``DATASETS`` tribble.

Each entry maps a per-game ``final.json`` block to a season-level dataset + its
``sportsdataverse-data`` release tag. ``player_box`` (skater+goalie) and ``pbp_lite``
(pbp minus CHANGE) are derived, not direct blocks.
"""

from __future__ import annotations

# (key, json_field, file_prefix, release_tag)
DATASETS: list[tuple[str, str, str, str]] = [
    ("pbp", "all_plays", "play_by_play", "nhl_pbp_full"),
    ("skater_box", "skater_stats", "skater_box", "nhl_skater_boxscores"),
    ("goalie_box", "goalie_stats", "goalie_box", "nhl_goalie_boxscores"),
    ("team_box", "team_box_parsed", "team_box", "nhl_team_boxscores"),
    ("game_info", "game_info", "game_info", "nhl_game_info"),
    ("game_rosters", "rosters", "game_rosters", "nhl_game_rosters"),
    ("shifts", "shifts", "shifts", "nhl_shifts"),
    ("scoring", "scoring", "scoring", "nhl_scoring"),
    ("penalties", "penalties", "penalties", "nhl_penalties"),
    ("scratches", "scratches", "scratches", "nhl_scratches"),
    ("linescore", "linescore", "linescore", "nhl_linescore"),
    ("three_stars", "decisions", "three_stars", "nhl_three_stars"),
    ("officials", "officials", "officials", "nhl_officials"),
    ("shots_by_period", "shots_by_period", "shots_by_period", "nhl_shots_by_period"),
    ("shootout", "shootout", "shootout_summary", "nhl_shootout"),
]

# Blocks that are a direct list-of-rows in final.json -> attach game_id/season/game_date.
STANDARD_KEYS = {"pbp", "skater_box", "goalie_box", "team_box", "game_info", "game_rosters", "shifts"}
