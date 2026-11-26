# Pregame snapshot fixtures (`tests/test_pregame.py`)

| File | Source | Content |
|---|---|---|
| `pregame_2026-11-25.json` | fastRhockey-nhl-raw stage 03 (`nhl_raw_03_pregame_snapshot.capture`) run on real ESPN Site v2 probes of `hockey/nhl/scoreboard?dates=20261125` and `hockey/nhl/injuries`, fetched every 15 min from 2026-11-26T00:10:47Z to 02:11:06Z | 9 probes, 4 kept captures, 14 events; ids are ESPN integers |
| `pregame_schedule_2026-11-25.parquet` | `nhl/schedules/parquet/nhl_schedule_2027.parquet`, rows with `game_date == "2026-11-25"` | 14 games; `game_id` Int32 as published |
| `pregame_rosters.parquet` | every `nhl/rosters/parquet/rosters_*.parquet` row whose folded `full_name` appears in the captures, plus one 2026-27 row per team | `season`, `team_abbr`, `team_id`, `player_id` (Int64), `full_name` |
