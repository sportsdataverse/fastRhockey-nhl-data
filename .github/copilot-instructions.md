<!-- DOCTOC SKIP -->

# fastRhockey-nhl-data Copilot Instructions

## Project Context

This repo is the R-side compile stage for NHL play-by-play data. It pulls
per-game JSON from
[`fastRhockey-nhl-raw`](https://github.com/sportsdataverse/fastRhockey-nhl-raw)
(via `raw.githubusercontent.com`), turns each `final/{game_id}.json` into
season-level tables, and uploads `.rds` + `.parquet` files to GitHub
Releases on
[`sportsdataverse-data`](https://github.com/sportsdataverse/sportsdataverse-data)
using `piggyback::pb_upload()`. Downstream, `fastRhockey::load_nhl_*()`
and `fastRhockey::update_nhl_db()` read those releases.

Pipeline:
`NHL API -> fastRhockey-nhl-raw -> fastRhockey-nhl-data [HERE] -> sportsdataverse-data -> fastRhockey`.

Package name (per `DESCRIPTION`): `fastRhockey.nhl.data`. Not on CRAN.

## Repository Workflow

- `main` is the default branch and the only release branch.
- CI entry: `bash scripts/daily_nhl_R_processor.sh -s <YYYY> -e <YYYY>`.
- The shell script loops the year range and calls
  `Rscript R/nhl_data_creation.R -s $i -e $i` per season.
- `R/nhl_data_creation.R` is the compile driver. New datasets are added
  by appending a row to the `DATASETS` tribble at the top of the file
  plus a matching `nhl/<key>/` subdir.
- Compile-side bugs that look like NHL schema drift belong in
  `sportsdataverse/fastRhockey` (parsing functions), not here.

## Build & Development Commands

```sh
# Full daily flow (CI entry point)
bash scripts/daily_nhl_R_processor.sh -s 2026 -e 2026

# Direct R invocation
Rscript R/nhl_data_creation.R -s 2026           # single season: 2025-26
Rscript R/nhl_data_creation.R -s 2024 -e 2026   # range (end-year inclusive)

# Bootstrap / backfill (one-time)
Rscript R/0000_create_fastRhockey_releases_init.R
Rscript R/0001_push_existing_release_data.R

# xG model retraining (rare; lives under hockeyR/ + R/build_xg_model.R)
Rscript R/build_xg_model.R
```

`-s` / `-e` are the **end year** of the season (`2026` = 2025-26). All
compiled filenames embed end year: `play_by_play_2026.rds`,
`nhl_schedule_2026.rds`, etc.

Output paths under `nhl/`:

- `nhl/pbp/play_by_play_{year}.{rds,parquet}` — full PBP
- `nhl/pbp_lite/play_by_play_{year}.{rds,parquet}` — column-pruned PBP
- `nhl/skater_box/skater_box_{year}.{rds,parquet}`
- `nhl/goalie_box/goalie_box_{year}.{rds,parquet}`
- `nhl/team_box/team_box_{year}.{rds,parquet}`
- `nhl/game_info/game_info_{year}.{rds,parquet}`
- `nhl/game_rosters/game_rosters_{year}.{rds,parquet}`
- `nhl/scratches/`, `nhl/three_stars/`, `nhl/linescore/`,
  `nhl/rosters/`, `nhl/schedules/`
- `nhl/nhl_games_in_data_repo.{rds,parquet}` — index consumed by
  `fastRhockey::nhl_schedule(include_data_flags = TRUE)`
- `nhl/nhl_schedule_master.{rds,parquet}` — concatenated schedule

## Release Tags (load-bearing)

Each `DATASETS` row pins a release tag on `sportsdataverse-data`. The
tags listed below are consumed by `fastRhockey::load_nhl_*()` and cannot
be renamed without a coordinated breaking change:

`nhl_pbp_full`, `nhl_pbp_lite`, `nhl_skater_boxscores`,
`nhl_goalie_boxscores`, `nhl_team_boxscores`, `nhl_player_boxscores`,
`nhl_game_info`, `nhl_game_rosters`, `nhl_rosters`, `nhl_schedules`,
`nhl_scoring`, `nhl_penalties`, `nhl_scratches`, `nhl_linescore`,
`nhl_three_stars`, `nhl_shifts`, `nhl_officials`, `nhl_shots_by_period`,
`nhl_shootout`.

## Code Style

- Follow tidyverse style: `snake_case`, 2-space indent.
- Use `cli::cli_alert_*` for status, never bare `message()` / `print()`.
- Parallelism via `furrr::future_map()`; configure the plan inside the
  driver, not at package load.
- Both `.rds` (R-native) and `.parquet` (Arrow) are written for every
  per-season file — keep them in sync.
- `httr::RETRY("GET", ...)` for any raw-cache fetches; rely on
  `jsonlite::fromJSON()` for parse.
- Don't add packages to `Imports:` that aren't on CRAN, on the `Remotes:`
  pins (`sportsdataverse/fastRhockey`,
  `sportsdataverse/sportsdataverse-data`, `ropensci/piggyback`), or
  reachable from the GitHub Actions R setup.

## CI Workflow

`.github/workflows/daily_nhl.yml`:

- Cron: `0 9 * 10-12 *`, `0 9 * 1-4 *`, `0 9 * 5-6 *` (NHL regular
  season + playoffs).
- `repository_dispatch` event-type `daily_nhl_data` — fired by
  `fastRhockey-nhl-raw` after its daily push. The trigger workflow
  regex-greps `commit_message` for two integers (start/end year). The
  raw-side commit message `"NHL Raw Update (Start: 2026 End: 2026)"`
  is therefore load-bearing.
- `workflow_dispatch` inputs: `start_year`, `end_year`. Empty inputs
  fall back to `fastRhockey:::most_recent_nhl_season()`.
- Daily-flow commit message is `"NHL Data Updated (Start: $i End:
  $i)"`. Keep the `Start:` / `End:` integers in the subject for
  any downstream automation parsers.

## Cross-Repo References

- Upstream raw cache: <https://github.com/sportsdataverse/fastRhockey-nhl-raw>
- Parsing + loaders: <https://github.com/sportsdataverse/fastRhockey>
- Shared SDV conventions: <https://github.com/sportsdataverse/fastRhockey/blob/main/CLAUDE.md>

## Conventional Commits

Use `type(scope): description`. Common types: `feat`, `fix`, `chore`,
`ci`, `docs`, `refactor`. Common scopes: `compile`, `xg`, `loader`,
`ci`. Use `type!:` or a `BREAKING CHANGE:` footer for breaking changes
(renaming a release tag, changing season conventions, etc.).

**Important: Never include AI agents or assistants (e.g., Claude,
Copilot, Cursor, GPT, Gemini) as co-authors on commits.** Omit all
`Co-Authored-By` trailers referencing AI tools. This applies whether the
change was generated, refactored, or reviewed with AI assistance — the
human author is the sole attributable contributor.
