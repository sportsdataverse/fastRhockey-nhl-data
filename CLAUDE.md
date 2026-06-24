# CLAUDE.md — fastRhockey-nhl-data Development Guide

## Repo Overview

`fastRhockey.nhl.data` (package name on `DESCRIPTION`) is the R-side
parser/compiler that turns per-game NHL JSON from
[fastRhockey-nhl-raw](https://github.com/sportsdataverse/fastRhockey-nhl-raw)
into season-level compiled datasets and uploads them as GitHub Releases on
[sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data).
The package depends on `sportsdataverse/fastRhockey` for parsing helpers and
`ropensci/piggyback` for release uploads. This repo is not on CRAN — it is a
data-processing workspace whose job is to read raw, compile clean, and push
to releases.

The downstream `load_nhl_*()` and `update_nhl_db()` helpers in `fastRhockey`
read from those releases via piggyback URLs, so the per-dataset release tags
listed below are load-bearing.

## Pipeline Position

```
NHL API --[python scrape]--> fastRhockey-nhl-raw
                                    | raw JSON
                                    v
                              fastRhockey-nhl-data [HERE]
                                    | release upload (piggyback)
                                    v
                              sportsdataverse-data (GitHub Releases)
                                    |
                                    v
                              fastRhockey::load_nhl_*()
```

The compile job pulls per-game `final/{game_id}.json` from
`https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main`,
extracts each per-game JSON field into its corresponding season table, and
uploads `.rds` + `.parquet` files to the release tags below using
`piggyback::pb_upload()`.

## Build & Development Commands

The repo is driven by `scripts/daily_nhl_R_processor.sh`, which calls
`R/nhl_data_creation.R` for each season in a range:

```sh
# Daily flow for a single end-year season (the CI entry point)
bash scripts/daily_nhl_R_processor.sh -s 2026 -e 2026

# Range of seasons
bash scripts/daily_nhl_R_processor.sh -s 2024 -e 2026

# Call the R script directly when iterating
Rscript R/nhl_data_creation.R -s 2026           # single season: 2025-26
Rscript R/nhl_data_creation.R -s 2024 -e 2026   # range: 2023-24 .. 2025-26
```

**Season convention**: `-s` / `-e` are the *end year* of the season (2026 =
2025-26). All compiled dataset filenames embed the end year:
`play_by_play_{end_year}.rds`, `nhl_schedule_{end_year}.rds`, etc.

Other R scripts in `R/`:

```r
# One-time bootstrap: create release tags on sportsdataverse-data
Rscript R/0000_create_fastRhockey_releases_init.R

# One-time backfill: push pre-existing data into the new release tags
Rscript R/0001_push_existing_release_data.R

# xG model retraining (rare; consumed by fastRhockey::helper_nhl_calculate_xg)
Rscript R/build_xg_model.R

# Re-compress old PBP RDS files to current schema (rare)
Rscript R/compress_pbp_data.R
```

The `hockeyR/` subdirectory holds a vendored copy of the `hockeyR` package
sources used for xG model retraining; it is not built/installed by this
repo's normal flow.

## Repo Layout

```
R/
  nhl_data_creation.R                # Main compile driver (one season -> 15 datasets)
  0000_create_fastRhockey_releases_init.R   # Bootstrap release tags
  0001_push_existing_release_data.R         # Backfill historical seasons into releases
  build_xg_model.R                   # Retrain XGBoost xG models -> models/
  compress_pbp_data.R                # Re-compress legacy PBP files
scripts/
  daily_nhl_R_processor.sh           # CI entry point; loops seasons + commits/pushes
nhl/                                 # Committed compiled output (one folder per dataset)
  pbp/, pbp_lite/, skater_box/, goalie_box/, player_box/, team_box/,
  game_info/, game_rosters/, rosters/, schedules/, scratches/,
  three_stars/, linescore/
  nhl_games_in_data_repo.{rds,parquet}   # Master "what we have" index used by fastRhockey
  nhl_schedule_master.{rds,parquet}       # Concatenated schedule across all seasons
hockeyR/                             # Vendored hockeyR package (for xG retraining only)
data/                                # Cached CV results from xG retraining
models/                              # Trained XGBoost JSON models (consumed by fastRhockey)
.github/workflows/daily_nhl.yml      # CI cron + repository_dispatch + workflow_dispatch
```

## Compiled Datasets

`R/nhl_data_creation.R` defines a `DATASETS` tribble at the top of the file
that drives compilation. Each row is `(key, json_field, file_prefix,
release_tag, description)`. The release tags below are the canonical names
on `sportsdataverse-data` and must match what `fastRhockey::load_nhl_*()`
expects:

| Key                | File prefix           | Release tag                  |
|--------------------|------------------------|------------------------------|
| `pbp`              | `play_by_play`         | `nhl_pbp_full`               |
| `skater_box`       | `skater_box`           | `nhl_skater_boxscores`       |
| `goalie_box`       | `goalie_box`           | `nhl_goalie_boxscores`       |
| `team_box`         | `team_box`             | `nhl_team_boxscores`         |
| `game_info`        | `game_info`            | `nhl_game_info`              |
| `game_rosters`     | `game_rosters`         | `nhl_game_rosters`           |
| `shifts`           | `shifts`               | `nhl_shifts`                 |
| `scoring`          | `scoring`              | `nhl_scoring`                |
| `penalties`        | `penalties`            | `nhl_penalties`              |
| `scratches`        | `scratches`            | `nhl_scratches`              |
| `linescore`        | `linescore`            | `nhl_linescore`              |
| `three_stars`      | `three_stars`          | `nhl_three_stars`            |
| `officials`        | `officials`            | `nhl_officials`              |
| `shots_by_period`  | `shots_by_period`      | `nhl_shots_by_period`        |
| `shootout`         | `shootout_summary`     | `nhl_shootout`               |

Add a new compiled dataset by appending one row to `DATASETS` and creating
the matching `nhl/<key>/` subdirectory; the rest of the compile loop is
data-driven. The corresponding loader on the `fastRhockey` package side
(`load_nhl_<key>()`) also needs a new catalog row in
`R/nhl_loaders.R`.

A "lite" play-by-play (`nhl_pbp_lite`) is generated from the full PBP by
dropping a configurable set of columns; it shares the same compile path as
`pbp` but writes to `nhl/pbp_lite/`.

## Daily CI Workflow

`.github/workflows/daily_nhl.yml`:

- **Cron cadence**:
  - `0 9 * 10-12 *` — regular season (Oct-Dec)
  - `0 9 * 1-4 *`   — regular season (Jan-Apr)
  - `0 9 * 5-6 *`   — playoffs (May-Jun)
- **`repository_dispatch`** event type `daily_nhl_data` — fired by
  `fastRhockey-nhl-raw` after its daily push. The dispatch payload's
  `commit_message` is regex-grepped for two integers, which become
  `START_YEAR` / `END_YEAR`. The raw-side commit format is `"NHL Raw
  Updated (Start: 2026 End: 2026)"` — the regex (`grep -o -E '[0-9]+'`,
  `head -1` / `tail -1`) only needs the first/last integers in the
  subject, so keep those two years present and outermost when changing
  the raw-side message.
- **`workflow_dispatch`** inputs: `start_year`, `end_year` strings.
- Empty inputs fall back to `fastRhockey:::most_recent_nhl_season()`.
- Calls `bash scripts/daily_nhl_R_processor.sh -s $START_YEAR -e $END_YEAR`.

The shell script commits with `"NHL Data Updated (Start: $i End: $i)"` per
season. That message format may also be parsed by downstream automation;
keep the `Start:`/`End:` integers in the subject.

## Conventions

- **Season is end year** everywhere user-facing. `2026` means the 2025-26
  season. File names embed end year only.
- **Compile script must be idempotent**. Re-running for a season should
  produce byte-identical output (modulo the timestamp embedded in the S3
  class via `make_fastRhockey_data()`).
- **Both `.rds` and `.parquet`** are written for every per-season file.
  The `.rds` is the master cached copy on disk; the parquet is for
  Python/Arrow consumers downstream.
- **CLI messaging** uses `cli::cli_alert_info/success/danger`. No `print()`
  or bare `message()` for status updates.
- **Parallelism** is via `furrr::future_map()` over `plan(multisession)`
  or `plan(multicore)` depending on environment. The compile script
  configures this near the top.
- **Schema drift is fastRhockey's problem, not this repo's.** If the NHL
  Web API drops or renames a field, the fix belongs in
  `sportsdataverse/fastRhockey` parsing functions. This repo only orchestrates.
- **Don't touch `hockeyR/`** during normal compile work; it is the
  xG-retraining sandbox only.

## Cross-Repo References

- Upstream raw cache: <https://github.com/sportsdataverse/fastRhockey-nhl-raw>
- Parsing functions + loaders: <https://github.com/sportsdataverse/fastRhockey>
- Release destination: <https://github.com/sportsdataverse/sportsdataverse-data>
- Shared SDV conventions: <https://github.com/sportsdataverse/fastRhockey/blob/main/CLAUDE.md>

## Project-Specific Gotchas

- The `Remotes:` field in `DESCRIPTION` pins
  `sportsdataverse/fastRhockey`, `sportsdataverse/sportsdataverse-data`,
  and `ropensci/piggyback`. CI installs from those; do not add packages
  here that are not present in those upstreams or on CRAN.
- The compile script reads raw JSON from `raw.githubusercontent.com`, not
  from a local clone of `fastRhockey-nhl-raw`. CI race conditions between
  the raw push landing and the data compile starting are avoided by the
  cron offset (raw scrapes earlier, data compile at `0 9 UTC`) and by
  the `repository_dispatch` mechanism described above.
- `nhl/nhl_games_in_data_repo.{rds,parquet}` is the index used by
  `fastRhockey::nhl_schedule(include_data_flags = TRUE)` to annotate
  live schedules with "do we have this in releases yet?" Adding a new
  dataset means adding a column to this index, and updating the
  `include_data_flags` consumer on the `fastRhockey` side to read it.
- Releases on `sportsdataverse-data` are append-only per season — the
  per-season asset is overwritten on re-compile, but the release tag
  itself stays put. Renaming a release tag is a breaking change to all
  downstream `load_nhl_*()` consumers.
- The xG retraining flow (`build_xg_model.R` + `hockeyR/`) writes JSON
  models into `models/` that `fastRhockey::.onLoad()` downloads on
  package load. Don't bump the model schema without coordinating with
  `helper_nhl_prepare_xg_data()` in `fastRhockey`.

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for
hand-authored changes:

```
feat(compile): add nhl_officials dataset row to DATASETS tribble
fix(compile): handle empty all_plays field in playoff games
chore(deps): bump fastRhockey pin in DESCRIPTION Remotes
ci: align cron windows with NHL playoff calendar
```

The **daily CI commit** uses the load-bearing umbrella format
`"NHL Data Updated (Start: <year> End: <year>)"` — do not retroactively
re-style those commits or downstream year-parsing will break.

Prefer scoped subjects (`feat(compile): ...`, `fix(xg): ...`). Use
`type!:` or a `BREAKING CHANGE:` footer for breaking changes (renaming
release tags, changing season conventions, etc.).

**Important: Never include AI agents or assistants (e.g., Claude, Copilot,
Cursor, GPT, Gemini) as co-authors on commits.** Omit all
`Co-Authored-By` trailers referencing AI tools. This applies whether the
change was generated, refactored, or reviewed with AI assistance — the
human author is the sole attributable contributor.
