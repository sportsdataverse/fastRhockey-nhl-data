<!-- DOCTOC SKIP -->

# fastRhockey-nhl-data Copilot Instructions

## Project Context

This repo is the compile stage for NHL play-by-play data. It reads
per-game JSON from
[`fastRhockey-nhl-raw`](https://github.com/sportsdataverse/fastRhockey-nhl-raw),
turns each `final/{game_id}.json` into season-level tables, and uploads
`.parquet` + `.rds` + `.csv` to GitHub Releases on
[`sportsdataverse-data`](https://github.com/sportsdataverse/sportsdataverse-data).
Downstream, `fastRhockey::load_nhl_*()` and
`fastRhockey::update_nhl_db()` read those releases.

**The Python reshaper (`python/nhl_data_build`) is production** as of the
2026-07-21 cutover, and is where the work happens. The R stack
(`R/nhl_data_creation.R`, `scripts/daily_nhl_R_processor.sh`,
`.github/workflows/daily_nhl.yml`) is **de-scheduled but actively
maintained** — dispatch-only, no cron.

**It is NOT retired.** Standing policy (2026-08-03): this repo carries both
pipelines. Python is primary; the R chain is kept as the methodological /
language equivalent; **both move together when either changes.** Adding,
renaming or removing a dataset on one side alone is a defect, and
`tests/test_r_python_parity.py` fails the build for it — it compares the
`DATASETS` registry in `python/nhl_data_build/config.py` against the
`DATASETS` tribble in `R/nhl_data_creation.R`, field by field.

Neither side is automatically authoritative. If the two disagree, that is a
review item: decide which pipeline is methodologically right, then update the
other. Do not "fix" the parity test by editing one registry to match.

Pipeline:
`NHL API -> fastRhockey-nhl-raw -> fastRhockey-nhl-data [HERE] -> sportsdataverse-data -> fastRhockey`.

Package name (per `DESCRIPTION`): `fastRhockey.nhl.data`. Not on CRAN.

## Repository Workflow

- `main` is the default branch and the only release branch.
- CI entry: `.github/workflows/daily_nhl_python.yml`.
- Droplet entry (sdv-orch systemd):
  `bash scripts/daily_nhl_python_processor.sh -s <YYYY> -e <YYYY>`.
- `python/nhl_data_build/` is the compile package. New datasets are added
  in its dataset registry plus a matching `nhl/<key>/` subdir.
- **Two call sites run the same stages** (`nhl_data_build.season` then
  `nhl_data_build.publish.publish_season`): the workflow above and the
  droplet driver. Their git/env plumbing differs on purpose — the runner
  sparse-checks-out nhl-raw and pushes with an explicit token; the
  droplet uses an absolute venv path (systemd's PATH has no `uv`) and
  commits per season. **A change to the stage sequence must land in
  both.**
- Compile-side bugs that look like NHL schema drift belong in
  `sportsdataverse/fastRhockey` (parsing functions), not here.

## Build & Development Commands

```sh
# Full daily flow (droplet entry point)
bash scripts/daily_nhl_python_processor.sh -s 2026 -e 2026

# Direct invocation (from python/)
uv run python -m nhl_data_build.season -s 2026 -e 2026 \
  --final-dir ../../fastRhockey-nhl-raw/nhl/json/final --out-dir ../nhl
uv run python -c "from nhl_data_build.publish import publish_season; publish_season('../nhl', 2026)"

# xG model retraining (rare). Two families — see README.md "Model registry"
# for artifacts, prerequisites and cadence before running either.
Rscript R/build_xg_model.R          # fastRhockey-side models -> models/
Rscript hockeyR/retrain_xg_models.R # vendored hockeyR 5v5 + special-teams
```

Maintained R equivalent (dispatch-only, no cron -- kept in parity with Python):

```sh
bash scripts/daily_nhl_R_processor.sh -s 2026 -e 2026
Rscript R/nhl_data_creation.R -s 2026
# one-time bootstraps, already run:
Rscript R/0000_create_fastRhockey_releases_init.R
Rscript R/0001_push_existing_release_data.R
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

Each row of the `DATASETS` registry in `python/nhl_data_build/config.py`
(kept in lockstep with the R `DATASETS` tribble the maintained R
equivalent reads -- `tests/test_r_python_parity.py` enforces it) pins a release tag on `sportsdataverse-data`. The tags below are
consumed by `fastRhockey::load_nhl_*()` and cannot be renamed without a
coordinated breaking change:

`nhl_pbp_full`, `nhl_pbp_lite`, `nhl_skater_boxscores`,
`nhl_goalie_boxscores`, `nhl_team_boxscores`, `nhl_player_boxscores`,
`nhl_game_info`, `nhl_game_rosters`, `nhl_rosters`, `nhl_schedules`,
`nhl_scoring`, `nhl_penalties`, `nhl_scratches`, `nhl_linescore`,
`nhl_three_stars`, `nhl_shifts`, `nhl_officials`, `nhl_shots_by_period`,
`nhl_shootout`.

## Code Style

Python (`python/nhl_data_build/` — the production surface):

- polars 1.x modern API only; `snake_case`; fully typed new modules.
- Both `.parquet` and `.rds` are written for every per-season file —
  keep them in sync (`.rds` via the sdv-py native writer; `serialize_rds`
  R shims are retired ecosystem-wide).
- Season args are always CLI (`-s`/`-e`), never hardcoded; end-year
  convention throughout.

R (maintained equivalent + the xG retraining sandbox):

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

`.github/workflows/daily_nhl_python.yml` (production):

- Cron: `0 9 * 10-12 *`, `0 9 * 1-4 *`, `0 9 * 5-6 *` (NHL regular
  season + playoffs).
- `repository_dispatch` event-type `daily_nhl_data` — fired by
  `fastRhockey-nhl-raw` after its daily push. The workflow regex-greps
  `client_payload.commit_message` for two integers (start/end year), so
  the raw-side commit message `"NHL Raw Updated (Start: 2026 End: 2026)"`
  is load-bearing.
- **Exactly one workflow may receive `daily_nhl_data`.** It was listed
  here *and* in the R workflow once, and the two compilers raced on the
  same release — `gh release upload --clobber` deletes an asset before
  re-uploading, so each run 404'd on the other's asset mid-flight
  (2026-07-22). The R workflow's `repository_dispatch` was removed.
- `workflow_dispatch` inputs: `start_year`, `end_year`, `publish`. Empty
  years fall back to the date-derived NHL end-year (Aug rollover).
- Droplet-driver commit subject is `"NHL Data Updated (Start: $i End:
  $i)"`; the workflow's is `"ci(data): compile NHL datasets (Python)
  <start>-<end>"`. Keep the year integers in both — downstream
  automation parses them.

`.github/workflows/daily_nhl.yml` is the de-scheduled R workflow: no cron, no
`repository_dispatch`, `workflow_dispatch` only.

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
