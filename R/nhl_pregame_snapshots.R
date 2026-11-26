#!/usr/bin/env Rscript
# R twin of python/nhl_data_build/pregame.py: compile fastRhockey-nhl-raw's pregame captures
# (nhl/pregame/{END_YEAR}/{YYYY-MM-DD}.json, stage 03 there) into nhl_pregame_snapshots, one
# row per state interval. The two must produce the same table: tests/test_pregame_r_parity.py
# runs both on the real captured day in tests/fixtures/ and compares them column by column.
#
# A state is (game, team, kind, player, status). valid_from is the first capture that showed
# it, valid_to the first capture that no longer did (NA while current); read it as of a time t
# with valid_from <= t < valid_to. ESPN ids are bridged to NHL ids: the event by ET date + both
# team abbreviations, the team by abbreviation, the player by a folded name against this
# season's team roster, then the last three seasons' rosters when the name is unique. Rows
# that do not bridge keep an NA id.
#
#   Rscript R/nhl_pregame_snapshots.R -s 2027 --pregame-dir ../fastRhockey-nhl-raw/nhl/pregame --out-dir nhl
#
# The missing-capture gate (pregame.missing_capture_dates) is Python-only: this rollback twin
# compiles what exists and exits 0.
# Writes nhl/pregame_snapshots/parquet/nhl_pregame_snapshots_{END_YEAR}.parquet. Publishing to
# the release is the Python call sites' job (publish._PUBLISH).

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(arrow)
  library(cli)
})

# ESPN scoreboard abbreviations that differ from the NHL schedule's (python: ESPN_TO_NHL_ABBR)
.pg_abbr <- c(NJ = "NJD", TB = "TBL", SJ = "SJS", LA = "LAK")

.pg_schema <- arrow::schema(
  season = arrow::int32(), game_date = arrow::utf8(), game_id = arrow::int64(),
  espn_event_id = arrow::int64(), start_time_utc = arrow::timestamp("us", "UTC"),
  team_id = arrow::int64(), team_abbr = arrow::utf8(), kind = arrow::utf8(),
  player_id = arrow::int64(), espn_athlete_id = arrow::int64(), player_name = arrow::utf8(),
  status = arrow::utf8(), valid_from = arrow::timestamp("us", "UTC"),
  valid_to = arrow::timestamp("us", "UTC"), source = arrow::utf8()
)

.nz <- function(x) if (is.null(x)) NA else x

# "2026-11-26T00:00Z" / "2026-11-26T00:10:47Z" -> seconds since the epoch (UTC)
.pg_ts <- function(x) {
  x <- unlist(lapply(x, .nz))
  fmt <- ifelse(lengths(regmatches(x, gregexpr(":", x))) == 2, "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ")
  as.numeric(as.POSIXct(x, format = fmt, tz = "UTC"))
}

#' Fold a name to a join key: NFKD, non-ASCII dropped, lowercase, letters only (python: name_key)
pregame_name_key <- function(x) {
  folded <- stringi::stri_trans_nfkd(x)
  folded <- gsub("[^\\x01-\\x7f]", "", folded, perl = TRUE)
  out <- gsub("[^a-z]", "", tolower(folded))
  out[is.na(x) | out == ""] <- NA_character_
  out
}

#' One day's kept captures -> ESPN-keyed state intervals (python: intervals)
pregame_intervals <- function(doc) {
  caps <- doc$captures
  caps <- caps[order(vapply(caps, function(c) c$captured_at, ""))]
  teams <- list()
  for (e in doc$events) {
    teams[[as.character(e$home_id)]] <- e$event_id
    teams[[as.character(e$away_id)]] <- e$event_id
  }
  state <- function(event, team, kind, athlete, name, status, source) {
    list(espn_event_id = .nz(event), espn_team_id = .nz(team), kind = kind,
         espn_athlete_id = .nz(athlete), player_name = .nz(name), status = .nz(status), source = source)
  }
  key <- function(s) paste(vapply(s, as.character, ""), collapse = "\x1f")
  open <- list()
  out <- list()
  for (cap in caps) {
    at <- .pg_ts(cap$captured_at)
    seen <- list()
    for (p in cap$probables) {
      s <- state(p$event_id, p$team_id, "probable_goalie", p$athlete_id, p$athlete_display_name,
                 p$status, "espn_scoreboard")
      seen[[key(s)]] <- s
    }
    for (i in cap$injuries) {
      ev <- if (is.null(i$team_id)) NULL else teams[[as.character(i$team_id)]]
      if (!is.null(ev)) { # only teams playing on this date
        s <- state(ev, i$team_id, "injury", i$athlete_id, i$athlete_display_name, i$status, "espn_injuries")
        seen[[key(s)]] <- s
      }
    }
    for (k in setdiff(names(open), names(seen))) {
      out[[length(out) + 1]] <- c(open[[k]]$state, valid_from = open[[k]]$from, valid_to = at)
      open[[k]] <- NULL
    }
    for (k in setdiff(names(seen), names(open))) open[[k]] <- list(state = seen[[k]], from = at)
  }
  for (k in names(open)) out[[length(out) + 1]] <- c(open[[k]]$state, valid_from = open[[k]]$from, valid_to = NA_real_)
  if (!length(out)) return(NULL)
  dplyr::bind_rows(lapply(out, function(r) {
    tibble::tibble(
      espn_event_id = as.numeric(r$espn_event_id), espn_team_id = as.numeric(r$espn_team_id),
      kind = r$kind, espn_athlete_id = as.numeric(r$espn_athlete_id),
      player_name = as.character(r$player_name), status = as.character(r$status), source = r$source,
      valid_from = as.numeric(r$valid_from), valid_to = as.numeric(r$valid_to)
    )
  }))
}

.pg_unique_on <- function(df, by) {
  if (anyDuplicated(df[by])) cli::cli_abort("pregame bridge: {.val {by}} is not unique on the right side")
  df
}

#' One raw day file -> rows in .pg_schema order, ESPN ids bridged to NHL ids (python: compile_day)
pregame_compile_day <- function(doc, schedule, rosters) {
  rows <- pregame_intervals(doc)
  if (is.null(rows)) return(NULL)
  day <- doc$date
  season <- as.integer(substr(day, 1, 4)) + as.integer(as.integer(substr(day, 6, 7)) >= 9)
  nhl <- function(a) ifelse(a %in% names(.pg_abbr), .pg_abbr[a], a)
  ev <- dplyr::bind_rows(lapply(doc$events, function(e) tibble::tibble(
    espn_event_id = as.numeric(e$event_id), start_time_utc = .pg_ts(.nz(e$event_date)),
    home = unname(nhl(.nz(e$home_abbreviation))), away = unname(nhl(.nz(e$away_abbreviation))),
    home_id = as.numeric(.nz(e$home_id)), away_id = as.numeric(.nz(e$away_id))
  )))
  games <- schedule |>
    dplyr::filter(.data$game_date == day) |>
    dplyr::transmute(game_id = as.numeric(.data$game_id), home = .data$home_team_abbr, away = .data$away_team_abbr)
  ev <- dplyr::left_join(ev, .pg_unique_on(games, c("home", "away")), by = c("home", "away"), na_matches = "never")
  team_abbr <- dplyr::distinct(dplyr::bind_rows(
    dplyr::transmute(ev, espn_team_id = .data$home_id, team_abbr = .data$home),
    dplyr::transmute(ev, espn_team_id = .data$away_id, team_abbr = .data$away)
  ))
  team_ids <- rosters |>
    dplyr::filter(.data$season == !!season) |>
    dplyr::group_by(.data$team_abbr) |>
    dplyr::summarise(team_id = as.numeric(dplyr::first(.data$team_id)), .groups = "drop")
  keyed <- dplyr::mutate(rosters, key = pregame_name_key(.data$full_name), player_id = as.numeric(.data$player_id))
  by_team <- keyed |>
    dplyr::filter(.data$season == !!season) |>
    dplyr::distinct(.data$team_abbr, .data$player_id, .data$key) |>
    dplyr::add_count(.data$team_abbr, .data$key) |>
    dplyr::filter(.data$n == 1) |>
    dplyr::select(-"n")
  league <- keyed |>
    dplyr::filter(.data$season >= !!season - 2, .data$season <= !!season) |>
    dplyr::distinct(.data$player_id, .data$key) |>
    dplyr::add_count(.data$key) |>
    dplyr::filter(.data$n == 1) |>
    dplyr::transmute(key = .data$key, league_pid = .data$player_id)

  rows |>
    dplyr::mutate(key = pregame_name_key(.data$player_name)) |>
    dplyr::left_join(.pg_unique_on(dplyr::select(ev, "espn_event_id", "game_id", "start_time_utc"), "espn_event_id"),
                     by = "espn_event_id", na_matches = "never") |>
    dplyr::left_join(.pg_unique_on(team_abbr, "espn_team_id"), by = "espn_team_id", na_matches = "never") |>
    dplyr::left_join(.pg_unique_on(team_ids, "team_abbr"), by = "team_abbr", na_matches = "never") |>
    dplyr::left_join(.pg_unique_on(by_team, c("team_abbr", "key")), by = c("team_abbr", "key"), na_matches = "never") |>
    dplyr::left_join(.pg_unique_on(league, "key"), by = "key", na_matches = "never") |>
    dplyr::mutate(player_id = dplyr::coalesce(.data$player_id, .data$league_pid), season = !!season, game_date = !!day) |>
    dplyr::select(dplyr::all_of(names(.pg_schema))) |>
    # polars' sort order: nulls first, then ascending (C-locale strings)
    dplyr::arrange(!is.na(.data$game_id), .data$game_id, !is.na(.data$team_id), .data$team_id, .data$kind,
                   !is.na(.data$player_name), .data$player_name, !is.na(.data$espn_athlete_id),
                   .data$espn_athlete_id, !is.na(.data$status), .data$status, .data$valid_from)
}

#' Every day file of end_year -> one table, written as parquet with the shared schema
pregame_compile_season <- function(pregame_dir, end_year, out_dir) {
  if (!dir.exists(pregame_dir)) cli::cli_abort("pregame dir not found: {pregame_dir}")
  files <- sort(list.files(file.path(pregame_dir, end_year), pattern = "\\.json$", full.names = TRUE))
  if (!length(files)) {
    cli::cli_alert_info("[pregame] {end_year}: no captures")
    return(invisible(NULL))
  }
  schedule <- arrow::read_parquet(file.path(out_dir, "schedules", "parquet", glue::glue("nhl_schedule_{end_year}.parquet")))
  rosters <- dplyr::bind_rows(lapply(
    sort(list.files(file.path(out_dir, "rosters", "parquet"), pattern = "^rosters_.*\\.parquet$", full.names = TRUE)),
    function(f) {
      r <- arrow::read_parquet(f, col_select = c("season", "team_abbr", "team_id", "player_id", "full_name"))
      dplyr::mutate(r, season = as.integer(.data$season), team_id = as.numeric(.data$team_id),
                    player_id = as.numeric(.data$player_id))
    }
  ))
  df <- dplyr::bind_rows(lapply(files, function(f) {
    pregame_compile_day(jsonlite::fromJSON(f, simplifyVector = FALSE), schedule, rosters)
  }))
  if (!nrow(df)) return(invisible(NULL))
  to_ts <- function(x) as.POSIXct(x, origin = "1970-01-01", tz = "UTC")
  df <- dplyr::mutate(df, dplyr::across(c("start_time_utc", "valid_from", "valid_to"), to_ts))
  tbl <- arrow::Table$create(df)$cast(.pg_schema)
  dest <- file.path(out_dir, "pregame_snapshots", "parquet", glue::glue("nhl_pregame_snapshots_{end_year}.parquet"))
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(tbl, dest)
  cli::cli_alert_success(
    "[pregame] {end_year}: {nrow(df)} rows, {sum(is.na(df$game_id))} without game_id, {sum(is.na(df$player_id))} without player_id"
  )
  invisible(dest)
}

if (sys.nframe() == 0L) {
  opts <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--season"), type = "integer", help = "END year (2027 = 2026-27)"),
    optparse::make_option("--pregame-dir", dest = "pregame_dir", type = "character"),
    optparse::make_option("--out-dir", dest = "out_dir", type = "character", default = "nhl")
  )))
  if (is.null(opts$season) || is.null(opts$pregame_dir)) stop("usage: -s <END_YEAR> --pregame-dir <dir> [--out-dir nhl]")
  pregame_compile_season(opts$pregame_dir, opts$season, opts$out_dir)
}
