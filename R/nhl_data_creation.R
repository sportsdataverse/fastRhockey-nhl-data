## Compile NHL season datasets from fastRhockey-nhl-raw repo
##
## NOTE ON SEASON CONVENTION:
##   -s / -e refer to the *end year* of the season (e.g., 2026 for 2025-26).
##   All compiled dataset files are named using the end year convention:
##     play_by_play_{end_year}.rds, nhl_schedule_{end_year}.rds, etc.
##
## Usage:
##   Rscript R/nhl_data_creation.R -s 2026           (single season: 2025-26)
##   Rscript R/nhl_data_creation.R -s 2024 -e 2026   (range: 2023-24 through 2025-26)
##
## Reads from: sportsdataverse/fastRhockey-nhl-raw (schedules + final game JSON)
## Produces:   PBP (full/lite), skater_box, goalie_box, player_box, team_box,
##             game_info, game_rosters, scoring, penalties, three_stars,
##             scratches, linescore, shifts, rosters, schedules, master files
## Uploads to: sportsdataverse/sportsdataverse-data (GitHub releases)

suppressPackageStartupMessages(library(fastRhockey))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(furrr))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(cli))

cli::cli_alert_info("=== NHL Data Creation started ===")

option_list <- list(
  optparse::make_option(
    c("-s", "--start_year"),
    action = "store",
    default = fastRhockey::most_recent_nhl_season(),
    type = "integer",
    help = "Start season's end year to process, e.g. 2026 for 2025-26 [default: most recent]"
  ),
  optparse::make_option(
    c("-e", "--end_year"),
    action = "store",
    default = NA_integer_,
    type = "integer",
    help = "End season's end year to process [default: same as start_year]"
  )
)

opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))
options(stringsAsFactors = FALSE)
options(scipen = 999)

if (is.na(opt$end_year)) opt$end_year <- opt$start_year
years_vec <- opt$start_year:opt$end_year
cli::cli_alert_info("Processing seasons: {paste(years_vec, collapse=', ')}")

RAW_BASE <- "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main"


# ═══════════════════════════════════════════════════════════════════════
# Compile-spec table
#
# Each row defines one season-level dataset compiled from the per-game
# final JSON. The order matches what we export per season; new datasets
# only need a row added here plus a sub-dir under `nhl/`.
# ═══════════════════════════════════════════════════════════════════════

DATASETS <- tibble::tribble(
  ~key,              ~json_field,        ~file_prefix,        ~release_tag,             ~description,
  "pbp",             "all_plays",        "play_by_play",      "nhl_pbp_full",           "NHL play-by-play data (full)",
  "skater_box",      "skater_stats",     "skater_box",        "nhl_skater_boxscores",   "NHL skater boxscores",
  "goalie_box",      "goalie_stats",     "goalie_box",        "nhl_goalie_boxscores",   "NHL goalie boxscores",
  "team_box",        "team_box_parsed",  "team_box",          "nhl_team_boxscores",     "NHL team boxscores",
  "game_info",       "game_info",        "game_info",         "nhl_game_info",          "NHL game info",
  "game_rosters",    "rosters",          "game_rosters",      "nhl_game_rosters",       "NHL per-game rosters",
  "shifts",          "shifts",           "shifts",            "nhl_shifts",             "NHL shifts",
  "scoring",         "scoring",          "scoring",           "nhl_scoring",            "NHL scoring summary",
  "penalties",       "penalties",        "penalties",         "nhl_penalties",          "NHL penalty summary",
  "scratches",       "scratches",        "scratches",         "nhl_scratches",          "NHL scratches",
  "linescore",       "linescore",        "linescore",         "nhl_linescore",          "NHL linescore",
  "three_stars",     "decisions",        "three_stars",       "nhl_three_stars",        "NHL three stars / decisions",
  "officials",       "officials",        "officials",         "nhl_officials",          "NHL on-ice officials",
  "shots_by_period", "shots_by_period",  "shots_by_period",   "nhl_shots_by_period",    "NHL shots by period",
  "shootout",        "shootout",         "shootout_summary",  "nhl_shootout",           "NHL shootout summary"
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

.rds_from_url <- function(url) {
  tryCatch(
    {
      con <- url(url)
      on.exit(close(con))
      readRDS(con)
    },
    error = function(e) {
      cli::cli_alert_danger("Failed to read RDS from {url}: {conditionMessage(e)}")
      NULL
    }
  )
}

.json_from_url <- function(url) {
  tryCatch(
    {
      jsonlite::fromJSON(url, simplifyVector = TRUE, flatten = FALSE)
    },
    error = function(e) NULL
  )
}

# Pull all configured datasets out of one parsed game JSON in a single pass.
# Returns a named list mapping dataset key -> data.frame (possibly NULL).
.extract_all <- function(game_json) {
  if (is.null(game_json)) {
    return(setNames(
      vector("list", nrow(DATASETS) + 1),
      c(DATASETS$key, "player_box")
    ))
  }

  out <- setNames(
    vector("list", nrow(DATASETS) + 1),
    c(DATASETS$key, "player_box")
  )

  for (i in seq_len(nrow(DATASETS))) {
    key <- DATASETS$key[i]
    field <- DATASETS$json_field[i]
    val <- game_json[[field]]

    if (key == "scoring" || key == "penalties") {
      # These are nested list-of-period structures; flatten to data.frame
      if (is.list(val) && length(val) > 0) {
        parts <- purrr::map(val, function(period_block) {
          period_desc <- period_block$periodDescriptor
          items <- if (key == "scoring") period_block$goals else period_block$penalties
          if (is.data.frame(items) && nrow(items) > 0) {
            items$period_number <- period_desc$number
            items$period_type <- period_desc$periodType
            items
          } else {
            NULL
          }
        })
        parts <- purrr::compact(parts)
        if (length(parts) > 0) {
          out[[key]] <- tryCatch(dplyr::bind_rows(parts), error = function(e) parts[[1]])
        }
      }
    } else if (key == "linescore") {
      # Linescore is a nested list; flatten key parts into a single-row data.frame
      if (is.list(val) && length(val) > 0) {
        ls_row <- tryCatch(
          {
            home_team <- val$teams$home$team
            away_team <- val$teams$away$team
            tibble::tibble(
              game_id = game_json$game_info$game_id %||%
                game_json$game_info[[1]]$game_id %||% NA_integer_,
              home_team_id = home_team$id,
              home_team_abbr = home_team$abbreviation,
              home_goals = val$teams$home$goals,
              home_shots = val$teams$home$shotsOnGoal,
              away_team_id = away_team$id,
              away_team_abbr = away_team$abbreviation,
              away_goals = val$teams$away$goals,
              away_shots = val$teams$away$shotsOnGoal,
              has_shootout = val$shootout$hasShootout %||% FALSE
            )
          },
          error = function(e) NULL
        )
        out[[key]] <- ls_row
      }
    } else if (key == "three_stars") {
      # Decisions is a nested list too; flatten
      if (is.list(val) && !is.null(val$threeStars)) {
        ts_df <- tryCatch(
          {
            ts <- val$threeStars
            if (is.data.frame(ts)) {
              game_id_val <- game_json$game_info$game_id %||%
                game_json$game_info[[1]]$game_id %||% NA_integer_
              ts$game_id <- game_id_val
              ts$winner_id <- val$winner$id
              ts$winner_name <- val$winner$name
              ts$loser_id <- val$loser$id
              ts$loser_name <- val$loser$name
              ts
            } else {
              NULL
            }
          },
          error = function(e) NULL
        )
        out[[key]] <- ts_df
      }
    } else if (key == "scratches") {
      # Scratches is a list of {id, firstName, lastName}; convert to data.frame
      if (is.list(val) && length(val) > 0) {
        sc_df <- tryCatch(
          {
            df <- if (is.data.frame(val)) val else dplyr::bind_rows(val)
            if (nrow(df) > 0) {
              game_id_val <- game_json$game_info$game_id %||%
                game_json$game_info[[1]]$game_id %||% NA_integer_
              df$game_id <- game_id_val
            }
            df
          },
          error = function(e) NULL
        )
        out[[key]] <- sc_df
      }
    } else if (key %in% c("officials", "shots_by_period", "shootout")) {
      # All three are lists-of-named-lists produced by scrape_nhl_raw helpers.
      # bind_rows + attach game_id; some installs may surface them as
      # data.frames already (when jsonlite simplifies).
      if (length(val) > 0) {
        df <- tryCatch(
          if (is.data.frame(val)) val else dplyr::bind_rows(val),
          error = function(e) NULL
        )
        if (!is.null(df) && nrow(df) > 0) {
          game_id_val <- game_json$game_info$game_id %||%
            game_json$game_info[[1]]$game_id %||% NA_integer_
          df$game_id <- game_id_val
          # Pull season + game_date in for downstream joins
          df$season <- game_json$game_info$season %||%
            game_json$game_info[[1]]$season %||% NA_integer_
          df$game_date <- game_json$game_info$game_date %||%
            game_json$game_info[[1]]$game_date %||% NA_character_
          out[[key]] <- df
        }
      }
    } else {
      # Standard case: json_field is a data.frame or list convertible to one
      if (is.data.frame(val) && nrow(val) > 0) {
        out[[key]] <- val
      } else if (is.list(val) && length(val) > 0 && !is.data.frame(val)) {
        # Try to convert single-row game_info etc. to tibble
        df <- tryCatch(dplyr::bind_rows(val), error = function(e) NULL)
        if (!is.null(df) && nrow(df) > 0) out[[key]] <- df
      }
    }
  }

  # Player box = bind of skater_box + goalie_box
  parts <- purrr::compact(list(out[["skater_box"]], out[["goalie_box"]]))
  if (length(parts) > 0) {
    out[["player_box"]] <- tryCatch(dplyr::bind_rows(parts), error = function(e) parts[[1]])
  }

  out
}

.save_dataset <- function(df, dir_base, name, season) {
  rds_dir <- file.path(dir_base, "rds")
  parquet_dir <- file.path(dir_base, "parquet")
  for (d in c(rds_dir, parquet_dir)) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  saveRDS(df, file.path(rds_dir, glue("{name}_{season}.rds")), compress = "xz")
  arrow::write_parquet(df, file.path(parquet_dir, glue("{name}_{season}.parquet")),
    compression = "gzip"
  )
}

# Cache release-existence checks so we hit `gh` once per tag, not once per file.
.release_cache <- new.env(parent = emptyenv())
.release_exists <- function(release_tag,
                            repo = "sportsdataverse/sportsdataverse-data") {
  key <- paste0(repo, "@", release_tag)
  if (exists(key, envir = .release_cache, inherits = FALSE)) {
    return(get(key, envir = .release_cache, inherits = FALSE))
  }
  ok <- tryCatch(
    {
      res <- suppressWarnings(system2(
        "gh",
        c("release", "view", release_tag, "-R", repo, "--json", "tagName"),
        stdout = TRUE, stderr = TRUE
      ))
      st <- attr(res, "status")
      (is.null(st) || st == 0) &&
        length(res) > 0 &&
        !any(grepl("release not found", res, ignore.case = TRUE))
    },
    error = function(e) FALSE
  )
  assign(key, ok, envir = .release_cache)
  ok
}

.upload_to_release <- function(df, file_name, release_tag, description) {
  if (!.release_exists(release_tag)) {
    cli::cli_alert_warning(
      "Release tag {.val {release_tag}} does not exist on sportsdataverse-data; skipping upload of {.val {file_name}}. Create the release once with `gh release create {release_tag} -R sportsdataverse/sportsdataverse-data --notes 'init'` and re-run."
    )
    return(invisible(NULL))
  }
  retry_rate <- purrr::rate_backoff(pause_base = 1, pause_min = 60, max_times = 10)
  tryCatch(
    purrr::insistently(
      sportsdataversedata::sportsdataverse_save,
      rate = retry_rate, quiet = FALSE
    )(
      data_frame = df,
      file_name = file_name,
      sportsdataverse_type = description,
      release_tag = release_tag,
      pkg_function = glue("fastRhockey::load_nhl_{gsub('nhl_', '', release_tag)}()"),
      file_types = c("rds", "csv", "parquet"),
      .token = Sys.getenv("GITHUB_PAT",
                          unset = system("gh auth token", intern = TRUE))
    ),
    error = function(e) {
      cli::cli_alert_warning("Failed to upload {file_name} to {release_tag}: {conditionMessage(e)}")
    }
  )
}


# ═══════════════════════════════════════════════════════════════════════
# Main loop: per season
# ═══════════════════════════════════════════════════════════════════════

invisible(purrr::map(years_vec, function(season_year) {
  season_start <- season_year - 1
  season_label <- paste0(season_start, "-", substr(as.character(season_year), 3, 4))
  cli::cli_h1("Processing {season_label} season")


  # ──────────────────────────────────────────────────────────────────────
  # STEP 1: Fetch schedule from nhl-raw repo
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Downloading {season_label} schedule from nhl-raw",
    msg_done = "Downloaded {season_label} schedule"
  )

  sched <- .rds_from_url(glue("{RAW_BASE}/nhl/schedules/rds/nhl_schedule_{season_year}.rds"))

  if (is.null(sched)) {
    cli::cli_alert_danger("Could not fetch schedule for {season_label}. Skipping.")
    return(NULL)
  }

  for (d in c("nhl/schedules/rds", "nhl/schedules/parquet")) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  saveRDS(sched, glue("nhl/schedules/rds/nhl_schedule_{season_year}.rds"))
  arrow::write_parquet(sched, glue("nhl/schedules/parquet/nhl_schedule_{season_year}.parquet"),
    compression = "gzip"
  )

  season_json_games <- sched %>% dplyr::filter(.data$game_json == TRUE)
  season_game_list <- season_json_games$game_id
  season_game_urls <- season_json_games$game_json_url

  cli::cli_alert_info("{length(season_game_list)} games with final JSON in raw repo")

  if (length(season_game_list) == 0) {
    cli::cli_alert_warning("No games with JSON. Skipping.")
    return(NULL)
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 2: Single-pass extraction of every per-game dataset
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Reading {length(season_game_urls)} game JSONs and extracting datasets",
    msg_done = "Extracted per-game datasets"
  )

  future::plan(future::multisession, workers = 6)

  # Bind helpers + spec table to local names for furrr's globals detection
  json_from_url <- .json_from_url
  extract_all   <- .extract_all
  datasets_spec <- DATASETS

  per_game <- furrr::future_map(
    season_game_urls,
    function(url) {
      tryCatch(extract_all(json_from_url(url)),
        error = function(e) NULL)
    },
    .options = furrr::furrr_options(
      seed = TRUE,
      globals = list(
        json_from_url = json_from_url,
        extract_all = extract_all,
        DATASETS = datasets_spec
      ),
      packages = c("jsonlite", "dplyr", "purrr", "tibble")
    )
  )

  # Pivot list-of-named-lists into named-list-of-frames, one per dataset key.
  all_keys <- c(DATASETS$key, "player_box")

  compiled <- purrr::map(all_keys, function(key) {
    parts <- purrr::map(per_game, ~ .x[[key]])
    parts <- purrr::compact(parts)
    if (length(parts) == 0) return(NULL)
    tryCatch(
      dplyr::bind_rows(parts) |> dplyr::distinct(),
      error = function(e) {
        cli::cli_alert_warning("bind_rows failed for {key}: {conditionMessage(e)}")
        parts[[1]]
      }
    )
  })
  names(compiled) <- all_keys


  # ──────────────────────────────────────────────────────────────────────
  # STEP 3: Save + upload each dataset from the spec table
  # ──────────────────────────────────────────────────────────────────────

  for (i in seq_len(nrow(DATASETS))) {
    key   <- DATASETS$key[i]
    pref  <- DATASETS$file_prefix[i]
    rtag  <- DATASETS$release_tag[i]
    desc  <- DATASETS$description[i]
    df    <- compiled[[key]]

    if (is.null(df) || nrow(df) == 0) {
      cli::cli_alert_info("{key}: 0 rows -> skipped")
      next
    }

    cli::cli_alert_info("{key}: {nrow(df)} rows")
    .save_dataset(df, file.path("nhl", key), pref, season_year)
    .upload_to_release(df, glue("{pref}_{season_year}"), rtag, desc)
  }

  # PBP lite (PBP without CHANGE events)
  pbp_full <- compiled[["pbp"]]
  if (!is.null(pbp_full) && nrow(pbp_full) > 0) {
    pbp_lite <- pbp_full |> dplyr::filter(.data$event_type != "CHANGE")
    .save_dataset(pbp_lite, "nhl/pbp_lite", "play_by_play_lite", season_year)
    .upload_to_release(pbp_lite, glue("play_by_play_{season_year}_lite"),
      "nhl_pbp_lite", "NHL play-by-play data (lite)")
    cli::cli_alert_info("pbp_lite: {nrow(pbp_lite)} rows")
  }

  # Player box (combined skater + goalie)
  player_box <- compiled[["player_box"]]
  if (!is.null(player_box) && nrow(player_box) > 0) {
    .save_dataset(player_box, "nhl/player_box", "player_box", season_year)
    .upload_to_release(player_box, glue("player_box_{season_year}"),
      "nhl_player_boxscores", "NHL player boxscores")
    cli::cli_alert_info("player_box: {nrow(player_box)} rows")
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 4: Compile season rosters (unique players from game_rosters)
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Compiling {season_label} season rosters",
    msg_done = "Compiled {season_label} season rosters"
  )

  season_rosters <- compiled[["game_rosters"]]
  if (!is.null(season_rosters) && nrow(season_rosters) > 0) {
    rosters_unique <- season_rosters |>
      dplyr::select(-dplyr::any_of("game_id")) |>
      dplyr::distinct()
    rosters_unique$season <- season_year
    .save_dataset(rosters_unique, "nhl/rosters", "rosters", season_year)
    .upload_to_release(rosters_unique, glue("rosters_{season_year}"),
      "nhl_rosters", "NHL rosters")
    cli::cli_alert_info("rosters: {nrow(rosters_unique)} unique entries")
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 5: Update schedule with data availability flags
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Updating {season_label} schedule flags",
    msg_done = "Updated {season_label} schedule flags"
  )

  ids_with <- function(key) {
    df <- compiled[[key]]
    if (is.null(df) || !"game_id" %in% names(df)) integer(0)
    else as.integer(unique(df$game_id))
  }

  final_sched <- sched |>
    dplyr::mutate(
      PBP             = as.integer(.data$game_id) %in% ids_with("pbp"),
      team_box        = as.integer(.data$game_id) %in% ids_with("team_box"),
      player_box      = as.integer(.data$game_id) %in% ids_with("player_box"),
      skater_box      = as.integer(.data$game_id) %in% ids_with("skater_box"),
      goalie_box      = as.integer(.data$game_id) %in% ids_with("goalie_box"),
      game_info       = as.integer(.data$game_id) %in% ids_with("game_info"),
      game_rosters    = as.integer(.data$game_id) %in% ids_with("game_rosters"),
      scoring         = as.integer(.data$game_id) %in% ids_with("scoring"),
      penalties       = as.integer(.data$game_id) %in% ids_with("penalties"),
      scratches       = as.integer(.data$game_id) %in% ids_with("scratches"),
      linescore       = as.integer(.data$game_id) %in% ids_with("linescore"),
      three_stars     = as.integer(.data$game_id) %in% ids_with("three_stars"),
      shifts          = as.integer(.data$game_id) %in% ids_with("shifts"),
      officials       = as.integer(.data$game_id) %in% ids_with("officials"),
      shots_by_period = as.integer(.data$game_id) %in% ids_with("shots_by_period"),
      shootout        = as.integer(.data$game_id) %in% ids_with("shootout")
    ) |>
    dplyr::distinct() |>
    dplyr::arrange(dplyr::desc(.data$game_date))

  saveRDS(final_sched, glue("nhl/schedules/rds/nhl_schedule_{season_year}.rds"))
  arrow::write_parquet(final_sched,
    glue("nhl/schedules/parquet/nhl_schedule_{season_year}.parquet"),
    compression = "gzip"
  )

  .upload_to_release(
    final_sched, glue("nhl_schedule_{season_year}"),
    "nhl_schedules", "NHL schedule"
  )

  cli::cli_alert_success("Done with {season_label}")

  rm(compiled, per_game, final_sched, sched)
  gc()
  NULL
}))


# ═══════════════════════════════════════════════════════════════════════
# Build cross-season master files
# ═══════════════════════════════════════════════════════════════════════

cli::cli_progress_step(
  msg = "Building master schedule + nhl_games_in_data_repo",
  msg_done = "Master files built!"
)

sched_files <- list.files("nhl/schedules/rds", pattern = "\\.rds$", full.names = TRUE)
sched_all <- purrr::map(sched_files, readRDS) |>
  dplyr::bind_rows() |>
  dplyr::arrange(dplyr::desc(.data$game_date))

saveRDS(sched_all, "nhl/nhl_schedule_master.rds", compress = "xz")
arrow::write_parquet(sched_all, "nhl/nhl_schedule_master.parquet", compression = "gzip")

games_in_repo <- sched_all |>
  dplyr::filter(.data$PBP == TRUE) |>
  dplyr::arrange(dplyr::desc(.data$game_date))

if (!dir.exists("nhl")) dir.create("nhl")
saveRDS(games_in_repo, "nhl/nhl_games_in_data_repo.rds", compress = "xz")
arrow::write_parquet(games_in_repo, "nhl/nhl_games_in_data_repo.parquet", compression = "gzip")

.upload_to_release(sched_all, "nhl_schedule_master", "nhl_schedules", "NHL schedules")
.upload_to_release(
  games_in_repo, "nhl_games_in_data_repo",
  "nhl_schedules", "NHL games available in fastRhockey data repo"
)

cli::cli_alert_success("{nrow(sched_all)} total schedule rows, {nrow(games_in_repo)} with PBP")

cli::cli_alert_info("=== NHL Data Creation complete ===")
cli::cli_h1("All done!")
