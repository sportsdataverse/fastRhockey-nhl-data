## Compile NHL season datasets from fastRhockey-nhl-raw repo
## Usage:
##   Rscript R/nhl_data_creation.R -s 2024           (single season)
##   Rscript R/nhl_data_creation.R -s 2023 -e 2024   (range of seasons)
##
## Reads from: sportsdataverse/fastRhockey-nhl-raw (schedules + final game JSON)
## Produces:   PBP, team_box, player_box, rosters, schedules, master files
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

# ── Logging ──────────────────────────────────────────────────────────────
LOG_FILE <- "fastRhockey_nhl_data_logfile.txt"
logging <- function(msg, level = "INFO") {
  entry <- paste0(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), level, ": ", msg)
  cat(entry, "\n", file = LOG_FILE, append = TRUE)
}
logging("=== NHL Data Creation started ===")

option_list <- list(
  optparse::make_option(
    c("-s", "--start_year"),
    action = "store",
    default = fastRhockey:::most_recent_nhl_season(),
    type = "integer",
    help = "Start year of the seasons to process [default: current season]"
  ),
  optparse::make_option(
    c("-e", "--end_year"),
    action = "store",
    default = NA_integer_,
    type = "integer",
    help = "End year of the seasons to process [default: same as start_year]"
  )
)

opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))
options(stringsAsFactors = FALSE)
options(scipen = 999)

if (is.na(opt$end_year)) opt$end_year <- opt$start_year
years_vec <- opt$start_year:opt$end_year
logging(glue("Processing seasons: {paste(years_vec, collapse=', ')}"))

RAW_BASE <- "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main"


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
      logging(glue("Failed to read RDS from {url}: {conditionMessage(e)}"), "ERROR")
      NULL
    }
  )
}

.json_from_url <- function(url) {
  tryCatch(
    {
      jsonlite::fromJSON(url, simplifyVector = TRUE, flatten = TRUE)
    },
    error = function(e) NULL
  )
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

.upload_to_release <- function(df, file_name, release_tag, description) {
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
      logging(glue("Failed to upload {file_name} to {release_tag}: {conditionMessage(e)}"), "WARN")
    }
  )
}


# ═══════════════════════════════════════════════════════════════════════
# Main loop: per season
# ═══════════════════════════════════════════════════════════════════════

all_games <- purrr::map(years_vec, function(season_year) {
  season_label <- paste0(
    season_year, "-",
    substr(as.character(season_year + 1), 3, 4)
  )
  cli::cli_h1("Processing {season_label} season")
  logging(glue("=== {season_label} season ==="))


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
    logging(glue("Could not fetch schedule for {season_label}"), "ERROR")
    return(NULL)
  }

  for (d in c("nhl/schedules/rds", "nhl/schedules/parquet")) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  saveRDS(sched, glue("nhl/schedules/rds/nhl_schedule_{season_year}.rds"))
  arrow::write_parquet(sched, glue("nhl/schedules/parquet/nhl_schedule_{season_year}.parquet"),
    compression = "gzip"
  )

  season_json_games <- sched %>% dplyr::filter(game_json == TRUE)
  season_game_list <- season_json_games$game_id
  season_game_urls <- season_json_games$game_json_url

  logging(glue("{length(season_game_list)} games with final JSON in raw repo"))
  cli::cli_alert_info("{length(season_game_list)} games with final JSON in raw repo")

  if (length(season_game_list) == 0) {
    cli::cli_alert_warning("No games with JSON. Skipping.")
    logging("No games with JSON, skipping season", "WARN")
    return(NULL)
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 2: Compile play-by-play
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Compiling {season_label} PBP ({length(season_game_list)} games)",
    msg_done = "Compiled {season_label} PBP!"
  )

  future::plan(future::multisession, workers = 6)

  season_pbp <- furrr::future_map_dfr(
    season_game_urls,
    function(url) {
      tryCatch(
        {
          game_json <- .json_from_url(url)
          if (is.null(game_json)) {
            return(NULL)
          }
          pbp <- game_json$all_plays
          if (is.data.frame(pbp) && nrow(pbp) > 0) {
            return(pbp)
          }
          NULL
        },
        error = function(e) NULL
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )

  season_pbp <- dplyr::distinct(season_pbp)
  logging(glue("{nrow(season_pbp)} PBP events compiled"))
  cli::cli_alert_info("{nrow(season_pbp)} PBP events")

  if (nrow(season_pbp) > 0) {
    season_first <- as.character(season_year)
    season_last <- substr(as.character(season_year + 1), 3, 4)
    pbp_name <- glue("play_by_play_{season_first}_{season_last}")
    pbp_lite <- season_pbp |> dplyr::filter(event_type != "CHANGE")

    for (sub in c(
      "nhl/pbp/full/rds", "nhl/pbp/full/parquet",
      "nhl/pbp/lite/rds", "nhl/pbp/lite/parquet"
    )) {
      if (!dir.exists(sub)) dir.create(sub, recursive = TRUE)
    }
    season_pbp |> saveRDS(glue("nhl/pbp/full/rds/{pbp_name}.rds"), compress = "xz")
    pbp_lite |> saveRDS(glue("nhl/pbp/lite/rds/{pbp_name}_lite.rds"), compress = "xz")
    season_pbp |> arrow::write_parquet(glue("nhl/pbp/full/parquet/{pbp_name}.parquet"), compression = "gzip")
    pbp_lite |> arrow::write_parquet(glue("nhl/pbp/lite/parquet/{pbp_name}_lite.parquet"), compression = "gzip")

    # Upload to sportsdataverse-data releases
    logging(glue("Uploading {pbp_name} to sportsdataverse-data releases"))
    .upload_to_release(season_pbp, pbp_name, "nhl_pbp_full", "NHL play-by-play data (full)")
    .upload_to_release(pbp_lite, glue("{pbp_name}_lite"), "nhl_pbp_lite", "NHL play-by-play data (lite)")
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 3: Compile team + player boxscores
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Compiling {season_label} boxscores",
    msg_done = "Compiled {season_label} boxscores!"
  )

  season_team_box <- furrr::future_map_dfr(
    season_game_urls,
    function(url) {
      tryCatch(
        {
          game_json <- .json_from_url(url)
          if (is.null(game_json)) {
            return(NULL)
          }
          tb <- game_json$team_box_parsed
          if (is.data.frame(tb) && nrow(tb) > 0) {
            return(tb)
          }
          NULL
        },
        error = function(e) NULL
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )

  season_skater_box <- furrr::future_map_dfr(
    season_game_urls,
    function(url) {
      tryCatch(
        {
          game_json <- .json_from_url(url)
          if (is.null(game_json)) {
            return(NULL)
          }
          sk <- game_json$skater_stats
          if (is.data.frame(sk) && nrow(sk) > 0) {
            return(sk)
          }
          NULL
        },
        error = function(e) NULL
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )

  season_goalie_box <- furrr::future_map_dfr(
    season_game_urls,
    function(url) {
      tryCatch(
        {
          game_json <- .json_from_url(url)
          if (is.null(game_json)) {
            return(NULL)
          }
          gl <- game_json$goalie_stats
          if (is.data.frame(gl) && nrow(gl) > 0) {
            return(gl)
          }
          NULL
        },
        error = function(e) NULL
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )

  season_player_box <- dplyr::bind_rows(season_skater_box, season_goalie_box)

  if (nrow(season_team_box) > 0) {
    .save_dataset(season_team_box, "nhl/team_box", "team_box", season_year)
    logging(glue("{nrow(season_team_box)} team_box rows"))
    cli::cli_alert_info("{nrow(season_team_box)} team_box rows")
    .upload_to_release(
      season_team_box, glue("team_box_{season_year}"),
      "nhl_team_boxscores", "NHL team boxscores"
    )
  }

  if (nrow(season_player_box) > 0) {
    .save_dataset(season_player_box, "nhl/player_box", "player_box", season_year)
    logging(glue("{nrow(season_skater_box)} skater + {nrow(season_goalie_box)} goalie rows"))
    cli::cli_alert_info("{nrow(season_skater_box)} skater + {nrow(season_goalie_box)} goalie rows")
    .upload_to_release(
      season_player_box, glue("player_box_{season_year}"),
      "nhl_player_boxscores", "NHL player boxscores"
    )
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 4: Compile rosters
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Compiling {season_label} rosters",
    msg_done = "Compiled {season_label} rosters!"
  )

  season_rosters <- furrr::future_map_dfr(
    season_game_urls,
    function(url) {
      tryCatch(
        {
          game_json <- .json_from_url(url)
          if (is.null(game_json)) {
            return(NULL)
          }
          rosters <- game_json$rosters
          if (is.data.frame(rosters) && nrow(rosters) > 0) {
            return(rosters)
          }
          NULL
        },
        error = function(e) NULL
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )

  season_rosters_unique <- season_rosters %>%
    dplyr::select(-dplyr::any_of("game_id")) %>%
    dplyr::distinct()
  season_rosters_unique$season <- season_year

  if (nrow(season_rosters_unique) > 0) {
    .save_dataset(season_rosters_unique, "nhl/rosters", "rosters", season_year)
    logging(glue("{nrow(season_rosters_unique)} unique roster entries"))
    cli::cli_alert_info("{nrow(season_rosters_unique)} unique roster entries")
    .upload_to_release(season_rosters_unique, glue("rosters_{season_year}"),
                       "nhl_rosters", "NHL rosters")
  }


  # ──────────────────────────────────────────────────────────────────────
  # STEP 5: Update schedule with data availability flags
  # ──────────────────────────────────────────────────────────────────────

  cli::cli_progress_step(
    msg = "Updating {season_label} schedule flags",
    msg_done = "Updated {season_label} schedule flags"
  )

  pbp_ids <- if (nrow(season_pbp) > 0) unique(season_pbp$game_id) else integer(0)
  team_box_ids <- if (nrow(season_team_box) > 0) unique(season_team_box$game_id) else integer(0)
  player_ids <- if (nrow(season_player_box) > 0) unique(season_player_box$game_id) else integer(0)

  final_sched <- sched %>%
    dplyr::mutate(
      PBP        = game_id %in% pbp_ids,
      team_box   = game_id %in% team_box_ids,
      player_box = game_id %in% player_ids
    ) %>%
    dplyr::distinct() %>%
    dplyr::arrange(dplyr::desc(game_date))

  saveRDS(final_sched, glue("nhl/schedules/rds/nhl_schedule_{season_year}.rds"))
  arrow::write_parquet(final_sched,
    glue("nhl/schedules/parquet/nhl_schedule_{season_year}.parquet"),
    compression = "gzip"
  )

  cli::cli_alert_success("Done with {season_label}")
  logging(glue("Completed {season_label}: {nrow(season_pbp)} PBP, {nrow(season_team_box)} team_box, {nrow(season_player_box)} player_box"))

  rm(
    season_pbp, pbp_lite, season_team_box, season_skater_box,
    season_goalie_box, season_player_box, season_rosters,
    season_rosters_unique, final_sched, sched
  )
  gc()

  return(NULL)
}) # end purrr::map


# ═══════════════════════════════════════════════════════════════════════
# Build cross-season master files
# ═══════════════════════════════════════════════════════════════════════

cli::cli_progress_step(
  msg = "Building master schedule + nhl_games_in_data_repo",
  msg_done = "Master files built!"
)

sched_files <- list.files("nhl/schedules/rds", pattern = "\\.rds$", full.names = TRUE)
sched_all <- purrr::map_dfr(sched_files, readRDS) %>%
  dplyr::arrange(dplyr::desc(game_date))

saveRDS(sched_all, "nhl/nhl_schedule_master.rds", compress = "xz")
arrow::write_parquet(sched_all, "nhl/nhl_schedule_master.parquet", compression = "gzip")

games_in_repo <- sched_all %>%
  dplyr::filter(PBP == TRUE) %>%
  dplyr::arrange(dplyr::desc(game_date))

if (!dir.exists("nhl")) dir.create("nhl")
saveRDS(games_in_repo, "nhl/nhl_games_in_data_repo.rds", compress = "xz")
arrow::write_parquet(games_in_repo, "nhl/nhl_games_in_data_repo.parquet", compression = "gzip")

# Upload schedules to release
.upload_to_release(sched_all, "nhl_schedule_master", "nhl_schedules", "NHL schedules")

logging(glue("Master: {nrow(sched_all)} schedule rows, {nrow(games_in_repo)} with PBP"))
cli::cli_alert_success("{nrow(sched_all)} total schedule rows, {nrow(games_in_repo)} with PBP")

logging("=== NHL Data Creation complete ===")
cli::cli_h1("All done!")
