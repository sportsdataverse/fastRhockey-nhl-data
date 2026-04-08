## Push existing local NHL data files to sportsdataverse-data releases.
##
## NOTE: All files use the *end year* convention (e.g. `play_by_play_2026.rds`
## for the 2025-26 season) which matches `most_recent_nhl_season()`.

## In CI, R_LIBS is set so we honor lib.loc; locally we fall back to default libs.
lib_path <- Sys.getenv("R_LIBS", unset = "")
.lib_args <- if (nzchar(lib_path) && dir.exists(lib_path)) list(lib.loc = lib_path) else list()
.lib <- function(pkg) do.call(library, c(list(pkg), .lib_args))

suppressPackageStartupMessages(suppressMessages(.lib("dplyr")))
suppressPackageStartupMessages(suppressMessages(.lib("magrittr")))
suppressPackageStartupMessages(suppressMessages(.lib("jsonlite")))
suppressPackageStartupMessages(suppressMessages(.lib("purrr")))
suppressPackageStartupMessages(suppressMessages(.lib("progressr")))
suppressPackageStartupMessages(suppressMessages(.lib("data.table")))
suppressPackageStartupMessages(suppressMessages(.lib("arrow")))
suppressPackageStartupMessages(suppressMessages(.lib("glue")))
suppressPackageStartupMessages(suppressMessages(.lib("stringr")))


# --- Schedules (per-season) ---
sched_list <- list.files(path = "nhl/schedules/rds/")
purrr::walk(sched_list, function(x) {
  sched <- readRDS(paste0("nhl/schedules/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Schedule from fastRhockey data repository", Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = sched,
    file_name = glue::glue("nhl_schedule_{y}"),
    sportsdataverse_type = "schedule data",
    release_tag = "nhl_schedules",
    pkg_function = "fastRhockey::load_nhl_schedule()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

# --- Master schedule + games-in-repo index ---
if (file.exists("nhl/nhl_schedule_master.rds")) {
  master_sched <- readRDS("nhl/nhl_schedule_master.rds") |>
    fastRhockey:::make_fastRhockey_data(
      "NHL master schedule from fastRhockey data repository", Sys.time()
    )
  sportsdataversedata::sportsdataverse_save(
    data_frame = master_sched,
    file_name = "nhl_schedule_master",
    sportsdataverse_type = "schedule data",
    release_tag = "nhl_schedules",
    pkg_function = "fastRhockey::load_nhl_schedule()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
}

if (file.exists("nhl/nhl_games_in_data_repo.rds")) {
  games_in_repo <- readRDS("nhl/nhl_games_in_data_repo.rds") |>
    fastRhockey:::make_fastRhockey_data(
      "NHL games available in fastRhockey data repo", Sys.time()
    )
  sportsdataversedata::sportsdataverse_save(
    data_frame = games_in_repo,
    file_name = "nhl_games_in_data_repo",
    sportsdataverse_type = "schedule data",
    release_tag = "nhl_schedules",
    pkg_function = "fastRhockey:::load_nhl_games()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
}

# --- PBP full ---
pbp_list <- list.files(path = "nhl/pbp/full/rds/")
purrr::walk(pbp_list, function(x) {
  pbp <- readRDS(paste0("nhl/pbp/full/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Play-by-Play (full, with shifts) from fastRhockey data repository",
      Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = pbp,
    file_name = glue::glue("play_by_play_{y}"),
    sportsdataverse_type = "Play-by-Play data",
    release_tag = "nhl_pbp_full",
    pkg_function = "fastRhockey::load_nhl_pbp()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

# --- PBP lite ---
pbp_list <- list.files(path = "nhl/pbp/lite/rds/")
purrr::walk(pbp_list, function(x) {
  pbp <- readRDS(paste0("nhl/pbp/lite/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Play-by-Play (lite, no CHANGE events) from fastRhockey data repository",
      Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = pbp,
    file_name = glue::glue("play_by_play_{y}_lite"),
    sportsdataverse_type = "Play-by-Play data",
    release_tag = "nhl_pbp_lite",
    pkg_function = "fastRhockey::load_nhl_pbp_lite()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

# --- Team box ---
team_box_list <- list.files(path = "nhl/team_box/rds/")
purrr::walk(team_box_list, function(x) {
  team_box <- readRDS(paste0("nhl/team_box/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Team Boxscores from fastRhockey data repository", Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = team_box,
    file_name = glue::glue("team_box_{y}"),
    sportsdataverse_type = "Team Boxscores data",
    release_tag = "nhl_team_boxscores",
    pkg_function = "fastRhockey::load_nhl_team_box()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

# --- Player box ---
player_box_list <- list.files(path = "nhl/player_box/rds/")
purrr::walk(player_box_list, function(x) {
  player_box <- readRDS(paste0("nhl/player_box/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Player Boxscores from fastRhockey data repository", Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = player_box,
    file_name = glue::glue("player_box_{y}"),
    sportsdataverse_type = "Player Boxscores data",
    release_tag = "nhl_player_boxscores",
    pkg_function = "fastRhockey::load_nhl_player_box()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

# --- Rosters ---
roster_list <- list.files(path = "nhl/rosters/rds/")
purrr::walk(roster_list, function(x) {
  rosters <- readRDS(paste0("nhl/rosters/rds/", x)) |>
    fastRhockey:::make_fastRhockey_data(
      "NHL Rosters from fastRhockey data repository", Sys.time()
    )
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = rosters,
    file_name = glue::glue("rosters_{y}"),
    sportsdataverse_type = "Rosters data",
    release_tag = "nhl_rosters",
    pkg_function = "fastRhockey::load_nhl_rosters()",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})
