lib_path <- Sys.getenv("R_LIBS")
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", lib = Sys.getenv("R_LIBS"), repos = "http://cran.us.r-project.org")
}
suppressPackageStartupMessages(suppressMessages(library(dplyr, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(magrittr, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(jsonlite, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(purrr, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(progressr, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(data.table, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(arrow, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(glue, lib.loc = lib_path)))
suppressPackageStartupMessages(suppressMessages(library(optparse, lib.loc = lib_path)))



sched_list <- list.files(path = glue::glue("nhl/schedules/rds/"))
sched_g <- purrr::map(sched_list, function(x) {
  sched <- readRDS(paste0("nhl/schedules/rds/", x)) %>%
    dplyr::mutate(
      id = as.integer(.data$id),
      game_id = as.integer(.data$game_id),
      status_display_clock = as.character(.data$status_display_clock)
    )

  sched <- sched %>%
    fastRhockey:::make_fastRhockey_data("NHL Schedule from fastRhockey data repository", Sys.time())
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = sched,
    file_name = glue::glue("nhl_schedule_{y}"),
    sportsdataverse_type = "schedule data",
    release_tag = "nhl_schedules",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})
rm(sched_g)

pbp_list <- list.files(path = glue::glue("nhl/pbp/full/rds/"))
pbp_g <- purrr::map(pbp_list, function(x) {
  pbp <- readRDS(paste0("nhl/pbp/full/rds/", x))

  pbp <- pbp %>%
    fastRhockey:::make_fastRhockey_data("NHL Play-by-Play - Full version with Game Shifts from fastRhockey data repository", Sys.time())
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = pbp,
    file_name = glue::glue("nhl_play_by_play_{y}"),
    sportsdataverse_type = "Play-by-Play data",
    release_tag = "nhl_pbp_full",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})
rm(pbp_g)

pbp_list <- list.files(path = glue::glue("nhl/pbp/lite/rds/"))
pbp_g <- purrr::map(pbp_list, function(x) {
  pbp <- readRDS(paste0("nhl/pbp/lite/rds/", x))

  pbp <- pbp %>%
    fastRhockey:::make_fastRhockey_data("NHL Play-by-Play - Lite version without Game Shifts from fastRhockey data repository", Sys.time())
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = pbp,
    file_name = glue::glue("nhl_play_by_play_{y}"),
    sportsdataverse_type = "Play-by-Play data",
    release_tag = "nhl_pbp_lite",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})
rm(pbp_g)

team_box_list <- list.files(path = glue::glue("nhl/team_box/rds/"))
team_box_g <- purrr::map(team_box_list, function(x) {
  team_box <- readRDS(paste0("nhl/team_box/rds/", x))
  team_box <- team_box %>%
    fastRhockey:::make_fastRhockey_data("NHL Team Boxscores from fastRhockey data repository", Sys.time())
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = team_box,
    file_name = glue::glue("nhl_team_box_{y}"),
    sportsdataverse_type = "Team Boxscores data",
    release_tag = "nhl_team_boxscores",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

rm(team_box_g)

player_box_list <- list.files(path = glue::glue("nhl/player_box/rds/"))
player_box_g <- purrr::map(player_box_list, function(x) {
  player_box <- readRDS(paste0("nhl/player_box/rds/", x))
  player_box <- player_box %>%
    fastRhockey:::make_fastRhockey_data("NHL Player Boxscores from fastRhockey data repository", Sys.time())
  y <- stringr::str_extract(x, "\\d+")
  sportsdataversedata::sportsdataverse_save(
    data_frame = player_box,
    file_name = glue::glue("nhl_player_box_{y}"),
    sportsdataverse_type = "Player Boxscores data",
    release_tag = "nhl_player_boxscores",
    file_types = c("rds", "csv", "parquet"),
    .token = Sys.getenv("GITHUB_PAT")
  )
})

rm(player_box_g)
