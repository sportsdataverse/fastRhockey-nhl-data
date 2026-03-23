## Scrape a full season of play-by-play data
## Usage: Rscript scrape_full_season.R <season_end_year>
## Example: Rscript scrape_full_season.R 2025   (scrapes 2024-25 season)
## Note: hockeyR uses the ending year, so season=2025 means 2024-25

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide a season end year (e.g., 2025 for the 2024-25 season)")
}
season_year <- as.integer(args[1])

cat(
  "=== Scraping",
  paste0(season_year - 1, "-", substr(season_year, 3, 4)),
  "season ===\n"
)

library(hockeyR)
library(dplyr)
library(glue)
library(purrr)
library(furrr)
library(future)
library(readr)

# Get all game IDs for the season
cat("Fetching game IDs...\n")
games <- hockeyR::get_game_ids(season = season_year)

# Only scrape games that have already been played
games <- dplyr::filter(games, date < Sys.Date())
cat(glue("Found {nrow(games)} completed games to scrape"), "\n")

if (nrow(games) == 0) {
  cat("No completed games found for this season. Exiting.\n")
  quit(status = 0)
}

# Scrape play-by-play for each game
cat("Scraping play-by-play data (this may take a while)...\n")
future::plan(future::multisession, workers = 6) # Adjust workers as needed

library(progressr)

p <- progressr::progressor(steps = nrow(games))

progressr::with_progress({
  pbp_updated <- furrr::future_map_dfr(
    .x = games$game_id,
    .f = function(gid) {
      tryCatch(
        {
          result <- hockeyR::scrape_game(gid)
          p()
          result
        },
        error = function(e) {
          cat(
            "  WARNING: Failed to scrape game",
            gid,
            "-",
            conditionMessage(e),
            "\n"
          )
          p()
          NULL
        }
      )
    },
    .options = furrr::furrr_options(packages = "hockeyR")
  )
})

cat(glue("Scraped {nrow(pbp_updated)} play-by-play events"), "\n")

# Deduplicate
pbp_updated <- dplyr::distinct(pbp_updated)

# Determine filename — season_year is the ending year, so 2025 → "2024_25"
season_first <- as.character(season_year - 1)
season_last <- substr(as.character(season_year), 3, 4)

# Save into data/ directory (create if needed)
if (!dir.exists("nhl")) {
  dir.create("nhl")
}
if (!dir.exists("nhl/pbp")){
  dir.create("nhl/pbp")
  dir.create("nhl/pbp/full")
  dir.create("nhl/pbp/full/rds")
  dir.create("nhl/pbp/full/parquet")
  dir.create("nhl/pbp/lite")
  dir.create("nhl/pbp/lite/rds")
  dir.create("nhl/pbp/lite/parquet")
}
filename <- glue("play_by_play_{season_first}_{season_last}")

cat(glue("Saving to {filename}.*"), "\n")

# Create lite version (no line change events)
pbp_lite <- pbp_updated |>
  dplyr::filter(event_type != "CHANGE")

# Save in all 4 formats
pbp_updated |> saveRDS(glue("nhl/pbp/full/rds/{filename}.rds"), compress = "xz")
pbp_lite |> saveRDS(glue("nhl/pbp/lite/rds/{filename}_lite.rds"), compress = "xz")
pbp_updated |> arrow::write_parquet(glue("nhl/pbp/full/parquet/{filename}.parquet"), compression = "gzip")
pbp_lite |> arrow::write_parquet(glue("nhl/pbp/lite/parquet/{filename}_lite.parquet"), compression = "gzip")


cat("=== Done! ===\n")
cat(glue("Full:  {filename}.rds, {filename}.parquet"), "\n")
cat(glue("Lite:  {filename}_lite.rds, {filename}_lite.parquet"), "\n")
