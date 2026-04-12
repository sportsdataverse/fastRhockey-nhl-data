#--- NHL Data Release Initialization -----
# Run once to create the release tags on sportsdataverse-data.
# Each entry below corresponds to a dataset compiled by
# R/nhl_data_creation.R and uploaded via .upload_to_release().

REPO <- "sportsdataverse/sportsdataverse-data"
TOKEN <- Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))

releases <- tibble::tribble(
  ~tag,                     ~body,
  "nhl_schedules",          "NHL Schedules Data (from NHL API)",
  "nhl_pbp_full",           "NHL Play-by-Play Data — Full version with line changes and shifts (from NHL API)",
  "nhl_pbp_lite",           "NHL Play-by-Play Data — Lite version without line changes (from NHL API)",
  "nhl_player_boxscores",   "NHL Player Boxscores Data (from NHL API)",
  "nhl_skater_boxscores",   "NHL Skater Boxscores Data (from NHL API)",
  "nhl_goalie_boxscores",   "NHL Goalie Boxscores Data (from NHL API)",
  "nhl_team_boxscores",     "NHL Team Boxscores Data (from NHL API)",
  "nhl_rosters",            "NHL Rosters Data (from NHL API)",
  "nhl_game_rosters",       "NHL Per-Game Rosters Data (from NHL API)",
  "nhl_game_info",          "NHL Game Info Data (from NHL API)",
  "nhl_scoring",            "NHL Scoring Summary Data (from NHL API)",
  "nhl_penalties",          "NHL Penalty Summary Data (from NHL API)",
  "nhl_three_stars",        "NHL Three Stars / Decisions Data (from NHL API)",
  "nhl_scratches",          "NHL Scratches Data (from NHL API)",
  "nhl_linescore",          "NHL Linescore Data (from NHL API)",
  "nhl_shifts",             "NHL Shifts Data (from NHL API)",
  "nhl_officials",          "NHL On-Ice Officials Data — referees + linesmen, one row per official per game (from NHL API)",
  "nhl_shots_by_period",    "NHL Shots by Period Data — one row per team per period per game (from NHL API)",
  "nhl_shootout",           "NHL Shootout Summary Data — one row per shootout attempt (from NHL API)"
)

for (i in seq_len(nrow(releases))) {
  tag <- releases$tag[i]
  body <- releases$body[i]
  message("Creating release: ", tag)
  tryCatch(
    piggyback::pb_release_create(
      repo = REPO,
      tag = tag,
      name = tag,
      body = body,
      .token = TOKEN
    ),
    error = function(e) {
      message("  -> ", conditionMessage(e))
    }
  )
}
