#--- NHL Data -----
piggyback::pb_release_create(
  repo = "sportsdataverse/sportsdataverse-data",
  tag = "nhl_schedules",
  name = "nhl_schedules",
  body = "NHL Schedules Data (from NHL API)",
  .token = Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))
)

piggyback::pb_release_create(
  repo = "sportsdataverse/sportsdataverse-data",
  tag = "nhl_team_boxscores",
  name = "nhl_team_boxscores",
  body = "NHL Team Boxscores Data (from NHL API)",
  .token = Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))
)

piggyback::pb_release_create(
  repo = "sportsdataverse/sportsdataverse-data",
  tag = "nhl_player_boxscores",
  name = "nhl_player_boxscores",
  body = "NHL Player Boxscores Data (from NHL API)",
  .token = Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))
)


piggyback::pb_release_create(
  repo = "sportsdataverse/sportsdataverse-data",
  tag = "nhl_pbp_lite",
  name = "nhl_pbp_lite",
  body = "NHL Play-by-Play Data (from NHL API) - Lite version without game shifts",
  .token = Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))
)

piggyback::pb_release_create(
  repo = "sportsdataverse/sportsdataverse-data",
  tag = "nhl_pbp_full",
  name = "nhl_pbp_full",
  body = "NHL Play-by-Play Data (from NHL API) - Full version with game shifts",
  .token = Sys.getenv("GITHUB_PAT", unset = system("gh auth token", intern = TRUE))
)
