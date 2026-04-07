## retrain_xg_models.R
## Retrains hockeyR xG models using modern xgboost format.
## Run from the hockeyR repo root:
##   Rscript retrain_xg_models.R
##
## Reads historical PBP data from ../hockeyR-data/pbp_data/
## Outputs new R/sysdata.rda with xgboost models saved in modern format.

.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")

library(dplyr)
library(purrr)
library(tidyr)
library(janitor)
library(xgboost)

cat("=== Retraining hockeyR xG Models ===\n")
cat("xgboost version:", as.character(packageVersion("xgboost")), "\n\n")

# --------------------------------------------------------------------------
# 1. Load historical PBP data (2010-11 through 2023-24 as training data)
# --------------------------------------------------------------------------
pbp_dir <- "../hockeyR-data/pbp_data"
rds_files <- list.files(pbp_dir, pattern = "\\.rds$", full.names = TRUE)
# Exclude lite files and exclude 2024-25 / 2025-26 (future seasons)
rds_files <- rds_files[!grepl("_lite\\.rds$", rds_files)]
rds_files <- rds_files[!grepl("2024_25|2025_26", rds_files)]
rds_files <- sort(rds_files)
cat("Loading", length(rds_files), "seasons of PBP data...\n")
cat(paste0("  ", basename(rds_files), collapse = "\n"), "\n\n")

# Columns needed for feature engineering
needed_cols <- c(
    "season",
    "game_id",
    "event_id",
    "event_type",
    "secondary_type",
    "period_type",
    "period",
    "game_seconds",
    "x",
    "y",
    "x_fixed",
    "y_fixed",
    "event_team",
    "event_team_abbr",
    "home_abbr",
    "away_abbr",
    "home_skaters",
    "away_skaters",
    "shot_distance",
    "shot_angle",
    "empty_net",
    "strength_state"
)

pbp_all <- purrr::map_dfr(rds_files, function(f) {
    cat("  Loading", basename(f), "...")
    df <- readRDS(f)

    # Keep only needed columns (some may not exist in all seasons)
    avail <- intersect(needed_cols, names(df))
    df <- df[, avail, drop = FALSE]

    # Add missing columns as NA
    for (col in setdiff(needed_cols, avail)) {
        df[[col]] <- NA
    }

    # Coerce to consistent types to allow binding across all seasons
    df$season <- as.character(df$season)
    df$game_id <- as.character(df$game_id)
    df$event_id <- as.integer(df$event_id)
    df$game_seconds <- as.numeric(df$game_seconds)
    df$x <- as.numeric(df$x)
    df$y <- as.numeric(df$y)
    df$x_fixed <- as.numeric(df$x_fixed)
    df$y_fixed <- as.numeric(df$y_fixed)
    df$home_skaters <- as.numeric(df$home_skaters)
    df$away_skaters <- as.numeric(df$away_skaters)
    df$shot_distance <- as.numeric(df$shot_distance)
    df$shot_angle <- as.numeric(df$shot_angle)
    df$empty_net <- as.logical(df$empty_net)
    df$period <- as.integer(df$period)

    cat(" [", nrow(df), "rows ]\n")
    df
})
cat("Total rows:", nrow(pbp_all), "\n\n")

# --------------------------------------------------------------------------
# 2. Feature engineering (replicates helper_nhl_prepare_xg_data.R logic)
# --------------------------------------------------------------------------
cat("Engineering features...\n")

`%not_in%` <- function(x, table) !(x %in% table)

model_df <- pbp_all %>%
    # filter out shootouts
    filter(period_type != "SHOOTOUT") %>%
    # remove penalty shots
    filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) %>%
    # remove shift change events
    filter(event_type != "CHANGE") %>%
    # add model feature variables
    group_by(game_id, period) %>%
    mutate(
        last_event_type = lag(event_type),
        last_event_team = lag(event_team),
        time_since_last = game_seconds - lag(game_seconds),
        last_x = lag(x),
        last_y = lag(y),
        distance_from_last = round(
            sqrt(((y - last_y)^2) + ((x - last_x)^2)),
            1
        ),
        event_zone = case_when(
            x >= -25 & x <= 25 ~ "NZ",
            (x_fixed < -25 & event_team_abbr == home_abbr) |
                (x_fixed > 25 & event_team_abbr == away_abbr) ~ "DZ",
            (x_fixed > 25 & event_team_abbr == home_abbr) |
                (x_fixed < -25 & event_team_abbr == away_abbr) ~ "OZ"
        ),
        last_event_zone = lag(event_zone)
    ) %>%
    ungroup() %>%
    # filter to only unblocked shots
    filter(event_type %in% c("SHOT", "MISSED_SHOT", "GOAL")) %>%
    # filter valid last events
    filter(
        last_event_type %in%
            c(
                "FACEOFF",
                "GIVEAWAY",
                "TAKEAWAY",
                "BLOCKED_SHOT",
                "HIT",
                "MISSED_SHOT",
                "SHOT",
                "STOP",
                "PENALTY",
                "GOAL"
            )
    ) %>%
    mutate(
        era_2011_2013 = ifelse(
            season %in% c("20102011", "20112012", "20122013"),
            1,
            0
        ),
        era_2014_2018 = ifelse(
            season %in%
                c("20132014", "20142015", "20152016", "20162017", "20172018"),
            1,
            0
        ),
        era_2019_2021 = ifelse(
            season %in% c("20182019", "20192020", "20202021"),
            1,
            0
        ),
        era_2022_on = ifelse(as.numeric(season) > 20202021, 1, 0),
        # ST model features
        event_team_skaters = ifelse(
            event_team_abbr == home_abbr,
            home_skaters,
            away_skaters
        ),
        opponent_team_skaters = ifelse(
            event_team_abbr == home_abbr,
            away_skaters,
            home_skaters
        ),
        total_skaters_on = event_team_skaters + opponent_team_skaters,
        event_team_advantage = event_team_skaters - opponent_team_skaters,
        # 5v5 model features
        rebound = ifelse(
            last_event_type %in%
                c("SHOT", "MISSED_SHOT", "GOAL") &
                time_since_last <= 2,
            1,
            0
        ),
        rush = ifelse(
            last_event_zone %in% c("NZ", "DZ") & time_since_last <= 4,
            1,
            0
        ),
        cross_ice_event = ifelse(
            last_event_zone == "OZ" &
                ((last_y > 3 & y < -3) | (last_y < -3 & y > 3)) &
                time_since_last <= 2,
            1,
            0
        ),
        empty_net = ifelse(is.na(empty_net) | empty_net == FALSE, FALSE, TRUE),
        shot_type = secondary_type,
        goal = ifelse(event_type == "GOAL", 1, 0)
    ) %>%
    select(
        season,
        game_id,
        event_id,
        strength_state,
        shot_distance,
        shot_angle,
        empty_net,
        last_event_type,
        last_event_team,
        time_since_last,
        last_x,
        last_y,
        distance_from_last,
        event_zone,
        last_event_zone,
        era_2011_2013,
        era_2014_2018,
        era_2019_2021,
        era_2022_on,
        event_team_skaters,
        opponent_team_skaters,
        total_skaters_on,
        event_team_advantage,
        rebound,
        rush,
        cross_ice_event,
        shot_type,
        goal
    ) %>%
    # one-hot encode shot_type and last_event_type
    mutate(type_value = 1L, last_value = 1L) %>%
    tidyr::pivot_wider(
        names_from = shot_type,
        values_from = type_value,
        values_fill = 0L,
        values_fn = max
    ) %>%
    tidyr::pivot_wider(
        names_from = last_event_type,
        values_from = last_value,
        values_fill = 0L,
        values_fn = max,
        names_prefix = "last_"
    ) %>%
    janitor::clean_names() %>%
    select(
        -last_event_team,
        -event_zone,
        -last_event_zone,
        -event_team_skaters,
        -opponent_team_skaters
    )

# Drop NA column if created from NA shot_type values
if ("na" %in% names(model_df)) {
    model_df <- select(model_df, -na)
}

cat("Feature matrix dimensions:", nrow(model_df), "x", ncol(model_df), "\n")
cat("Goal rate:", round(mean(model_df$goal), 4), "\n\n")

# --------------------------------------------------------------------------
# 3. Train 5v5 model
# --------------------------------------------------------------------------
cat("Training 5v5 model...\n")

df_5v5 <- model_df %>%
    filter(strength_state == "5v5") %>%
    select(
        -season,
        -game_id,
        -event_id,
        -strength_state,
        -total_skaters_on,
        -event_team_advantage
    )

# Ensure numeric
df_5v5 <- df_5v5 %>% mutate(across(everything(), as.numeric))

cat("  5v5 shots:", nrow(df_5v5), "| Goals:", sum(df_5v5$goal), "\n")

feat_names_5v5 <- setdiff(names(df_5v5), "goal")

dtrain_5v5 <- xgb.DMatrix(
    data = as.matrix(df_5v5[, feat_names_5v5]),
    label = df_5v5$goal
)

params_5v5 <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,
    scale_pos_weight = (sum(df_5v5$goal == 0) / sum(df_5v5$goal == 1))
)

set.seed(42)
xg_model_5v5 <- xgb.train(
    params = params_5v5,
    data = dtrain_5v5,
    nrounds = 500,
    verbose = 1,
    print_every_n = 50
)

cat(
    "5v5 model trained. Features:",
    length(xgboost:::xgb.feature_names(xg_model_5v5)),
    "\n\n"
)

# --------------------------------------------------------------------------
# 4. Train special teams (ST) model
# --------------------------------------------------------------------------
cat("Training special teams model...\n")

df_st <- model_df %>%
    filter(strength_state != "5v5") %>%
    select(-season, -game_id, -event_id, -strength_state)

df_st <- df_st %>% mutate(across(everything(), as.numeric))

cat("  ST shots:", nrow(df_st), "| Goals:", sum(df_st$goal), "\n")

feat_names_st <- setdiff(names(df_st), "goal")

dtrain_st <- xgb.DMatrix(
    data = as.matrix(df_st[, feat_names_st]),
    label = df_st$goal
)

params_st <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    scale_pos_weight = (sum(df_st$goal == 0) / sum(df_st$goal == 1))
)

set.seed(42)
xg_model_st <- xgb.train(
    params = params_st,
    data = dtrain_st,
    nrounds = 500,
    verbose = 1,
    print_every_n = 50
)

cat(
    "ST model trained. Features:",
    length(xgboost:::xgb.feature_names(xg_model_st)),
    "\n\n"
)

# --------------------------------------------------------------------------
# 5. Penalty shot xG constant
# --------------------------------------------------------------------------
# Historical penalty shot conversion rate (roughly 32-33%)
xg_model_ps <- 0.326

cat("Penalty shot xG constant:", xg_model_ps, "\n\n")

# --------------------------------------------------------------------------
# 6. Save models as JSON to inst/extdata/  (portable modern format)
#    AND rebuild sysdata.rda
# --------------------------------------------------------------------------
if (!dir.exists("inst/extdata")) {
    dir.create("inst/extdata", recursive = TRUE)
}

cat("Saving models as JSON...\n")
xgb.save(xg_model_5v5, "inst/extdata/xg_model_5v5.json")
xgb.save(xg_model_st, "inst/extdata/xg_model_st.json")
cat("JSON models saved to inst/extdata/\n\n")

# --------------------------------------------------------------------------
# 7. Restore team_abbr_yearly and xg_model_ps from extracted .rds files
# --------------------------------------------------------------------------
cat("Restoring team_abbr_yearly and xg_model_ps from extracted files...\n")

team_abbr_yearly_path <- "R/team_abbr_yearly.rds"
xg_model_ps_path <- "R/xg_model_ps.rds"

if (file.exists(team_abbr_yearly_path)) {
    team_abbr_yearly <- readRDS(team_abbr_yearly_path)
    cat("  Loaded team_abbr_yearly:", nrow(team_abbr_yearly), "rows\n")
} else {
    stop("R/team_abbr_yearly.rds not found - run extract_team_abbr.R first!")
}

if (file.exists(xg_model_ps_path)) {
    xg_model_ps <- readRDS(xg_model_ps_path)
    cat("  xg_model_ps:", xg_model_ps, "\n")
} else {
    xg_model_ps <- 0.3183531 # value extracted from original sysdata.rda
    cat("  Using hardcoded xg_model_ps:", xg_model_ps, "\n")
}

# --------------------------------------------------------------------------
# 8. Save updated sysdata.rda
#    Reload models from JSON (ensures clean modern serialization)
# --------------------------------------------------------------------------
cat("Loading freshly saved JSON models back for sysdata.rda...\n")
xg_model_5v5 <- xgb.load("inst/extdata/xg_model_5v5.json")
xg_model_st <- xgb.load("inst/extdata/xg_model_st.json")

# Store feature names as explicit character vectors (robust against future xgboost API changes)
xg_feature_names_5v5 <- xgboost:::xgb.feature_names(xg_model_5v5)
xg_feature_names_st <- xgboost:::xgb.feature_names(xg_model_st)
cat("5v5 feature names count:", length(xg_feature_names_5v5), "\n")
cat("ST feature names count:", length(xg_feature_names_st), "\n")

cat("Saving updated R/sysdata.rda...\n")
save(
    xg_model_5v5,
    xg_model_st,
    xg_feature_names_5v5,
    xg_feature_names_st,
    xg_model_ps,
    team_abbr_yearly,
    file = "R/sysdata.rda",
    compress = "bzip2"
)

cat("\n=== Done! ===\n")
cat("Updated files:\n")
cat(
    "  R/sysdata.rda              - updated xgboost models (modern format) + team_abbr_yearly\n"
)
cat("  inst/extdata/xg_model_5v5.json - 5v5 model JSON\n")
cat("  inst/extdata/xg_model_st.json  - special teams model JSON\n")
cat("\nNext steps:\n")
cat(
    "  1. Run extract_team_abbr_yearly.R to rebuild team_abbr_yearly properly\n"
)
cat(
    "  2. Reinstall hockeyR: install.packages('.', repos=NULL, type='source')\n"
)
cat("  3. Re-run scrape_full_season.R\n")
