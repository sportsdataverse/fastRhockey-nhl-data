# retrain_xg_models.R
#
# Training script for fastRhockey xG models.
# Save this in the fastRhockey-nhl-data repository and run from
# the repo root:
#
#   Rscript xg_models/retrain_xg_models.R
#
# Outputs (to xg_models/):
#   - xg_model_5v5.json        XGBoost JSON model (5v5)
#   - xg_model_st.json         XGBoost JSON model (special teams)
#   - xg_model_meta.rds        Metadata: feature names + penalty shot constant
#   - cv_results_5v5.rds       Cross-validation results (5v5)
#   - cv_results_st.rds        Cross-validation results (special teams)
#
# Requirements:
#   install.packages(c("xgboost", "dplyr", "tidyr", "janitor", "readr",
#                       "arrow", "purrr", "stringr", "glue"))
#
# Data source: fastRhockey-nhl-data PBP lite parquets or hockeyR-data .rds files
# The script expects either:
#   (a) hockeyR-data .rds files from ../hockeyR-data/pbp_data/
#   (b) fastRhockey-nhl-data lite parquets from nhl/pbp/lite/parquet/
#
# --------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(xgboost)
    library(dplyr)
    library(tidyr)
    library(janitor)
    library(purrr)
    library(glue)
})

# ----- Configuration -----
output_dir <- "xg_models"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

SEED <- 42
set.seed(SEED)

# Columns required from PBP data
NEEDED_COLS <- c(
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

# ----- 1. Load PBP Data -----
message(glue("{Sys.time()}: Loading PBP data..."))

# Try hockeyR-data .rds files first (these have the most complete data)
rds_dir <- "nhl/pbp/full/rds"
lite_dir <- "nhl/pbp/lite/parquet"

if (dir.exists(rds_dir)) {
    rds_files <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)
    message(glue("  Found {length(rds_files)} .rds files in {rds_dir}"))
    all_pbp <- purrr::map_dfr(rds_files, function(f) {
        message(glue("  Loading {basename(f)}..."))
        df <- readRDS(f)
        # Map column names to our standard
        if ("home_team_abbr" %in% names(df) && !("home_abbr" %in% names(df))) {
            df$home_abbr <- df$home_team_abbr
        }
        if ("away_team_abbr" %in% names(df) && !("away_abbr" %in% names(df))) {
            df$away_abbr <- df$away_team_abbr
        }
        # Coerce season to character to avoid type mismatch across seasons
        if ("season" %in% names(df)) {
            df$season <- as.character(df$season)
        }
        # Select only needed columns (if available)
        available <- intersect(NEEDED_COLS, names(df))
        df[, available, drop = FALSE]
    })
} else if (dir.exists(lite_dir)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
        stop(
            "Package 'arrow' is required to read parquet files.",
            call. = FALSE
        )
    }
    pq_files <- list.files(lite_dir, pattern = "\\.parquet$", full.names = TRUE)
    message(glue("  Found {length(pq_files)} parquet files in {lite_dir}"))
    all_pbp <- purrr::map_dfr(pq_files, function(f) {
        message(glue("  Loading {basename(f)}..."))
        df <- arrow::read_parquet(f)
        available <- intersect(NEEDED_COLS, names(df))
        df[, available, drop = FALSE]
    })
} else {
    stop(
        "No PBP data found.\n",
        "Expected either: ",
        rds_dir,
        "/ (.rds) or ",
        lite_dir,
        "/ (.parquet)",
        call. = FALSE
    )
}

message(glue("  Total rows loaded: {nrow(all_pbp)}"))

# ----- 2. Feature Engineering -----
message(glue("{Sys.time()}: Engineering features..."))

# Coerce types
all_pbp <- all_pbp %>%
    mutate(
        season = as.character(season),
        game_id = as.character(game_id),
        event_id = as.integer(event_id),
        period = as.integer(period),
        game_seconds = as.numeric(game_seconds),
        x = as.numeric(x),
        y = as.numeric(y),
        x_fixed = as.numeric(x_fixed),
        shot_distance = as.numeric(shot_distance),
        shot_angle = as.numeric(shot_angle),
        home_skaters = as.integer(home_skaters),
        away_skaters = as.integer(away_skaters),
        empty_net = as.logical(empty_net)
    )

# Feature engineering pipeline (identical to helper_nhl_prepare_xg_data.R)
model_df <- all_pbp %>%
    # Remove shootouts
    filter(period_type != "SHOOTOUT") %>%
    # Remove penalty shots
    filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) %>%
    # Remove shift changes
    filter(event_type != "CHANGE") %>%
    # Lag features
    group_by(game_id, period) %>%
    mutate(
        last_event_type = lag(event_type),
        last_event_team = lag(event_team_abbr),
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
    # Keep only shot attempts (unblocked)
    filter(event_type %in% c("SHOT", "MISSED_SHOT", "GOAL")) %>%
    # Valid last-event types
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
    # Era dummies
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
        # Skater features (for ST model)
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
        # Tactical features
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
    # One-hot encode categoricals
    mutate(type_value = 1L, last_value = 1L) %>%
    pivot_wider(
        names_from = shot_type,
        values_from = type_value,
        values_fill = 0L,
        values_fn = max
    ) %>%
    pivot_wider(
        names_from = last_event_type,
        values_from = last_value,
        values_fill = 0L,
        values_fn = max,
        names_prefix = "last_"
    ) %>%
    clean_names() %>%
    select(
        -last_event_team,
        -event_zone,
        -last_event_zone,
        -event_team_skaters,
        -opponent_team_skaters
    )

# Drop NA column from missing shot_type values
if ("na" %in% names(model_df)) {
    model_df <- select(model_df, -na)
}

# Remove rows with NA in critical features
model_df <- model_df %>%
    filter(
        !is.na(shot_distance),
        !is.na(shot_angle),
        !is.na(time_since_last)
    )

message(glue("  Rows after feature engineering: {nrow(model_df)}"))
message(glue("  Goal rate: {round(mean(model_df$goal), 4)}"))

# ----- 3. Train 5v5 Model -----
message(glue("\n{Sys.time()}: Training 5v5 model..."))

data_5v5 <- model_df %>%
    filter(strength_state == "5v5") %>%
    select(
        -season,
        -game_id,
        -event_id,
        -strength_state,
        -total_skaters_on,
        -event_team_advantage
    )

feat_names_5v5 <- setdiff(names(data_5v5), "goal")
message(glue(
    "  5v5 features ({length(feat_names_5v5)}): {paste(feat_names_5v5, collapse = ', ')}"
))
message(glue("  5v5 training rows: {nrow(data_5v5)}"))

labels_5v5 <- data_5v5$goal
mat_5v5 <- as.matrix(data_5v5[, feat_names_5v5])
dtrain_5v5 <- xgb.DMatrix(data = mat_5v5, label = labels_5v5)

# Scale positive weight
pos_weight_5v5 <- sum(labels_5v5 == 0) / sum(labels_5v5 == 1)

params_5v5 <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,
    scale_pos_weight = pos_weight_5v5,
    seed = SEED
)

# Cross-validation
message("  Running 5-fold CV...")
cv_5v5 <- xgb.cv(
    params = params_5v5,
    data = dtrain_5v5,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    print_every_n = 100,
    verbose = 1
)

best_nrounds_5v5 <- cv_5v5$best_iteration
message(glue("  Best 5v5 nrounds (from CV): {best_nrounds_5v5}"))

# Train final model (use 500 or CV best, whichever is smaller fallback)
if (is.null(best_nrounds_5v5) || best_nrounds_5v5 < 1) {
    best_nrounds_5v5 <- 500
}

model_5v5 <- xgb.train(
    params = params_5v5,
    data = dtrain_5v5,
    nrounds = best_nrounds_5v5
)

message(glue("  5v5 model trained with {best_nrounds_5v5} rounds"))

# ----- 4. Train Special Teams Model -----
message(glue("\n{Sys.time()}: Training special teams model..."))

data_st <- model_df %>%
    filter(strength_state != "5v5") %>%
    select(-season, -game_id, -event_id, -strength_state)

feat_names_st <- setdiff(names(data_st), "goal")
message(glue(
    "  ST features ({length(feat_names_st)}): {paste(feat_names_st, collapse = ', ')}"
))
message(glue("  ST training rows: {nrow(data_st)}"))

labels_st <- data_st$goal
mat_st <- as.matrix(data_st[, feat_names_st])
dtrain_st <- xgb.DMatrix(data = mat_st, label = labels_st)

pos_weight_st <- sum(labels_st == 0) / sum(labels_st == 1)

params_st <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    scale_pos_weight = pos_weight_st,
    seed = SEED
)

message("  Running 5-fold CV...")
cv_st <- xgb.cv(
    params = params_st,
    data = dtrain_st,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    print_every_n = 100,
    verbose = 1
)

best_nrounds_st <- cv_st$best_iteration
message(glue("  Best ST nrounds (from CV): {best_nrounds_st}"))

if (is.null(best_nrounds_st) || best_nrounds_st < 1) {
    best_nrounds_st <- 500
}

model_st <- xgb.train(
    params = params_st,
    data = dtrain_st,
    nrounds = best_nrounds_st
)

message(glue("  ST model trained with {best_nrounds_st} rounds"))

# ----- 5. Penalty Shot Constant -----
# Penalty shot xG is a fixed value based on historical conversion rate
penalty_shots <- all_pbp %>%
    filter(
        !is.na(secondary_type),
        secondary_type == "Penalty Shot",
        event_type %in% c("SHOT", "MISSED_SHOT", "GOAL")
    )

if (nrow(penalty_shots) > 0) {
    xg_model_ps <- mean(penalty_shots$event_type == "GOAL")
} else {
    xg_model_ps <- 0.326 # Historical fallback
}
message(glue("\n  Penalty shot xG constant: {round(xg_model_ps, 4)}"))

# ----- 6. Save Models & Metadata -----
message(glue("\n{Sys.time()}: Saving models to {output_dir}/..."))

# Save JSON models
xgb.save(model_5v5, file.path(output_dir, "xg_model_5v5.json"))
xgb.save(model_st, file.path(output_dir, "xg_model_st.json"))

# Save metadata RDS (feature names + penalty shot constant)
meta <- list(
    xg_feature_names_5v5 = feat_names_5v5,
    xg_feature_names_st = feat_names_st,
    xg_model_ps = xg_model_ps,
    training_date = Sys.time(),
    training_rows_5v5 = nrow(data_5v5),
    training_rows_st = nrow(data_st),
    nrounds_5v5 = best_nrounds_5v5,
    nrounds_st = best_nrounds_st,
    params_5v5 = params_5v5,
    params_st = params_st
)
saveRDS(meta, file.path(output_dir, "xg_model_meta.rds"))

# Save CV results
saveRDS(cv_5v5, file.path(output_dir, "cv_results_5v5.rds"))
saveRDS(cv_st, file.path(output_dir, "cv_results_st.rds"))

message(glue("\n{Sys.time()}: Done! Files saved:"))
message(glue("  {output_dir}/xg_model_5v5.json"))
message(glue("  {output_dir}/xg_model_st.json"))
message(glue("  {output_dir}/xg_model_meta.rds"))
message(glue("  {output_dir}/cv_results_5v5.rds"))
message(glue("  {output_dir}/cv_results_st.rds"))
message(glue(
    "\nCommit and push these files to GitHub so fastRhockey can download them."
))
