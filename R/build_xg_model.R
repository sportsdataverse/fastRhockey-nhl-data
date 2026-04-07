#################################
#                               #
#   fastRhockey xG Model        #
#                               #
#   Expected Goals Model         #
#   Builder & Analysis Script    #
#                               #
#   Analogous to hockeyR-models  #
#   R/build_xg_model.R           #
#                               #
#################################
#
# Run from fastRhockey-nhl-data repo root:
#   Rscript --vanilla R/build_xg_model.R
#
# Outputs:
#   models/xg_model_5v5.json                 XGBoost JSON (5v5)
#   models/xg_model_st.json                  XGBoost JSON (special teams)
#   models/xg_model_meta.rds                 Feature names + penalty shot constant
#   data/cv_results_5v5_final.rds            CV results (5v5)
#   data/cv_results_st_final.rds             CV results (special teams)
#   figures/fastRhockey_xg_5v5_feature_importance.png
#   figures/fastRhockey_xg_st_feature_importance.png
#
# Era groupings (by season ending year):
#   era_2011_2013 : 2010-11 through 2012-13
#   era_2014_2018 : 2013-14 through 2017-18
#   era_2019_2021 : 2018-19 through 2020-21
#   era_2022_2024 : 2021-22 through 2023-24
#   era_2025_on   : 2024-25 and beyond
#
# Key design choice: Only unblocked shots (Fenwick events: SHOT, MISSED_SHOT,
# GOAL) are used. Blocked shots are excluded because the recorded location is
# the block location, not the shooter's location.
#
# --------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(stringr)
    library(glue)
    library(xgboost, exclude = "slice")
    library(janitor)
})

`%not_in%` <- purrr::negate(`%in%`)

# ---------- Output directories ----------
for (d in c("models", "data", "figures")) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

############ LOAD DATA ##############

message(glue("{Sys.time()}: Loading PBP data..."))

rds_dir <- "nhl/pbp/full/rds"
if (!dir.exists(rds_dir)) {
    stop("PBP data directory not found: ", rds_dir, call. = FALSE)
}

rds_files <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)
message(glue("  Found {length(rds_files)} season files in {rds_dir}"))

pbp_all <- purrr::map_dfr(rds_files, function(f) {
    message(glue("  Loading {basename(f)}..."))
    df <- readRDS(f)

    # --- Column name harmonisation ---
    # fastRhockey-nhl-data RDS files may use either hockeyR names
    # (event_team, home_name, away_name) or new-API names
    # (event_team_abbr, home_abbr, away_abbr, home_team_abbr, etc.)
    # Standardise to: event_team, home_team, away_team

    # event_team
    if (!("event_team" %in% names(df))) {
        if ("event_team_abbr" %in% names(df)) {
            df$event_team <- df$event_team_abbr
        } else if ("event_owner_team_abbr" %in% names(df)) {
            df$event_team <- df$event_owner_team_abbr
        }
    }

    # home_team
    if (!("home_team" %in% names(df))) {
        if ("home_name" %in% names(df)) {
            df$home_team <- df$home_name
        } else if ("home_abbr" %in% names(df)) {
            df$home_team <- df$home_abbr
        } else if ("home_team_abbr" %in% names(df)) {
            df$home_team <- df$home_team_abbr
        }
    }

    # away_team
    if (!("away_team" %in% names(df))) {
        if ("away_name" %in% names(df)) {
            df$away_team <- df$away_name
        } else if ("away_abbr" %in% names(df)) {
            df$away_team <- df$away_abbr
        } else if ("away_team_abbr" %in% names(df)) {
            df$away_team <- df$away_team_abbr
        }
    }

    # Ensure consistent column types across seasons
    if ("season" %in% names(df)) {
        df$season <- as.character(df$season)
    }
    if ("game_id" %in% names(df)) {
        df$game_id <- as.character(df$game_id)
    }
    if ("game_date" %in% names(df)) {
        df$game_date <- as.character(df$game_date)
    }
    if ("x" %in% names(df)) {
        df$x <- as.numeric(df$x)
    }
    if ("y" %in% names(df)) {
        df$y <- as.numeric(df$y)
    }
    if ("home_skaters" %in% names(df)) {
        df$home_skaters <- as.numeric(df$home_skaters)
    }
    if ("away_skaters" %in% names(df)) {
        df$away_skaters <- as.numeric(df$away_skaters)
    }
    if ("home_id" %in% names(df)) {
        df$home_id <- as.character(df$home_id)
    }
    if ("away_id" %in% names(df)) {
        df$away_id <- as.character(df$away_id)
    }
    if ("description" %in% names(df)) {
        df$description <- as.character(df$description)
    }

    # Create event_id if missing (from game_id + event_idx)
    if (!("event_id" %in% names(df)) && "event_idx" %in% names(df)) {
        df <- df %>%
            mutate(
                event_idx_pad = stringr::str_pad(
                    event_idx,
                    width = 4,
                    side = "left",
                    pad = "0"
                ),
                event_id = as.numeric(paste0(game_id, event_idx_pad))
            ) %>%
            select(-event_idx_pad)
    }

    df
})

message(glue("  Total rows loaded: {nrow(pbp_all)}"))
message(glue(
    "  Seasons: {paste(sort(unique(pbp_all$season)), collapse = ', ')}"
))

############# DATA CLEANING #############

# Define unblocked shot types — no BLOCKED_SHOT
fenwick <- c("SHOT", "MISSED_SHOT", "GOAL")

# Valid strength states
real_strengths <- c(
    "5v5",
    "5v4",
    "5v3",
    "6v5",
    "6v4",
    "6v3",
    "4v3",
    "3v3",
    "4v4",
    "4v5",
    "3v5",
    "5v6",
    "4v6",
    "3v6",
    "3v4"
)

# --- Normalize secondary_type across NHL API eras ---
# Old API (2010-2022): Title Case ("Wrist Shot", "Snap Shot", etc.)
# New API (2023+): lowercase abbreviated ("wrist", "snap", etc.)
# Normalize everything to the old Title Case convention so that
# pivot_wider produces one column per shot type, not duplicates.
pbp_all <- pbp_all %>%
    mutate(
        secondary_type = case_when(
            is.na(secondary_type) ~ NA_character_,
            tolower(secondary_type) == "wrist" ~ "Wrist Shot",
            tolower(secondary_type) == "wrist shot" ~ "Wrist Shot",
            tolower(secondary_type) == "snap" ~ "Snap Shot",
            tolower(secondary_type) == "snap shot" ~ "Snap Shot",
            tolower(secondary_type) == "slap" ~ "Slap Shot",
            tolower(secondary_type) == "slap shot" ~ "Slap Shot",
            tolower(secondary_type) == "backhand" ~ "Backhand",
            tolower(secondary_type) == "deflected" ~ "Deflected",
            tolower(secondary_type) == "tip-in" ~ "Tip-In",
            tolower(secondary_type) == "wrap-around" ~ "Wrap-around",
            tolower(secondary_type) == "bat" ~ "Batted",
            tolower(secondary_type) == "batted" ~ "Batted",
            tolower(secondary_type) == "poke" ~ "Poke",
            tolower(secondary_type) == "between-legs" ~ "Between Legs",
            tolower(secondary_type) == "between legs" ~ "Between Legs",
            tolower(secondary_type) == "cradle" ~ "Cradle",
            tolower(secondary_type) == "penalty shot" ~ "Penalty Shot",
            TRUE ~ secondary_type
        )
    )

n_types <- length(unique(pbp_all$secondary_type[
    !is.na(pbp_all$secondary_type)
]))
message(glue(
    "  Normalized secondary_type to {n_types} canonical values: ",
    "{paste(sort(unique(pbp_all$secondary_type[!is.na(pbp_all$secondary_type)])), collapse = ', ')}"
))

# --- Strength state check (for logging) ---
if ("strength_state" %in% names(pbp_all)) {
    strength_check <- pbp_all %>%
        filter(event_type == "GOAL") %>%
        filter(!is.na(strength_state)) %>%
        mutate(real = ifelse(strength_state %in% real_strengths, 1L, 0L)) %>%
        group_by(real) %>%
        summarise(goals = n(), .groups = "drop")

    good_goals <- strength_check %>% filter(real == 1L) %>% pull(goals)
    bad_goals <- strength_check %>% filter(real == 0L) %>% pull(goals)
    if (length(bad_goals) == 0) {
        bad_goals <- 0
    }
    total_goals <- sum(strength_check$goals)

    message(glue(
        "\n  Strength state check: {good_goals} valid / {bad_goals} invalid ",
        "out of {total_goals} total goals ",
        "({round(100 * bad_goals / total_goals, 3)}% bad)"
    ))
}

############# PENALTY SHOT 'MODEL' ##########

message(glue("\n{Sys.time()}: Computing penalty shot xG constant..."))

pens <- pbp_all %>%
    filter(
        period_type == "SHOOTOUT" |
            (!is.na(secondary_type) & secondary_type == "Penalty Shot")
    ) %>%
    filter(event_type %in% fenwick)

if (nrow(pens) > 0) {
    ps_xg <- mean(pens$event_type == "GOAL")
} else {
    ps_xg <- 0.326 # historical fallback
}

message(glue("  Penalty shot / shootout attempts: {nrow(pens)}"))
message(glue("  Conversion rate (xG constant): {round(ps_xg, 4)}"))

############ 5v5 MODEL ############

message(glue("\n{Sys.time()}: Preparing 5v5 model data..."))

# features (before one-hot encoding)
model_feats_5v5 <- c(
    "shot_distance",
    "shot_angle",
    "shot_type",
    "rebound",
    "rush",
    "last_event_type",
    "time_since_last",
    "distance_from_last",
    "cross_ice_event",
    "empty_net",
    "last_x",
    "last_y",
    # era dummies added via starts_with("era")
    # target
    "goal"
)

pbp_shots_5v5 <- pbp_all %>%
    # filter out shootouts
    filter(period_type != "SHOOTOUT") %>%
    # remove penalty shots
    filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) %>%
    # add model feature variables
    group_by(game_id) %>%
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
            (x_fixed < -25 & event_team == home_team) |
                (x_fixed > 25 & event_team == away_team) ~ "DZ",
            (x_fixed > 25 & event_team == home_team) |
                (x_fixed < -25 & event_team == away_team) ~ "OZ"
        ),
        last_event_zone = lag(event_zone)
    ) %>%
    ungroup() %>%
    # filter to only unblocked shots (no BLOCKED_SHOT)
    filter(event_type %in% fenwick) %>%
    # get rid of oddball last_events (e.g., EARLY_INTERMISSION_START)
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
    # add more feature variables
    mutate(
        # --- ERA DUMMIES ---
        # era_2011_2013: goalie pad era
        era_2011_2013 = ifelse(
            season %in% c("20102011", "20112012", "20122013"),
            1,
            0
        ),
        # era_2014_2018: goalie pad reduction, shallower nets
        era_2014_2018 = ifelse(
            season %in%
                c(
                    "20132014",
                    "20142015",
                    "20152016",
                    "20162017",
                    "20172018"
                ),
            1,
            0
        ),
        # era_2019_2021: goalie chest/arm protector reduction
        era_2019_2021 = ifelse(
            season %in% c("20182019", "20192020", "20202021"),
            1,
            0
        ),
        # era_2022_2024: cross-checking emphasis era
        era_2022_2024 = ifelse(
            season %in% c("20212022", "20222023", "20232024"),
            1,
            0
        ),
        # era_2025_on: latest rules / trends
        era_2025_on = ifelse(
            as.numeric(season) > 20232024,
            1,
            0
        ),
        # tactical features
        rebound = ifelse(
            last_event_type %in% fenwick & time_since_last <= 2,
            1,
            0
        ),
        rush = ifelse(
            last_event_zone %in% c("NZ", "DZ") & time_since_last <= 4,
            1,
            0
        ),
        cross_ice_event = ifelse(
            # indicates goalie had to move from one post to the other
            last_event_zone == "OZ" &
                ((lag(y) > 3 & y < -3) | (lag(y) < -3 & y > 3)) &
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
        starts_with("era"),
        all_of(model_feats_5v5)
    ) %>%
    # one-hot encode categorical variables
    mutate(type_value = 1, last_value = 1) %>%
    pivot_wider(
        names_from = shot_type,
        values_from = type_value,
        values_fill = 0
    ) %>%
    pivot_wider(
        names_from = last_event_type,
        values_from = last_value,
        values_fill = 0,
        names_prefix = "last_"
    ) %>%
    clean_names() %>%
    select(-any_of("na"))

n_before <- nrow(pbp_shots_5v5)
pbp_shots_5v5 <- na.omit(pbp_shots_5v5)
n_after <- nrow(pbp_shots_5v5)

message(glue("  5v5 rows before NA removal: {n_before}"))
message(glue("  5v5 rows after NA removal:  {n_after}"))
message(glue(
    "  Removed {n_before - n_after} rows ({round(100 * (n_before - n_after) / n_before, 2)}%)"
))
message(glue("  Goal rate: {round(mean(pbp_shots_5v5$goal), 4)}"))

# --- Train/test split (grouped by game_id) ---

set.seed(37) # yanni gogo gang (following hockeyR-models)

game_ids_5v5 <- unique(pbp_shots_5v5$game_id)
n_train <- floor(0.8 * length(game_ids_5v5))
train_game_ids <- sample(game_ids_5v5, n_train)

train_shots_5v5 <- pbp_shots_5v5 %>% filter(game_id %in% train_game_ids)
test_shots_5v5 <- pbp_shots_5v5 %>% filter(game_id %not_in% train_game_ids)

message(glue(
    "  Train games: {n_train}, Test games: {length(game_ids_5v5) - n_train}"
))
message(glue(
    "  Train rows: {nrow(train_shots_5v5)}, Test rows: {nrow(test_shots_5v5)}"
))

# Create grouped CV folds (5 folds, grouped by game)
train_game_ids_5v5 <- unique(train_shots_5v5$game_id)
fold_assignment_5v5 <- sample(rep(1:5, length.out = length(train_game_ids_5v5)))
names(fold_assignment_5v5) <- train_game_ids_5v5
train_shots_5v5$fold <- fold_assignment_5v5[train_shots_5v5$game_id]
folds_5v5 <- lapply(1:5, function(k) which(train_shots_5v5$fold == k))
train_shots_5v5 <- select(train_shots_5v5, -fold)

# Drop ID columns before training
train_shots_5v5 <- train_shots_5v5 %>%
    select(-event_id, -season, -game_id)
test_shots_5v5 <- test_shots_5v5 %>%
    select(-event_id, -season, -game_id)

feat_names_5v5 <- setdiff(names(train_shots_5v5), "goal")
message(glue(
    "  5v5 features ({length(feat_names_5v5)}): {paste(feat_names_5v5, collapse = ', ')}"
))

train_set_5v5 <- train_shots_5v5 %>% select(-goal) %>% data.matrix()
train_labels_5v5 <- train_shots_5v5 %>% select(goal) %>% data.matrix()
test_set_5v5 <- test_shots_5v5 %>% select(-goal) %>% data.matrix()
test_labels_5v5 <- test_shots_5v5 %>% select(goal) %>% data.matrix()

dtrain_5v5 <- xgb.DMatrix(data = train_set_5v5, label = train_labels_5v5)
dtest_5v5 <- xgb.DMatrix(data = test_set_5v5, label = test_labels_5v5)

# --- Hyperparameter tuning (grid search over min_child_weight) ---
# Base params from hockeyR-models tuning. Only searching min_child_weight.

message(glue(
    "\n{Sys.time()}: 5v5 hyperparameter tuning (min_child_weight grid)..."
))

grid_search_5v5 <- expand.grid(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 4,
    eta = .06,
    gamma = 1,
    subsample = .8,
    colsample_bytree = .8,
    min_child_weight = 1:10,
    stringsAsFactors = FALSE
)

param_tune_5v5 <- function(param) {
    cv_model <- xgb.cv(
        data = dtrain_5v5,
        params = as.list(param),
        folds = folds_5v5,
        nrounds = 1000,
        verbose = FALSE,
        early_stopping_rounds = 30
    )
    # best_iteration is NULL when early stopping doesn't trigger
    best_iter <- cv_model$best_iteration
    if (is.null(best_iter)) {
        best_iter <- nrow(cv_model$evaluation_log)
    }
    tibble(
        logloss = min(cv_model$evaluation_log$test_logloss_mean),
        nrounds = best_iter,
        min_child_weight = param$min_child_weight
    )
}

cv_grid_5v5 <- purrr::map_dfr(
    .x = seq_len(nrow(grid_search_5v5)),
    ~ {
        message(glue(
            "  Tuning min_child_weight = {grid_search_5v5[.x, 'min_child_weight']}..."
        ))
        param_tune_5v5(grid_search_5v5[.x, ])
    }
)

best_mcw_5v5 <- cv_grid_5v5 %>%
    arrange(logloss) %>%
    slice(1)

message(glue(
    "\n  Best min_child_weight: {best_mcw_5v5$min_child_weight} ",
    "(logloss = {round(best_mcw_5v5$logloss, 6)}, nrounds = {best_mcw_5v5$nrounds})"
))

############# 5v5: TRAIN FINAL MODEL WITH TUNED PARAMS ################

message(glue("\n{Sys.time()}: Training final 5v5 model..."))

params_5v5 <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eval_metric = "auc",
    max_depth = 4,
    eta = .06,
    gamma = 1,
    subsample = .8,
    colsample_bytree = .8,
    min_child_weight = best_mcw_5v5$min_child_weight
)

# Final CV to get nrounds and metrics
cv_results_5v5 <- xgb.cv(
    data = dtrain_5v5,
    params = params_5v5,
    folds = folds_5v5,
    nrounds = 1500,
    verbose = TRUE,
    print_every_n = 50,
    early_stopping_rounds = 30
)

saveRDS(cv_results_5v5, "data/cv_results_5v5_final.rds")

rounds_5v5 <- cv_results_5v5$best_iteration
if (is.null(rounds_5v5)) {
    rounds_5v5 <- nrow(cv_results_5v5$evaluation_log)
}
cv_logloss_5v5 <- cv_results_5v5$evaluation_log$test_logloss_mean[rounds_5v5]
cv_auc_5v5 <- cv_results_5v5$evaluation_log$test_auc_mean[rounds_5v5]

message(glue("  5v5 CV log-loss: {round(cv_logloss_5v5, 4)}"))
message(glue("  5v5 CV AUC:      {round(cv_auc_5v5, 4)}"))
message(glue("  Best nrounds:    {rounds_5v5}"))

# Train final model
xg_model_5v5 <- xgb.train(
    data = dtrain_5v5,
    params = params_5v5,
    nrounds = rounds_5v5
)

# Feature importance
importance_5v5 <- xgb.importance(feat_names_5v5, model = xg_model_5v5)

png(
    "figures/fastRhockey_xg_5v5_feature_importance.png",
    width = 8,
    height = 5,
    units = "in",
    res = 500
)
par(mar = c(5, 12, 3, 2))
imp_5v5_sorted <- importance_5v5[order(importance_5v5$Gain), ]
barplot(
    imp_5v5_sorted$Gain,
    names.arg = imp_5v5_sorted$Feature,
    horiz = TRUE,
    las = 1,
    col = "#99D9D9",
    border = "#001628",
    xlab = "Importance (Gain)",
    main = "fastRhockey 5v5 Expected Goals model feature importance",
    cex.names = 0.6
)
dev.off()

message("  Saved figures/fastRhockey_xg_5v5_feature_importance.png")

# Save model
xgb.save(xg_model_5v5, "models/xg_model_5v5.json")
message("  Saved models/xg_model_5v5.json")

# --- Evaluate on holdout test set ---
preds_5v5 <- predict(xg_model_5v5, dtest_5v5)
test_logloss_5v5 <- -mean(
    test_labels_5v5 *
        log(pmax(preds_5v5, 1e-15)) +
        (1 - test_labels_5v5) * log(pmax(1 - preds_5v5, 1e-15))
)

# Simple AUC calculation (rank-based, no extra packages needed)
.calc_auc <- function(pred, actual) {
    n1 <- as.double(sum(actual == 1))
    n0 <- as.double(sum(actual == 0))
    ranks <- rank(pred)
    auc <- (sum(ranks[actual == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc
}
test_auc_5v5 <- .calc_auc(preds_5v5, as.vector(test_labels_5v5))

message(glue("\n  5v5 Test set log-loss: {round(test_logloss_5v5, 4)}"))
message(glue("  5v5 Test set AUC:      {round(test_auc_5v5, 4)}"))

############ SPECIAL TEAMS MODEL ########################

message(glue("\n{Sys.time()}: Preparing special teams model data..."))

# features (before one-hot encoding)
model_feats_st <- c(
    "shot_distance",
    "shot_angle",
    "shot_type",
    "rebound",
    "rush",
    "last_event_type",
    "time_since_last",
    "distance_from_last",
    "cross_ice_event",
    "empty_net",
    "last_x",
    "last_y",
    # new for uneven / short-even strengths
    "total_skaters_on",
    "event_team_advantage",
    # target
    "goal"
)

# Include only specific non-5v5 strength states
# Exclude 6v3 (too small sample, like hockeyR-models)
st_strengths <- c(
    "5v4",
    "5v3",
    "6v5",
    "6v4",
    "4v4",
    "4v3",
    "3v3",
    "4v5",
    "3v5",
    "5v6",
    "4v6",
    "3v4"
)

pbp_shots_st <- pbp_all %>%
    # filter to valid non-5v5 strength states
    filter(strength_state %in% st_strengths) %>%
    # filter out shootouts and OT shootout periods
    filter(period_type != "SHOOTOUT") %>%
    # remove penalty shots
    filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) %>%
    # add model feature variables
    group_by(game_id) %>%
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
            (x_fixed < -25 & event_team == home_team) |
                (x_fixed > 25 & event_team == away_team) ~ "DZ",
            (x_fixed > 25 & event_team == home_team) |
                (x_fixed < -25 & event_team == away_team) ~ "OZ"
        ),
        last_event_zone = lag(event_zone)
    ) %>%
    ungroup() %>%
    # filter to only unblocked shots (no BLOCKED_SHOT)
    filter(event_type %in% fenwick) %>%
    # valid last_event_type
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
    # add features
    mutate(
        # --- ERA DUMMIES ---
        era_2011_2013 = ifelse(
            season %in% c("20102011", "20112012", "20122013"),
            1,
            0
        ),
        era_2014_2018 = ifelse(
            season %in%
                c(
                    "20132014",
                    "20142015",
                    "20152016",
                    "20162017",
                    "20172018"
                ),
            1,
            0
        ),
        era_2019_2021 = ifelse(
            season %in% c("20182019", "20192020", "20202021"),
            1,
            0
        ),
        era_2022_2024 = ifelse(
            season %in% c("20212022", "20222023", "20232024"),
            1,
            0
        ),
        era_2025_on = ifelse(
            as.numeric(season) > 20232024,
            1,
            0
        ),
        # ST-specific features
        event_team_skaters = ifelse(
            event_team == home_team,
            home_skaters,
            away_skaters
        ),
        opponent_team_skaters = ifelse(
            event_team == home_team,
            away_skaters,
            home_skaters
        ),
        total_skaters_on = event_team_skaters + opponent_team_skaters,
        event_team_advantage = event_team_skaters - opponent_team_skaters,
        # tactical features
        rebound = ifelse(
            last_event_type %in% fenwick & time_since_last <= 2,
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
                ((lag(y) > 3 & y < -3) | (lag(y) < -3 & y > 3)) &
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
        starts_with("era"),
        all_of(model_feats_st)
    ) %>%
    # one-hot encode categoricals
    mutate(type_value = 1, last_value = 1) %>%
    pivot_wider(
        names_from = shot_type,
        values_from = type_value,
        values_fill = 0
    ) %>%
    pivot_wider(
        names_from = last_event_type,
        values_from = last_value,
        values_fill = 0,
        names_prefix = "last_"
    ) %>%
    clean_names() %>%
    select(-any_of("na"))

n_before_st <- nrow(pbp_shots_st)
pbp_shots_st <- na.omit(pbp_shots_st)
n_after_st <- nrow(pbp_shots_st)

message(glue("  ST rows before NA removal: {n_before_st}"))
message(glue("  ST rows after NA removal:  {n_after_st}"))
message(glue(
    "  Removed {n_before_st - n_after_st} rows ({round(100 * (n_before_st - n_after_st) / n_before_st, 2)}%)"
))
message(glue("  Goal rate: {round(mean(pbp_shots_st$goal), 4)}"))

# --- Train/test split ---

set.seed(37)

game_ids_st <- unique(pbp_shots_st$game_id)
n_train_st <- floor(0.8 * length(game_ids_st))
train_game_ids_st <- sample(game_ids_st, n_train_st)

train_shots_st <- pbp_shots_st %>% filter(game_id %in% train_game_ids_st)
test_shots_st <- pbp_shots_st %>% filter(game_id %not_in% train_game_ids_st)

message(glue(
    "  Train games: {n_train_st}, Test games: {length(game_ids_st) - n_train_st}"
))
message(glue(
    "  Train rows: {nrow(train_shots_st)}, Test rows: {nrow(test_shots_st)}"
))

# Grouped CV folds
train_game_ids_st_vec <- unique(train_shots_st$game_id)
fold_assignment_st <- sample(rep(
    1:5,
    length.out = length(train_game_ids_st_vec)
))
names(fold_assignment_st) <- train_game_ids_st_vec
train_shots_st$fold <- fold_assignment_st[train_shots_st$game_id]
folds_st <- lapply(1:5, function(k) which(train_shots_st$fold == k))
train_shots_st <- select(train_shots_st, -fold)

train_shots_st <- train_shots_st %>% select(-event_id, -season, -game_id)
test_shots_st <- test_shots_st %>% select(-event_id, -season, -game_id)

feat_names_st <- setdiff(names(train_shots_st), "goal")
message(glue(
    "  ST features ({length(feat_names_st)}): {paste(feat_names_st, collapse = ', ')}"
))

train_set_st <- train_shots_st %>% select(-goal) %>% data.matrix()
train_labels_st <- train_shots_st %>% select(goal) %>% data.matrix()
test_set_st <- test_shots_st %>% select(-goal) %>% data.matrix()
test_labels_st <- test_shots_st %>% select(goal) %>% data.matrix()

dtrain_st <- xgb.DMatrix(data = train_set_st, label = train_labels_st)
dtest_st <- xgb.DMatrix(data = test_set_st, label = test_labels_st)

############# ST: TRAIN FINAL MODEL ################

message(glue("\n{Sys.time()}: Training final special teams model..."))

params_st <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eval_metric = "auc",
    max_depth = 4,
    eta = .06,
    gamma = 1,
    subsample = .8,
    colsample_bytree = .8,
    min_child_weight = best_mcw_5v5$min_child_weight
)

cv_results_st <- xgb.cv(
    data = dtrain_st,
    params = params_st,
    folds = folds_st,
    nrounds = 1500,
    verbose = TRUE,
    print_every_n = 50,
    early_stopping_rounds = 30
)

saveRDS(cv_results_st, "data/cv_results_st_final.rds")

rounds_st <- cv_results_st$best_iteration
if (is.null(rounds_st)) {
    rounds_st <- nrow(cv_results_st$evaluation_log)
}
cv_logloss_st <- cv_results_st$evaluation_log$test_logloss_mean[rounds_st]
cv_auc_st <- cv_results_st$evaluation_log$test_auc_mean[rounds_st]

message(glue("  ST CV log-loss: {round(cv_logloss_st, 4)}"))
message(glue("  ST CV AUC:      {round(cv_auc_st, 4)}"))
message(glue("  Best nrounds:   {rounds_st}"))

# Train final model
xg_model_st <- xgb.train(
    data = dtrain_st,
    params = params_st,
    nrounds = rounds_st
)

# Feature importance
importance_st <- xgb.importance(feat_names_st, model = xg_model_st)

png(
    "figures/fastRhockey_xg_st_feature_importance.png",
    width = 8,
    height = 5,
    units = "in",
    res = 500
)
par(mar = c(5, 12, 3, 2))
imp_st_sorted <- importance_st[order(importance_st$Gain), ]
barplot(
    imp_st_sorted$Gain,
    names.arg = imp_st_sorted$Feature,
    horiz = TRUE,
    las = 1,
    col = "#99D9D9",
    border = "#001628",
    xlab = "Importance (Gain)",
    main = "fastRhockey Special Teams Expected Goals model feature importance",
    cex.names = 0.6
)
dev.off()

message("  Saved figures/fastRhockey_xg_st_feature_importance.png")

# Save model
xgb.save(xg_model_st, "models/xg_model_st.json")
message("  Saved models/xg_model_st.json")

# --- Evaluate on holdout test set ---
preds_st <- predict(xg_model_st, dtest_st)
test_logloss_st <- -mean(
    test_labels_st *
        log(pmax(preds_st, 1e-15)) +
        (1 - test_labels_st) * log(pmax(1 - preds_st, 1e-15))
)
test_auc_st <- .calc_auc(preds_st, as.vector(test_labels_st))

message(glue("\n  ST Test set log-loss: {round(test_logloss_st, 4)}"))
message(glue("  ST Test set AUC:      {round(test_auc_st, 4)}"))

############# SAVE METADATA #################

message(glue("\n{Sys.time()}: Saving metadata..."))

meta <- list(
    xg_feature_names_5v5 = feat_names_5v5,
    xg_feature_names_st = feat_names_st,
    xg_model_ps = ps_xg,
    training_date = Sys.time(),
    training_rows_5v5 = nrow(train_shots_5v5),
    training_rows_st = nrow(train_shots_st),
    nrounds_5v5 = rounds_5v5,
    nrounds_st = rounds_st,
    params_5v5 = params_5v5,
    params_st = params_st,
    cv_logloss_5v5 = cv_logloss_5v5,
    cv_auc_5v5 = cv_auc_5v5,
    cv_logloss_st = cv_logloss_st,
    cv_auc_st = cv_auc_st,
    test_logloss_5v5 = test_logloss_5v5,
    test_auc_5v5 = test_auc_5v5,
    test_logloss_st = test_logloss_st,
    test_auc_st = test_auc_st,
    eras = c(
        "era_2011_2013",
        "era_2014_2018",
        "era_2019_2021",
        "era_2022_2024",
        "era_2025_on"
    )
)

saveRDS(meta, "models/xg_model_meta.rds")
message("  Saved models/xg_model_meta.rds")

############# SUMMARY #################

message(glue("\n{'=' |> strrep(60)}"))
message("fastRhockey xG Model Training Summary")
message(glue("{'=' |> strrep(60)}"))
message(glue(""))
message(glue("| Model | CV Log-loss | CV AUC | Test Log-loss | Test AUC |"))
message(glue("|:-----:|:-----------:|:------:|:-------------:|:--------:|"))
message(glue(
    "|  5v5  |   {format(round(cv_logloss_5v5, 4), width = 6)}    |",
    " {format(round(cv_auc_5v5, 4), width = 6)} |",
    "     {format(round(test_logloss_5v5, 4), width = 6)}    |",
    "  {format(round(test_auc_5v5, 4), width = 6)} |"
))
message(glue(
    "|  ST   |   {format(round(cv_logloss_st, 4), width = 6)}    |",
    " {format(round(cv_auc_st, 4), width = 6)} |",
    "     {format(round(test_logloss_st, 4), width = 6)}    |",
    "  {format(round(test_auc_st, 4), width = 6)} |"
))
message(glue("|  PS   |     —       |   —    |       —       |    —     |"))
message(glue(""))
message(glue("Penalty shot xG constant: {round(ps_xg, 4)}"))
message(glue(""))
message(glue("Era groupings:"))
message(glue("  era_2011_2013 : 2010-11 through 2012-13"))
message(glue("  era_2014_2018 : 2013-14 through 2017-18"))
message(glue("  era_2019_2021 : 2018-19 through 2020-21"))
message(glue("  era_2022_2024 : 2021-22 through 2023-24"))
message(glue("  era_2025_on   : 2024-25 and beyond"))
message(glue(""))
message(glue("Files saved:"))
message(glue("  models/xg_model_5v5.json"))
message(glue("  models/xg_model_st.json"))
message(glue("  models/xg_model_meta.rds"))
message(glue("  data/cv_results_5v5_final.rds"))
message(glue("  data/cv_results_st_final.rds"))
message(glue("  figures/fastRhockey_xg_5v5_feature_importance.png"))
message(glue("  figures/fastRhockey_xg_st_feature_importance.png"))
message(glue(""))
message(glue("{Sys.time()}: Done!"))
