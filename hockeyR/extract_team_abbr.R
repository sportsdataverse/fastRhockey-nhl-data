## extract_team_abbr.R
## Extracts team_abbr_yearly from old sysdata.rda without triggering xgboost
## Run BEFORE any xgboost library is loaded!

cat("Attempting to extract team_abbr_yearly from sysdata.rda\n")
cat("xgboost NOT loaded yet\n")

# Load the rda into a dedicated environment
env <- new.env(parent = emptyenv())
tryCatch(
    load("R/sysdata.rda", envir = env),
    error = function(e) cat("load() error:", conditionMessage(e), "\n")
)
cat("Objects found:", paste(ls(env), collapse = ", "), "\n")

# Access only the safe objects
if (exists("team_abbr_yearly", envir = env)) {
    team_abbr_yearly <- get("team_abbr_yearly", envir = env)
    cat("team_abbr_yearly rows:", nrow(team_abbr_yearly), "\n")
    cat("columns:", paste(names(team_abbr_yearly), collapse = ", "), "\n")
    head_str <- capture.output(head(team_abbr_yearly, 5))
    cat(paste(head_str, collapse = "\n"), "\n")
    saveRDS(team_abbr_yearly, "R/team_abbr_yearly.rds")
    cat("Saved to R/team_abbr_yearly.rds\n")
} else {
    cat("team_abbr_yearly not found in sysdata.rda\n")
}

if (exists("xg_model_ps", envir = env)) {
    xg_model_ps <- get("xg_model_ps", envir = env)
    cat("xg_model_ps:", xg_model_ps, "\n")
    saveRDS(xg_model_ps, "R/xg_model_ps.rds")
    cat("Saved to R/xg_model_ps.rds\n")
}

cat("Done\n")
