## extract_model_raw_bytes.R
## Attempts to extract raw bytes from old xgboost models stored in sysdata.rda
## WITHOUT triggering the xgboost C++ deserialization that causes segfault

cat("Attempting to extract raw model bytes from old sysdata.rda\n")
cat("xgboost version:", as.character(packageVersion("xgboost")), "\n\n")

env <- new.env(parent = emptyenv())
load("R/sysdata.rda", envir = env)
cat("Loaded. Objects:", paste(ls(env), collapse = ", "), "\n\n")

# Try to re-serialize the model objects to raw bytes
# This calls the $serialize hook of the xgb.Booster R6 object
# If lucky, we get the old compressed model bytes we can load with xgb.load.raw()

cat("Attempting to serialize xg_model_5v5 to raw bytes...\n")
tryCatch(
    {
        raw_5v5 <- base::serialize(env$xg_model_5v5, NULL)
        cat("Serialized 5v5 model:", length(raw_5v5), "bytes\n")
        saveRDS(raw_5v5, "R/xg_model_5v5_raw.rds")
        cat("Saved raw bytes to R/xg_model_5v5_raw.rds\n")
    },
    error = function(e) {
        cat("Error:", conditionMessage(e), "\n")
    }
)

cat("\nAttempting to serialize xg_model_st to raw bytes...\n")
tryCatch(
    {
        raw_st <- base::serialize(env$xg_model_st, NULL)
        cat("Serialized ST model:", length(raw_st), "bytes\n")
        saveRDS(raw_st, "R/xg_model_st_raw.rds")
        cat("Saved raw bytes to R/xg_model_st_raw.rds\n")
    },
    error = function(e) {
        cat("Error:", conditionMessage(e), "\n")
    }
)

cat("\nDone\n")
