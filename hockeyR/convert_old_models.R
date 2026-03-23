## convert_old_models.R
## Converts old xgboost model objects to modern xgboost JSON format
## Using raw bytes extracted from R's serialization format

.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")

library(xgboost)
cat("xgboost version:", as.character(packageVersion("xgboost")), "\n\n")

# Load the raw serialized R objects
cat("Loading raw serialized bytes...\n")
raw_5v5 <- readRDS("R/xg_model_5v5_raw.rds")
raw_st <- readRDS("R/xg_model_st_raw.rds")
cat("5v5 bytes:", length(raw_5v5), "\n")
cat("ST bytes:", length(raw_st), "\n\n")

# The raw_5v5 is base::serialize() output — it's an R serialization of the
# xgb.Booster R6 object. We need to unserialize it to get the R6 object back.
# Then extract the internal 'raw' or 'xgb_model' field with the model bytes.

cat("Deserializing 5v5 model R object...\n")
tryCatch(
    {
        obj_5v5 <- unserialize(raw_5v5)
        cat("Class:", class(obj_5v5), "\n")
        cat("Names/fields:", paste(names(obj_5v5), collapse = ", "), "\n")

        # In old xgboost R6 objects, the model bytes are in obj$raw (or .__enclos_env__$private$raw)
        # Try various access methods
        if (!is.null(obj_5v5$raw)) {
            cat("Found obj$raw field, length:", length(obj_5v5$raw), "\n")
            model_bytes_5v5 <- obj_5v5$raw
        } else if (!is.null(obj_5v5$xgb_model)) {
            cat(
                "Found obj$xgb_model field, length:",
                length(obj_5v5$xgb_model),
                "\n"
            )
            model_bytes_5v5 <- obj_5v5$xgb_model
        } else {
            cat("Could not find model bytes inside R6 object\n")
            cat(
                "All accessible names:",
                paste(names(obj_5v5), collapse = ", "),
                "\n"
            )
            model_bytes_5v5 <- NULL
        }

        if (!is.null(model_bytes_5v5)) {
            cat("Trying xgb.load.raw (no format arg) ...\n")
            xg_model_5v5 <- tryCatch(
                xgb.load.raw(model_bytes_5v5),
                error = function(e) {
                    cat("xgb.load.raw error:", conditionMessage(e), "\n")
                    NULL
                }
            )
            if (!is.null(xg_model_5v5)) {
                cat(
                    "SUCCESS: xg_model_5v5 loaded! Features:",
                    length(xg_model_5v5$feature_names),
                    "\n"
                )
                xgb.save(xg_model_5v5, "inst/extdata/xg_model_5v5.json")
                cat("Saved to inst/extdata/xg_model_5v5.json\n")
            } else {
                # Try treating raw bytes as saved xgb format (from xgb.save.raw of old version)
                cat("Trying xgb.load.raw with ubj format...\n")
                xg_model_5v5 <- tryCatch(
                    xgb.load.raw(model_bytes_5v5, raw_format = "ubj"),
                    error = function(e) {
                        cat("ubj error:", conditionMessage(e), "\n")
                        NULL
                    }
                )
                if (!is.null(xg_model_5v5)) {
                    xgb.save(xg_model_5v5, "inst/extdata/xg_model_5v5.json")
                    cat("Saved to inst/extdata/xg_model_5v5.json\n")
                }
            }
        }
    },
    error = function(e) {
        cat("Error:", conditionMessage(e), "\n")
    }
)

cat("\nDeserializing ST model R object...\n")
tryCatch(
    {
        obj_st <- unserialize(raw_st)
        cat("Class:", class(obj_st), "\n")

        if (!is.null(obj_st$raw)) {
            model_bytes_st <- obj_st$raw
            cat("Found obj$raw field, length:", length(model_bytes_st), "\n")
        } else if (!is.null(obj_st$xgb_model)) {
            model_bytes_st <- obj_st$xgb_model
            cat(
                "Found obj$xgb_model field, length:",
                length(model_bytes_st),
                "\n"
            )
        } else {
            model_bytes_st <- NULL
            cat("Could not find model bytes inside R6 object\n")
        }

        if (!is.null(model_bytes_st)) {
            cat("Trying xgb.load.raw (no format arg) ...\n")
            xg_model_st <- tryCatch(
                xgb.load.raw(model_bytes_st),
                error = function(e) {
                    cat("xgb.load.raw error:", conditionMessage(e), "\n")
                    NULL
                }
            )
            if (!is.null(xg_model_st)) {
                cat(
                    "SUCCESS: xg_model_st loaded! Features:",
                    length(xg_model_st$feature_names),
                    "\n"
                )
                xgb.save(xg_model_st, "inst/extdata/xg_model_st.json")
                cat("Saved to inst/extdata/xg_model_st.json\n")
            } else {
                cat("Trying xgb.load.raw with ubj format...\n")
                xg_model_st <- tryCatch(
                    xgb.load.raw(model_bytes_st, raw_format = "ubj"),
                    error = function(e) {
                        cat("ubj error:", conditionMessage(e), "\n")
                        NULL
                    }
                )
                if (!is.null(xg_model_st)) {
                    xgb.save(xg_model_st, "inst/extdata/xg_model_st.json")
                    cat("Saved to inst/extdata/xg_model_st.json\n")
                }
            }
        }
    },
    error = function(e) {
        cat("Error:", conditionMessage(e), "\n")
    }
)

cat("\nDone\n")
