library(hockeyR)
cat("hockeyR loaded OK\n")
# Access internal objects via package namespace
ns <- asNamespace("hockeyR")
cat("xg_feature_names_5v5:", length(ns$xg_feature_names_5v5), "features\n")
cat("xg_feature_names_st:", length(ns$xg_feature_names_st), "features\n")
cat("xg_model_5v5 class:", class(ns$xg_model_5v5), "\n")
cat("Testing scrape_game for one game (2022030175)...\n")
result <- tryCatch(
    scrape_game(2022030175),
    error = function(e) {
        cat("ERROR:", conditionMessage(e), "\n")
        NULL
    }
)
if (!is.null(result)) {
    cat("Rows:", nrow(result), "\n")
    cat("xg col exists:", "xg" %in% names(result), "\n")
    cat("Non-NA xg:", sum(!is.na(result$xg)), "\n")
    cat("xg range:", round(range(result$xg, na.rm = TRUE), 4), "\n")
    cat("SUCCESS\n")
} else {
    cat("FAILED - result was NULL\n")
}
