rds_dir <- "nhl/pbp/full/rds"
rds_files <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)

fenwick <- c("SHOT", "MISSED_SHOT", "GOAL")

for (f in rds_files) {
    df <- readRDS(f)
    cat("\n===", basename(f), "===\n")
    if ("secondary_type" %in% names(df)) {
        mask <- df$event_type %in% fenwick
        vals <- sort(unique(df$secondary_type[mask]))
        cat("  shot types:", paste(vals, collapse = " | "), "\n")
    }
    if ("event_type" %in% names(df)) {
        cat(
            "  event types:",
            paste(sort(unique(df$event_type)), collapse = " | "),
            "\n"
        )
    }
}
