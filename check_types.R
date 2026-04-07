df1 <- readRDS("nhl/pbp/full/rds/play_by_play_2010_11.rds")
df14 <- readRDS("nhl/pbp/full/rds/play_by_play_2023_24.rds")
df16 <- readRDS("nhl/pbp/full/rds/play_by_play_2025_26.rds")

common_1_14 <- intersect(names(df1), names(df14))
types1 <- sapply(df1[common_1_14], function(x) paste(class(x), collapse = "/"))
types14 <- sapply(df14[common_1_14], function(x) {
    paste(class(x), collapse = "/")
})
mismatches_1_14 <- common_1_14[types1 != types14]
cat("Mismatches between season 1 and 14:\n")
for (col in mismatches_1_14) {
    cat("  ", col, ":", types1[col], "vs", types14[col], "\n")
}

common_1_16 <- intersect(names(df1), names(df16))
types1b <- sapply(df1[common_1_16], function(x) paste(class(x), collapse = "/"))
types16 <- sapply(df16[common_1_16], function(x) {
    paste(class(x), collapse = "/")
})
mismatches_1_16 <- common_1_16[types1b != types16]
cat("\nMismatches between season 1 and 16:\n")
for (col in mismatches_1_16) {
    cat("  ", col, ":", types1b[col], "vs", types16[col], "\n")
}

cat("\nDone.\n")
