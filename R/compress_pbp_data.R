# compress_pbp_data.R
# Re-saves all .rds files in fastRhockey-nhl-data/nhl/pbp with xz compression
# (xz provides the best compression ratio; these are archival files so
#  the slower write speed is acceptable)

pbp_root <- file.path(
    getwd(),
    "nhl",
    "pbp"
)

files <- list.files(
    pbp_root,
    pattern = "\\.rds$",
    recursive = TRUE,
    full.names = TRUE
)

cat(sprintf("Found %d .rds files under %s\n\n", length(files), pbp_root))
library(dplyr)
total_before <- 0
total_after <- 0


for (f in files) {
    size_before <- file.size(f)
    total_before <- total_before + size_before

    data <- readRDS(f)
    saveRDS(data, f, compress = "xz")

    size_after <- file.size(f)
    total_after <- total_after + size_after

    cat(sprintf(
        "%-55s  %5.1f MB -> %5.1f MB  (%+.0f%%)\n",
        basename(f),
        size_before / 1e6,
        size_after / 1e6,
        (size_after - size_before) / size_before * 100
    ))
}

cat(sprintf(
    "\nTotal: %.1f MB -> %.1f MB  (saved %.1f MB, %+.0f%%)\n",
    total_before / 1e6,
    total_after / 1e6,
    (total_before - total_after) / 1e6,
    (total_after - total_before) / total_before * 100
))
