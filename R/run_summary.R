#!/usr/bin/env Rscript
# Run summary: parse the per-season tracked logs written by the daily processor
# and emit (a) a cli-formatted summary to stdout -- visible in the GitHub Action
# logs -- and (b) a markdown summary to $GITHUB_STEP_SUMMARY when set, so the run
# Summary tab shows which releases updated plus any remaining warnings/errors.
#
# Usage: Rscript R/run_summary.R -s 2025 -e 2025
# League + log prefix are auto-detected from logs/<prefix>_logfile_<year>.log.

suppressPackageStartupMessages({
  library(optparse)
  library(cli)
  library(glue)
})

option_list <- list(
  make_option(c("-s", "--start_year"), type = "integer", default = NA),
  make_option(c("-e", "--end_year"), type = "integer", default = NA)
)
opt <- parse_args(OptionParser(option_list = option_list))

# --- detect the data-log prefix (e.g. fastRhockey_nhl_data) -------------------
all_logs <- list.files("logs", pattern = "_data_logfile_[0-9]{4}\\.log$",
                       full.names = FALSE)
if (length(all_logs) == 0) {
  cli::cli_alert_info("No data logs found in logs/; nothing to summarize.")
  quit(status = 0)
}
prefix <- sub("_logfile_[0-9]{4}\\.log$", "", all_logs[1])
m <- regmatches(prefix, regexec("^fastRhockey_([a-z]+)_data$", prefix))[[1]]
league <- if (length(m) == 2) toupper(m[2]) else "fastRhockey"

# Seasons: explicit range, else every season that has a log on disk.
if (!is.na(opt$s) && !is.na(opt$e)) {
  seasons <- opt$s:opt$e
} else {
  seasons <- sort(as.integer(sub(
    paste0("^", prefix, "_logfile_([0-9]{4})\\.log$"), "\\1", all_logs
  )))
}

# --- helpers ------------------------------------------------------------------
# Releases that uploaded: `sportsdataverse_save` prints
#   "Uploaded <n> to sportsdataverse/sportsdataverse-data @ <release_tag>".
extract_uploaded <- function(lines) {
  m <- regmatches(lines, regexpr("Uploaded [0-9]+ to \\S+ @ \\S+", lines))
  m <- m[nzchar(m)]
  tags <- sub("^Uploaded [0-9]+ to \\S+ @ (\\S+).*$", "\\1", m)
  sort(unique(tags))
}

# Per-dataset row counts: the compile prints "<key>: <n> rows".
extract_rows <- function(lines) {
  m <- regmatches(lines, regexpr("^[a-z_0-9]+: [0-9]+ rows$", lines))
  m[nzchar(m)]
}

# Datasets with no data: "<key>: 0 rows -> skipped".
extract_skipped <- function(lines) {
  m <- regmatches(lines, regexpr("^[a-z_0-9]+: 0 rows -> skipped$", lines))
  sub(": 0 rows -> skipped$", "", m[nzchar(m)])
}

# cli warnings render as "! <message>"; also catch the upload-failure and
# missing-release warnings explicitly.
extract_warnings <- function(lines) {
  w <- lines[
    grepl("^!\\s", lines) |
      grepl("Failed to upload", lines) |
      grepl("does not exist on sportsdataverse-data", lines) |
      grepl("pipeline failed", lines, ignore.case = TRUE)
  ]
  trimws(unique(w))
}

extract_errors <- function(lines) {
  e <- lines[
    grepl("Execution halted", lines) |
      grepl("^Error[: ]", lines) |
      grepl("^✖|^x ", lines) |
      grepl("::error", lines) |
      grepl("Could not fetch schedule", lines)
  ]
  trimws(unique(e))
}

# --- per-season report --------------------------------------------------------
cli::cli_h1("{league} Data — Run Summary ({min(seasons)}-{max(seasons)})")

tot_up <- 0L
tot_w <- 0L
tot_e <- 0L
md <- c(glue("## {league} Data — Run Summary ({min(seasons)}-{max(seasons)})"), "")

for (y in seasons) {
  logf <- glue("logs/{prefix}_logfile_{y}.log")
  if (!file.exists(logf)) {
    cli::cli_alert_info("Season {y}: no log on disk — skipped")
    next
  }
  lines <- readLines(logf, warn = FALSE)
  uploaded <- extract_uploaded(lines)
  rows <- extract_rows(lines)
  skipped <- extract_skipped(lines)
  warns <- extract_warnings(lines)
  errs <- extract_errors(lines)
  tot_up <- tot_up + length(uploaded)
  tot_w <- tot_w + length(warns)
  tot_e <- tot_e + length(errs)

  cli::cli_h2("Season {y}")
  if (length(uploaded)) {
    cli::cli_alert_success("{length(uploaded)} releases uploaded: {paste(uploaded, collapse = ', ')}")
  } else {
    cli::cli_alert_danger("No releases uploaded")
  }
  if (length(rows)) cli::cli_alert_info("Row counts: {paste(rows, collapse = ' | ')}")
  if (length(skipped)) cli::cli_alert_warning("{length(skipped)} dataset(s) with 0 rows: {paste(skipped, collapse = ', ')}")
  if (length(warns)) {
    cli::cli_alert_warning("{length(warns)} warning line(s):")
    cli::cli_ul(utils::head(warns, 8))
  }
  if (length(errs)) {
    cli::cli_alert_danger("{length(errs)} error line(s):")
    cli::cli_ul(utils::head(errs, 5))
  }
  if (!length(warns) && !length(errs)) cli::cli_alert_success("No warnings or errors")

  md <- c(md,
    glue("### Season {y}"),
    glue("- ✅ **{length(uploaded)} releases uploaded**: {ifelse(length(uploaded), paste(uploaded, collapse = ', '), '_none_')}"),
    if (length(skipped)) glue("- ℹ️ {length(skipped)} dataset(s) with 0 rows: {paste(skipped, collapse = ', ')}") else NULL,
    if (length(warns)) c(glue("- ⚠️ **{length(warns)} warning line(s)**:"),
                         paste0("  - `", utils::head(warns, 8), "`")) else glue("- ✅ no warnings"),
    if (length(errs)) c(glue("- ❌ **{length(errs)} error line(s)**:"),
                        paste0("  - `", utils::head(errs, 5), "`")) else glue("- ✅ no errors"),
    ""
  )
}

cli::cli_rule()
status_fn <- if (tot_e > 0) cli::cli_alert_danger else if (tot_w > 0) cli::cli_alert_warning else cli::cli_alert_success
status_fn("Totals across {length(seasons)} season(s): {tot_up} releases uploaded, {tot_w} warning line(s), {tot_e} error line(s)")

md <- c(md, "---",
  glue("**Totals across {length(seasons)} season(s):** {tot_up} releases uploaded · {tot_w} warning line(s) · {tot_e} error line(s)"))

gh <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(gh)) {
  tryCatch(writeLines(md, gh), error = function(e) {})
}
