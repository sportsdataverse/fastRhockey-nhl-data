#!/bin/bash
# Process NHL datasets from fastRhockey-nhl-raw repo
# Usage: bash scripts/daily_nhl_R_processor.sh -s 2025 -e 2025


# Commit + push, surviving a remote that moved while the build was running.
#
# Pulling BEFORE staging can only abort: the build has just rewritten tracked
# parquet/csv/json, so `git pull` refuses with "Your local changes would be
# overwritten by merge". The old form then committed anyway, pushed into a
# non-fast-forward rejection, and swallowed it -- a GREEN job that published
# nothing (wehoop-wnba-data 32192069433/32192069566, hoopR-nba-data 32204419012).
#
# Stage and commit FIRST so the tree is clean, then reconcile. `rebase --merge`
# rather than `pull --rebase`: the default am backend base64-encodes every blob
# it replays, which crawls on these binary-asset repos.
sdv_commit_push() {
  local msg="$1"; shift
  git add -- "$@" >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    echo "nothing to commit for: $msg"
    return 0
  fi
  git commit -m "$msg" >/dev/null || { echo "::warning ::commit failed: $msg"; return 1; }
  local attempt
  for attempt in 1 2 3; do
    if git push origin HEAD >/dev/null 2>&1; then
      echo "pushed: $msg (attempt $attempt)"
      return 0
    fi
    echo "push rejected (attempt $attempt); syncing with origin"
    git fetch --quiet origin main || true
    if ! git rebase --merge origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1 || true
      echo "::error ::cannot rebase onto origin/main for: $msg"
      return 1
    fi
  done
  echo "::error ::push still rejected after 3 attempts: $msg"
  return 1
}

while getopts s:e: flag
do
    case "${flag}" in
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
    esac
done

if [ -z "$START_YEAR" ] || [ -z "$END_YEAR" ]; then
    echo "Usage: $0 -s <start_year> -e <end_year>"
    exit 1
fi

mkdir -p logs
for i in $(seq "${START_YEAR}" "${END_YEAR}")
do
    LOGFILE="logs/fastRhockey_nhl_data_logfile_${i}.log"
    TMPLOG=$(mktemp "/tmp/fastRhockey_nhl_data_logfile_${i}.XXXXXX.log")
    echo "=== Processing NHL data for season $i ==="
    # Tee inside the block writes to /tmp (untracked) so the `git pull` calls
    # don't trip over their own log output being written to a tracked file.
    # The block sets ${RSCRIPT_RC} so we can surface a non-zero R exit code
    # to the workflow rather than silently masking it with a successful git
    # push.
    {
        git pull >> /dev/null
        git config --local user.email "action@github.com"
        git config --local user.name "Github Action"
        Rscript R/nhl_data_creation.R -s $i -e $i
        echo "RSCRIPT_RC=$?" > "/tmp/_rscript_rc_${i}"
        # If R failed, leave any partially-written files alone so the next
        # daily run can pick them up via the resilient per-dataset
        # tryCatch in the R script; still try the git plumbing so the
        # successful datasets get committed + pushed.
        sdv_commit_push "NHL Data Updated (Start: $i End: $i)" nhl || PUSH_RC=1
    } 2>&1 | tee "$TMPLOG"
    RSCRIPT_RC=$(cat "/tmp/_rscript_rc_${i}" 2>/dev/null | sed 's/RSCRIPT_RC=//')
    rm -f "/tmp/_rscript_rc_${i}"

    # Block is finished and pushed; tee has closed $TMPLOG. Now copy the log
    # into its tracked location and commit/push it on its own.
    cp "$TMPLOG" "$LOGFILE"
    git stash -u --quiet 2>/dev/null || true
    git pull --rebase >> /dev/null || true
    git stash pop --quiet 2>/dev/null || true
    sdv_commit_push "NHL Data log update (Start: $i End: $i)" $LOGFILE || PUSH_RC=1
    rm -f "$TMPLOG"

    # Propagate the R script's exit code so the workflow correctly reports
    # failure if a season compile errored. Don't `exit` immediately --
    # iterate the rest of the requested seasons first.
    if [ "${RSCRIPT_RC:-0}" != "0" ]; then
        echo "::error ::Rscript R/nhl_data_creation.R for season $i exited with code $RSCRIPT_RC"
        ANY_FAILED=1
    fi
done

# ---- Run summary: updated releases + remaining warnings/errors ----
# Prints a cli summary to the Action log and (when set) writes markdown to
# $GITHUB_STEP_SUMMARY so the run's Summary tab shows what landed and what didn't.
Rscript R/run_summary.R -s "$START_YEAR" -e "$END_YEAR" || true

if [ "${ANY_FAILED:-0}" != "0" ]; then
    echo "::error ::At least one season's Rscript exited non-zero. See per-season logs."
    exit 1
fi

# A rejected push is a FAILED run, not a green one. Release assets upload on a
# separate path and can succeed while the repo mirror is left stale.
if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale."
  exit 1
fi
