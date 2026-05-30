#!/bin/bash
# Process NHL datasets from fastRhockey-nhl-raw repo
# Usage: bash scripts/daily_nhl_R_processor.sh -s 2025 -e 2025

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
        git pull >> /dev/null
        git add nhl/* >> /dev/null
        git pull >> /dev/null
        git add . >> /dev/null
        git commit -m "NHL Data Updated (Start: $i End: $i)" || echo "No changes to commit"
        git pull >> /dev/null
        git push >> /dev/null
    } 2>&1 | tee "$TMPLOG"
    RSCRIPT_RC=$(cat "/tmp/_rscript_rc_${i}" 2>/dev/null | sed 's/RSCRIPT_RC=//')
    rm -f "/tmp/_rscript_rc_${i}"

    # Block is finished and pushed; tee has closed $TMPLOG. Now copy the log
    # into its tracked location and commit/push it on its own.
    cp "$TMPLOG" "$LOGFILE"
    git stash -u --quiet 2>/dev/null || true
    git pull --rebase >> /dev/null || true
    git stash pop --quiet 2>/dev/null || true
    git add "$LOGFILE"
    git commit -m "NHL Data log update (Start: $i End: $i)" >> /dev/null || echo "No log changes to commit"
    git push >> /dev/null
    rm -f "$TMPLOG"

    # Propagate the R script's exit code so the workflow correctly reports
    # failure if a season compile errored. Don't `exit` immediately --
    # iterate the rest of the requested seasons first.
    if [ "${RSCRIPT_RC:-0}" != "0" ]; then
        echo "::error ::Rscript R/nhl_data_creation.R for season $i exited with code $RSCRIPT_RC"
        ANY_FAILED=1
    fi
done

if [ "${ANY_FAILED:-0}" != "0" ]; then
    echo "::error ::At least one season's Rscript exited non-zero. See per-season logs."
    exit 1
fi
