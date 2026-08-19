#!/bin/bash
# Compile NHL datasets with the Python reshaper (python/nhl_data_build).
#
# Drop-in replacement for daily_nhl_R_processor.sh: same -s/-e contract, same
# per-season commit subject, same exit-code propagation, so sdv-orch's
# `data.build` stage can call either one.
#
#   bash scripts/daily_nhl_python_processor.sh -s 2026 -e 2026
#
# SECOND CALL SITE: .github/workflows/daily_nhl_python.yml runs the same two
# stages (nhl_data_build.season -> publish.publish_season) on the GH runner.
# The stage sequence is duplicated on purpose -- the git/env plumbing genuinely
# differs (that workflow sparse-checks-out nhl-raw and pushes with an explicit
# token; this script uses an absolute venv path because systemd's PATH has no
# `uv`). Change the stages in BOTH, or CI and the droplet will drift apart.
#
# Memory note: a full season compile was OOM-killed here on 2026-07-22 at 13.7GB
# RSS, which blocked this script until sportsdataverse-py #296. The compile itself
# was never the problem -- build_season batches 250 games and the reader streams,
# putting it at a 5.46GB peak. The kill was inside write_rds, which buffered the
# entire serialized frame (one small bytes object per value) and then joined that
# into a second contiguous copy: ~6.8GB for the 1.1M x 94 pbp frame. #296 streams
# it instead, and the same compile now finishes at a 2.60GB write-phase peak.
#
# Keep an eye on it if NHL datasets grow substantially: peak now tracks the
# largest single dataset held as a frame, not the serialized stream.
#
# Reads the raw finals from the sibling fastRhockey-nhl-raw checkout (the
# `raw.scrape` stage runs first and self-commits there), writes parquet into
# nhl/, and uploads parquet + rds + csv to the nhl_* releases. rds/csv are
# gitignored -- the releases are their distribution channel.

set -uo pipefail


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

while getopts s:e: flag; do
    case "${flag}" in
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
        *) echo "Usage: $0 -s <start_year> -e <end_year>"; exit 1;;
    esac
done

if [ -z "${START_YEAR:-}" ] || [ -z "${END_YEAR:-}" ]; then
    echo "Usage: $0 -s <start_year> -e <end_year>"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS_ROOT="${SDV_REPOS:-/mnt/sdv_repos}"
FINAL_DIR="${NHL_RAW_FINAL_DIR:-${REPOS_ROOT}/fastRhockey-nhl-raw/nhl/json/final}"
OUT_DIR="${REPO_DIR}/nhl"

# Call the project venv's interpreter by absolute path rather than going through
# `uv run`. sdv-orch invokes this from a systemd unit, whose PATH is the systemd
# default and does NOT include /root/.local/bin where uv lives -- `uv` exits 127
# there while working fine in an interactive shell.
PYBIN="${NHL_PYBIN:-${REPO_DIR}/python/.venv/bin/python}"

# Fail before touching git if the upstream checkout isn't where we expect. A
# missing final dir would otherwise compile zero games and "succeed", quietly
# publishing nothing.
if [ ! -d "${FINAL_DIR}" ]; then
    echo "::error ::raw finals not found at ${FINAL_DIR}"
    exit 1
fi

if [ ! -x "${PYBIN}" ]; then
    echo "::error ::python venv not found at ${PYBIN} -- run 'uv sync' in ${REPO_DIR}/python"
    exit 1
fi

cd "${REPO_DIR}" || exit 1
mkdir -p logs

ANY_FAILED=0
for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
    LOGFILE="logs/fastRhockey_nhl_data_logfile_${i}.log"
    TMPLOG=$(mktemp "/tmp/fastRhockey_nhl_data_logfile_${i}.XXXXXX.log")
    echo "=== Processing NHL data for season $i (Python) ==="

    # Tee inside the block writes to /tmp (untracked) so the `git pull` calls
    # don't trip over their own log output being written to a tracked file.
    {
        git pull >> /dev/null
        git config --local user.email "action@github.com"
        git config --local user.name "Github Action"

        # nhl_data_build is not installed into the venv, so it only imports with
        # python/ as cwd -- hence the subshell. Paths are absolute for that reason.
        ( cd python && "${PYBIN}" -m nhl_data_build.season \
                -s "$i" -e "$i" --final-dir "${FINAL_DIR}" --out-dir "${OUT_DIR}" )
        echo "COMPILE_RC=$?" > "/tmp/_nhl_compile_rc_${i}"

        # Publish only what compiled. Uploading is idempotent (--clobber), so a
        # partial season still ships the datasets that built.
        ( cd python && "${PYBIN}" -c "
from nhl_data_build.publish import publish_season
print(len(publish_season('${OUT_DIR}', ${i})), 'assets uploaded')
" )

        sdv_commit_push "NHL Data Updated (Start: $i End: $i)" nhl || PUSH_RC=1
    } 2>&1 | tee "$TMPLOG"

    COMPILE_RC=$(sed 's/COMPILE_RC=//' "/tmp/_nhl_compile_rc_${i}" 2>/dev/null)
    rm -f "/tmp/_nhl_compile_rc_${i}"

    cp "$TMPLOG" "$LOGFILE"
    git stash -u --quiet 2>/dev/null || true
    git pull --rebase >> /dev/null || true
    git stash pop --quiet 2>/dev/null || true
    sdv_commit_push "NHL Data log update (Start: $i End: $i)" $LOGFILE || PUSH_RC=1
    rm -f "$TMPLOG"

    # Surface a failed compile rather than masking it with a successful push;
    # finish the remaining seasons first.
    if [ "${COMPILE_RC:-0}" != "0" ]; then
        echo "::error ::nhl_data_build.season for season $i exited with code ${COMPILE_RC}"
        ANY_FAILED=1
    fi
done

if [ "${ANY_FAILED}" != "0" ]; then
    echo "::error ::At least one season's compile exited non-zero. See per-season logs."
    exit 1
fi

# A rejected push is a FAILED run, not a green one. Release assets upload on a
# separate path and can succeed while the repo mirror is left stale.
if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale."
  exit 1
fi
