#!/bin/bash
# Compile NHL datasets with the Python reshaper (python/nhl_data_build).
#
# Drop-in replacement for daily_nhl_R_processor.sh: same -s/-e contract, same
# per-season commit subject, same exit-code propagation, so sdv-orch's
# `data.build` stage can call either one.
#
#   bash scripts/daily_nhl_python_processor.sh -s 2026 -e 2026
#
# ⚠ NOT WIRED INTO sdv-orch YET -- pending a dependency bump. A full 2026 compile
# was OOM-killed on the droplet on 2026-07-22 at 13.7GB RSS:
#   Out of memory: Killed process (python3) total-vm:31471552kB anon-rss:13774772kB
#
# The compile is not the problem -- profiling puts build_season (which already
# batches 250 games and streams the reader) at a 5.46GB peak. The kill was inside
# sportsdataverse write_rds: it buffered the entire serialized frame as one small
# bytes object per value, then joined that into a second contiguous copy, costing
# ~6.8GB for the 1.1M x 94 pbp frame. Streaming fix in sdv-py PR #296 brings it to
# -0.03GB with a 2.60GB write-phase peak.
#
# To finish the cutover: merge #296, re-pin sportsdataverse in python/pyproject.toml
# + uv.lock, verify a full-season run, then point sdv-orch's data.build stage here.
# Until then the R processor remains the scheduled path.
#
# Reads the raw finals from the sibling fastRhockey-nhl-raw checkout (the
# `raw.scrape` stage runs first and self-commits there), writes parquet into
# nhl/, and uploads parquet + rds + csv to the nhl_* releases. rds/csv are
# gitignored -- the releases are their distribution channel.

set -uo pipefail

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

        git pull >> /dev/null
        git add nhl >> /dev/null
        git commit -m "NHL Data Updated (Start: $i End: $i)" || echo "No changes to commit"
        git pull >> /dev/null
        git push >> /dev/null
    } 2>&1 | tee "$TMPLOG"

    COMPILE_RC=$(sed 's/COMPILE_RC=//' "/tmp/_nhl_compile_rc_${i}" 2>/dev/null)
    rm -f "/tmp/_nhl_compile_rc_${i}"

    cp "$TMPLOG" "$LOGFILE"
    git stash -u --quiet 2>/dev/null || true
    git pull --rebase >> /dev/null || true
    git stash pop --quiet 2>/dev/null || true
    git add "$LOGFILE"
    git commit -m "NHL Data log update (Start: $i End: $i)" >> /dev/null || echo "No log changes to commit"
    git push >> /dev/null
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
