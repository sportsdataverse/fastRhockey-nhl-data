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

for i in $(seq "${START_YEAR}" "${END_YEAR}")
do
    echo "=== Processing NHL data for season $i ==="
    git pull >> /dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    Rscript R/nhl_data_creation.R -s $i -e $i
    git pull >> /dev/null
    git add nhl/* >> /dev/null
    git add fastRhockey_nhl_data_logfile.txt >> /dev/null
    git pull >> /dev/null
    git add . >> /dev/null
    git commit -m "NHL Data Updated (Start: $i End: $i)" || echo "No changes to commit"
    git pull >> /dev/null
    git push >> /dev/null
done
