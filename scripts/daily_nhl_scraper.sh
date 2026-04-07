#!/bin/bash
# Scrape NHL play-by-play data for one or more seasons using fastRhockey
# Usage: bash scripts/daily_nhl_scraper.sh -s 2024 -e 2025
#   -s START_YEAR : first season start year to scrape (e.g. 2024 for 2024-25)
#   -e END_YEAR   : last season start year to scrape (e.g. 2025 for 2025-26)
#   -r RESCRAPE   : (optional) unused, kept for backward compatibility

while getopts s:e:r: flag
do
    case "${flag}" in
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
        r) RESCRAPE=${OPTARG};;
    esac
done

if [ -z "$START_YEAR" ] || [ -z "$END_YEAR" ]; then
    echo "Usage: $0 -s <start_year> -e <end_year> [-r <rescrape>]"
    exit 1
fi

git pull > /dev/null

for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    echo "--- Scraping season ${year}-$(printf '%02d' $(( (year + 1) % 100 ))) ---"
    Rscript R/scrape_full_season.R "$year"
done

git add . > /dev/null
git pull > /dev/null
git commit -m "NHL Play-by-Play and Schedules update (Start: $START_YEAR End: $END_YEAR)" > /dev/null || echo "No changes to commit"
git pull > /dev/null
git push > /dev/null
