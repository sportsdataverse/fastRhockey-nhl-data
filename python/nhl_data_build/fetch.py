"""Fetch fastRhockey-nhl-raw per-game finals over HTTP into a local cache.

Track B of the data-repo standard: a ``-data`` CI job fetches individual game
JSON over HTTP rather than checking out the raw sibling — ``nhl/json/final``
holds ~22k committed files and a checkout that works today is a timeout
waiting for the season that adds enough games.

Enumeration comes from the raw repo's per-season schedule parquet
(``nhl/schedules/parquet/nhl_schedule_{end_year}.parquet``) rather than a
directory listing: the GitHub contents API caps a directory at 1000 entries,
and ``final/`` is flat. The parquet is read with pyarrow — the files are
R/arrow-written and their latin1 key metadata trips polars' strict UTF-8
parquet reader.

The cache dir keeps the ``final/{gid}.json`` shape ``--final-dir`` already
expects, so ``season.py`` is untouched. Mirrors
``cfb_data_ingest/fetch.py::fetch_final``: read-through cache, corrupt-cache
guard, fail-soft per game with counters.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main"


def _default_opener(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except Exception:  # noqa: BLE001 — a missing/failed asset is a per-item skip
        return None


def season_game_ids(
    season_end_year: int,
    *,
    raw_base: str = RAW_BASE,
    opener: Callable[[str], bytes | None] | None = None,
) -> list[int]:
    """Game ids for one season from the raw repo's schedule parquet."""
    import pyarrow.parquet as pq

    op = opener or _default_opener
    url = f"{raw_base}/nhl/schedules/parquet/nhl_schedule_{season_end_year}.parquet"
    raw = op(url)
    if raw is None:
        raise FileNotFoundError(f"schedule parquet not found: {url}")
    table = pq.read_table(io.BytesIO(raw), columns=["game_id"])
    ids = sorted({int(v) for v in table.column("game_id").to_pylist() if v is not None})
    return ids


def fetch_finals(
    season_end_year: int,
    dest_dir: str | Path,
    *,
    raw_base: str = RAW_BASE,
    refresh: bool = False,
    opener: Callable[[str], bytes | None] | None = None,
) -> dict:
    """Fetch each ``final/{gid}.json`` for one season into ``dest_dir``.

    Fail-soft per game (a game with no final yet is ``missing``, never fatal);
    a cached file is revalidated with ``json.loads`` before being trusted, so a
    half-written entry from an interrupted run refetches instead of poisoning
    every later build.
    """
    op = opener or _default_opener
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    fetched = skipped = missing = 0
    ids = season_game_ids(season_end_year, raw_base=raw_base, opener=opener)
    for gid in ids:
        out = dest / f"{gid}.json"
        if out.exists() and not refresh:
            try:
                json.loads(out.read_text(encoding="utf-8"))
                skipped += 1
                continue
            except Exception:  # noqa: BLE001 — corrupt cache entry: refetch
                pass
        raw = op(f"{raw_base}/nhl/json/final/{gid}.json")
        if raw is None:
            missing += 1
            continue
        try:
            json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 — never persist an unparseable payload
            missing += 1
            continue
        out.write_bytes(raw)
        fetched += 1
    summary = {"fetched": fetched, "skipped": skipped, "missing": missing, "total": len(ids)}
    log.info("season %s: %s", season_end_year, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-s", "--start", type=int, required=True, help="start season end-year")
    ap.add_argument("-e", "--end", type=int, help="end season end-year (default: --start)")
    ap.add_argument("--dest", required=True, help="cache dir for final/{gid}.json")
    ap.add_argument("--raw-base", default=RAW_BASE)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = args.end or args.start
    for year in range(args.start, end + 1):
        summary = fetch_finals(year, args.dest, raw_base=args.raw_base, refresh=args.refresh)
        print(f"season {year}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
